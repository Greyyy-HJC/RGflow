"""Training and held-out evaluation for conditional inverse-blocking flows."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from .action import Phi4Action
from .affine_flow import AffineFlowConfig, ConditionalAffineFlow, model_from_checkpoint
from .blocking import apply_kernel, inverse_kernel, kernel_transform
from .inverse_blocking import assemble_details, extract_details
from .operator_diagnostics import (
    KERNEL_OBSERVABLE_NAMES,
    distribution_metrics,
    kernel_observable_series,
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def load_configurations(path: Path) -> np.ndarray:
    with np.load(path) as data:
        configurations = np.asarray(data["configurations"], dtype=np.float64)
    if configurations.ndim != 4 or configurations.shape[0] < 4:
        raise ValueError(f"{path} must contain at least four chains")
    return configurations


def load_kernel(path: Path) -> tuple[np.ndarray, dict[str, Any], str]:
    raw = path.read_bytes()
    metadata = json.loads(raw)
    kernel = np.asarray(metadata["matrix"], dtype=np.float64)
    return kernel, metadata, hashlib.sha256(raw).hexdigest()


def _augment_psi(psi: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    augmented = np.asarray(psi)
    rotations = int(rng.integers(4))
    augmented = np.rot90(augmented, rotations, axes=(-2, -1))
    if rng.random() < 0.5:
        augmented = np.flip(augmented, axis=-2)
    shift = rng.integers(0, augmented.shape[-1], size=2)
    augmented = np.roll(augmented, tuple(int(x) for x in shift), axis=(-2, -1))
    if rng.random() < 0.5:
        augmented = -augmented
    return np.ascontiguousarray(augmented)


def _augment_coarse(coarse: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    augmented = np.asarray(coarse)
    augmented = np.rot90(augmented, int(rng.integers(4)), axes=(-2, -1))
    if rng.random() < 0.5:
        augmented = np.flip(augmented, axis=-1)
    shift = rng.integers(0, augmented.shape[-1], size=2)
    augmented = np.roll(augmented, tuple(int(x) for x in shift), axis=(-2, -1))
    if rng.random() < 0.5:
        augmented = -augmented
    return np.ascontiguousarray(augmented)


def _torch_assemble(coarse: Tensor, details: Tensor) -> Tensor:
    size = coarse.shape[-1]
    psi = coarse.new_empty((coarse.shape[0], 2 * size, 2 * size))
    psi[:, 0::2, 0::2] = coarse
    psi[:, 0::2, 1::2] = details[:, 0]
    psi[:, 1::2, 0::2] = details[:, 1]
    psi[:, 1::2, 1::2] = details[:, 2]
    return psi


def _torch_inverse(psi: Tensor, kernel_fft: Tensor) -> Tensor:
    transformed = torch.fft.fft2(psi)
    return torch.fft.ifft2(transformed / kernel_fft).real


def _torch_action(field: Tensor, action: Phi4Action) -> Tensor:
    field_sq = field.square()
    potential = field_sq + action.lam * (field_sq - 1.0).square()
    hopping = field * torch.roll(field, -1, dims=-2)
    hopping += field * torch.roll(field, -1, dims=-1)
    return (potential - 2.0 * action.kappa * hopping).sum(dim=(-2, -1))


def _torch_observables(psi: Tensor, fine: Tensor, action: Phi4Action) -> Tensor:
    axes = (-2, -1)
    psi2 = psi.square().mean(dim=axes)
    psi4 = psi.pow(4).mean(dim=axes)
    c10 = 0.5 * (
        (psi * torch.roll(psi, -1, dims=-2)).mean(dim=axes)
        + (psi * torch.roll(psi, -1, dims=-1)).mean(dim=axes)
    )
    c11 = (
        psi * torch.roll(torch.roll(psi, -1, dims=-2), -1, dims=-1)
    ).mean(dim=axes)
    c20 = 0.5 * (
        (psi * torch.roll(psi, -2, dims=-2)).mean(dim=axes)
        + (psi * torch.roll(psi, -2, dims=-1)).mean(dim=axes)
    )
    fine_sq = fine.square()
    potential = fine_sq + action.lam * (fine_sq - 1.0).square()
    hopping = fine * torch.roll(fine, -1, dims=-2)
    hopping += fine * torch.roll(fine, -1, dims=-1)
    action_density = (potential - 2.0 * action.kappa * hopping).mean(dim=axes)
    return torch.stack((psi2, psi4, c10, c11, c20, action_density), dim=1)


def _normalization(psi_train: np.ndarray) -> dict[str, Any]:
    coarse, details = extract_details(psi_train)
    return {
        "coarse_mean": float(np.mean(coarse)),
        "coarse_std": float(np.std(coarse)),
        "detail_mean": np.mean(details, axis=(0, 2, 3)).tolist(),
        "detail_std": np.std(details, axis=(0, 2, 3)).tolist(),
    }


def _standardize(
    coarse: Tensor,
    details: Tensor | None,
    normalization: dict[str, Any],
) -> tuple[Tensor, Tensor | None]:
    coarse_std = (coarse - normalization["coarse_mean"]) / normalization[
        "coarse_std"
    ]
    if details is None:
        return coarse_std, None
    mean = details.new_tensor(normalization["detail_mean"])[None, :, None, None]
    std = details.new_tensor(normalization["detail_std"])[None, :, None, None]
    return coarse_std, (details - mean) / std


def _unstandardize_details(details: Tensor, normalization: dict[str, Any]) -> Tensor:
    mean = details.new_tensor(normalization["detail_mean"])[None, :, None, None]
    std = details.new_tensor(normalization["detail_std"])[None, :, None, None]
    return details * std + mean


def _observable_scales(
    psi_train: np.ndarray,
    fine_train: np.ndarray,
    action: Phi4Action,
) -> dict[str, Any]:
    psi = torch.from_numpy(psi_train.astype(np.float32))
    fine = torch.from_numpy(fine_train.astype(np.float32))
    values = _torch_observables(psi, fine, action).numpy()
    return {
        "observable_std": np.maximum(np.std(values, axis=0), 1.0e-4).tolist(),
        "psi_site_std": float(np.std(psi_train)),
        "fine_site_std": float(np.std(fine_train)),
    }


def _auxiliary_loss(
    model: ConditionalAffineFlow,
    native_coarse: Tensor,
    target_psi: Tensor,
    normalization: dict[str, Any],
    scales: dict[str, Any],
    kernel_fft: Tensor,
    action: Phi4Action,
    *,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    coarse_std, _ = _standardize(native_coarse, None, normalization)
    generated_std, _, _ = model.sample(coarse_std, generator=generator)
    generated_details = _unstandardize_details(generated_std, normalization)
    generated_psi = _torch_assemble(native_coarse, generated_details)
    generated_fine = _torch_inverse(generated_psi, kernel_fft)
    target_fine = _torch_inverse(target_psi, kernel_fft)

    generated_obs = _torch_observables(generated_psi, generated_fine, action)
    target_obs = _torch_observables(target_psi, target_fine, action)
    observable_scale = generated_obs.new_tensor(scales["observable_std"])
    observable_loss = (
        (generated_obs.mean(dim=0) - target_obs.mean(dim=0)) / observable_scale
    ).square().mean()

    probabilities = generated_psi.new_tensor((0.01, 0.05, 0.95, 0.99))
    psi_difference = torch.quantile(generated_psi, probabilities) - torch.quantile(
        target_psi, probabilities
    )
    fine_probabilities = generated_fine.new_tensor((0.01, 0.05, 0.95, 0.99))
    fine_difference = torch.quantile(
        generated_fine, fine_probabilities
    ) - torch.quantile(target_fine, fine_probabilities)
    tail_loss = 0.5 * (
        (psi_difference / scales["psi_site_std"]).square().mean()
        + (fine_difference / scales["fine_site_std"]).square().mean()
    )
    return observable_loss, tail_loss


def physical_detail_log_prob(
    model: ConditionalAffineFlow,
    coarse: np.ndarray,
    details: np.ndarray,
    normalization: dict[str, Any],
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    values = []
    scale_logdet = coarse.shape[-1] ** 2 * float(
        np.sum(np.log(np.asarray(normalization["detail_std"])))
    )
    model.eval()
    with torch.no_grad():
        for start in range(0, len(coarse), batch_size):
            coarse_batch = torch.from_numpy(coarse[start : start + batch_size]).to(
                device=device, dtype=torch.float32
            )
            detail_batch = torch.from_numpy(details[start : start + batch_size]).to(
                device=device, dtype=torch.float32
            )
            coarse_std, detail_std = _standardize(
                coarse_batch, detail_batch, normalization
            )
            assert detail_std is not None
            values.append((model.log_prob(coarse_std, detail_std) - scale_logdet).cpu())
    return torch.cat(values).numpy().astype(np.float64)


def _validation_objective(
    model: ConditionalAffineFlow,
    psi: np.ndarray,
    native_coarse: np.ndarray,
    normalization: dict[str, Any],
    scales: dict[str, Any],
    kernel_fft: Tensor,
    action: Phi4Action,
    device: torch.device,
    batch_size: int,
    use_auxiliary: bool,
    seed: int,
) -> dict[str, float]:
    model.eval()
    log_probabilities = []
    dimension = 3 * native_coarse.shape[-1] ** 2
    with torch.no_grad():
        for start in range(0, len(psi), batch_size):
            target = torch.from_numpy(psi[start : start + batch_size]).to(
                device=device, dtype=torch.float32
            )
            coarse_np, details_np = extract_details(psi[start : start + batch_size])
            coarse = torch.from_numpy(coarse_np).to(device=device, dtype=torch.float32)
            details = torch.from_numpy(details_np).to(device=device, dtype=torch.float32)
            coarse_std, details_std = _standardize(coarse, details, normalization)
            assert details_std is not None
            log_probabilities.append(model.log_prob(coarse_std, details_std).cpu())
        nll = float(-torch.cat(log_probabilities).mean() / dimension)
        observable = 0.0
        tail = 0.0
        count = min(len(psi), len(native_coarse))
        generator = torch.Generator(device=device).manual_seed(seed)
        native = torch.from_numpy(native_coarse[:count]).to(
            device=device, dtype=torch.float32
        )
        native_std, _ = _standardize(native, None, normalization)
        generated_std, generated_logq_std, _ = model.sample(
            native_std, generator=generator
        )
        generated_details = _unstandardize_details(generated_std, normalization)
        generated_psi = _torch_assemble(native, generated_details)
        generated_fine = _torch_inverse(generated_psi, kernel_fft)
        scale_logdet = native.shape[-1] ** 2 * float(
            np.sum(np.log(np.asarray(normalization["detail_std"])))
        )
        log_weight = -_torch_action(generated_fine, action)
        log_weight -= generated_logq_std - scale_logdet
        normalized_weight = torch.softmax(log_weight - torch.max(log_weight), dim=0)
        importance_ess_ratio = float(
            1.0 / (count * torch.sum(normalized_weight.square()))
        )
        if use_auxiliary:
            target = torch.from_numpy(psi[:count]).to(
                device=device, dtype=torch.float32
            )
            obs_value, tail_value = _auxiliary_loss(
                model,
                native,
                target,
                normalization,
                scales,
                kernel_fft,
                action,
                generator=generator,
            )
            observable = float(obs_value)
            tail = float(tail_value)
    return {
        "nll_per_detail_dof": nll,
        "observable_loss": observable,
        "tail_loss": tail,
        "importance_ess_ratio": importance_ess_ratio,
        "total": nll + 0.05 * observable + 0.02 * tail,
    }


def sample_physical_details(
    model: ConditionalAffineFlow,
    coarse: np.ndarray,
    normalization: dict[str, Any],
    *,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    generator = torch.Generator(device=device).manual_seed(seed)
    details = []
    logq = []
    model.eval()
    scale_logdet = coarse.shape[-1] ** 2 * float(
        np.sum(np.log(np.asarray(normalization["detail_std"])))
    )
    with torch.no_grad():
        for start in range(0, len(coarse), batch_size):
            batch = torch.from_numpy(coarse[start : start + batch_size]).to(
                device=device, dtype=torch.float32
            )
            batch_std, _ = _standardize(batch, None, normalization)
            detail_std, density_std, _ = model.sample(
                batch_std,
                generator=generator,
            )
            details.append(_unstandardize_details(detail_std, normalization).cpu())
            logq.append((density_std - scale_logdet).cpu())
    return torch.cat(details).numpy().astype(np.float64), torch.cat(logq).numpy()


def _heldout_metrics(
    generated: np.ndarray,
    native: np.ndarray,
    action: Phi4Action,
) -> dict[str, Any]:
    generated_series = kernel_observable_series(generated, action)
    native_series = kernel_observable_series(native, action)
    distributions = {
        name: distribution_metrics(generated_series[name], native_series[name])
        for name in KERNEL_OBSERVABLE_NAMES
    }
    shifts = np.array(
        [distributions[name]["standardized_mean_shift"] for name in KERNEL_OBSERVABLE_NAMES]
    )
    return {
        "distributions": distributions,
        "rms_standardized_shift": float(np.sqrt(np.mean(shifts**2))),
        "max_abs_standardized_shift": float(np.max(np.abs(shifts))),
    }


def _save_training_figures(
    output: Path,
    history: list[dict[str, float]],
    generated: np.ndarray,
    native: np.ndarray,
    action: Phi4Action,
) -> None:
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(epochs, [row["training_nll"] for row in history], label="train")
    axes[0].plot(epochs, [row["validation_nll"] for row in history], label="validation")
    axes[0].set(xlabel="epoch", ylabel="NLL / detail dof", title="Conditional likelihood")
    axes[0].legend()
    axes[1].plot(epochs, [row["validation_observable"] for row in history], label="observable")
    axes[1].plot(epochs, [row["validation_tail"] for row in history], label="tail")
    axes[1].set(xlabel="epoch", ylabel="loss", title="Validation auxiliary losses")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output / "training_curves.pdf")
    figure.savefig(output / "training_curves.png", dpi=180)
    plt.close(figure)

    generated_series = kernel_observable_series(generated, action)
    native_series = kernel_observable_series(native, action)
    figure, axes = plt.subplots(2, 5, figsize=(18, 7))
    for axis, name in zip(axes.flat, KERNEL_OBSERVABLE_NAMES):
        combined = np.concatenate((generated_series[name], native_series[name]))
        bins = np.histogram_bin_edges(combined, bins=25)
        axis.hist(native_series[name], bins=bins, density=True, histtype="step", label="native")
        axis.hist(generated_series[name], bins=bins, density=True, alpha=0.4, label="flow sweep 0")
        axis.set_title(name)
    axes[0, 0].legend()
    figure.tight_layout()
    figure.savefig(output / "flow_observable_histograms.pdf")
    figure.savefig(output / "flow_observable_histograms.png", dpi=180)
    plt.close(figure)


def train_flow_pair(
    fine_path: Path,
    coarse_path: Path,
    kernel_path: Path,
    output: Path,
    *,
    epochs: int = 300,
    warmup_epochs: int = 100,
    batch_size: int = 32,
    patience: int = 40,
    learning_rate: float = 3.0e-4,
    weight_decay: float = 1.0e-5,
    coupling_layers: int = 6,
    hidden_channels: int = 32,
    kernel_size: int = 3,
    model_type: str = "affine",
    spline_layers: int = 4,
    spline_bins: int = 8,
    spline_tail_bound: float = 4.0,
    reverse_kl_weight: float = 0.0,
    initialize_from: Path | None = None,
    seed: int = 1234,
    device_name: str = "auto",
) -> dict[str, Any]:
    training_started = time.perf_counter()
    fine_chains = load_configurations(fine_path)
    coarse_chains = load_configurations(coarse_path)
    fine_size = fine_chains.shape[-1]
    coarse_size = coarse_chains.shape[-1]
    if fine_size != 2 * coarse_size:
        raise ValueError("flow training requires a factor-two fine/coarse pair")
    kernel, kernel_metadata, kernel_sha256 = load_kernel(kernel_path)
    psi_chains = apply_kernel(fine_chains, kernel)

    fine_train = fine_chains[:2].reshape(-1, fine_size, fine_size)
    psi_train = psi_chains[:2].reshape(-1, fine_size, fine_size)
    coarse_train = coarse_chains[:2].reshape(-1, coarse_size, coarse_size)
    psi_validation = psi_chains[2]
    coarse_validation = coarse_chains[2]
    fine_test = fine_chains[3]
    coarse_test = coarse_chains[3]

    normalization = _normalization(psi_train)
    action = Phi4Action()
    scales = _observable_scales(psi_train, fine_train, action)
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)

    config = AffineFlowConfig(
        lattice_size=coarse_size,
        coupling_layers=coupling_layers,
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        model_type=model_type,
        spline_layers=spline_layers,
        spline_bins=spline_bins,
        spline_tail_bound=spline_tail_bound,
    )
    model = ConditionalAffineFlow(config).to(device)
    if initialize_from is not None:
        initial = torch.load(initialize_from, map_location=device, weights_only=False)
        source = model_from_checkpoint(initial, device)
        missing, unexpected = model.load_state_dict(source.state_dict(), strict=False)
        if unexpected or any("conditioner.net" not in name for name in missing):
            raise ValueError("initializer is not architecture-compatible")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=15,
        min_lr=1.0e-6,
    )
    kernel_fft = torch.from_numpy(kernel_transform(kernel, fine_size)).to(
        device=device, dtype=torch.complex64
    )
    history: list[dict[str, float]] = []
    best_score: tuple[float, float, float] = (float("inf"),) * 3
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    epochs_without_improvement = 0
    dimension = 3 * coarse_size**2

    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(psi_train))
        training_nll = []
        training_observable = []
        training_tail = []
        training_reverse_kl = []
        auxiliary = epoch > warmup_epochs
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            target_psi_np = _augment_psi(psi_train[indices], rng)
            paired_coarse_np, paired_details_np = extract_details(target_psi_np)
            native_indices = rng.integers(0, len(coarse_train), size=len(indices))
            native_coarse_np = _augment_coarse(coarse_train[native_indices], rng)

            paired_coarse = torch.from_numpy(paired_coarse_np).to(
                device=device, dtype=torch.float32
            )
            paired_details = torch.from_numpy(paired_details_np).to(
                device=device, dtype=torch.float32
            )
            target_psi = torch.from_numpy(target_psi_np).to(
                device=device, dtype=torch.float32
            )
            native_coarse = torch.from_numpy(native_coarse_np).to(
                device=device, dtype=torch.float32
            )
            coarse_std, details_std = _standardize(
                paired_coarse, paired_details, normalization
            )
            assert details_std is not None
            nll = -model.log_prob(coarse_std, details_std).mean() / dimension
            observable_loss = nll.new_zeros(())
            tail_loss = nll.new_zeros(())
            if auxiliary:
                observable_loss, tail_loss = _auxiliary_loss(
                    model,
                    native_coarse,
                    target_psi,
                    normalization,
                    scales,
                    kernel_fft,
                    action,
                )
            reverse_kl = nll.new_zeros(())
            if auxiliary and reverse_kl_weight > 0.0:
                native_std, _ = _standardize(native_coarse, None, normalization)
                sampled_std, sampled_logq_std, _ = model.sample(native_std)
                sampled_details = _unstandardize_details(sampled_std, normalization)
                sampled_fine = _torch_inverse(
                    _torch_assemble(native_coarse, sampled_details), kernel_fft
                )
                scale_logdet = coarse_size**2 * float(
                    np.sum(np.log(np.asarray(normalization["detail_std"])))
                )
                reverse_kl = (
                    sampled_logq_std - scale_logdet
                    + _torch_action(sampled_fine, action)
                ).mean() / dimension
            loss = (
                nll
                + 0.05 * observable_loss
                + 0.02 * tail_loss
                + reverse_kl_weight * reverse_kl
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            training_nll.append(float(nll.detach()))
            training_observable.append(float(observable_loss.detach()))
            training_tail.append(float(tail_loss.detach()))
            training_reverse_kl.append(float(reverse_kl.detach()))

        validation = _validation_objective(
            model,
            psi_validation,
            coarse_validation,
            normalization,
            scales,
            kernel_fft,
            action,
            device,
            batch_size,
            True,
            seed + 10_000,
        )
        scheduler.step(validation["total"])
        row = {
            "epoch": float(epoch),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "training_nll": float(np.mean(training_nll)),
            "training_observable": float(np.mean(training_observable)),
            "training_tail": float(np.mean(training_tail)),
            "training_reverse_kl": float(np.mean(training_reverse_kl)),
            "validation_nll": validation["nll_per_detail_dof"],
            "validation_observable": validation["observable_loss"],
            "validation_tail": validation["tail_loss"],
            "validation_importance_ess_ratio": validation["importance_ess_ratio"],
            "validation_total": validation["total"],
        }
        history.append(row)
        selection_score = (
            -validation["importance_ess_ratio"],
            validation["total"],
            validation["nll_per_detail_dof"],
        )
        if best_state is None or selection_score < best_score:
            best_score = selection_score
            best_epoch = epoch
            best_state = deepcopy(
                {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                }
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        print(
            f"L{coarse_size}->L{fine_size} epoch {epoch:03d}: "
            f"train_nll={row['training_nll']:.5f} "
            f"val={validation['total']:.5f}",
            flush=True,
        )
        if auxiliary and epochs_without_improvement >= patience:
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 2,
        "model_type": model_type,
        "model_state": best_state,
        "optimizer_state": optimizer.state_dict(),
        "config": {
            "model": model.configuration(),
            "training": {
                "epochs_requested": epochs,
                "epochs_completed": len(history),
                "warmup_epochs": warmup_epochs,
                "batch_size": batch_size,
                "patience": patience,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "observable_weight": 0.05,
                "tail_weight": 0.02,
                "reverse_kl_weight": reverse_kl_weight,
                "initialize_from": str(initialize_from) if initialize_from else None,
                "seed": seed,
                "device": str(device),
            },
            "fine_source": str(fine_path),
            "coarse_source": str(coarse_path),
            "chain_split": {"train": [0, 1], "validation": [2], "test": [3]},
        },
        "normalization": normalization,
        "loss_scales": scales,
        "kernel": {
            "path": str(kernel_path),
            "sha256": kernel_sha256,
            "matrix": kernel.tolist(),
            "parameterization": kernel_metadata.get("parameterization"),
        },
        "best_epoch": best_epoch,
        "best_validation_objective": best_score[1],
        "best_validation_importance_ess_ratio": -best_score[0],
        "history": history,
    }
    generated_details, _ = sample_physical_details(
        model,
        coarse_test,
        normalization,
        device=device,
        batch_size=batch_size,
        seed=seed + 20_000,
    )
    generated_psi = assemble_details(coarse_test, generated_details)
    generated_fine = inverse_kernel(generated_psi, kernel)
    heldout = _heldout_metrics(generated_fine, fine_test, action)
    with torch.no_grad():
        coarse = torch.from_numpy(coarse_test[:8]).to(device=device, dtype=torch.float32)
        coarse_std, _ = _standardize(coarse, None, normalization)
        generator = torch.Generator(device=device).manual_seed(seed + 30_000)
        sampled, _, latent = model.sample(coarse_std, generator=generator)
        latent_back, inverse_logdet = model.inverse(coarse_std, sampled)
        _, forward_logdet = model.forward(coarse_std, latent)
        roundtrip = {
            "latent_max_abs_error": float(torch.max(torch.abs(latent_back - latent))),
            "logdet_max_abs_error": float(torch.max(torch.abs(forward_logdet + inverse_logdet))),
        }
    metrics = {
        "mapping": f"L{coarse_size}_to_L{fine_size}",
        "best_epoch": best_epoch,
        "best_validation_objective": best_score[1],
        "best_validation_importance_ess_ratio": -best_score[0],
        "model_type": model_type,
        "test": heldout,
        "roundtrip": roundtrip,
        "normalization": normalization,
        "kernel_sha256": kernel_sha256,
        "history": history,
    }
    metrics["training_wall_seconds"] = time.perf_counter() - training_started
    checkpoint["training_wall_seconds"] = metrics["training_wall_seconds"]
    torch.save(checkpoint, output / "checkpoint.pt")
    _write_json(output / "metrics.json", metrics)
    _save_training_figures(output, history, generated_fine, fine_test, action)
    return metrics
