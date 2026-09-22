"""Streaming CNN training for SU(3) factor-two downsampling."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from .downsampling import (
    LinkCoefficientCNN,
    block_links,
    smear_and_block,
    su3_errors,
    weights_from_path_logits,
)
from .io import read_nersc_gauge
from .observables import observable_names, observable_vector_pyquda, observable_vector_torch


OBSERVABLE_NAMES = observable_names()
VARIANCE_EPS = 1.0e-12


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.generic, torch.Tensor)):
        return _json_ready(value.item() if value.ndim == 0 else value.detach().cpu().numpy())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(_json_ready(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ensemble_paths(directory: Path) -> tuple[list[Path], dict]:
    manifest = json.loads((directory / "ensemble.json").read_text(encoding="utf-8"))
    paths = [directory / item["filename"] for item in manifest["configurations"]]
    if not paths or any(not path.exists() for path in paths):
        raise FileNotFoundError(f"missing NERSC configuration under {directory}")
    return paths, manifest


def split_indices(count: int) -> dict[str, list[int]]:
    return {
        "train": [index for index in range(count) if index % 5 in (0, 1, 2)],
        "validation": [index for index in range(count) if index % 5 == 3],
        "test": [index for index in range(count) if index % 5 == 4],
    }


class FeatureCache:
    """Disk cache for gauge-invariant CNN input features."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(path: Path) -> str:
        return hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:20]

    def get(self, path: Path, links: torch.Tensor, device: torch.device) -> torch.Tensor:
        from .downsampling import gauge_invariant_features

        cache_path = self.directory / f"{self._key(path)}.pt"
        stat = path.stat()
        if cache_path.exists():
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
            if cached["size"] == stat.st_size and cached["mtime_ns"] == stat.st_mtime_ns:
                self.hits += 1
                return cached["features"].to(device=device, dtype=torch.float32).unsqueeze(0)
        self.misses += 1
        with torch.no_grad():
            features = gauge_invariant_features(links.unsqueeze(0)).squeeze(0).cpu().half()
        torch.save({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "features": features}, cache_path)
        return features.to(device=device, dtype=torch.float32).unsqueeze(0)


def _load_links(path: Path, device: torch.device) -> torch.Tensor:
    links, _ = read_nersc_gauge(path)
    tensor = torch.from_numpy(links)
    if device.type == "cuda":
        tensor = tensor.pin_memory()
    return tensor.to(device=device, non_blocking=device.type == "cuda")


class LinkPrefetcher:
    """Read one NERSC file ahead while the current configuration is processed."""

    def __init__(self, paths: Iterable[Path], device: torch.device):
        self.paths = tuple(paths)
        self.device = device

    def __iter__(self):
        if not self.paths:
            return
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="nersc-prefetch") as pool:
            future = pool.submit(read_nersc_gauge, self.paths[0])
            for index, path in enumerate(self.paths):
                links, _ = future.result()
                if index + 1 < len(self.paths):
                    future = pool.submit(read_nersc_gauge, self.paths[index + 1])
                tensor = torch.from_numpy(links)
                if self.device.type == "cuda":
                    tensor = tensor.pin_memory()
                yield path, tensor.to(device=self.device, non_blocking=self.device.type == "cuda")


def _link_batches(
    paths: Iterable[Path], device: torch.device, feature_cache: FeatureCache | None,
    needs_features: bool, batch_size: int,
):
    links_batch = []
    features_batch = []
    for path, links in LinkPrefetcher(paths, device):
        links_batch.append(links)
        if needs_features and feature_cache:
            features_batch.append(feature_cache.get(path, links, device).squeeze(0))
        if len(links_batch) == batch_size:
            features = torch.stack(features_batch) if features_batch else None
            yield torch.stack(links_batch), features
            links_batch.clear()
            features_batch.clear()
    if links_batch:
        features = torch.stack(features_batch) if features_batch else None
        yield torch.stack(links_batch), features


def _projected_blocked_observables(
    paths: Iterable[Path],
    kernel: torch.Tensor | nn.Module,
    device: torch.device,
    *,
    requires_grad: bool,
    feature_cache: FeatureCache | None = None,
    batch_size: int = 2,
) -> torch.Tensor:
    values = []
    needs_features = isinstance(kernel, nn.Module)
    context = torch.enable_grad() if requires_grad else torch.no_grad()
    with context:
        for links, features in _link_batches(paths, device, feature_cache, needs_features, batch_size):
            blocked = smear_and_block(links, kernel, features)
            values.append(observable_vector_torch(blocked))
            del links, features, blocked
    return torch.cat(values, dim=0)


def _straight_blocked_observables(
    paths: Iterable[Path], device: torch.device, *, batch_size: int = 2,
) -> torch.Tensor:
    values = []
    with torch.no_grad():
        for links, _ in _link_batches(paths, device, None, False, batch_size):
            values.append(observable_vector_torch(block_links(links)))
    return torch.cat(values, dim=0)


def _baseline_logits(device: torch.device) -> torch.Tensor:
    return torch.tensor([40.0, -40.0, -40.0], dtype=torch.float32, device=device)


def _covariance_inverse(coarse: np.ndarray, baseline: np.ndarray, shrinkage: float) -> tuple[np.ndarray, np.ndarray]:
    covariance = np.cov(coarse, rowvar=False, ddof=1) / len(coarse)
    covariance += np.cov(baseline, rowvar=False, ddof=1) / len(baseline)
    diagonal = np.diag(np.diag(covariance))
    covariance = (1.0 - shrinkage) * covariance + shrinkage * diagonal
    covariance += np.eye(covariance.shape[0]) * max(float(np.median(np.diag(covariance))), 1.0e-30) * 1.0e-8
    return covariance, np.linalg.inv(covariance)


def _split_paths(paths: list[Path], indices: dict[str, list[int]], split: str) -> list[Path]:
    return [paths[index] for index in indices[split]]


def _stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    ddof = 1 if len(values) > 1 else 0
    variance = values.var(axis=0, ddof=ddof)
    return mean, variance, np.log(variance + VARIANCE_EPS)


def _torch_stats(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = values.mean(dim=0)
    variance = values.var(dim=0, unbiased=values.shape[0] > 1)
    return mean, variance, torch.log(variance + VARIANCE_EPS)


def _loss_components(values: np.ndarray, target: np.ndarray, inverse_covariance: np.ndarray, variance_scale: np.ndarray | None = None) -> dict:
    if variance_scale is None:
        variance_scale = np.ones(values.shape[1], dtype=np.float64)
    mean, variance, log_variance = _stats(values)
    target_mean, target_variance, target_log_variance = _stats(target)
    mean_difference = mean - target_mean
    mean_loss = float(mean_difference @ inverse_covariance @ mean_difference)
    variance_difference = (log_variance - target_log_variance) / variance_scale
    variance_loss = float(np.sum(variance_difference**2))
    return {
        "loss": mean_loss + variance_loss,
        "mean_loss": mean_loss,
        "variance_loss": variance_loss,
        "mean": mean,
        "variance": variance,
        "log_variance": log_variance,
        "target_mean": target_mean,
        "target_variance": target_variance,
        "target_log_variance": target_log_variance,
    }


def _metric_record(values: np.ndarray, target: np.ndarray, inverse_covariance: np.ndarray, variance_scale: np.ndarray | None = None) -> dict:
    if variance_scale is None:
        variance_scale = np.ones(values.shape[1], dtype=np.float64)
    record = _loss_components(values, target, inverse_covariance, variance_scale)
    mean = record["mean"]
    target_mean = record["target_mean"]
    ddof = 1 if len(values) > 1 else 0
    target_ddof = 1 if len(target) > 1 else 0
    standard_error = values.std(axis=0, ddof=ddof) / np.sqrt(len(values))
    target_error = target.std(axis=0, ddof=target_ddof) / np.sqrt(len(target))
    denominator = np.sqrt(standard_error**2 + target_error**2)
    mean_shift = np.divide(mean - target_mean, denominator, out=np.full_like(mean, np.nan), where=denominator > 0)
    variance_shift = (record["log_variance"] - record["target_log_variance"]) / variance_scale
    record.update(
        {
            "standard_error": standard_error,
            "target_standard_error": target_error,
            "standardized_shift": mean_shift,
            "standardized_log_variance_shift": variance_shift,
        }
    )
    return record


def _loss_coefficients(values: torch.Tensor, target_mean: torch.Tensor, target_log_variance: torch.Tensor, inverse_covariance: torch.Tensor, variance_scale: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    mean, variance, log_variance = _torch_stats(values)
    mean_difference = mean.double() - target_mean
    mean_loss = mean_difference @ inverse_covariance @ mean_difference
    variance_difference = (log_variance.double() - target_log_variance) / variance_scale
    variance_loss = (variance_difference**2).sum()
    n = values.shape[0]
    mean_gradient = 2.0 * (inverse_covariance @ mean_difference) / n
    log_variance_gradient = 2.0 * variance_difference / variance_scale
    variance_gradient = 2.0 * (values.double() - mean.double()) / ((n - 1) * (variance.double() + VARIANCE_EPS))
    coefficients = mean_gradient.unsqueeze(0) + log_variance_gradient.unsqueeze(0) * variance_gradient
    return coefficients, {"loss": float((mean_loss + variance_loss).detach().cpu()), "mean_loss": float(mean_loss.detach().cpu()), "variance_loss": float(variance_loss.detach().cpu())}


def _pyquda_gauge_from_links(links: np.ndarray, lattice: list[int]):
    from pyquda_utils.core import LatticeGauge, LatticeInfo

    info = LatticeInfo(lattice)
    links = np.ascontiguousarray(links, dtype=np.complex128)
    return LatticeGauge(info, info.evenodd(links, True))


def measure_reference_with_pyquda(paths: list[Path], lattice: list[int], resource_path: Path) -> np.ndarray:
    """Measure the native coarse ensemble through PyQUDA's gauge routines."""
    from pyquda_utils import core, io

    resource_path.mkdir(parents=True, exist_ok=True)
    core.init(None, lattice, backend="numpy", resource_path=str(resource_path))
    measurements = []
    for path in paths:
        gauge = io.readNERSCGauge(str(path), checksum=True, plaquette=False, link_trace=False, reunitarize_sigma=0)
        measurements.append(observable_vector_pyquda(gauge))
    return np.asarray(measurements, dtype=np.float64)


def measure_blocked_with_pyquda(paths: list[Path], kernel: torch.Tensor | nn.Module, lattice: list[int], device: torch.device, feature_cache: FeatureCache | None = None) -> tuple[np.ndarray, tuple[float, float]]:
    measurements = []
    max_unitarity = 0.0
    max_determinant = 0.0
    needs_features = isinstance(kernel, nn.Module)
    with torch.no_grad():
        for path, links in LinkPrefetcher(paths, device):
            features = feature_cache.get(path, links, device) if needs_features and feature_cache else None
            blocked = smear_and_block(links, kernel, features)
            unitarity, determinant = su3_errors(blocked)
            max_unitarity = max(max_unitarity, unitarity)
            max_determinant = max(max_determinant, determinant)
            gauge = _pyquda_gauge_from_links(blocked.cpu().numpy(), lattice)
            measurements.append(observable_vector_pyquda(gauge))
            del links, features, blocked, gauge
    return np.asarray(measurements), (max_unitarity, max_determinant)


def _compile_kernel(model: LinkCoefficientCNN, backend: str, sample_features: torch.Tensor | None) -> tuple[nn.Module, dict]:
    if backend == "eager":
        return model, {"requested": backend, "used": "eager", "compiled": False}
    if backend != "compile":
        raise ValueError(f"unsupported backend: {backend}")
    try:
        # ``reduce-overhead`` uses CUDA graphs and cannot safely accumulate
        # gradients from one compiled invocation per configuration.  The
        # default inductor mode still compiles the static CNN while retaining
        # ordinary autograd semantics for the streaming surrogate pass.
        compiled = torch.compile(model, dynamic=False, fullgraph=False, mode="default")
        if sample_features is not None:
            with torch.no_grad():
                compiled(sample_features)
            if sample_features.is_cuda:
                torch.cuda.synchronize()
        return compiled, {"requested": backend, "used": "compile", "compiled": True}
    except Exception as error:
        return model, {"requested": backend, "used": "eager", "compiled": False, "error": str(error)}


def _distribution_bins(values: list[np.ndarray], minimum: int = 12, maximum: int = 40) -> np.ndarray:
    pooled = np.concatenate([np.asarray(value).reshape(-1) for value in values])
    lower, upper = float(np.min(pooled)), float(np.max(pooled))
    if upper <= lower:
        return np.linspace(lower - 0.5, upper + 0.5, minimum + 1)
    width = 2.0 * (np.percentile(pooled, 75) - np.percentile(pooled, 25)) * len(pooled) ** (-1.0 / 3.0)
    count = minimum if width <= 0 else int(np.ceil((upper - lower) / width))
    count = max(minimum, min(maximum, count))
    return np.linspace(lower, upper, count + 1)


def _plot_diagnostics(path: Path, history: list[dict], test: dict[str, np.ndarray], names: tuple[str, ...]) -> None:
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / "scripts" / "heatbath"))
    from plot_settings import BLUE, FIG_SIZE, GREY, ORANGE, apply_plot_style

    apply_plot_style()
    figure, axes = plt.subplots(2, 3, figsize=(FIG_SIZE[0] * 2.2, FIG_SIZE[1] * 2.0))
    epochs = [row["epoch"] for row in history]
    axes[0, 0].plot(epochs, [row["train_loss"] for row in history], color=BLUE, label="train")
    axes[0, 0].plot(epochs, [row["validation_loss"] for row in history], color=ORANGE, label="validation")
    axes[0, 0].set(xlabel="epoch", ylabel="total loss", title="Training history")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].set_yscale("log")
    axes[0, 0].grid(linestyle=":")
    for index, name in enumerate(names):
        axis = axes.flat[index + 1]
        values = [test[label][:, index] for label in ("reference", "downsampled", "baseline")]
        bins = _distribution_bins(values)
        for label, color in (("reference", BLUE), ("downsampled", ORANGE), ("baseline", GREY)):
            axis.hist(test[label][:, index], bins=bins, density=True, histtype="step", linewidth=1.8, color=color, label=label)
        reference_mean = np.mean(test["reference"][:, index])
        downsampled_mean = np.mean(test["downsampled"][:, index])
        reference_error = np.std(test["reference"][:, index], ddof=1 if len(test["reference"]) > 1 else 0) / np.sqrt(len(test["reference"]))
        downsampled_error = np.std(test["downsampled"][:, index], ddof=1 if len(test["downsampled"]) > 1 else 0) / np.sqrt(len(test["downsampled"]))
        axis.axvspan(reference_mean - reference_error, reference_mean + reference_error, color=BLUE, alpha=0.12, linewidth=0)
        axis.axvspan(downsampled_mean - downsampled_error, downsampled_mean + downsampled_error, color=ORANGE, alpha=0.12, linewidth=0)
        axis.axvline(reference_mean, color=BLUE, linestyle=":")
        axis.axvline(downsampled_mean, color=ORANGE, linestyle=":")
        axis.set_title(name)
        axis.grid(linestyle=":")
        axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def train(
    fine_paths: list[Path], coarse_reference: np.ndarray, output: Path, *, lattice: list[int], device: torch.device,
    epochs: int = 100, patience: int = 12, learning_rate: float = 0.003, seed: int = 1234,
    shrinkage: float = 0.1, backend: str = "eager", feature_cache_dir: Path | None = None,
    batch_size: int = 2, configs_per_epoch: int | None = None,
    validation_configs: int | None = None,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    output.mkdir(parents=True, exist_ok=True)
    indices = split_indices(len(fine_paths))
    feature_cache = FeatureCache(feature_cache_dir or output / "feature_cache")
    baseline = {
        split: _straight_blocked_observables(
            _split_paths(fine_paths, indices, split), device, batch_size=batch_size
        ).cpu().numpy()
        for split in indices
    }
    targets = {split: coarse_reference[indices[split]] for split in indices}
    covariance, inverse_covariance = _covariance_inverse(targets["train"], baseline["train"], shrinkage)
    variance_scale = np.full(len(OBSERVABLE_NAMES), np.sqrt(2.0 / max(len(targets["train"]) - 1, 1)))
    inverse_covariance_torch = torch.as_tensor(inverse_covariance, dtype=torch.float64, device=device)
    target_mean_np, _, target_log_variance_np = _stats(targets["train"])
    target_mean_torch = torch.as_tensor(target_mean_np, dtype=torch.float64, device=device)
    target_log_variance_torch = torch.as_tensor(target_log_variance_np, dtype=torch.float64, device=device)
    variance_scale_torch = torch.as_tensor(variance_scale, dtype=torch.float64, device=device)
    model = LinkCoefficientCNN().to(device)
    sample_path = fine_paths[indices["train"][0]]
    sample_links = _load_links(sample_path, device)
    sample_features = feature_cache.get(sample_path, sample_links, device)
    kernel, backend_info = _compile_kernel(model, backend, sample_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    best_state = None
    best_validation = float("inf")
    stale = 0
    train_paths = _split_paths(fine_paths, indices, "train")
    validation_paths = _split_paths(fine_paths, indices, "validation")
    training_order = np.random.default_rng(seed).permutation(len(train_paths))
    if configs_per_epoch is None:
        configs_per_epoch = len(train_paths)
    if validation_configs is None:
        validation_configs = len(validation_paths)
    validation_epoch_paths = validation_paths[:validation_configs]
    for epoch in range(1, epochs + 1):
        start = ((epoch - 1) * configs_per_epoch) % len(train_paths)
        epoch_indices = np.resize(training_order, start + configs_per_epoch)[start:]
        epoch_paths = [train_paths[index] for index in epoch_indices]
        with torch.no_grad():
            train_values = _projected_blocked_observables(epoch_paths, kernel, device, requires_grad=False, feature_cache=feature_cache, batch_size=batch_size)
            coefficients, train_components = _loss_coefficients(train_values, target_mean_torch, target_log_variance_torch, inverse_covariance_torch, variance_scale_torch)
        optimizer.zero_grad(set_to_none=True)
        for (links, features), coefficient_start in zip(
            _link_batches(epoch_paths, device, feature_cache, True, batch_size),
            range(0, len(epoch_paths), batch_size),
        ):
            coefficient = coefficients[coefficient_start : coefficient_start + links.shape[0]]
            blocked = smear_and_block(links, kernel, features)
            observable = observable_vector_torch(blocked)
            (coefficient * observable.double()).sum().backward()
            del links, features, blocked, observable
        optimizer.step()
        with torch.no_grad():
            validation_values = _projected_blocked_observables(validation_epoch_paths, kernel, device, requires_grad=False, feature_cache=feature_cache, batch_size=batch_size).cpu().numpy()
            sample_weights = weights_from_path_logits(model(sample_features))
            weight_means = sample_weights.mean(dim=(0, 3, 4, 5, 6)).mean(dim=0).cpu().numpy()
            local_weight_std = float(sample_weights.std(dim=(3, 4, 5, 6)).mean().cpu())
        train_record = _metric_record(train_values.cpu().numpy(), targets["train"], inverse_covariance, variance_scale)
        validation_record = _metric_record(validation_values, targets["validation"], inverse_covariance, variance_scale)
        row = {
            "epoch": epoch, "train_loss": train_record["loss"], "train_mean_loss": train_record["mean_loss"],
            "train_variance_loss": train_record["variance_loss"], "validation_loss": validation_record["loss"],
            "validation_mean_loss": validation_record["mean_loss"], "validation_variance_loss": validation_record["variance_loss"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "straight_weight": weight_means[0], "staple_weight": weight_means[1],
            "rectangle_weight": weight_means[2], "local_weight_std": local_weight_std,
        }
        history.append(row)
        print(f"epoch {epoch:03d}: train={row['train_loss']:.6g} val={row['validation_loss']:.6g} mean={row['train_mean_loss']:.6g} variance={row['train_variance_loss']:.6g}", flush=True)
        if validation_record["loss"] < best_validation:
            best_validation = validation_record["loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    # Do final evaluation with the eager module.  The compiled wrapper can
    # invalidate/recompile graphs after loading a new state dict, making the
    # post-selection evaluation disproportionately slower than training.
    evaluation_kernel = model.eval()
    torch.save({"state_dict": best_state, "epoch": best_epoch}, output / "kernel.pt")
    trained = {
        split: _projected_blocked_observables(
            _split_paths(fine_paths, indices, split), evaluation_kernel, device,
            requires_grad=False, feature_cache=feature_cache, batch_size=batch_size,
        ).cpu().numpy()
        for split in indices
    }
    records = {split: {"baseline": _metric_record(baseline[split], targets[split], inverse_covariance, variance_scale), "trained": _metric_record(trained[split], targets[split], inverse_covariance, variance_scale)} for split in indices}
    test_pass = records["test"]["trained"]["loss"] < records["test"]["baseline"]["loss"]
    kernel_record = {
        "type": "SU3_APE_style_local_CNN_path_convolution",
        "architecture": {"class": "LinkCoefficientCNN", "hidden_channels": 16, "output_channels": 3, "initial_weights": [0.30, 0.65, 0.05], "local_logit_scale": 0.1},
        "path_channels": ["straight", "six_staples", "six_transverse_1x2_rectangles"],
        "normalization": "per-link softmax over straight, staple, rectangle weights; each aggregate has six paths",
        "projection": "polar SVD projection followed by determinant-one phase correction",
        "blocking": "all-even anchors, product of two projected links",
        "reversibility": "not exact; projection and factor-two blocking are lossy",
        "observables": OBSERVABLE_NAMES, "data_split": indices, "best_epoch": best_epoch,
        "optimizer": {"name": "Adam", "learning_rate": learning_rate, "epochs": epochs, "patience": patience, "seed": seed, "batch_size": batch_size, "configs_per_epoch": configs_per_epoch, "validation_configs": validation_configs},
        "input_pipeline": "NERSC reader with one-configuration threaded prefetch and pinned host-to-CUDA transfer",
        "backend": backend_info, "feature_cache": {"directory": feature_cache.directory, "hits": feature_cache.hits, "misses": feature_cache.misses},
        "covariance": covariance, "variance_scale": variance_scale,
    }
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    _write_json(output / "kernel.json", kernel_record)
    metrics = {"observables": OBSERVABLE_NAMES, "acceptance": {"test_trained_loss_below_baseline": test_pass, "all": test_pass}, "splits": records, "best_epoch": best_epoch, "loss_definition": "covariance-weighted mean plus standardized log-variance mismatch"}
    _write_json(output / "metrics.json", metrics)
    _plot_diagnostics(output / "diagnostics.pdf", history, {"reference": targets["test"], "baseline": baseline["test"], "downsampled": trained["test"]}, OBSERVABLE_NAMES)
    return {"kernel": kernel_record, "metrics": metrics, "kernel_model": model, "trained_values": trained, "baseline_values": baseline, "reference_values": coarse_reference, "indices": indices, "feature_cache": feature_cache}
