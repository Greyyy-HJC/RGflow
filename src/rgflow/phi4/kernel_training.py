"""Train and evaluate an approximate perfect-blocking kernel."""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from .action import Phi4Action
from .blocking import (
    ETA,
    KERNEL_SUM,
    ORBIT_NAMES,
    block_field,
    identity_kernel,
    kernel_from_parameters,
    kernel_transform,
    spectrum_diagnostics,
)
from .operator_diagnostics import (
    distribution_metrics,
    save_operator_distribution_figure,
)


FloatArray = NDArray[np.float64]
FEATURE_NAMES = ("phi2", "phi4", "NN", "diag", "2nn", "m2", "G_pmin")


def feature_series(fields: FloatArray) -> FloatArray:
    """Return the per-configuration features used to train the kernel."""
    fields = np.asarray(fields, dtype=np.float64)
    axes = (-2, -1)
    magnetization = np.mean(fields, axis=axes)
    phi2 = np.mean(fields**2, axis=axes)
    phi4 = np.mean(fields**4, axis=axes)
    nearest = 0.5 * (
        np.mean(fields * np.roll(fields, -1, axis=-2), axis=axes)
        + np.mean(fields * np.roll(fields, -1, axis=-1), axis=axes)
    )
    diagonal = 0.5 * (
        np.mean(
            fields * np.roll(np.roll(fields, -1, axis=-2), -1, axis=-1),
            axis=axes,
        )
        + np.mean(
            fields * np.roll(np.roll(fields, -1, axis=-2), 1, axis=-1),
            axis=axes,
        )
    )
    distance_two = 0.5 * (
        np.mean(fields * np.roll(fields, -2, axis=-2), axis=axes)
        + np.mean(fields * np.roll(fields, -2, axis=-1), axis=axes)
    )
    transformed = np.fft.fft2(fields, axes=axes)
    volume = fields.shape[-1] ** 2
    minimum_momentum = 0.5 * (
        np.abs(transformed[..., 1, 0]) ** 2
        + np.abs(transformed[..., 0, 1]) ** 2
    ) / volume
    return np.stack(
        (
            phi2,
            phi4,
            nearest,
            diagonal,
            distance_two,
            magnetization**2,
            minimum_momentum,
        ),
        axis=-1,
    )


def _blocking_design(fields: FloatArray) -> tuple[FloatArray, FloatArray]:
    base = block_field(fields, identity_kernel())
    basis = np.stack(
        [
            block_field(
                fields,
                kernel_from_parameters(np.eye(5, dtype=np.float64)[index], total=0.0),
            )
            for index in range(5)
        ],
        axis=0,
    )
    return base, basis


def _apply_design(
    design: tuple[FloatArray, FloatArray],
    parameters: FloatArray,
) -> FloatArray:
    base, basis = design
    return base + np.tensordot(parameters, basis, axes=(0, 0))


def _whitening_matrix(features: FloatArray, shrinkage: float) -> FloatArray:
    covariance = np.cov(features, rowvar=False, ddof=1)
    diagonal = np.diag(np.diag(covariance))
    covariance = (1.0 - shrinkage) * covariance + shrinkage * diagonal
    scale = float(np.median(np.diag(covariance)))
    covariance += np.eye(covariance.shape[0]) * scale * 1.0e-10
    return np.linalg.inv(covariance)


def _matching_loss(
    blocked: FloatArray,
    target_mean: FloatArray,
    inverse_covariance: FloatArray,
) -> float:
    difference = np.mean(feature_series(blocked), axis=0) - target_mean
    return float(difference @ inverse_covariance @ difference)


def _spectrum_penalty(parameters: FloatArray) -> float:
    kernel = kernel_from_parameters(parameters)
    diagnostics = spectrum_diagnostics(kernel, 64)
    minimum_violation = max(0.0, 0.5 - diagnostics["min_abs_K"])
    condition_violation = max(0.0, diagnostics["condition_number"] - 3.0)
    return 1.0e4 * minimum_violation**2 + 1.0e2 * condition_violation**2


def _ensemble_scalars(fields: FloatArray, action: Phi4Action) -> dict[str, float]:
    fields = np.asarray(fields, dtype=np.float64)
    volume = fields.shape[-1] ** 2
    magnetization = np.mean(fields, axis=(-2, -1))
    magnetization_sq = magnetization**2
    magnetization_fourth = magnetization_sq**2
    transformed = np.fft.fft2(fields, axes=(-2, -1))
    minimum_momentum = float(
        np.mean(
            0.5
            * (
                np.abs(transformed[..., 1, 0]) ** 2
                + np.abs(transformed[..., 0, 1]) ** 2
            )
            / volume
        )
    )
    connected_zero = float(
        volume * (np.mean(magnetization_sq) - np.mean(magnetization) ** 2)
    )
    ratio = connected_zero / minimum_momentum
    xi_over_l = (
        np.sqrt(ratio - 1.0)
        / (2.0 * np.sin(np.pi / fields.shape[-1]))
        / fields.shape[-1]
        if ratio > 1.0
        else float("nan")
    )
    return {
        "action_density": float(np.mean(action(fields) / volume)),
        "magnetization": float(np.mean(magnetization)),
        "abs_magnetization": float(np.mean(np.abs(magnetization))),
        "susceptibility": connected_zero,
        "binder_cumulant": float(
            1.0
            - np.mean(magnetization_fourth)
            / (3.0 * np.mean(magnetization_sq) ** 2)
        ),
        "xi_over_L": float(xi_over_l),
    }


def evaluate_pair(
    fine: FloatArray,
    coarse: FloatArray,
    kernel: FloatArray,
    action: Phi4Action,
) -> dict[str, Any]:
    """Compare a blocked fine ensemble with a native coarse ensemble."""
    blocked = block_field(fine, kernel)
    blocked_features = feature_series(blocked)
    coarse_features = feature_series(coarse)
    features = {
        name: distribution_metrics(blocked_features[:, index], coarse_features[:, index])
        for index, name in enumerate(FEATURE_NAMES)
    }
    shifts = np.array(
        [features[name]["standardized_mean_shift"] for name in FEATURE_NAMES]
    )
    action_blocked = action(blocked) / (blocked.shape[-1] ** 2)
    action_coarse = action(coarse) / (coarse.shape[-1] ** 2)
    magnetization_blocked = np.mean(blocked, axis=(-2, -1))
    magnetization_coarse = np.mean(coarse, axis=(-2, -1))
    return {
        "summary": {
            "rms_standardized_shift": float(np.sqrt(np.mean(shifts**2))),
            "max_abs_standardized_shift": float(np.max(np.abs(shifts))),
        },
        "features": features,
        "diagnostic_distributions": {
            "action_density": distribution_metrics(action_blocked, action_coarse),
            "magnetization": distribution_metrics(
                magnetization_blocked, magnetization_coarse
            ),
            "abs_magnetization": distribution_metrics(
                np.abs(magnetization_blocked), np.abs(magnetization_coarse)
            ),
        },
        "ensemble_observables": {
            "blocked": _ensemble_scalars(blocked, action),
            "target": _ensemble_scalars(coarse, action),
        },
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            _json_ready(value),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _save_diagnostics(
    path: Path,
    fine_test: FloatArray,
    coarse_test: FloatArray,
    kernel: FloatArray,
    identity: FloatArray,
    metrics: dict[str, Any],
    action: Phi4Action,
) -> None:
    blocked = block_field(fine_test, kernel)
    blocked_identity = block_field(fine_test, identity)
    blocked_features = feature_series(blocked)
    coarse_features = feature_series(coarse_test)
    shifts = [
        metrics["L16_to_L8"]["trained"]["test"]["features"][name][
            "standardized_mean_shift"
        ]
        for name in FEATURE_NAMES
    ]
    identity_shifts = [
        metrics["L16_to_L8"]["identity"]["test"]["features"][name][
            "standardized_mean_shift"
        ]
        for name in FEATURE_NAMES
    ]

    figure, axes = plt.subplots(2, 3, figsize=(15.0, 8.5))
    for axis, index, name in zip(
        axes[0, :2],
        (0, 1),
        FEATURE_NAMES[:2],
    ):
        axis.hist(
            coarse_features[:, index],
            bins=30,
            density=True,
            histtype="step",
            label="native L8",
        )
        axis.hist(
            feature_series(blocked_identity)[:, index],
            bins=30,
            density=True,
            histtype="step",
            label="identity",
        )
        axis.hist(
            blocked_features[:, index],
            bins=30,
            density=True,
            histtype="step",
            label="trained K",
        )
        axis.set_title(name)
        axis.legend(frameon=False)

    action_volume = coarse_test.shape[-1] ** 2
    axes[0, 2].hist(
        action(coarse_test) / action_volume,
        bins=30,
        density=True,
        histtype="step",
        label="native L8",
    )
    axes[0, 2].hist(
        action(blocked) / action_volume,
        bins=30,
        density=True,
        histtype="step",
        label="trained K",
    )
    axes[0, 2].set_title("action density")
    axes[0, 2].legend(frameon=False)

    positions = np.arange(len(FEATURE_NAMES))
    width = 0.38
    axes[1, 0].bar(positions - width / 2, identity_shifts, width, label="identity")
    axes[1, 0].bar(positions + width / 2, shifts, width, label="trained K")
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set_xticks(positions, FEATURE_NAMES, rotation=45, ha="right")
    axes[1, 0].set_ylabel("standardized mean shift")
    axes[1, 0].legend(frameon=False)

    image = axes[1, 1].imshow(kernel, cmap="coolwarm")
    axes[1, 1].set_title("trained 5x5 kernel")
    figure.colorbar(image, ax=axes[1, 1], fraction=0.046)

    spectrum = np.fft.fftshift(np.abs(kernel_transform(kernel, 128)))
    spectrum_image = axes[1, 2].imshow(spectrum, cmap="viridis")
    axes[1, 2].set_title(r"$|K(p)|$ on dense grid")
    figure.colorbar(spectrum_image, ax=axes[1, 2], fraction=0.046)

    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def train_blocking_kernel(
    fine: FloatArray,
    coarse: FloatArray,
    transfer_fine: FloatArray,
    *,
    action: Phi4Action,
    starts: int = 12,
    max_iterations: int = 400,
    seed: int = 1234,
    shrinkage: float = 0.1,
) -> tuple[dict[str, Any], dict[str, Any], FloatArray]:
    """Train on chains 0-1, select on chain 2, and evaluate on chain 3."""
    train_fine = fine[:2].reshape(-1, fine.shape[-1], fine.shape[-1])
    train_coarse = coarse[:2].reshape(-1, coarse.shape[-1], coarse.shape[-1])
    validation_fine = fine[2]
    validation_coarse = coarse[2]
    test_fine = fine[3]
    test_coarse = coarse[3]
    transfer_test = transfer_fine[3]

    train_design = _blocking_design(train_fine)
    validation_design = _blocking_design(validation_fine)
    target_features = feature_series(train_coarse)
    target_mean = np.mean(target_features, axis=0)
    inverse_covariance = _whitening_matrix(target_features, shrinkage)

    def objective(parameters: FloatArray) -> float:
        blocked = _apply_design(train_design, parameters)
        return (
            _matching_loss(blocked, target_mean, inverse_covariance)
            + _spectrum_penalty(parameters)
        )

    rng = np.random.default_rng(seed)
    initial_points = np.vstack(
        (
            np.zeros((1, 5), dtype=np.float64),
            np.clip(rng.normal(0.0, 0.025, size=(starts - 1, 5)), -0.1, 0.1),
        )
    )
    candidates = []
    results = []
    for start_index, initial in enumerate(initial_points):
        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            bounds=[(-0.1, 0.1)] * 5,
            options={"maxiter": max_iterations, "ftol": 1.0e-12},
        )
        parameters = np.asarray(result.x, dtype=np.float64)
        kernel = kernel_from_parameters(parameters)
        validation_loss = _matching_loss(
            _apply_design(validation_design, parameters),
            np.mean(feature_series(validation_coarse), axis=0),
            inverse_covariance,
        )
        spectrum = spectrum_diagnostics(kernel, 64)
        feasible = (
            spectrum["min_abs_K"] >= 0.5
            and spectrum["condition_number"] <= 3.0
        )
        selection_score = validation_loss + (0.0 if feasible else 1.0e6)
        candidates.append(
            {
                "start": start_index,
                "success": bool(result.success),
                "iterations": int(result.nit),
                "training_objective": float(result.fun),
                "validation_loss": validation_loss,
                "selection_score": selection_score,
                "parameters": parameters,
                "spectrum_L64_grid": spectrum,
            }
        )
        results.append(parameters)

    selected_index = int(np.argmin([item["selection_score"] for item in candidates]))
    parameters = results[selected_index]
    kernel = kernel_from_parameters(parameters)
    identity = identity_kernel()

    split_fields = {
        "train": (train_fine, train_coarse),
        "validation": (validation_fine, validation_coarse),
        "test": (test_fine, test_coarse),
    }
    pair_metrics: dict[str, Any] = {"identity": {}, "trained": {}}
    for label, current_kernel in (("identity", identity), ("trained", kernel)):
        for split, (split_fine, split_coarse) in split_fields.items():
            pair_metrics[label][split] = evaluate_pair(
                split_fine, split_coarse, current_kernel, action
            )

    transfer_metrics = {
        "identity": evaluate_pair(transfer_test, test_fine, identity, action),
        "trained": evaluate_pair(transfer_test, test_fine, kernel, action),
    }
    spectra = {
        f"L{size}": spectrum_diagnostics(kernel, size)
        for size in (8, 16, 32, 256)
    }
    test_summary = pair_metrics["trained"]["test"]["summary"]
    identity_test_summary = pair_metrics["identity"]["test"]["summary"]
    transfer_summary = transfer_metrics["trained"]["summary"]
    spectrum_pass = all(
        item["min_abs_K"] >= 0.5 and item["condition_number"] <= 3.0
        for item in spectra.values()
    )
    acceptance = {
        "test_rms_shift": test_summary["rms_standardized_shift"] <= 0.25,
        "identity_improvement": (
            test_summary["rms_standardized_shift"]
            <= 0.5 * identity_test_summary["rms_standardized_shift"]
        ),
        "test_max_shift": test_summary["max_abs_standardized_shift"] <= 0.5,
        "spectrum": spectrum_pass,
        "transfer_rms_shift": transfer_summary["rms_standardized_shift"] <= 0.25,
    }
    acceptance["all"] = all(acceptance.values())

    kernel_record = {
        "type": "D4_symmetric_5x5",
        "eta": ETA,
        "normalization": "sum(K) = 2^(eta/2)",
        "kernel_sum": KERNEL_SUM,
        "kernel_coefficients_include_eta_scale": True,
        "downsampling": "even-even",
        "orbit_parameters": dict(zip(ORBIT_NAMES, parameters)),
        "center_coefficient": float(kernel[2, 2]),
        "matrix": kernel,
        "data_split": {
            "training_chains": [0, 1],
            "validation_chain": 2,
            "test_chain": 3,
            "transfer_test_chain": 3,
        },
        "optimizer": {
            "method": "L-BFGS-B",
            "bounds": [-0.1, 0.1],
            "starts": starts,
            "max_iterations": max_iterations,
            "seed": seed,
            "covariance_shrinkage": shrinkage,
        },
        "selected_start": selected_index,
        "candidates": candidates,
        "spectrum": spectra,
    }
    metrics = {
        "L16_to_L8": pair_metrics,
        "L32_to_L16_transfer_test": transfer_metrics,
        "acceptance": acceptance,
    }
    return kernel_record, metrics, kernel


def run_training(
    fine_path: Path,
    coarse_path: Path,
    transfer_fine_path: Path,
    output: Path,
    *,
    starts: int = 12,
    max_iterations: int = 400,
    seed: int = 1234,
    bootstrap_samples: int = 2000,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load ensembles, train a kernel, and write the operational artifacts."""
    with np.load(fine_path) as data:
        fine = np.asarray(data["configurations"], dtype=np.float64)
        kappa = float(data["kappa"])
        lam = float(data["lam"])
    with np.load(coarse_path) as data:
        coarse = np.asarray(data["configurations"], dtype=np.float64)
        coarse_kappa = float(data["kappa"])
        coarse_lam = float(data["lam"])
    with np.load(transfer_fine_path) as data:
        transfer_fine = np.asarray(data["configurations"], dtype=np.float64)
        transfer_kappa = float(data["kappa"])
        transfer_lam = float(data["lam"])

    if fine.shape[0] != 4 or coarse.shape[0] != 4 or transfer_fine.shape[0] != 4:
        raise ValueError("kernel training requires exactly four chains per ensemble")
    if fine.shape[-1] != 2 * coarse.shape[-1]:
        raise ValueError("fine lattice size must be twice the coarse lattice size")
    if transfer_fine.shape[-1] != 2 * fine.shape[-1]:
        raise ValueError("transfer lattice size must be twice the fine lattice size")
    if (coarse_kappa, coarse_lam) != (kappa, lam) or (
        transfer_kappa,
        transfer_lam,
    ) != (kappa, lam):
        raise ValueError("all ensembles must use the same kappa and lambda")

    action = Phi4Action(kappa=kappa, lam=lam)
    kernel_record, metrics, kernel = train_blocking_kernel(
        fine,
        coarse,
        transfer_fine,
        action=action,
        starts=starts,
        max_iterations=max_iterations,
        seed=seed,
    )
    kernel_record["theory"] = {"kappa": kappa, "lambda": lam}
    kernel_record["sources"] = {
        "fine": fine_path,
        "coarse": coarse_path,
        "transfer_fine": transfer_fine_path,
    }

    output.mkdir(parents=True, exist_ok=True)
    metrics["operator_distributions"] = save_operator_distribution_figure(
        output / "operator_distributions.pdf",
        coarse,
        block_field(fine, kernel),
        action,
        bootstrap_samples=bootstrap_samples,
    )
    _write_json(output / "kernel.json", kernel_record)
    _write_json(output / "metrics.json", metrics)
    _save_diagnostics(
        output / "diagnostics.pdf",
        fine[3],
        coarse[3],
        kernel,
        identity_kernel(),
        metrics,
        action,
    )
    return kernel_record, metrics
