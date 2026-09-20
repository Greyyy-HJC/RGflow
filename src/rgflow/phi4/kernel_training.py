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
    SOS_PARAMETERS_PER_CHANNEL,
    block_field,
    identity_kernel,
    kernel_from_parameters,
    kernel_transform,
    orbit_parameters_from_kernel,
    sos_filters_from_parameters,
    sos_kernel_from_parameters,
    spectrum_diagnostics,
)
from .operator_diagnostics import (
    KERNEL_OBSERVABLE_NAMES,
    distribution_metrics,
    kernel_observable_series,
    save_kernel_observable_histograms,
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


SUPPORT_QUANTILES = np.array(
    [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99],
    dtype=np.float64,
)


def _distribution_targets(
    fields: FloatArray,
    action: Phi4Action,
) -> tuple[dict[str, FloatArray], dict[str, float]]:
    series = kernel_observable_series(fields, action)
    quantiles = {
        name: np.quantile(series[name], SUPPORT_QUANTILES)
        for name in KERNEL_OBSERVABLE_NAMES
    }
    scales = {
        name: max(float(np.std(series[name], ddof=1)), 1.0e-12)
        for name in KERNEL_OBSERVABLE_NAMES
    }
    return quantiles, scales


def _distribution_support_loss(
    fields: FloatArray,
    targets: tuple[dict[str, FloatArray], dict[str, float]],
    action: Phi4Action,
) -> float:
    target_quantiles, target_scales = targets
    series = kernel_observable_series(fields, action)
    losses = [
        np.mean(
            (
                (
                    np.quantile(series[name], SUPPORT_QUANTILES)
                    - target_quantiles[name]
                )
                / target_scales[name]
            )
            ** 2
        )
        for name in KERNEL_OBSERVABLE_NAMES
    ]
    return float(np.mean(losses))


def _spectrum_penalty(kernel: FloatArray) -> float:
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
    pair_metrics: dict[str, Any],
    action: Phi4Action,
) -> None:
    blocked = block_field(fine_test, kernel)
    blocked_identity = block_field(fine_test, identity)
    blocked_features = feature_series(blocked)
    coarse_features = feature_series(coarse_test)
    shifts = [
        pair_metrics["trained"]["test"]["features"][name]["standardized_mean_shift"]
        for name in FEATURE_NAMES
    ]
    identity_shifts = [
        pair_metrics["identity"]["test"]["features"][name]["standardized_mean_shift"]
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
    support_weight: float = 1.0,
    sos_channels: int = 2,
    sos_floor_fraction: float = 0.05,
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
    distribution_targets = _distribution_targets(train_coarse, action)

    parameter_count = SOS_PARAMETERS_PER_CHANNEL * sos_channels
    parameter_bound = 0.5
    initial_scale = 0.08

    def build_kernel(parameters: FloatArray) -> FloatArray:
        return sos_kernel_from_parameters(
            parameters,
            channels=sos_channels,
            floor_fraction=sos_floor_fraction,
        )

    def design_coefficients(parameters: FloatArray) -> FloatArray:
        return orbit_parameters_from_kernel(build_kernel(parameters))

    def objective(parameters: FloatArray) -> float:
        kernel = build_kernel(parameters)
        blocked = _apply_design(train_design, design_coefficients(parameters))
        return (
            _matching_loss(blocked, target_mean, inverse_covariance)
            + support_weight
            * _distribution_support_loss(blocked, distribution_targets, action)
            + _spectrum_penalty(kernel)
        )

    rng = np.random.default_rng(seed)
    initial_points = np.vstack(
        (
            np.zeros((1, parameter_count), dtype=np.float64),
            np.clip(
                rng.normal(
                    0.0,
                    initial_scale,
                    size=(starts - 1, parameter_count),
                ),
                -parameter_bound,
                parameter_bound,
            ),
        )
    )
    candidates = []
    results = []
    for start_index, initial in enumerate(initial_points):
        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            bounds=[(-parameter_bound, parameter_bound)] * parameter_count,
            options={"maxiter": max_iterations, "ftol": 1.0e-12},
        )
        parameters = np.asarray(result.x, dtype=np.float64)
        kernel = build_kernel(parameters)
        orbit_parameters = orbit_parameters_from_kernel(kernel)
        training_blocked = _apply_design(train_design, orbit_parameters)
        validation_blocked = _apply_design(validation_design, orbit_parameters)
        training_mean_loss = _matching_loss(
            training_blocked,
            target_mean,
            inverse_covariance,
        )
        training_distribution_loss = _distribution_support_loss(
            training_blocked,
            distribution_targets,
            action,
        )
        validation_mean_loss = _matching_loss(
            validation_blocked,
            np.mean(feature_series(validation_coarse), axis=0),
            inverse_covariance,
        )
        validation_distribution_loss = _distribution_support_loss(
            validation_blocked,
            _distribution_targets(validation_coarse, action),
            action,
        )
        validation_loss = (
            validation_mean_loss + support_weight * validation_distribution_loss
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
                "training_mean_loss": training_mean_loss,
                "training_distribution_loss": training_distribution_loss,
                "validation_loss": validation_loss,
                "validation_mean_loss": validation_mean_loss,
                "validation_distribution_loss": validation_distribution_loss,
                "selection_score": selection_score,
                "parameters": parameters,
                "orbit_parameters": dict(zip(ORBIT_NAMES, orbit_parameters)),
                "spectrum_L64_grid": spectrum,
            }
        )
        results.append(parameters)

    selected_index = int(np.argmin([item["selection_score"] for item in candidates]))
    parameters = results[selected_index]
    kernel = build_kernel(parameters)
    orbit_parameters = orbit_parameters_from_kernel(kernel)
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
        "type": "D4_symmetric_5x5_multichannel_SOS",
        "parameterization": "multichannel_sos",
        "invertibility": "by_construction_positive_spectrum",
        "eta": ETA,
        "normalization": "sum(K) = 2^(eta/2)",
        "kernel_sum": KERNEL_SUM,
        "kernel_coefficients_include_eta_scale": True,
        "downsampling": "even-even",
        "orbit_parameters": dict(zip(ORBIT_NAMES, orbit_parameters)),
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
            "bounds": [-parameter_bound, parameter_bound],
            "starts": starts,
            "max_iterations": max_iterations,
            "seed": seed,
            "covariance_shrinkage": shrinkage,
            "distribution_support_weight": support_weight,
            "distribution_support_quantiles": SUPPORT_QUANTILES,
            "distribution_support_observables": KERNEL_OBSERVABLE_NAMES,
        },
        "selected_start": selected_index,
        "sos": {
            "channels": sos_channels,
            "parameters_per_channel": SOS_PARAMETERS_PER_CHANNEL,
            "floor_fraction": sos_floor_fraction,
            "guaranteed_spectral_floor": KERNEL_SUM * sos_floor_fraction,
            "factor_sums": [1.0] + [0.0] * (sos_channels - 1),
            "filters": sos_filters_from_parameters(parameters, sos_channels),
        },
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
    support_weight: float = 1.0,
    sos_channels: int = 2,
    sos_floor_fraction: float = 0.05,
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
        support_weight=support_weight,
        sos_channels=sos_channels,
        sos_floor_fraction=sos_floor_fraction,
    )
    kernel_record["theory"] = {"kappa": kappa, "lambda": lam}
    kernel_record["sources"] = {
        "fine": fine_path,
        "coarse": coarse_path,
        "transfer_fine": transfer_fine_path,
    }

    output.mkdir(parents=True, exist_ok=True)
    blocked = block_field(fine, kernel)
    metrics["kernel_observable_distributions"] = save_kernel_observable_histograms(
        output / "kernel_observable_histograms.pdf",
        coarse,
        blocked,
        action,
    )
    metrics["operator_distributions"] = save_operator_distribution_figure(
        output / "operator_distributions.pdf",
        coarse,
        blocked,
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
        metrics["L16_to_L8"],
        action,
    )
    return kernel_record, metrics


def _physical_training_vector(
    fields: FloatArray,
    action: Phi4Action,
) -> FloatArray:
    """Return stable ensemble-level critical observables for optimization."""
    scalars = _ensemble_scalars(fields, action)
    lattice_size = fields.shape[-1]
    return np.array(
        [
            scalars["binder_cumulant"],
            scalars["susceptibility"] / lattice_size ** (2.0 - ETA),
        ],
        dtype=np.float64,
    )


def _physical_training_loss(
    fields: FloatArray,
    target: FloatArray,
    scales: FloatArray,
    action: Phi4Action,
) -> float:
    difference = (_physical_training_vector(fields, action) - target) / scales
    return float(np.mean(difference**2))


def _conditioning_penalty(
    kernel: FloatArray,
    target_condition: float = 2.0,
) -> float:
    condition = spectrum_diagnostics(kernel, 64)["condition_number"]
    return max(0.0, condition - target_condition) ** 2


def _load_training_ensemble(path: Path) -> tuple[FloatArray, float, float]:
    with np.load(path) as data:
        return (
            np.asarray(data["configurations"], dtype=np.float64),
            float(data["kappa"]),
            float(data["lam"]),
        )


def train_multivolume_blocking_kernel(
    training_pairs: dict[str, tuple[FloatArray, FloatArray]],
    transfer_pairs: dict[str, tuple[FloatArray, FloatArray]],
    *,
    action: Phi4Action,
    starts: int = 12,
    max_iterations: int = 400,
    optimization_samples_per_chain: int = 128,
    seed: int = 1234,
    shrinkage: float = 0.1,
    support_weight: float = 1.0,
    physical_weight: float = 1.0,
    conditioning_weight: float = 0.05,
    sos_channels: int = 2,
    sos_floor_fraction: float = 0.05,
) -> tuple[dict[str, Any], dict[str, Any], FloatArray]:
    """Train one kernel on equally weighted one-step volume pairs."""
    prepared: dict[str, dict[str, Any]] = {}
    for label, (fine, coarse) in training_pairs.items():
        train_fine = fine[:2].reshape(-1, fine.shape[-1], fine.shape[-1])
        train_coarse = coarse[:2].reshape(-1, coarse.shape[-1], coarse.shape[-1])
        sample_count = min(
            optimization_samples_per_chain,
            fine.shape[1],
            coarse.shape[1],
        )
        objective_fine = fine[:2, :sample_count].reshape(
            -1,
            fine.shape[-1],
            fine.shape[-1],
        )
        objective_coarse = coarse[:2, :sample_count].reshape(
            -1,
            coarse.shape[-1],
            coarse.shape[-1],
        )
        validation_fine = fine[2]
        validation_coarse = coarse[2]
        target_features = feature_series(objective_coarse)
        physical_target = _physical_training_vector(objective_coarse, action)
        validation_physical_target = _physical_training_vector(
            validation_coarse,
            action,
        )
        prepared[label] = {
            "optimization_samples_per_chain": sample_count,
            "fields": {
                "train": (train_fine, train_coarse),
                "validation": (validation_fine, validation_coarse),
                "test": (fine[3], coarse[3]),
            },
            "train_design": _blocking_design(objective_fine),
            "validation_design": _blocking_design(validation_fine),
            "target_mean": np.mean(target_features, axis=0),
            "inverse_covariance": _whitening_matrix(target_features, shrinkage),
            "distribution_targets": _distribution_targets(objective_coarse, action),
            "physical_target": physical_target,
            "physical_scales": np.maximum(np.abs(physical_target), 0.1),
            "validation_target_mean": np.mean(
                feature_series(validation_coarse),
                axis=0,
            ),
            "validation_distribution_targets": _distribution_targets(
                validation_coarse,
                action,
            ),
            "validation_physical_target": validation_physical_target,
            "validation_physical_scales": np.maximum(
                np.abs(validation_physical_target),
                0.1,
            ),
        }

    parameter_count = SOS_PARAMETERS_PER_CHANNEL * sos_channels
    parameter_bound = 0.5
    initial_scale = 0.08

    def build_kernel(parameters: FloatArray) -> FloatArray:
        return sos_kernel_from_parameters(
            parameters,
            channels=sos_channels,
            floor_fraction=sos_floor_fraction,
        )

    def pair_loss(
        blocked: FloatArray,
        *,
        target_mean: FloatArray,
        inverse_covariance: FloatArray,
        distribution_targets: tuple[dict[str, FloatArray], dict[str, float]],
        physical_target: FloatArray,
        physical_scales: FloatArray,
    ) -> tuple[float, float, float]:
        mean_loss = _matching_loss(blocked, target_mean, inverse_covariance)
        distribution_loss = _distribution_support_loss(
            blocked,
            distribution_targets,
            action,
        )
        physical_loss = _physical_training_loss(
            blocked,
            physical_target,
            physical_scales,
            action,
        )
        return mean_loss, distribution_loss, physical_loss

    def total_pair_loss(losses: tuple[float, float, float]) -> float:
        mean_loss, distribution_loss, physical_loss = losses
        return (
            mean_loss
            + support_weight * distribution_loss
            + physical_weight * physical_loss
        )

    def objective(parameters: FloatArray) -> float:
        kernel = build_kernel(parameters)
        coefficients = orbit_parameters_from_kernel(kernel)
        losses = [
            total_pair_loss(
                pair_loss(
                    _apply_design(item["train_design"], coefficients),
                    target_mean=item["target_mean"],
                    inverse_covariance=item["inverse_covariance"],
                    distribution_targets=item["distribution_targets"],
                    physical_target=item["physical_target"],
                    physical_scales=item["physical_scales"],
                )
            )
            for item in prepared.values()
        ]
        return (
            float(np.mean(losses))
            + _spectrum_penalty(kernel)
            + conditioning_weight * _conditioning_penalty(kernel)
        )

    rng = np.random.default_rng(seed)
    initial_points = np.vstack(
        (
            np.zeros((1, parameter_count), dtype=np.float64),
            np.clip(
                rng.normal(
                    0.0,
                    initial_scale,
                    size=(starts - 1, parameter_count),
                ),
                -parameter_bound,
                parameter_bound,
            ),
        )
    )
    candidates = []
    results = []
    for start_index, initial in enumerate(initial_points):
        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            bounds=[(-parameter_bound, parameter_bound)] * parameter_count,
            options={"maxiter": max_iterations, "ftol": 1.0e-12},
        )
        parameters = np.asarray(result.x, dtype=np.float64)
        kernel = build_kernel(parameters)
        coefficients = orbit_parameters_from_kernel(kernel)
        training_losses = {}
        validation_losses = {}
        for label, item in prepared.items():
            training_losses[label] = pair_loss(
                _apply_design(item["train_design"], coefficients),
                target_mean=item["target_mean"],
                inverse_covariance=item["inverse_covariance"],
                distribution_targets=item["distribution_targets"],
                physical_target=item["physical_target"],
                physical_scales=item["physical_scales"],
            )
            validation_losses[label] = pair_loss(
                _apply_design(item["validation_design"], coefficients),
                target_mean=item["validation_target_mean"],
                inverse_covariance=item["inverse_covariance"],
                distribution_targets=item["validation_distribution_targets"],
                physical_target=item["validation_physical_target"],
                physical_scales=item["validation_physical_scales"],
            )
        training_loss = float(
            np.mean([total_pair_loss(losses) for losses in training_losses.values()])
        )
        validation_loss = float(
            np.mean([total_pair_loss(losses) for losses in validation_losses.values()])
        )
        spectrum = spectrum_diagnostics(kernel, 64)
        feasible = (
            spectrum["min_abs_K"] >= 0.5
            and spectrum["condition_number"] <= 3.0
        )
        selection_score = (
            validation_loss
            + conditioning_weight * _conditioning_penalty(kernel)
            + (0.0 if feasible else 1.0e6)
        )
        candidates.append(
            {
                "start": start_index,
                "success": bool(result.success),
                "iterations": int(result.nit),
                "training_objective": float(result.fun),
                "training_loss": training_loss,
                "validation_loss": validation_loss,
                "selection_score": selection_score,
                "training_pair_losses": {
                    label: {
                        "mean": losses[0],
                        "distribution": losses[1],
                        "physical": losses[2],
                    }
                    for label, losses in training_losses.items()
                },
                "validation_pair_losses": {
                    label: {
                        "mean": losses[0],
                        "distribution": losses[1],
                        "physical": losses[2],
                    }
                    for label, losses in validation_losses.items()
                },
                "parameters": parameters,
                "orbit_parameters": dict(zip(ORBIT_NAMES, coefficients)),
                "spectrum_L64_grid": spectrum,
            }
        )
        results.append(parameters)

    selected_index = int(np.argmin([item["selection_score"] for item in candidates]))
    parameters = results[selected_index]
    kernel = build_kernel(parameters)
    orbit_parameters = orbit_parameters_from_kernel(kernel)
    identity = identity_kernel()

    training_metrics: dict[str, Any] = {}
    for pair_label, item in prepared.items():
        training_metrics[pair_label] = {"identity": {}, "trained": {}}
        for kernel_label, current_kernel in (
            ("identity", identity),
            ("trained", kernel),
        ):
            for split, (fine, coarse) in item["fields"].items():
                training_metrics[pair_label][kernel_label][split] = evaluate_pair(
                    fine,
                    coarse,
                    current_kernel,
                    action,
                )

    transfer_metrics = {
        pair_label: {
            "identity": evaluate_pair(fine[3], coarse[3], identity, action),
            "trained": evaluate_pair(fine[3], coarse[3], kernel, action),
        }
        for pair_label, (fine, coarse) in transfer_pairs.items()
    }
    spectrum_sizes = {
        field.shape[-1]
        for pair in (*training_pairs.values(), *transfer_pairs.values())
        for field in pair
    }
    spectra = {
        f"L{size}": spectrum_diagnostics(kernel, size)
        for size in sorted((*spectrum_sizes, 256))
    }
    training_test = [
        pair["trained"]["test"]["summary"]
        for pair in training_metrics.values()
    ]
    identity_test = [
        pair["identity"]["test"]["summary"]
        for pair in training_metrics.values()
    ]
    transfer_test = [
        pair["trained"]["summary"]
        for pair in transfer_metrics.values()
    ]
    spectrum_pass = all(
        item["min_abs_K"] >= 0.5 and item["condition_number"] <= 3.0
        for item in spectra.values()
    )
    acceptance = {
        "training_test_rms_shift": max(
            item["rms_standardized_shift"] for item in training_test
        )
        <= 0.25,
        "training_identity_improvement": all(
            trained["rms_standardized_shift"]
            <= 0.5 * baseline["rms_standardized_shift"]
            for trained, baseline in zip(training_test, identity_test)
        ),
        "training_test_max_shift": max(
            item["max_abs_standardized_shift"] for item in training_test
        )
        <= 0.5,
        "spectrum": spectrum_pass,
        "transfer_test_rms_shift": max(
            item["rms_standardized_shift"] for item in transfer_test
        )
        <= 0.25,
    }
    acceptance["all"] = all(acceptance.values())

    kernel_record = {
        "type": "D4_symmetric_5x5_multichannel_SOS",
        "parameterization": "multichannel_sos",
        "training_strategy": "equal_weight_multivolume_one_step",
        "invertibility": "by_construction_positive_spectrum",
        "eta": ETA,
        "normalization": "sum(K) = 2^(eta/2)",
        "kernel_sum": KERNEL_SUM,
        "kernel_coefficients_include_eta_scale": True,
        "downsampling": "even-even",
        "orbit_parameters": dict(zip(ORBIT_NAMES, orbit_parameters)),
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
            "bounds": [-parameter_bound, parameter_bound],
            "starts": starts,
            "max_iterations": max_iterations,
            "optimization_samples_per_chain": {
                label: item["optimization_samples_per_chain"]
                for label, item in prepared.items()
            },
            "seed": seed,
            "covariance_shrinkage": shrinkage,
            "volume_weighting": "equal_per_training_pair",
            "distribution_support_weight": support_weight,
            "physical_observable_weight": physical_weight,
            "physical_observables": [
                "binder_cumulant",
                "susceptibility_over_L_to_2_minus_eta",
            ],
            "conditioning_weight": conditioning_weight,
            "conditioning_target": 2.0,
            "distribution_support_quantiles": SUPPORT_QUANTILES,
            "distribution_support_observables": KERNEL_OBSERVABLE_NAMES,
        },
        "selected_start": selected_index,
        "sos": {
            "channels": sos_channels,
            "parameters_per_channel": SOS_PARAMETERS_PER_CHANNEL,
            "floor_fraction": sos_floor_fraction,
            "guaranteed_spectral_floor": KERNEL_SUM * sos_floor_fraction,
            "factor_sums": [1.0] + [0.0] * (sos_channels - 1),
            "filters": sos_filters_from_parameters(parameters, sos_channels),
        },
        "candidates": candidates,
        "spectrum": spectra,
    }
    metrics = {
        "training_pairs": training_metrics,
        "transfer_pairs": transfer_metrics,
        "acceptance": acceptance,
    }
    return kernel_record, metrics, kernel


def run_multivolume_training(
    training_pair_paths: list[tuple[Path, Path]],
    transfer_pair_paths: list[tuple[Path, Path]],
    output: Path,
    *,
    starts: int = 6,
    max_iterations: int = 250,
    optimization_samples_per_chain: int = 128,
    seed: int = 1234,
    bootstrap_samples: int = 2000,
    support_weight: float = 1.0,
    physical_weight: float = 1.0,
    conditioning_weight: float = 0.05,
    sos_channels: int = 2,
    sos_floor_fraction: float = 0.05,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load volume pairs, train one-step losses, and write artifacts."""
    theory: tuple[float, float] | None = None

    def load_pairs(
        paths: list[tuple[Path, Path]],
    ) -> dict[str, tuple[FloatArray, FloatArray]]:
        nonlocal theory
        loaded = {}
        for fine_path, coarse_path in paths:
            fine, fine_kappa, fine_lam = _load_training_ensemble(fine_path)
            coarse, coarse_kappa, coarse_lam = _load_training_ensemble(coarse_path)
            if fine.shape[0] != 4 or coarse.shape[0] != 4:
                raise ValueError(
                    "kernel training requires exactly four chains per ensemble"
                )
            if fine.shape[-1] != 2 * coarse.shape[-1]:
                raise ValueError(
                    "each fine lattice size must be twice its coarse lattice size"
                )
            pair_theory = (fine_kappa, fine_lam)
            if (coarse_kappa, coarse_lam) != pair_theory:
                raise ValueError("each volume pair must use one kappa and lambda")
            if theory is None:
                theory = pair_theory
            elif theory != pair_theory:
                raise ValueError("all volume pairs must use the same kappa and lambda")
            label = f"L{fine.shape[-1]}_to_L{coarse.shape[-1]}"
            if label in loaded:
                raise ValueError(f"duplicate volume pair: {label}")
            loaded[label] = (fine, coarse)
        return loaded

    training_pairs = load_pairs(training_pair_paths)
    transfer_pairs = load_pairs(transfer_pair_paths)
    if not training_pairs or not transfer_pairs or theory is None:
        raise ValueError("at least one training and one transfer pair are required")

    action = Phi4Action(kappa=theory[0], lam=theory[1])
    kernel_record, metrics, kernel = train_multivolume_blocking_kernel(
        training_pairs,
        transfer_pairs,
        action=action,
        starts=starts,
        max_iterations=max_iterations,
        optimization_samples_per_chain=optimization_samples_per_chain,
        seed=seed,
        support_weight=support_weight,
        physical_weight=physical_weight,
        conditioning_weight=conditioning_weight,
        sos_channels=sos_channels,
        sos_floor_fraction=sos_floor_fraction,
    )
    kernel_record["theory"] = {"kappa": theory[0], "lambda": theory[1]}
    kernel_record["sources"] = {
        "training_pairs": {
            label: {"fine": paths[0], "coarse": paths[1]}
            for label, paths in zip(training_pairs, training_pair_paths)
        },
        "transfer_pairs": {
            label: {"fine": paths[0], "coarse": paths[1]}
            for label, paths in zip(transfer_pairs, transfer_pair_paths)
        },
    }

    primary_label = next(iter(training_pairs))
    primary_fine, primary_coarse = training_pairs[primary_label]
    output.mkdir(parents=True, exist_ok=True)
    blocked = block_field(primary_fine, kernel)
    metrics["kernel_observable_distributions"] = save_kernel_observable_histograms(
        output / "kernel_observable_histograms.pdf",
        primary_coarse,
        blocked,
        action,
    )
    metrics["operator_distributions"] = save_operator_distribution_figure(
        output / "operator_distributions.pdf",
        primary_coarse,
        blocked,
        action,
        bootstrap_samples=bootstrap_samples,
    )
    _write_json(output / "kernel.json", kernel_record)
    _write_json(output / "metrics.json", metrics)
    _save_diagnostics(
        output / "diagnostics.pdf",
        primary_fine[3],
        primary_coarse[3],
        kernel,
        identity_kernel(),
        metrics["training_pairs"][primary_label],
        action,
    )
    return kernel_record, metrics
