"""Exact physical-coordinate MH after conditional-flow inverse blocking."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from rgflow.diagnostics import integrated_autocorrelation_time

from .action import Phi4Action
from .affine_flow import model_from_checkpoint
from .blocking import apply_kernel, inverse_kernel
from .flow_training import (
    load_configurations,
    physical_detail_log_prob,
    sample_physical_details,
)
from .inverse_blocking import (
    DETAIL_NAMES,
    assemble_details,
    extract_details,
    reblocking_error,
)
from .sampling import wolff_cluster_update
from .operator_diagnostics import (
    KERNEL_OBSERVABLE_NAMES,
    bootstrap_ensemble_observables,
    kernel_observable_series,
)


@dataclass
class InverseBlockingState:
    coarse: np.ndarray
    details: np.ndarray
    psi: np.ndarray
    fine: np.ndarray
    fine_action: np.ndarray


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def build_state(
    coarse: np.ndarray,
    details: np.ndarray,
    kernel: np.ndarray,
    action: Phi4Action,
) -> InverseBlockingState:
    coarse = np.asarray(coarse, dtype=np.float64)
    details = np.asarray(details, dtype=np.float64)
    psi = assemble_details(coarse, details)
    fine = inverse_kernel(psi, kernel)
    return InverseBlockingState(coarse, details, psi, fine, action(fine))


def choose_state(
    old: InverseBlockingState,
    proposed: InverseBlockingState,
    accepted: np.ndarray,
) -> InverseBlockingState:
    def choose(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        shape = (-1,) + (1,) * (left.ndim - 1)
        return np.where(accepted.reshape(shape), right, left)

    return InverseBlockingState(
        coarse=choose(old.coarse, proposed.coarse),
        details=choose(old.details, proposed.details),
        psi=choose(old.psi, proposed.psi),
        fine=choose(old.fine, proposed.fine),
        fine_action=choose(old.fine_action, proposed.fine_action),
    )


def coarse_target_transition(
    coarse: np.ndarray,
    active: np.ndarray,
    *,
    sigma: float,
    action: Phi4Action,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, int]:
    """One reversible parallel local-MH transition targeting exp(-S_c)."""
    proposal = coarse.copy()
    noise = rng.standard_normal(coarse.shape)
    proposal[:, active] += sigma * noise[:, active]
    delta = action.local_delta(coarse, proposal)
    take = np.log(rng.random(coarse.shape)) < np.minimum(0.0, -delta)
    take[:, ~active] = False
    transitioned = np.where(take, proposal, coarse)
    return transitioned, int(np.sum(take)), int(len(coarse) * np.sum(active))


def detail_log_acceptance(old_action: np.ndarray, new_action: np.ndarray) -> np.ndarray:
    return -new_action + old_action


def coarse_log_acceptance(
    old_fine_action: np.ndarray,
    new_fine_action: np.ndarray,
    old_coarse_action: np.ndarray,
    new_coarse_action: np.ndarray,
) -> np.ndarray:
    return (
        -new_fine_action
        + old_fine_action
        + new_coarse_action
        - old_coarse_action
    )


def _accept(log_acceptance: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.log(rng.random(len(log_acceptance))) < np.minimum(0.0, log_acceptance)


def _checked_state(
    coarse: np.ndarray,
    details: np.ndarray,
    kernel: np.ndarray,
    action: Phi4Action,
    tolerance: float,
) -> tuple[InverseBlockingState, float]:
    state = build_state(coarse, details, kernel, action)
    error = float(np.max(reblocking_error(state.fine, state.coarse, kernel)))
    if error > tolerance:
        raise RuntimeError(f"inverse-blocking reblocking error {error:.3e}")
    return state, error


def mh_sweep(
    state: InverseBlockingState,
    kernel: np.ndarray,
    action: Phi4Action,
    widths: dict[str, float],
    rng: np.random.Generator,
    *,
    divide: int = 2,
    detail_passes: int = 1,
    reblocking_tolerance: float = 1.0e-9,
) -> tuple[InverseBlockingState, dict[str, Any]]:
    lattice_size = state.coarse.shape[-1]
    yy, xx = np.indices((lattice_size, lattice_size))
    subsets = [
        (row, column, (yy % divide == row) & (xx % divide == column))
        for row in range(divide)
        for column in range(divide)
    ]
    counts = {
        name: {"outer_accepts": 0, "outer_attempts": 0}
        for name in ("coarse", *DETAIL_NAMES)
    }
    counts["coarse"].update({"inner_accepts": 0, "inner_attempts": 0})
    max_reblocking_error = 0.0

    for row, column, active in subsets:
        old_coarse_action = action(state.coarse)
        coarse, inner_accepts, inner_attempts = coarse_target_transition(
            state.coarse,
            active,
            sigma=widths["coarse"],
            action=action,
            rng=rng,
        )
        proposed, error = _checked_state(
            coarse,
            state.details,
            kernel,
            action,
            reblocking_tolerance,
        )
        loga = coarse_log_acceptance(
            state.fine_action,
            proposed.fine_action,
            old_coarse_action,
            action(proposed.coarse),
        )
        accepted = _accept(loga, rng)
        state = choose_state(state, proposed, accepted)
        counts["coarse"]["outer_accepts"] += int(np.sum(accepted))
        counts["coarse"]["outer_attempts"] += len(accepted)
        counts["coarse"]["inner_accepts"] += inner_accepts
        counts["coarse"]["inner_attempts"] += inner_attempts
        counts["coarse"].setdefault("subsets", []).append(
            {"row": row, "column": column, "acceptance": float(np.mean(accepted))}
        )
        max_reblocking_error = max(max_reblocking_error, error)

    for _ in range(detail_passes):
        for sector, name in enumerate(DETAIL_NAMES):
            for row, column, active in subsets:
                details = state.details.copy()
                noise = rng.standard_normal(details[:, sector].shape)
                details[:, sector, active] += widths[name] * noise[:, active]
                proposed, error = _checked_state(
                    state.coarse,
                    details,
                    kernel,
                    action,
                    reblocking_tolerance,
                )
                loga = detail_log_acceptance(state.fine_action, proposed.fine_action)
                accepted = _accept(loga, rng)
                state = choose_state(state, proposed, accepted)
                counts[name]["outer_accepts"] += int(np.sum(accepted))
                counts[name]["outer_attempts"] += len(accepted)
                counts[name].setdefault("subsets", []).append(
                    {
                        "row": row,
                        "column": column,
                        "acceptance": float(np.mean(accepted)),
                    }
                )
                max_reblocking_error = max(max_reblocking_error, error)

    summary = {
        name: {
            **values,
            "outer_acceptance": values["outer_accepts"] / values["outer_attempts"],
        }
        for name, values in counts.items()
    }
    summary["coarse"]["inner_acceptance"] = (
        summary["coarse"]["inner_accepts"]
        / summary["coarse"]["inner_attempts"]
    )
    return state, {
        "coordinates": summary,
        "max_reblocking_error": max_reblocking_error,
    }



def flow_refresh_log_acceptance(
    old_action: np.ndarray,
    new_action: np.ndarray,
    old_logq: np.ndarray,
    new_logq: np.ndarray,
) -> np.ndarray:
    return -new_action + old_action + old_logq - new_logq


def flow_refresh_transition(
    state: InverseBlockingState,
    model: torch.nn.Module,
    normalization: dict[str, Any],
    kernel: np.ndarray,
    action: Phi4Action,
    rng: np.random.Generator,
    *,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> tuple[InverseBlockingState, dict[str, Any]]:
    old_logq = physical_detail_log_prob(
        model,
        state.coarse,
        state.details,
        normalization,
        device=device,
        batch_size=batch_size,
    )
    details, new_logq = sample_physical_details(
        model,
        state.coarse,
        normalization,
        device=device,
        batch_size=batch_size,
        seed=seed,
    )
    proposed = build_state(state.coarse, details, kernel, action)
    loga = flow_refresh_log_acceptance(
        state.fine_action, proposed.fine_action, old_logq, new_logq
    )
    accepted = _accept(loga, rng)
    return choose_state(state, proposed, accepted), {
        "acceptance": float(np.mean(accepted)),
        "accepts": int(np.sum(accepted)),
        "attempts": len(accepted),
    }


def wolff_state_transition(
    state: InverseBlockingState,
    kernel: np.ndarray,
    action: Phi4Action,
    rng: np.random.Generator,
) -> tuple[InverseBlockingState, dict[str, Any]]:
    fine = state.fine.copy()
    cluster_sizes = np.asarray(
        [wolff_cluster_update(field, action.kappa, rng) for field in fine],
        dtype=np.int64,
    )
    psi = apply_kernel(fine, kernel)
    coarse, details = extract_details(psi)
    transitioned = InverseBlockingState(coarse, details, psi, fine, action(fine))
    return transitioned, {
        "mean_cluster_fraction": float(
            np.mean(cluster_sizes) / fine.shape[-1] ** 2
        ),
        "cluster_sizes": cluster_sizes,
    }


def _torch_action(field: torch.Tensor, action: Phi4Action) -> torch.Tensor:
    field_sq = field.square()
    potential = field_sq + action.lam * (field_sq - 1.0).square()
    hopping = field * torch.roll(field, -1, dims=-2)
    hopping += field * torch.roll(field, -1, dims=-1)
    return (potential - 2.0 * action.kappa * hopping).sum(dim=(-2, -1))


def hmc_transition(
    state: InverseBlockingState,
    kernel: np.ndarray,
    action: Phi4Action,
    rng: np.random.Generator,
    *,
    device: torch.device,
    step_size: float,
    minimum_steps: int = 8,
    maximum_steps: int = 16,
) -> tuple[InverseBlockingState, dict[str, Any]]:
    position = torch.from_numpy(state.fine).to(device=device, dtype=torch.float64)
    momentum = torch.from_numpy(rng.standard_normal(state.fine.shape)).to(
        device=device, dtype=torch.float64
    )
    initial_momentum = momentum.clone()
    steps = int(rng.integers(minimum_steps, maximum_steps + 1))

    def gradient(value: torch.Tensor) -> torch.Tensor:
        value = value.detach().requires_grad_(True)
        return torch.autograd.grad(_torch_action(value, action).sum(), value)[0]

    momentum = momentum - 0.5 * step_size * gradient(position)
    for step in range(steps):
        position = position + step_size * momentum
        if step + 1 < steps:
            momentum = momentum - step_size * gradient(position)
    momentum = momentum - 0.5 * step_size * gradient(position)
    proposed_fine = position.detach().cpu().numpy()
    proposed_psi = apply_kernel(proposed_fine, kernel)
    proposed_coarse, proposed_details = extract_details(proposed_psi)
    proposed = InverseBlockingState(
        proposed_coarse,
        proposed_details,
        proposed_psi,
        proposed_fine,
        action(proposed_fine),
    )
    old_hamiltonian = state.fine_action + 0.5 * np.sum(
        initial_momentum.cpu().numpy() ** 2, axis=(-2, -1)
    )
    new_hamiltonian = proposed.fine_action + 0.5 * np.sum(
        momentum.detach().cpu().numpy() ** 2, axis=(-2, -1)
    )
    accepted = _accept(old_hamiltonian - new_hamiltonian, rng)
    return choose_state(state, proposed, accepted), {
        "acceptance": float(np.mean(accepted)),
        "accepts": int(np.sum(accepted)),
        "attempts": len(accepted),
        "leapfrog_steps": steps,
        "mean_delta_h": float(np.mean(new_hamiltonian - old_hamiltonian)),
    }


def correction_sweep(
    state: InverseBlockingState,
    correction: str,
    kernel: np.ndarray,
    action: Phi4Action,
    widths: dict[str, float],
    rng: np.random.Generator,
    *,
    detail_passes: int,
    divide: int,
    model: torch.nn.Module,
    normalization: dict[str, Any],
    device: torch.device,
    batch_size: int,
    refresh_seed: int,
    hmc_step_size: float,
) -> tuple[InverseBlockingState, dict[str, Any]]:
    diagnostics: dict[str, Any] = {}
    if correction == "hmc-wolff":
        state, diagnostics["hmc"] = hmc_transition(
            state,
            kernel,
            action,
            rng,
            device=device,
            step_size=hmc_step_size,
        )
    else:
        if correction == "flow-detail-wolff":
            state, diagnostics["flow_refresh"] = flow_refresh_transition(
                state,
                model,
                normalization,
                kernel,
                action,
                rng,
                device=device,
                batch_size=batch_size,
                seed=refresh_seed,
            )
        state, coordinate = mh_sweep(
            state,
            kernel,
            action,
            widths,
            rng,
            divide=divide,
            detail_passes=detail_passes,
        )
        diagnostics.update(coordinate)
    if correction in {
        "coordinate-wolff",
        "detail-wolff",
        "flow-detail-wolff",
        "hmc-wolff",
    }:
        state, diagnostics["wolff"] = wolff_state_transition(
            state, kernel, action, rng
        )
    return state, diagnostics


def _chain_observables(
    state: InverseBlockingState, action: Phi4Action
) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(value, dtype=np.float64)
        for name, value in kernel_observable_series(state.fine, action).items()
    }


def trajectory_diagnostics(
    histories: dict[str, np.ndarray],
    wall_seconds: float,
    *,
    measurement_start: int = 0,
) -> dict[str, Any]:
    observables = {}
    for name, values in histories.items():
        stationary = values[:, measurement_start:]
        tau = integrated_autocorrelation_time(stationary)
        samples = stationary.shape[0] * stationary.shape[1]
        ess = samples / (2.0 * tau)
        observables[name] = {
            "tau_int": float(tau),
            "ess": float(ess),
            "ess_per_second": float(ess / wall_seconds),
        }
    return {
        "measurement_start": measurement_start,
        "wall_seconds": wall_seconds,
        "observables": observables,
        "worst_ess_per_second": min(
            item["ess_per_second"] for item in observables.values()
        ),
    }


def _ensemble_scalars(fields: np.ndarray, action: Phi4Action) -> dict[str, float]:
    size = fields.shape[-1]
    volume = size**2
    magnetization = np.mean(fields, axis=(-2, -1))
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
    susceptibility = float(
        volume * (np.mean(magnetization**2) - np.mean(magnetization) ** 2)
    )
    ratio = susceptibility / minimum_momentum
    xi_over_l = (
        float(np.sqrt(ratio - 1.0) / (2.0 * np.sin(np.pi / size)) / size)
        if ratio > 1.0
        else float("nan")
    )
    mean_m2 = float(np.mean(magnetization**2))
    return {
        "action_density": float(np.mean(action(fields) / volume)),
        "binder_cumulant": float(
            1.0 - np.mean(magnetization**4) / (3.0 * mean_m2**2)
        ),
        "xi_over_L": xi_over_l,
        "susceptibility": susceptibility,
        "scaled_susceptibility": susceptibility / size ** (2.0 - 0.25),
    }


def _history_row(
    sweep: int,
    state: InverseBlockingState,
    action: Phi4Action,
    acceptance: dict[str, Any] | None,
) -> dict[str, Any]:
    series = kernel_observable_series(state.fine, action)
    return {
        "sweep": sweep,
        "observables": {name: float(np.mean(series[name])) for name in KERNEL_OBSERVABLE_NAMES},
        "physical": _ensemble_scalars(state.fine, action),
        "acceptance": acceptance,
    }


def _save_state(path: Path, state: InverseBlockingState, source_indices: np.ndarray) -> None:
    np.savez_compressed(
        path,
        fine=state.fine,
        psi=state.psi,
        coarse=state.coarse,
        details=state.details,
        fine_action=state.fine_action,
        source_indices=source_indices,
    )


def _convergence_summary(
    generated: np.ndarray,
    native_chain: np.ndarray,
    action: Phi4Action,
    seed: int,
) -> dict[str, Any]:
    generated_series = kernel_observable_series(generated, action)
    native_series = kernel_observable_series(native_chain, action)
    shifts = {}
    for name in KERNEL_OBSERVABLE_NAMES:
        pooled_std = np.sqrt(
            0.5
            * (
                np.var(generated_series[name], ddof=1)
                + np.var(native_series[name], ddof=1)
            )
        )
        shifts[name] = float(
            (np.mean(generated_series[name]) - np.mean(native_series[name]))
            / pooled_std
        )
    values = np.asarray(list(shifts.values()))

    native_bootstrap = bootstrap_ensemble_observables(
        native_chain[None], samples=500, block_length=2, seed=seed
    )
    generated_bootstrap = bootstrap_ensemble_observables(
        generated[None], samples=500, block_length=1, seed=seed + 1
    )
    physical_checks = {}
    size = generated.shape[-1]
    for name in ("binder_cumulant", "xi_over_L", "susceptibility"):
        native_values = native_bootstrap[name][np.isfinite(native_bootstrap[name])]
        generated_values = generated_bootstrap[name][
            np.isfinite(generated_bootstrap[name])
        ]
        if len(native_values) < 2 or len(generated_values) < 2:
            difference = float("nan")
            combined_error = float("nan")
        else:
            difference = float(np.mean(generated_values) - np.mean(native_values))
            combined_error = float(
                np.sqrt(
                    np.var(generated_values, ddof=1)
                    + np.var(native_values, ddof=1)
                )
            )
        label = "scaled_susceptibility" if name == "susceptibility" else name
        scale = size ** (2.0 - 0.25) if name == "susceptibility" else 1.0
        physical_checks[label] = {
            "difference": difference / scale,
            "combined_standard_error": combined_error / scale,
            "within_2sigma": bool(
                np.isfinite(combined_error)
                and abs(difference) <= 2.0 * combined_error
            ),
        }
    rms = float(np.sqrt(np.mean(values**2)))
    maximum = float(np.max(np.abs(values)))
    converged = (
        rms <= 0.20
        and maximum <= 0.50
        and all(item["within_2sigma"] for item in physical_checks.values())
    )
    return {
        "converged": bool(converged),
        "rms_standardized_mean_shift": rms,
        "max_abs_standardized_mean_shift": maximum,
        "standardized_mean_shifts": shifts,
        "physical_checks": physical_checks,
        "interpretation": (
            "finite-chain convergence criteria passed"
            if converged
            else "exact MH kernel established; finite chain should be extended"
        ),
    }


def run_inverse_mh(
    checkpoint_path: Path,
    coarse_path: Path,
    output: Path,
    *,
    native_fine_path: Path | None = None,
    chains: int = 128,
    sweeps: int = 400,
    calibration_chains: int = 32,
    calibration_sweeps: int = 50,
    divide: int = 2,
    initial_width: float = 0.04,
    batch_size: int = 32,
    seed: int = 20260824,
    device_name: str = "auto",
    save_sweeps: tuple[int, ...] = (0, 1, 2, 5, 10, 20, 50, 100, 200, 400),
    correction: str = "coordinate",
    detail_passes: int = 1,
    initializer: str = "flow",
    evaluation_interval: int = 5,
) -> dict[str, Any]:
    corrections = {
        "coordinate",
        "coordinate-wolff",
        "detail-wolff",
        "flow-detail-wolff",
        "hmc-wolff",
    }
    if correction not in corrections:
        raise ValueError(f"unknown correction {correction!r}")
    if initializer not in {"flow", "gaussian"}:
        raise ValueError("initializer must be flow or gaussian")
    if divide != 2:
        raise ValueError("the exact parallel local coarse kernel requires divide=2")
    if detail_passes <= 0 or evaluation_interval <= 0:
        raise ValueError("detail passes and evaluation interval must be positive")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = model_from_checkpoint(checkpoint, device)
    normalization = checkpoint["normalization"]
    kernel = np.asarray(checkpoint["kernel"]["matrix"], dtype=np.float64)
    if native_fine_path is None:
        native_fine_path = Path(checkpoint["config"]["fine_source"])
    coarse_chains = load_configurations(coarse_path)
    native_fine_chains = load_configurations(native_fine_path)
    coarse_test = coarse_chains[3]
    native_test = native_fine_chains[3]
    if chains > len(coarse_test):
        raise ValueError("requested chains exceed held-out native coarse configurations")

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    action = Phi4Action()
    rng = np.random.default_rng(seed)
    calibration_indices = rng.choice(
        len(coarse_test), size=min(calibration_chains, chains), replace=False
    )
    calibration_coarse = coarse_test[calibration_indices]
    calibration_details, _ = sample_physical_details(
        model,
        calibration_coarse,
        normalization,
        device=device,
        batch_size=batch_size,
        seed=seed + 1,
    )
    calibration_state = build_state(
        calibration_coarse, calibration_details, kernel, action
    )
    widths = {name: initial_width for name in ("coarse", *DETAIL_NAMES)}
    hmc_step_size = initial_width
    calibration_history = []
    calibration_rng = np.random.default_rng(seed + 2)
    for sweep in range(1, calibration_sweeps + 1):
        if correction == "hmc-wolff":
            calibration_state, hmc_diagnostics = hmc_transition(
                calibration_state,
                kernel,
                action,
                calibration_rng,
                device=device,
                step_size=hmc_step_size,
            )
            rate = hmc_diagnostics["acceptance"]
            gain = 0.5 / np.sqrt(sweep)
            hmc_step_size = float(
                np.clip(hmc_step_size * np.exp(gain * (rate - 0.8)), 1.0e-4, 0.5)
            )
            calibration_history.append(
                {
                    "sweep": sweep,
                    "hmc_step_size": hmc_step_size,
                    "acceptance": rate,
                }
            )
        else:
            calibration_state, diagnostics = mh_sweep(
                calibration_state,
                kernel,
                action,
                widths,
                calibration_rng,
                divide=divide,
                detail_passes=detail_passes,
            )
            gain = 0.5 / np.sqrt(sweep)
            rates = {
                name: diagnostics["coordinates"][name]["outer_acceptance"]
                for name in widths
            }
            for name in widths:
                widths[name] = float(
                    np.clip(
                        widths[name] * np.exp(gain * (rates[name] - 0.5)),
                        1.0e-3,
                        0.2,
                    )
                )
            calibration_history.append(
                {"sweep": sweep, "widths": widths.copy(), "outer_acceptance": rates}
            )

    source_indices = rng.choice(len(coarse_test), size=chains, replace=False)
    production_coarse = coarse_test[source_indices]
    synchronize()
    initialization_started = time.perf_counter()
    if initializer == "flow":
        production_details, flow_logq = sample_physical_details(
            model,
            production_coarse,
            normalization,
            device=device,
            batch_size=batch_size,
            seed=seed + 3,
        )
    else:
        detail_mean = np.asarray(normalization["detail_mean"])[None, :, None, None]
        detail_std = np.asarray(normalization["detail_std"])[None, :, None, None]
        production_details = detail_mean + detail_std * rng.standard_normal(
            (chains, 3, *production_coarse.shape[-2:])
        )
        flow_logq = np.full(chains, np.nan)
    synchronize()
    initialization_seconds = time.perf_counter() - initialization_started

    state = build_state(production_coarse, production_details, kernel, action)
    initial_error = float(np.max(reblocking_error(state.fine, state.coarse, kernel)))
    if initial_error > 1.0e-9:
        raise RuntimeError(f"initial reblocking error {initial_error:.3e}")

    output.mkdir(parents=True, exist_ok=True)
    checkpoint_directory = output / "checkpoints"
    checkpoint_directory.mkdir(exist_ok=True)
    requested_saves = {value for value in save_sweeps if value <= sweeps} | {0, sweeps}
    _save_state(checkpoint_directory / "checkpoint_sweep_0000.npz", state, source_indices)
    history = [_history_row(0, state, action, None)]
    initial_observables = _chain_observables(state, action)
    trajectory = {name: [values] for name, values in initial_observables.items()}
    acceptance_history = []
    convergence_history = []
    consecutive_passes = 0
    time_to_tolerance: int | None = None
    production_rng = np.random.default_rng(seed + 4)
    synchronize()
    production_wall_started = time.perf_counter()
    transition_seconds = 0.0
    sweep_elapsed = [0.0]
    for sweep in range(1, sweeps + 1):
        synchronize()
        transition_started = time.perf_counter()
        state, diagnostics = correction_sweep(
            state,
            correction,
            kernel,
            action,
            widths,
            production_rng,
            detail_passes=detail_passes,
            divide=divide,
            model=model,
            normalization=normalization,
            device=device,
            batch_size=batch_size,
            refresh_seed=seed + 100_000 + sweep,
            hmc_step_size=hmc_step_size,
        )
        synchronize()
        transition_seconds += time.perf_counter() - transition_started
        diagnostics["sweep"] = sweep
        sweep_elapsed.append(transition_seconds)
        acceptance_history.append(diagnostics)
        history.append(_history_row(sweep, state, action, diagnostics))
        for name, values in _chain_observables(state, action).items():
            trajectory[name].append(values)
        if native_fine_path is not None and sweep % evaluation_interval == 0:
            check = _convergence_summary(state.fine, native_test, action, seed + sweep)
            check["sweep"] = sweep
            convergence_history.append(check)
            consecutive_passes = consecutive_passes + 1 if check["converged"] else 0
            if consecutive_passes == 3 and time_to_tolerance is None:
                time_to_tolerance = sweep
        if sweep in requested_saves:
            _save_state(
                checkpoint_directory / f"checkpoint_sweep_{sweep:04d}.npz",
                state,
                source_indices,
            )
        if sweep == 1 or sweep % 10 == 0 or sweep == sweeps:
            print(
                f"{correction} sweep {sweep:04d}/{sweeps}",
                flush=True,
            )
    synchronize()
    production_wall_seconds = time.perf_counter() - production_wall_started
    production_seconds = transition_seconds

    history_arrays = {
        name: np.stack(values, axis=1) for name, values in trajectory.items()
    }
    np.savez_compressed(output / "observable_history.npz", **history_arrays)
    convergence = _convergence_summary(state.fine, native_test, action, seed + 5)
    measurement_start = time_to_tolerance if time_to_tolerance is not None else sweeps // 2
    stationary_seconds = max(
        production_seconds - sweep_elapsed[measurement_start], 1.0e-12
    )
    efficiency = trajectory_diagnostics(
        history_arrays,
        stationary_seconds,
        measurement_start=measurement_start,
    )
    efficiency["total_transition_seconds"] = production_seconds
    efficiency["total_wall_seconds_with_diagnostics"] = production_wall_seconds
    efficiency["thermalization_transition_seconds"] = sweep_elapsed[
        measurement_start
    ]
    training_seconds = float(checkpoint.get("training_wall_seconds", 0.0))
    online_seconds_per_effective_sample = (
        stationary_seconds / min(
            item["ess"] for item in efficiency["observables"].values()
        )
    )
    setup_seconds = training_seconds + initialization_seconds
    run_config = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_format_version": checkpoint.get("format_version", 1),
        "model_type": checkpoint.get("model_type", "affine"),
        "coarse_source": str(coarse_path),
        "native_fine_source": str(native_fine_path),
        "chains": chains,
        "sweeps": sweeps,
        "calibration_chains": min(calibration_chains, chains),
        "calibration_sweeps": calibration_sweeps,
        "divide": divide,
        "proposal_widths": widths,
        "hmc_step_size": hmc_step_size if correction == "hmc-wolff" else None,
        "correction": correction,
        "detail_passes": detail_passes,
        "initializer": initializer,
        "wolff_updates_per_sweep": int(correction.endswith("-wolff")),
        "seed": seed,
        "device": str(device),
        "flow_used_after_sweep_zero": correction == "flow-detail-wolff",
        "evaluation_interval": evaluation_interval,
        "acceptance_ratios": {
            "coarse": "-S_f(new)+S_f(old)+S_c(new)-S_c(old)",
            "detail": "-S_f(new)+S_f(old)",
            "flow_refresh": "-S_f(new)+S_f(old)+logq(old|c)-logq(new|c)",
        },
        "saved_sweeps": sorted(requested_saves),
    }
    metrics = {
        "run_config": run_config,
        "calibration_history": calibration_history,
        "initial": {
            "flow_logq_mean": (
                float(np.nanmean(flow_logq))
                if np.any(np.isfinite(flow_logq))
                else None
            ),
            "max_reblocking_error": initial_error,
            "initialization_seconds": initialization_seconds,
            "flow_only_configs_per_second": chains / initialization_seconds,
            "one_shot_final_configs_per_second": chains
            / (initialization_seconds + production_seconds),
            "one_shot_has_time_series_ess": False,
        },
        "history": history,
        "convergence_history": convergence_history,
        "time_to_tolerance_sweeps": time_to_tolerance,
        "convergence": convergence,
        "efficiency": efficiency,
        "cost": {
            "training_seconds": training_seconds,
            "coarse_sampling_seconds": 0.0,
            "flow_initialization_seconds": initialization_seconds,
            "correction_seconds": production_seconds,
            "wall_seconds_with_diagnostics": production_wall_seconds,
            "setup_seconds": setup_seconds,
            "online_seconds_per_effective_sample": online_seconds_per_effective_sample,
            "end_to_end_seconds": setup_seconds + production_seconds,
        },
    }
    _write_json(output / "run_config.json", run_config)
    _write_json(output / "acceptance_history.json", acceptance_history)
    _write_json(output / "metrics.json", metrics)
    return metrics
