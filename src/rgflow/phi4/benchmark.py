"""Unified online and end-to-end benchmarks for 2D phi-four sampling."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from rgflow.diagnostics import integrated_autocorrelation_time

from .action import Phi4Action
from .flow_training import load_configurations
from .inverse_mh import run_inverse_mh, trajectory_diagnostics
from .operator_diagnostics import KERNEL_OBSERVABLE_NAMES, kernel_observable_series
from .sampling import radial_metropolis_sweep, wolff_cluster_update


METHODS = {
    "coordinate": ("coordinate", 1, "flow"),
    "coordinate-wolff": ("coordinate-wolff", 1, "flow"),
    "detail2-wolff": ("detail-wolff", 2, "flow"),
    "detail4-wolff": ("detail-wolff", 4, "flow"),
    "flow-detail2-wolff": ("flow-detail-wolff", 2, "flow"),
    "flow-detail4-wolff": ("flow-detail-wolff", 4, "flow"),
    "detail2-wolff-gaussian": ("detail-wolff", 2, "gaussian"),
    "detail4-wolff-gaussian": ("detail-wolff", 4, "gaussian"),
    "hmc-wolff": ("hmc-wolff", 1, "flow"),
}


def _write_json(path: Path, payload: Any) -> None:
    def default(value: Any) -> Any:
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"cannot serialize {type(value).__name__}")

    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=default) + "\n",
        encoding="utf-8",
    )


def run_native_baseline(
    native_path: Path,
    output: Path,
    *,
    chains: int,
    sweeps: int,
    seed: int,
    proposal_width: float = 1.0,
) -> dict[str, Any]:
    native = load_configurations(native_path)[3]
    metadata_path = native_path.with_suffix(".json")
    if metadata_path.exists() and proposal_width == 1.0:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        proposal_width = float(
            metadata.get("sampler", {}).get("proposal_width", proposal_width)
        )
    if chains > len(native):
        raise ValueError("requested benchmark chains exceed held-out native samples")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(native), chains, replace=False)
    fields = native[indices].copy()
    action = Phi4Action()
    first = kernel_observable_series(fields, action)
    histories = {name: [np.asarray(first[name])] for name in KERNEL_OBSERVABLE_NAMES}
    accepted = proposed = cluster_sites = 0
    wall_started = time.perf_counter()
    transition_seconds = 0.0
    for _ in range(sweeps):
        transition_started = time.perf_counter()
        take, attempts = radial_metropolis_sweep(
            fields, action, proposal_width, rng
        )
        accepted += take
        proposed += attempts
        for field in fields:
            cluster_sites += wolff_cluster_update(field, action.kappa, rng)
        transition_seconds += time.perf_counter() - transition_started
        values = kernel_observable_series(fields, action)
        for name in histories:
            histories[name].append(np.asarray(values[name]))
    wall_seconds = time.perf_counter() - wall_started
    arrays = {name: np.stack(values, axis=1) for name, values in histories.items()}
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "observable_history.npz", **arrays)
    metrics = {
        "sampler": "native-radial-wolff",
        "chains": chains,
        "sweeps": sweeps,
        "proposal_width": proposal_width,
        "radial_acceptance": accepted / proposed,
        "mean_cluster_fraction": cluster_sites / (chains * sweeps * fields.shape[-1] ** 2),
        "efficiency": trajectory_diagnostics(arrays, transition_seconds),
        "cost": {
            "setup_seconds": 0.0,
            "correction_seconds": transition_seconds,
            "wall_seconds_with_diagnostics": wall_seconds,
            "online_seconds_per_effective_sample": transition_seconds
            / min(
                value["ess"]
                for value in trajectory_diagnostics(arrays, transition_seconds)[
                    "observables"
                ].values()
            ),
        },
    }
    _write_json(output / "metrics.json", metrics)
    return metrics


def _bootstrap_efficiency_ratio(
    candidate_history: Path,
    native_history: Path,
    candidate_seconds: float,
    native_seconds: float,
    *,
    measurement_start: int,
    samples: int,
    seed: int,
) -> dict[str, float]:
    with np.load(candidate_history) as data:
        candidate = {name: np.asarray(data[name])[:, measurement_start:] for name in data}
    with np.load(native_history) as data:
        native = {name: np.asarray(data[name]) for name in data}
    chain_count = next(iter(candidate.values())).shape[0]
    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(samples):
        indices = rng.integers(0, chain_count, chain_count)
        candidate_worst = min(
            values.shape[0] * values.shape[1]
            / (2.0 * integrated_autocorrelation_time(values[indices]))
            / candidate_seconds
            for values in candidate.values()
        )
        native_worst = min(
            values.shape[0] * values.shape[1]
            / (2.0 * integrated_autocorrelation_time(values[indices]))
            / native_seconds
            for values in native.values()
        )
        ratios.append(candidate_worst / native_worst)
    low, high = np.quantile(ratios, (0.025, 0.975))
    return {
        "median": float(np.median(ratios)),
        "lower_95": float(low),
        "upper_95": float(high),
    }


def run_benchmark(
    cases: list[tuple[Path, Path, Path]],
    output_root: Path,
    *,
    methods: tuple[str, ...],
    chains: int,
    sweeps: int,
    calibration_sweeps: int,
    batch_size: int,
    seed: int,
    device_name: str,
    bootstrap_samples: int = 500,
    coarse_sampling_seconds: float = 0.0,
) -> dict[str, Any]:
    unknown = set(methods) - METHODS.keys()
    if unknown:
        raise ValueError(f"unknown benchmark methods: {sorted(unknown)}")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    root = output_root / run_id
    root.mkdir(parents=True)
    report: dict[str, Any] = {
        "run_id": run_id,
        "benchmark_config": {
            "cases": [
                [str(checkpoint), str(coarse), str(native)]
                for checkpoint, coarse, native in cases
            ],
            "methods": list(methods),
            "chains": chains,
            "sweeps": sweeps,
            "calibration_sweeps": calibration_sweeps,
            "batch_size": batch_size,
            "bootstrap_samples": bootstrap_samples,
            "coarse_sampling_seconds": coarse_sampling_seconds,
            "seed": seed,
            "device": device_name,
        },
        "cases": {},
    }
    for case_index, (checkpoint, coarse, native) in enumerate(cases):
        fine_size = load_configurations(native).shape[-1]
        coarse_size = load_configurations(coarse).shape[-1]
        case_name = f"L{coarse_size}_to_L{fine_size}"
        case_root = root / case_name
        native_metrics = run_native_baseline(
            native,
            case_root / "native",
            chains=chains,
            sweeps=sweeps,
            seed=seed + 10_000 * case_index,
        )
        case_report: dict[str, Any] = {
            "native": native_metrics,
            "methods": {},
        }
        native_worst = native_metrics["efficiency"]["worst_ess_per_second"]
        native_seconds = native_metrics["efficiency"]["wall_seconds"]
        for method_index, method in enumerate(methods):
            correction, detail_passes, initializer = METHODS[method]
            method_root = case_root / method
            metrics = run_inverse_mh(
                checkpoint,
                coarse,
                method_root,
                native_fine_path=native,
                chains=chains,
                sweeps=sweeps,
                calibration_sweeps=calibration_sweeps,
                batch_size=batch_size,
                seed=seed + 10_000 * case_index + 100 * (method_index + 1),
                device_name=device_name,
                correction=correction,
                detail_passes=detail_passes,
                initializer=initializer,
                save_sweeps=(0, 5, 10, 20, 50, 100, sweeps),
            )
            worst = metrics["efficiency"]["worst_ess_per_second"]
            ratio = worst / native_worst
            interval = _bootstrap_efficiency_ratio(
                method_root / "observable_history.npz",
                case_root / "native" / "observable_history.npz",
                metrics["efficiency"]["wall_seconds"],
                native_seconds,
                measurement_start=metrics["efficiency"]["measurement_start"],
                samples=bootstrap_samples,
                seed=seed + 1_000_000 + method_index,
            )
            native_per_ess = native_metrics["cost"][
                "online_seconds_per_effective_sample"
            ]
            inverse_per_ess = metrics["cost"]["online_seconds_per_effective_sample"]
            setup = metrics["cost"]["setup_seconds"] + coarse_sampling_seconds
            saving = native_per_ess - inverse_per_ess
            break_even = setup / saving if saving > 0.0 else None
            case_report["methods"][method] = {
                "metrics_path": str(method_root / "metrics.json"),
                "time_to_tolerance_sweeps": metrics["time_to_tolerance_sweeps"],
                "converged": metrics["convergence"]["converged"],
                "worst_ess_per_second": worst,
                "efficiency_ratio_to_native": ratio,
                "paired_bootstrap_ratio": interval,
                "break_even_effective_samples": break_even,
                "passes_correctness_gate": metrics["convergence"]["converged"],
                "passes_efficiency_gate": ratio >= 2.0 and interval["lower_95"] > 1.0,
                "passes_thermalization_gate": (
                    metrics["convergence"]["converged"]
                    and metrics["time_to_tolerance_sweeps"] is not None
                    and metrics["time_to_tolerance_sweeps"] <= 100
                ),
            }
        non_hmc = {
            name: value
            for name, value in case_report["methods"].items()
            if name != "hmc-wolff" and "gaussian" not in name
        }
        if "hmc-wolff" in case_report["methods"] and non_hmc:
            best_name, best = max(
                non_hmc.items(), key=lambda item: item[1]["worst_ess_per_second"]
            )
            hmc = case_report["methods"]["hmc-wolff"]
            hmc_time = hmc["time_to_tolerance_sweeps"]
            best_time = best["time_to_tolerance_sweeps"]
            hmc["screening"] = {
                "best_non_hmc": best_name,
                "advance_to_multiseed": bool(
                    hmc["worst_ess_per_second"] > best["worst_ess_per_second"]
                    and hmc_time is not None
                    and (best_time is None or hmc_time < best_time)
                ),
            }
        report["cases"][case_name] = case_report
        _write_json(case_root / "summary.json", case_report)
    _write_json(root / "summary.json", report)
    return report



def run_benchmark_replicates(
    cases: list[tuple[Path, Path, Path]],
    output_root: Path,
    *,
    replicates: int,
    **kwargs: Any,
) -> dict[str, Any]:
    base_seed = int(kwargs.pop("seed"))
    reports = [
        run_benchmark(
            cases,
            output_root,
            seed=base_seed + replicate,
            **kwargs,
        )
        for replicate in range(replicates)
    ]
    aggregate: dict[str, Any] = {}
    for case_name in reports[0]["cases"]:
        aggregate[case_name] = {}
        method_names = reports[0]["cases"][case_name]["methods"]
        for method in method_names:
            values = [
                report["cases"][case_name]["methods"][method] for report in reports
            ]
            aggregate[case_name][method] = {
                "seeds": [base_seed + index for index in range(replicates)],
                "all_final_correct": all(
                    value["passes_correctness_gate"] for value in values
                ),
                "all_thermalized_by_100": all(
                    value["passes_thermalization_gate"] for value in values
                ),
                "all_efficiency_gates_pass": all(
                    value["passes_efficiency_gate"] for value in values
                ),
                "median_efficiency_ratio_to_native": float(
                    np.median(
                        [value["efficiency_ratio_to_native"] for value in values]
                    )
                ),
                "minimum_bootstrap_lower_95": float(
                    min(value["paired_bootstrap_ratio"]["lower_95"] for value in values)
                ),
                "run_ids": [report["run_id"] for report in reports],
            }
    suite_id = datetime.now(timezone.utc).strftime("confirmation_%Y%m%dT%H%M%S.%fZ")
    suite = {
        "suite_id": suite_id,
        "replicates": replicates,
        "runs": [report["run_id"] for report in reports],
        "acceptance": aggregate,
    }
    _write_json(output_root / f"{suite_id}.json", suite)
    return suite
