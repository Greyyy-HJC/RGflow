#!/usr/bin/env python3
"""Evaluate a completed stout, polynomial stout, or field-transform checkpoint."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
from pathlib import Path

import numpy as np
import torch

from rgflow.su3.downsampling import (
    GaugeEquivariantFieldTransform, PolynomialStoutKernel, StoutKernel,
    block_links, block_polynomial_basis, polynomial_blocking_basis, su3_errors,
)
from rgflow.su3.observables import observable_vector_torch
from rgflow.su3.training import (
    _covariance_inverse,
    _link_batches,
    _load_links,
    _metric_record,
    _plot_diagnostics,
    _projected_blocked_observables,
    _stout_blocked_observables,
    _split_paths,
    distribution_acceptance,
    ensemble_paths,
    measure_blocked_with_pyquda,
    measure_straight_blocked_with_pyquda,
    measure_stout_with_pyquda,
    measure_polynomial_with_pyquda,
    observable_names,
    split_indices,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FINE = ROOT / "artifacts" / "4dsu3" / "L24_beta6p20"
DEFAULT_COARSE = ROOT / "artifacts" / "4dsu3" / "L12_beta5p775"
DEFAULT_RUN = ROOT / "artifacts" / "4dsu3" / "downsample" / "L24_beta6p20_to_L12_beta5p775_polynomial_local_ls_v3"
DEFAULT_REFERENCE_RUN = ROOT / "artifacts" / "4dsu3" / "downsample" / "L24_beta6p20_to_L12_beta5p80_cnn_v2"


@torch.no_grad()
def _polynomial_configuration(path, parameter, hook, local):
    basis = polynomial_blocking_basis(_load_links(path, torch.device("cpu")), hook=hook, local=local)
    blocked = block_polynomial_basis(basis, parameter)
    return observable_vector_torch(blocked).numpy(), su3_errors(blocked)


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


def _history(run: Path) -> list[dict[str, float]]:
    if (run / "fit_history.json").exists():
        rows = json.loads((run / "fit_history.json").read_text(encoding="utf-8"))
        return [{"epoch": row["evaluation"] + 1, "train_loss": row["loss"], "validation_loss": np.nan} for row in rows]
    with (run / "history.csv").open(newline="", encoding="utf-8") as stream:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-dir", type=Path, default=DEFAULT_FINE)
    parser.add_argument("--coarse-dir", type=Path, default=DEFAULT_COARSE)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--reference-run", type=Path, default=DEFAULT_REFERENCE_RUN)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--max-configs", type=int, default=None, help="evaluate a prefix matching a smoke-test run")
    parser.add_argument("--reuse-values", action="store_true", help="reuse observable_distributions.npz from a completed evaluation")
    parser.add_argument("--workers", type=int, choices=(1, 2, 4), default=1,
                        help="parallel CPU evaluation of the retained polynomial basis; requires --device cpu")
    parser.add_argument("--pyquda-configs", type=int, default=None,
                        help="cross-check this many held-out configurations (default: complete test split)")
    args = parser.parse_args()

    fine_paths, _ = ensemble_paths(args.fine_dir.resolve())
    coarse_paths, coarse_manifest = ensemble_paths(args.coarse_dir.resolve())
    if args.max_configs is not None:
        if args.max_configs < 5 or args.max_configs > len(fine_paths):
            raise SystemExit("max-configs must be at least five and no larger than the ensemble")
        fine_paths = fine_paths[: args.max_configs]
        coarse_paths = coarse_paths[: args.max_configs]
    run = args.run.resolve()
    lattice = list(map(int, coarse_manifest["lattice_size"]))
    reference = np.load(run / "reference_observables.npz")["observables"]
    if reference.shape != (len(coarse_paths), 4):
        raise SystemExit(f"reference shape {reference.shape} does not match ensemble size {len(coarse_paths)}")
    reference_metrics = json.loads((args.reference_run.resolve() / "metrics.json").read_text(encoding="utf-8"))
    reference_test_loss = float(reference_metrics["splits"]["test"]["trained"]["loss"])
    reference_kernel = json.loads((args.reference_run / "kernel.json").read_text(encoding="utf-8"))
    reference_beta = reference_kernel["sources"]["coarse_manifest"]["beta"]
    reference_comparable = reference_beta == coarse_manifest["beta"]

    device = torch.device(args.device)
    indices = split_indices(len(fine_paths))
    baseline_path = run / "baseline_observables.npz"
    if baseline_path.exists():
        baseline_all = np.load(baseline_path)["observables"]
    else:
        values = []
        with torch.no_grad():
            for links in _link_batches(fine_paths, device, args.batch_size):
                values.append(observable_vector_torch(block_links(links)))
        baseline_all = torch.cat(values).cpu().numpy()
        np.savez(baseline_path, observables=baseline_all)

    kernel = json.loads((run / "kernel.json").read_text(encoding="utf-8"))
    architecture = kernel["architecture"]
    if architecture["class"] == "StoutKernel":
        model = StoutKernel(
            initial_weights=tuple(float(value) for value in architecture["initial_weights"])
        ).to(device)
        method = "stout"
    elif architecture["class"] == "PolynomialStoutKernel":
        model = PolynomialStoutKernel(
            initial_coefficients=tuple(float(value) for value in architecture["initial_coefficients"]),
            hook_coefficients=tuple(float(value) for value in architecture.get("initial_hook_coefficients", ())),
            local_coefficients=tuple(float(value) for value in architecture.get("initial_local_coefficients", ())),
        ).to(device)
        method = "stout-polynomial"
    else:
        model = GaugeEquivariantFieldTransform(
            hidden_channels=int(architecture["hidden_channels"]),
            step_scale=float(architecture["step_scale"]),
            initial_output_scale=float(architecture["initial_output_scale"]),
            flow_steps=int(architecture.get("flow_steps", 1)),
        ).to(device)
        method = "field-transform"
    checkpoint = torch.load(run / "kernel.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    if args.workers > 1 and (method != "stout-polynomial" or device.type != "cpu"):
        parser.error("parallel evaluation requires a polynomial checkpoint and --device cpu")
    if args.pyquda_configs is not None and args.pyquda_configs < 1:
        parser.error("pyquda-configs must be positive")
    evaluation_errors = None
    if args.reuse_values:
        saved = np.load(run / "observable_distributions.npz")
        trained_all = np.empty_like(baseline_all)
        for split in indices:
            trained_all[indices[split]] = saved[f"trained_{split}"]
    elif args.workers > 1:
        torch.set_num_threads(2)
        coefficients = architecture["coefficients"] + architecture.get("hook_coefficients", []) + architecture.get("local_coefficients", [])
        scales = [12.**n for n in (1, 2, 3, 4)]
        if architecture.get("hook_coefficients"):
            scales += [12.**n for n in (1, 2, 3)]
        if architecture.get("local_coefficients"):
            scales += [12.**n for n in (1, 2, 3, 4)]
        parameter = torch.tensor(np.asarray(coefficients) * scales, dtype=torch.float32)
        measure = partial(_polynomial_configuration, parameter=parameter,
                          hook=bool(architecture.get("hook_coefficients")), local=bool(architecture.get("local_coefficients")))
        values, errors = [], []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for position, (value, error) in enumerate(pool.map(measure, fine_paths)):
                values.append(value)
                errors.append(error)
                if (position + 1) % 10 == 0:
                    print(f"evaluated {position + 1}/{len(fine_paths)} configurations", flush=True)
        trained_all = np.asarray(values)
        evaluation_errors = np.max(errors, axis=0).tolist()
    else:
        if method == "stout":
            trained_all = _stout_blocked_observables(
                fine_paths, model, device, requires_grad=False, batch_size=args.batch_size
            ).cpu().numpy()
        elif method == "stout-polynomial":
            from rgflow.su3.training import _polynomial_blocked_observables

            trained_all = _polynomial_blocked_observables(
                fine_paths, model, device, requires_grad=False, batch_size=args.batch_size
            ).cpu().numpy()
        else:
            trained_all = _projected_blocked_observables(
                fine_paths, model, device, requires_grad=False, batch_size=args.batch_size
            ).cpu().numpy()

    targets = {split: reference[indices[split]] for split in indices}
    baseline = {split: baseline_all[indices[split]] for split in indices}
    trained = {split: trained_all[indices[split]] for split in indices}
    covariance, inverse_covariance = _covariance_inverse(targets["train"], baseline["train"], 0.1)
    variance_scale = np.full(4, np.sqrt(2.0 / (len(targets["train"]) - 1)))
    records = {
        split: {
            "baseline": _metric_record(baseline[split], targets[split], inverse_covariance, variance_scale),
            "trained": _metric_record(trained[split], targets[split], inverse_covariance, variance_scale),
        }
        for split in indices
    }
    test_pass = records["test"]["trained"]["chi2"] < records["test"]["baseline"]["chi2"]
    reference_pass = (records["test"]["trained"]["chi2"] < reference_test_loss
                      if reference_comparable else None)
    metrics = {
        "observables": observable_names(),
        "acceptance": {
            "test_trained_chi2_below_baseline": test_pass,
            "test_trained_loss_below_baseline": test_pass,
            "test_trained_chi2_below_reference_cnn": reference_pass,
            "test_trained_loss_below_reference_cnn": reference_pass,
            "reference_cnn_test_loss": reference_test_loss,
            "reference_cnn_coarse_beta": reference_beta,
            "reference_cnn_comparable": reference_comparable,
            **distribution_acceptance(records["test"]["trained"]),
            "all": test_pass and (reference_pass if reference_comparable else True) and distribution_acceptance(records["test"]["trained"])["distribution_match"],
        },
        "splits": records,
        "best_epoch": int(checkpoint["epoch"]),
        "chi2_definition": "covariance-weighted mean mismatch plus standardized log-variance mismatch",
        "loss_definition": "covariance-weighted mean plus standardized log-variance mismatch",
        "evaluation": {"device": str(device), "workers": args.workers, "su3_errors_all_configurations": evaluation_errors},
    }
    (run / "metrics.json").write_text(json.dumps(_json_ready(metrics), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    np.savez(
        run / "observable_distributions.npz",
        reference_test=targets["test"], baseline_test=baseline["test"], trained_test=trained["test"],
        reference_all=reference, reference_train=targets["train"], reference_validation=targets["validation"],
        baseline_train=baseline["train"], baseline_validation=baseline["validation"],
        trained_train=trained["train"], trained_validation=trained["validation"],
    )
    _plot_diagnostics(
        run / "diagnostics.pdf", _history(run),
        {"reference": targets["test"], "baseline": baseline["test"], "downsampled": trained["test"]},
        observable_names(),
    )

    from pyquda_utils import core

    core.init(None, lattice, backend="numpy", resource_path=str(run / ".quda-cache"))
    check_paths = _split_paths(fine_paths, indices, "test")[:args.pyquda_configs]
    check_count = len(check_paths)
    if method == "stout":
        pyquda_trained, trained_errors = measure_stout_with_pyquda(
            check_paths, model, lattice, device
        )
    elif method == "stout-polynomial":
        pyquda_trained, trained_errors = measure_polynomial_with_pyquda(
            check_paths, model, lattice, device
        )
    else:
        pyquda_trained, trained_errors = measure_blocked_with_pyquda(
            check_paths, model, lattice, device
        )
    pyquda_baseline, baseline_errors = measure_straight_blocked_with_pyquda(
        check_paths, lattice, device
    )
    metrics["pyquda_crosscheck"] = {
        "fine_indices": indices["test"][:check_count],
        "trained_max_abs_observable_difference": float(np.max(np.abs(trained["test"][:check_count] - pyquda_trained))),
        "baseline_max_abs_observable_difference": float(np.max(np.abs(baseline["test"][:check_count] - pyquda_baseline))),
        "trained_max_unitarity_error": trained_errors[0],
        "trained_max_determinant_error": trained_errors[1],
        "baseline_max_unitarity_error": baseline_errors[0],
        "baseline_max_determinant_error": baseline_errors[1],
    }
    (run / "metrics.json").write_text(json.dumps(_json_ready(metrics), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Evaluation complete: test trained={records['test']['trained']['loss']:.6g}, pass={metrics['acceptance']['all']}")


if __name__ == "__main__":
    main()
