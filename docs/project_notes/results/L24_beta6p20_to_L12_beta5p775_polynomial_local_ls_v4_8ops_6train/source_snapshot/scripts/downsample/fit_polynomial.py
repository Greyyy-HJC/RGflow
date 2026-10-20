#!/usr/bin/env python3
"""Fit six operator distributions and evaluate eight with polynomial blocking."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
import torch

from rgflow.su3.downsampling import (
    PolynomialStoutKernel, block_links, block_polynomial_basis,
    polynomial_blocking_basis, su3_errors,
)
from rgflow.su3.observables import (
    DOWNSAMPLING_OBSERVABLES, TRAINING_OBSERVABLES,
    observable_names, observable_vector_torch,
)
from rgflow.su3.perfect_blocking import optimize_perturbative_coefficients
from rgflow.su3.training import (
    _covariance_inverse, _json_ready, _load_links, _metric_record,
    distribution_acceptance, ensemble_paths, split_indices,
    measure_reference_with_pyquda, measure_polynomial_with_pyquda,
)


def configuration_moments(basis, x, device, *, names=None):
    names = observable_names() if names is None else tuple(names)
    parameter = torch.tensor(x, dtype=torch.float32, device=device, requires_grad=True)
    blocked = block_polynomial_basis(basis.to(device), parameter)
    values, jacobians = [], []
    # A stacked output sends zero gradients through every unselected loop.
    # Keep each scalar graph separate while reusing the blocked field.
    for channel, name in enumerate(names):
        value = observable_vector_torch(blocked, names=(name,))[0]
        jacobians.append(torch.autograd.grad(value, parameter, retain_graph=channel < len(names) - 1)[0])
        values.append(value.detach())
    return torch.stack(values).cpu().numpy(), torch.stack(jacobians).cpu().numpy()


def moment_residuals(values, jacobians, targets):
    """Residuals/Jacobian in target-sigma units, including very small IR moments."""
    values, jacobians, targets = (np.asarray(a, dtype=np.float64) for a in (values, jacobians, targets))
    mean, variance = values.mean(0), values.var(0, ddof=1)
    target_variance = targets.var(0, ddof=1)
    mean_scale = np.sqrt(target_variance * (1 / len(values) + 1 / len(targets)))
    variance_scale = np.sqrt(2 / (len(values) - 1) + 2 / (len(targets) - 1))
    epsilon = 1e-12 * target_variance
    mean_jacobian = jacobians.mean(0)
    variance_jacobian = 2 * np.einsum("ni,nij->ij", values - mean, jacobians) / (len(values) - 1)
    residual = np.concatenate((
        (mean - targets.mean(0)) / mean_scale,
        np.log((variance + epsilon) / (target_variance + epsilon)) / variance_scale,
    ))
    jacobian = np.concatenate((
        mean_jacobian / mean_scale[:, None],
        variance_jacobian / (variance[:, None] + epsilon[:, None]) / variance_scale,
    ))
    return residual, jacobian


def grouped_acceptance(record, names, training_names):
    groups = {
        "training": [i for i, name in enumerate(names) if name in training_names],
        "held_out": [i for i, name in enumerate(names) if name not in training_names],
        "all": list(range(len(names))),
    }
    return {
        group: distribution_acceptance({key: np.asarray(record[key])[indices] for key in (
            "mean_shift_in_target_sigma", "std_ratio", "standardized_shift",
        )})
        for group, indices in groups.items() if indices
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-dir", type=Path, default=root / "artifacts/4dsu3/L24_beta6p20")
    parser.add_argument("--coarse-dir", type=Path, default=root / "artifacts/4dsu3/L12_beta5p775")
    parser.add_argument("--reference-run", type=Path, help="archive provenance only; four-column caches are never reused")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--train-configs", type=int, default=60, help="fixed training subset; 0 uses all training configurations")
    parser.add_argument("--max-nfev", type=int, default=100)
    parser.add_argument("--hook", action="store_true")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--initial-run", type=Path, help="checkpoint trained without either held-out operator")
    parser.add_argument("--pyquda-check", action="store_true")
    args = parser.parse_args()
    if args.train_configs < 0 or args.train_configs == 1 or args.max_nfev < 1:
        parser.error("train-configs must be zero or at least two; max-nfev must be positive")
    names, training_names = DOWNSAMPLING_OBSERVABLES, TRAINING_OBSERVABLES
    training_columns = [names.index(name) for name in training_names]
    metadata = {
        "observables": names, "training_observables": training_names,
        "held_out_observables": [name for name in names if name not in training_names],
    }
    initial_kernel = None
    if args.initial_run:
        initial_kernel = json.loads((args.initial_run / "kernel.json").read_text())
        previous_training = initial_kernel.get("training_observables", observable_names())
        if set(previous_training) - set(training_names):
            parser.error("initial checkpoint used held-out operators; use perturbative initialization")
    torch.set_num_threads(2)
    device = torch.device(args.device)
    moment_device = torch.device("cpu") if args.workers > 1 else device
    fit_name = "polynomial_local_ls_v4_8ops_6train" if args.local else "polynomial_ls_v4_8ops_6train"
    output = (args.output or root / "artifacts/4dsu3/downsample" / f"{args.fine_dir.name}_to_{args.coarse_dir.name}_{fit_name}").resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "kernel.pt").exists():
        parser.error("output already contains a checkpoint; choose a new directory")
    paths, fine_manifest = ensemble_paths(args.fine_dir.resolve())
    coarse_paths, coarse_manifest = ensemble_paths(args.coarse_dir.resolve())
    if len(paths) != len(coarse_paths) or fine_manifest["lattice_size"] != [2 * size for size in coarse_manifest["lattice_size"]]:
        parser.error("fine/coarse ensembles must have equal counts and factor-two lattice extents")
    source_paths = [Path(__file__), *[root / "src/rgflow/su3" / name for name in ("observables.py", "training.py", "downsampling.py", "perfect_blocking.py", "linalg.py", "io.py")]]
    for source in source_paths:
        snapshot = output / "source_snapshot" / source.relative_to(root)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(source.read_bytes())
    (output / "source_hashes.json").write_text(json.dumps({str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}, indent=2) + "\n")
    (output / "training_arguments.json").write_text(json.dumps(_json_ready({**vars(args), **metadata, "command": sys.argv}), indent=2) + "\n")
    (output / "environment.json").write_text(json.dumps({"python": platform.python_version(), "executable": sys.executable, "torch": torch.__version__, "numpy": np.__version__}, indent=2) + "\n")
    indices = split_indices(len(paths))
    print(f"Measuring {len(coarse_paths)} native coarse configurations at beta={coarse_manifest['beta']}", flush=True)
    reference = measure_reference_with_pyquda(coarse_paths, coarse_manifest["lattice_size"], args.coarse_dir.resolve() / ".quda-cache", names=names)
    np.savez(output / "reference_observables.npz", observables=reference, names=names)
    order = np.random.default_rng(args.seed).permutation(indices["train"])
    fit_indices = order[:args.train_configs] if args.train_configs else order
    targets = reference[indices["train"]][:, training_columns]
    hook_end = 7 if args.hook else 4
    scales = np.asarray([12.**n for n in (1, 2, 3, 4)] + ([12.**n for n in (1, 2, 3)] if args.hook else []) + ([12.**n for n in (1, 2, 3, 4)] if args.local else []))
    if initial_kernel:
        architecture = initial_kernel["architecture"]
        initial = np.asarray(architecture["coefficients"] + ((architecture.get("hook_coefficients") or [0.] * 3) if args.hook else []) + ((architecture.get("local_coefficients") or [0.] * 4) if args.local else []))
        perturbative_residual = None
    else:
        initial_tensor, perturbative_residual = optimize_perturbative_coefficients(grid_size=8, steps=100)
        initial = np.concatenate((initial_tensor.numpy(), np.zeros(len(scales) - 4)))
    x0 = initial * scales
    print(f"initial coefficients={initial.tolist()}, perturbative residual={perturbative_residual}", flush=True)
    pool = ThreadPoolExecutor(max_workers=args.workers) if args.workers > 1 else None

    @torch.no_grad()
    def build_basis(index):
        return polynomial_blocking_basis(_load_links(paths[index], moment_device), hook=args.hook, local=args.local).cpu()

    basis_cache = {}
    bases = pool.map(build_basis, fit_indices) if pool else map(build_basis, fit_indices)
    for position, (index, basis) in enumerate(zip(fit_indices, bases)):
        basis_cache[int(index)] = basis
        if (position + 1) % 10 == 0:
            print(f"cached {position + 1}/{len(fit_indices)} training configurations", flush=True)
    history, last_x, last_result = [], None, None

    def objective(x):
        nonlocal last_x, last_result
        if last_x is not None and np.array_equal(x, last_x):
            return last_result
        start = time.perf_counter()
        measure = partial(configuration_moments, x=x, device=moment_device, names=training_names)
        bases = (basis_cache[int(index)] for index in fit_indices)
        values, jacobians = zip(*(pool.map(measure, bases) if pool else map(measure, bases)))
        residual, jacobian = moment_residuals(values, jacobians, targets)
        record = {
            "evaluation": len(history), "loss": float(residual @ residual),
            "mean_residual": residual[:len(training_names)].tolist(),
            "variance_residual": residual[len(training_names):].tolist(),
            "coefficients": (x / scales).tolist(), "seconds": time.perf_counter() - start,
        }
        history.append(record)
        (output / "fit_history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(f"fit {len(history):03d}: loss={record['loss']:.6g}, mean z={residual[:len(training_names)].round(3)}, {record['seconds']:.1f}s", flush=True)
        last_x, last_result = x.copy(), (residual, jacobian)
        return last_result

    fit = least_squares(lambda x: objective(x)[0], x0, jac=lambda x: objective(x)[1], max_nfev=args.max_nfev, x_scale="jac", ftol=1e-5, xtol=1e-5, gtol=1e-5)
    coefficients = fit.x / scales
    model = PolynomialStoutKernel(tuple(coefficients[:4]), tuple(coefficients[4:hook_end]), tuple(coefficients[hook_end:]))
    torch.save({"state_dict": model.state_dict(), "epoch": len(history), **metadata}, output / "kernel.pt")
    kernel = {
        "method": "stout-polynomial", **metadata,
        "architecture": {
            "class": "PolynomialStoutKernel", "degree": 4,
            "initial_coefficients": initial[:4].tolist(), "initial_hook_coefficients": initial[4:hook_end].tolist(), "initial_local_coefficients": initial[hook_end:].tolist(),
            "coefficients": coefficients[:4].tolist(), "hook_coefficients": coefficients[4:hook_end].tolist(), "local_coefficients": coefficients[hook_end:].tolist(),
        },
        "optimizer": {"name": "least_squares", "fit_indices": fit_indices.tolist(), "max_nfev": args.max_nfev, "nfev": fit.nfev, "njev": fit.njev, "success": bool(fit.success), "message": fit.message, "moment_device": str(moment_device), "workers": args.workers, "basis_device": str(moment_device)},
        "data_split": indices, "perturbative_residual": perturbative_residual,
        "sources": {"fine_directory": str(args.fine_dir.resolve()), "coarse_directory": str(args.coarse_dir.resolve()), "fine_manifest": fine_manifest, "coarse_manifest": coarse_manifest},
        "initial_run": str(args.initial_run.resolve()) if args.initial_run else None,
        "reference_run": str(args.reference_run.resolve()) if args.reference_run else None,
        "mean_scale": np.sqrt(targets.var(0, ddof=1) * (1 / len(fit_indices) + 1 / len(targets))).tolist(),
        "variance_scale": float(np.sqrt(2 / (len(fit_indices) - 1) + 2 / (len(targets) - 1))),
        "variance_epsilon": (1e-12 * targets.var(0, ddof=1)).tolist(),
    }
    (output / "kernel.json").write_text(json.dumps(kernel, indent=2) + "\n")
    print(f"Frozen checkpoint: {fit.message}", flush=True)
    parameter = torch.tensor(fit.x, dtype=torch.float32, device=moment_device)

    @torch.no_grad()
    def evaluate_configuration(index):
        links = _load_links(paths[index], moment_device)
        baseline = observable_vector_torch(block_links(links), names=names).cpu().numpy()
        basis = basis_cache[index].to(moment_device) if index in basis_cache else polynomial_blocking_basis(links, hook=args.hook, local=args.local)
        blocked = block_polynomial_basis(basis, parameter)
        return baseline, observable_vector_torch(blocked, names=names).cpu().numpy(), su3_errors(blocked)

    baselines, trained_values, errors = [], [], []
    for position, (baseline, trained, error) in enumerate(pool.map(evaluate_configuration, range(len(paths))) if pool else map(evaluate_configuration, range(len(paths)))):
        baselines.append(baseline); trained_values.append(trained); errors.append(error)
        if (position + 1) % 10 == 0:
            print(f"evaluated {position + 1}/{len(paths)} configurations (all eight operators)", flush=True)
    if pool:
        pool.shutdown()
    baseline_all, trained_all = np.asarray(baselines), np.asarray(trained_values)
    np.savez(output / "baseline_observables.npz", observables=baseline_all, names=names)
    target_train = reference[indices["train"]]
    covariance, inverse = _covariance_inverse(target_train, baseline_all[indices["train"]], 0.1, observable_scale=target_train.std(0, ddof=1))
    variance_epsilon = 1e-12 * target_train.var(0, ddof=1)
    evaluation_variance_scale = np.full(len(names), np.sqrt(2 / (len(target_train) - 1)))
    distributions, records = {"reference_all": reference, "names": names}, {}
    for split, selected in indices.items():
        target, baseline, trained = reference[selected], baseline_all[selected], trained_all[selected]
        distributions.update({f"reference_{split}": target, f"baseline_{split}": baseline, f"trained_{split}": trained})
        records[split] = {label: _metric_record(values, target, inverse, evaluation_variance_scale, variance_epsilon=variance_epsilon) for label, values in (("baseline", baseline), ("trained", trained))}
        for record in records[split].values():
            record["acceptance_by_group"] = grouped_acceptance(record, names, training_names)
    acceptance = distribution_acceptance(records["test"]["trained"])
    acceptance["all"] = acceptance["distribution_match"]
    metrics = {
        **metadata, "splits": records, "acceptance": acceptance,
        "acceptance_by_group": records["test"]["trained"]["acceptance_by_group"],
        "evaluation": {"device": str(moment_device), "workers": args.workers, "su3_errors_all_configurations": np.max(errors, axis=0)},
        "covariance": covariance, "variance_epsilon": variance_epsilon,
        "note": "Six training operators only; two held-out operators are measured after freezing. Raw samples retain their physical normalization. No old four-operator aggregate loss comparison.",
    }
    np.savez(output / "observable_distributions.npz", **distributions)
    (output / "metrics.json").write_text(json.dumps(_json_ready(metrics), indent=2) + "\n")
    if args.pyquda_check:
        check_indices = indices["test"][:3]
        print("Cross-checking three held-out configurations with full-map PyQUDA", flush=True)
        pyquda_values, pyquda_errors = measure_polynomial_with_pyquda([paths[index] for index in check_indices], model.to(device), coarse_manifest["lattice_size"], device, projection_device="cpu", names=names)
        difference = np.abs(trained_all[check_indices] - pyquda_values)
        metrics["pyquda_crosscheck"] = {"fine_indices": check_indices, "projection_device": "cpu", "trained_max_abs_observable_difference": float(difference.max()), "per_observable_max_abs_difference": difference.max(0), "trained_values": trained_all[check_indices], "pyquda_values": pyquda_values, "trained_max_unitarity_error": pyquda_errors[0], "trained_max_determinant_error": pyquda_errors[1]}
        (output / "metrics.json").write_text(json.dumps(_json_ready(metrics), indent=2) + "\n")
    figure, axes = plt.subplots(2, 4, figsize=(18, 8))
    target = distributions["reference_test"]
    center, sigma = target.mean(0), target.std(0, ddof=1)
    for channel, axis in enumerate(axes.flat):
        samples = [target[:, channel], distributions["trained_test"][:, channel], distributions["baseline_test"][:, channel]]
        normalized = [(sample - center[channel]) / sigma[channel] for sample in samples]
        for values, label in zip(normalized, ("native coarse", "fitted polynomial", "naive blocking")):
            axis.hist(values, bins="fd", density=True, histtype="step", linewidth=1.5, label=label)
        role = "training" if names[channel] in training_names else "HELD OUT"
        axis.set(title=f"{names[channel]} ({role})", xlabel="(operator - target mean) / target std", ylabel="density")
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "distributions.pdf")
    figure.savefig(output / "distributions.png", dpi=160)
    plt.close(figure)
    print("Complete: " + json.dumps(metrics["acceptance_by_group"]), flush=True)


if __name__ == "__main__":
    main()
