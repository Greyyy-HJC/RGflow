#!/usr/bin/env python3
"""Fit the perfect-blocking polynomial with scaled ensemble least squares."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
import torch

from rgflow.su3.downsampling import (
    PolynomialStoutKernel, block_polynomial_basis, polynomial_blocking_basis,
    su3_errors,
)
from rgflow.su3.observables import observable_names, observable_vector_torch
from rgflow.su3.perfect_blocking import optimize_perturbative_coefficients
from rgflow.su3.training import (
    _covariance_inverse, _json_ready, _load_links, _metric_record,
    distribution_acceptance, ensemble_paths, split_indices,
)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-dir", type=Path, default=root / "artifacts/4dsu3/L24_beta6p20")
    parser.add_argument("--reference-run", type=Path, default=root / "artifacts/4dsu3/downsample/L24_beta6p20_to_L12_beta5p80_stout_v3")
    parser.add_argument("--output", type=Path, default=root / "artifacts/4dsu3/downsample/L24_beta6p20_to_L12_beta5p80_polynomial_ls_v1")
    parser.add_argument("--train-configs", type=int, default=60, help="fixed training subset; 0 uses all training configurations")
    parser.add_argument("--max-nfev", type=int, default=40)
    parser.add_argument("--hook", action="store_true")
    parser.add_argument("--local", action="store_true", help="add four plaquette-conditioned nonlinear coefficients")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--initial-run", type=Path, help="warm start from an existing polynomial checkpoint")
    args = parser.parse_args()
    if args.train_configs < 0 or args.train_configs == 1 or args.max_nfev < 1:
        parser.error("train-configs must be zero or at least two; max-nfev must be positive")
    torch.set_num_threads(2)
    device = torch.device(args.device)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths, _ = ensemble_paths(args.fine_dir.resolve())
    saved = np.load(args.reference_run / "observable_distributions.npz")
    reference = saved["reference_all"].astype(np.float64)
    if reference.shape != (len(paths), 4):
        raise ValueError("cached reference must cover the complete fine ensemble")
    indices = split_indices(len(paths))
    order = np.random.default_rng(args.seed).permutation(indices["train"])
    fit_indices = order[:args.train_configs] if args.train_configs else order
    targets = reference[indices["train"]]
    target_mean = targets.mean(0)
    target_variance = targets.var(0, ddof=1)
    # Relative moment errors, scaled by the sampling precision of both
    # independent ensembles. No covariance estimate from naive blocking.
    mean_scale = np.sqrt(target_variance * (1 / len(fit_indices) + 1 / len(targets)))
    variance_scale = np.sqrt(2 / (len(fit_indices) - 1) + 2 / (len(targets) - 1))
    hook_end = 7 if args.hook else 4
    scales = np.asarray([12., 12.**2, 12.**3, 12.**4] + ([12., 12.**2, 12.**3] if args.hook else []) + ([12., 12.**2, 12.**3, 12.**4] if args.local else []))
    if args.initial_run:
        architecture = json.loads((args.initial_run / "kernel.json").read_text())["architecture"]
        initial = np.asarray(architecture["coefficients"] + ((architecture.get("hook_coefficients") or [0.] * 3) if args.hook else []))
        if args.local:
            initial = np.concatenate((initial, architecture.get("local_coefficients") or [0.] * 4))
        perturbative_residual = None
    else:
        initial_tensor, perturbative_residual = optimize_perturbative_coefficients(grid_size=8, steps=100)
        initial = np.concatenate((initial_tensor.numpy(), np.zeros(3))) if args.hook else initial_tensor.numpy()
    if args.local and not args.initial_run:
        initial = np.concatenate((initial, np.zeros(4)))
    x0 = initial * scales
    print(f"initial coefficients={initial.tolist()}, perturbative residual={perturbative_residual}", flush=True)
    basis_cache = {}
    with torch.no_grad():
        for position, index in enumerate(fit_indices):
            basis_cache[int(index)] = polynomial_blocking_basis(_load_links(paths[index], device), hook=args.hook, local=args.local).cpu()
            if (position + 1) % 10 == 0:
                print(f"cached {position + 1}/{len(fit_indices)} training configurations", flush=True)
    history = []
    last_x = None
    last_result = None

    def objective(x):
        nonlocal last_x, last_result
        if last_x is not None and np.array_equal(x, last_x):
            return last_result
        start = time.perf_counter()
        values, jacobians = [], []
        parameter = torch.tensor(x, dtype=torch.float32, device=device, requires_grad=True)
        for index in fit_indices:
            basis = basis_cache[int(index)].to(device)
            observable = observable_vector_torch(block_polynomial_basis(basis, parameter))
            jacobian = torch.stack([
                torch.autograd.grad(observable[channel], parameter, retain_graph=channel < 3)[0]
                for channel in range(4)
            ])
            values.append(observable.detach().cpu().numpy())
            jacobians.append(jacobian.cpu().numpy())
        values = np.asarray(values, dtype=np.float64)
        jacobians = np.asarray(jacobians, dtype=np.float64)
        mean = values.mean(0)
        variance = values.var(0, ddof=1)
        mean_jacobian = jacobians.mean(0)
        variance_jacobian = 2 * np.einsum("ni,nij->ij", values - mean, jacobians) / (len(values) - 1)
        residual = np.concatenate(((mean - target_mean) / mean_scale, np.log((variance + 1e-12) / (target_variance + 1e-12)) / variance_scale))
        jacobian = np.concatenate((mean_jacobian / mean_scale[:, None], variance_jacobian / (variance[:, None] + 1e-12) / variance_scale))
        record = {
            "evaluation": len(history), "loss": float(residual @ residual),
            "mean_residual": residual[:4].tolist(),
            "variance_residual": residual[4:].tolist(),
            "coefficients": (x / scales).tolist(),
        }
        history.append(record)
        (output / "fit_history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(f"fit {len(history):03d}: loss={record['loss']:.6g}, mean z={residual[:4].round(3)}, {time.perf_counter() - start:.1f}s", flush=True)
        last_x, last_result = x.copy(), (residual, jacobian)
        return last_result

    fit = least_squares(
        lambda x: objective(x)[0], x0, jac=lambda x: objective(x)[1],
        max_nfev=args.max_nfev, x_scale="jac", ftol=1e-5, xtol=1e-5, gtol=1e-5,
    )
    coefficients = fit.x / scales
    model = PolynomialStoutKernel(tuple(coefficients[:4]), tuple(coefficients[4:hook_end]), tuple(coefficients[hook_end:]))
    torch.save({"state_dict": model.state_dict(), "epoch": len(history)}, output / "kernel.pt")
    kernel = {
        "method": "stout-polynomial",
        "architecture": {
            "class": "PolynomialStoutKernel", "degree": 4,
            "initial_coefficients": initial[:4].tolist(),
            "initial_hook_coefficients": initial[4:hook_end].tolist(),
            "initial_local_coefficients": initial[hook_end:].tolist(),
            "coefficients": coefficients[:4].tolist(),
            "hook_coefficients": coefficients[4:hook_end].tolist(),
            "local_coefficients": coefficients[hook_end:].tolist(),
        },
        "optimizer": {
            "name": "least_squares", "fit_indices": fit_indices.tolist(),
            "max_nfev": args.max_nfev, "nfev": fit.nfev, "njev": fit.njev,
            "success": bool(fit.success), "message": fit.message,
        },
        "data_split": indices, "perturbative_residual": perturbative_residual,
        "reference_run": str(args.reference_run.resolve()),
        "mean_scale": mean_scale.tolist(), "variance_scale": float(variance_scale),
    }
    (output / "kernel.json").write_text(json.dumps(kernel, indent=2) + "\n")
    np.savez(output / "reference_observables.npz", observables=reference)
    baseline_all = np.empty_like(reference)
    for split, selected in indices.items():
        baseline_all[selected] = saved[f"baseline_{split}"]
    np.savez(output / "baseline_observables.npz", observables=baseline_all)
    distributions = {"reference_all": reference}
    records = {}
    covariance, inverse = _covariance_inverse(targets, saved["baseline_train"], 0.1)
    evaluation_variance_scale = np.full(4, np.sqrt(2 / (len(targets) - 1)))
    errors = [0., 0.]
    fitted_parameter = torch.tensor(fit.x, dtype=torch.float32, device=device)
    with torch.no_grad():
        for split, split_indices_ in indices.items():
            values = []
            for position, index in enumerate(split_indices_):
                basis = basis_cache[index].to(device) if index in basis_cache else polynomial_blocking_basis(_load_links(paths[index], device), hook=args.hook, local=args.local)
                blocked = block_polynomial_basis(basis, fitted_parameter)
                values.append(observable_vector_torch(blocked).cpu().numpy())
                if position == 0:
                    errors = np.maximum(errors, su3_errors(blocked)).tolist()
                if (position + 1) % 30 == 0:
                    print(f"evaluated {split}: {position + 1}/{len(split_indices_)}", flush=True)
            trained = np.asarray(values, dtype=np.float64)
            target = reference[split_indices_]
            baseline = saved[f"baseline_{split}"]
            distributions.update({f"reference_{split}": target, f"baseline_{split}": baseline, f"trained_{split}": trained})
            records[split] = {
                "baseline": _metric_record(baseline, target, inverse, evaluation_variance_scale),
                "trained": _metric_record(trained, target, inverse, evaluation_variance_scale),
            }
            record = records[split]["trained"]
            print(f"{split}: mean z={record['standardized_shift']}, std ratio={record['std_ratio']}", flush=True)
    test = records["test"]["trained"]
    acceptance = distribution_acceptance(test)
    acceptance["all"] = acceptance["distribution_match"]
    metrics = {
        "observables": observable_names(), "splits": records, "acceptance": acceptance,
        "su3_errors_first_configuration_per_split": errors, "covariance": covariance,
        "note": "One target sigma measures distribution overlap; standardized_shift uses combined standard errors. Fit uses training data only.",
    }
    (output / "metrics.json").write_text(json.dumps(_json_ready(metrics), indent=2) + "\n")
    np.savez(output / "observable_distributions.npz", **distributions)
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    target = distributions["reference_test"]
    center, sigma = target.mean(0), target.std(0, ddof=1)
    for channel, axis in enumerate(axes.flat):
        samples = [target[:, channel], saved["trained_test"][:, channel], distributions["trained_test"][:, channel]]
        normalized = [(sample - center[channel]) / sigma[channel] for sample in samples]
        bins = np.histogram_bin_edges(np.concatenate(normalized), bins=30)
        for values, label in zip(normalized, ("native coarse", "previous fit", "local polynomial" if args.local else ("polynomial + hooks" if args.hook else "polynomial"))):
            axis.hist(values, bins=bins, density=True, histtype="step", linewidth=1.5, label=label)
        axis.set(title=observable_names()[channel], xlabel="(operator - target mean) / target std", ylabel="density")
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "distributions.pdf")
    figure.savefig(output / "distributions.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
