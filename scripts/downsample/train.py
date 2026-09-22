#!/usr/bin/env python3
"""Train the first L24 -> L12 SU(3) downsampling kernel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rgflow.su3.training import (
    _baseline_logits,
    _split_paths,
    ensemble_paths,
    measure_blocked_with_pyquda,
    measure_reference_with_pyquda,
    split_indices,
    train,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FINE = ROOT / "artifacts" / "4dsu3" / "L24_beta6p20"
DEFAULT_COARSE = ROOT / "artifacts" / "4dsu3" / "L12_beta5p80"
DEFAULT_OUTPUT = ROOT / "artifacts" / "4dsu3" / "downsample" / "L24_beta6p20_to_L12_beta5p80"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-dir", type=Path, default=DEFAULT_FINE)
    parser.add_argument("--coarse-dir", type=Path, default=DEFAULT_COARSE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--backend", choices=("eager", "compile"), default="compile")
    parser.add_argument("--feature-cache-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, choices=(1, 2), default=2)
    parser.add_argument("--configs-per-epoch", type=int, default=None, help="rotating stochastic subset of training configurations")
    parser.add_argument("--validation-configs", type=int, default=None, help="fixed validation subset used for checkpoint selection")
    parser.add_argument("--shrinkage", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-configs", type=int, default=None, help="use a prefix for a smoke test")
    parser.add_argument("--reuse-reference", action="store_true", help="reuse reference_observables.npz in the output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.epochs, args.patience) <= 0 or args.learning_rate <= 0:
        raise SystemExit("epochs, patience, and learning-rate must be positive")
    fine_paths, fine_manifest = ensemble_paths(args.fine_dir.resolve())
    coarse_paths, coarse_manifest = ensemble_paths(args.coarse_dir.resolve())
    if len(fine_paths) != len(coarse_paths):
        raise SystemExit("fine and coarse ensembles must contain the same number of configurations")
    if args.max_configs is not None:
        if args.max_configs < 5 or args.max_configs > len(fine_paths):
            raise SystemExit("max-configs must be at least five and no larger than the ensemble")
        fine_paths = fine_paths[: args.max_configs]
        coarse_paths = coarse_paths[: args.max_configs]
    lattice = list(map(int, coarse_manifest["lattice_size"]))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    reference_path = output / "reference_observables.npz"
    if args.reuse_reference and reference_path.exists():
        reference = np.load(reference_path)["observables"]
        if reference.shape != (len(coarse_paths), 4):
            raise SystemExit(f"cached reference has shape {reference.shape}, expected {(len(coarse_paths), 4)}")
        print(f"Using cached PyQUDA reference measurements for {len(coarse_paths)} configurations", flush=True)
        from pyquda_utils import core

        core.init(None, lattice, backend="numpy", resource_path=str(output / ".quda-cache"))
    else:
        print(f"Using device {device}; measuring {len(coarse_paths)} native coarse configurations with PyQUDA", flush=True)
        reference = measure_reference_with_pyquda(coarse_paths, lattice, output / ".quda-cache")
        np.savez(reference_path, observables=reference)

    result = train(
        fine_paths,
        reference,
        output,
        lattice=lattice,
        device=device,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        seed=args.seed,
        shrinkage=args.shrinkage,
        backend=args.backend,
        feature_cache_dir=args.feature_cache_dir.resolve() if args.feature_cache_dir else None,
        batch_size=args.batch_size,
        configs_per_epoch=args.configs_per_epoch,
        validation_configs=args.validation_configs,
    )
    indices = split_indices(len(fine_paths))
    test_paths = _split_paths(fine_paths, indices, "test")
    baseline_logits = _baseline_logits(device)
    trained_kernel = result["kernel_model"]
    torch_trained = result["trained_values"]["test"]
    torch_baseline = result["baseline_values"]["test"]
    pyquda_trained, trained_errors = measure_blocked_with_pyquda(
        test_paths, trained_kernel, lattice, device, result["feature_cache"]
    )
    pyquda_baseline, baseline_errors = measure_blocked_with_pyquda(
        test_paths, baseline_logits, lattice, device
    )
    np.savez(
        output / "observable_distributions.npz",
        reference_test=reference[indices["test"]],
        baseline_test=torch_baseline,
        trained_test=torch_trained,
        reference_all=reference,
        reference_train=reference[indices["train"]],
        reference_validation=reference[indices["validation"]],
        baseline_train=result["baseline_values"]["train"],
        baseline_validation=result["baseline_values"]["validation"],
        trained_train=result["trained_values"]["train"],
        trained_validation=result["trained_values"]["validation"],
    )
    metrics_path = output / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["pyquda_crosscheck"] = {
        "trained_max_abs_observable_difference": float(np.max(np.abs(torch_trained - pyquda_trained))),
        "baseline_max_abs_observable_difference": float(np.max(np.abs(torch_baseline - pyquda_baseline))),
        "trained_max_unitarity_error": trained_errors[0],
        "trained_max_determinant_error": trained_errors[1],
        "baseline_max_unitarity_error": baseline_errors[0],
        "baseline_max_determinant_error": baseline_errors[1],
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    kernel_path = output / "kernel.json"
    kernel = json.loads(kernel_path.read_text(encoding="utf-8"))
    kernel["sources"] = {
        "fine_manifest": fine_manifest,
        "coarse_manifest": coarse_manifest,
        "fine_sha256": [_sha256(path) for path in fine_paths],
        "coarse_sha256": [_sha256(path) for path in coarse_paths],
    }
    kernel_path.write_text(json.dumps(kernel, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    status = "PASS" if metrics["acceptance"]["all"] else "FAIL"
    trained_loss = metrics["splits"]["test"]["trained"]["loss"]
    baseline_loss = metrics["splits"]["test"]["baseline"]["loss"]
    print(
        f"Downsampling training {status}: test loss {trained_loss:.6g} "
        f"(baseline {baseline_loss:.6g}), backend={kernel['backend']['used']}",
        flush=True,
    )


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
