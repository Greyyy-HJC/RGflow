#!/usr/bin/env python3
"""Evaluate a completed CNN checkpoint without recompiling the training graph."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from rgflow.su3.downsampling import LinkCoefficientCNN, block_links
from rgflow.su3.observables import observable_vector_torch
from rgflow.su3.training import (
    FeatureCache,
    _baseline_logits,
    _covariance_inverse,
    _metric_record,
    _link_batches,
    _plot_diagnostics,
    _projected_blocked_observables,
    _split_paths,
    ensemble_paths,
    measure_blocked_with_pyquda,
    observable_names,
    split_indices,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FINE = ROOT / "artifacts" / "4dsu3" / "L24_beta6p20"
DEFAULT_COARSE = ROOT / "artifacts" / "4dsu3" / "L12_beta5p80"
DEFAULT_RUN = ROOT / "artifacts" / "4dsu3" / "downsample" / "L24_beta6p20_to_L12_beta5p80"


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _history_rows() -> list[dict]:
    # Captured from the eight completed training epochs before the final eager evaluation.
    values = [
        (1, 9.17550e6, 9.16673e6, 9.17484e6, 653.979),
        (2, 9.16077e6, 9.15193e6, 9.16012e6, 653.555),
        (3, 9.14598e6, 9.13706e6, 9.14533e6, 653.127),
        (4, 9.13112e6, 9.12211e6, 9.13047e6, 652.699),
        (5, 9.11618e6, 9.10710e6, 9.11553e6, 652.267),
        (6, 9.10116e6, 9.09199e6, 9.10051e6, 651.833),
        (7, 9.08608e6, 9.07682e6, 9.08542e6, 651.397),
        (8, 9.07090e6, 9.06156e6, 9.07025e6, 650.958),
    ]
    return [
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mean_loss": train_mean,
            "train_variance_loss": variance,
            "validation_loss": validation_loss,
            "validation_mean_loss": float("nan"),
            "validation_variance_loss": float("nan"),
            "learning_rate": 0.003,
        }
        for epoch, train_loss, validation_loss, train_mean, variance in values
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-dir", type=Path, default=DEFAULT_FINE)
    parser.add_argument("--coarse-dir", type=Path, default=DEFAULT_COARSE)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, choices=(1, 2), default=2)
    parser.add_argument("--reuse-values", action="store_true", help="reuse observable_distributions.npz from a completed evaluation")
    args = parser.parse_args()

    fine_paths, fine_manifest = ensemble_paths(args.fine_dir.resolve())
    coarse_paths, coarse_manifest = ensemble_paths(args.coarse_dir.resolve())
    run = args.run.resolve()
    lattice = list(map(int, coarse_manifest["lattice_size"]))
    reference = np.load(run / "reference_observables.npz")["observables"]
    if reference.shape != (len(coarse_paths), 4):
        raise SystemExit(f"reference shape {reference.shape} does not match ensemble size {len(coarse_paths)}")

    device = torch.device(args.device)
    indices = split_indices(len(fine_paths))
    cache = FeatureCache(run / "feature_cache")
    baseline_path = run / "baseline_observables.npz"
    if baseline_path.exists():
        baseline_all = np.load(baseline_path)["observables"]
    else:
        baseline_values = []
        with torch.no_grad():
            for links, _ in _link_batches(fine_paths, device, None, False, args.batch_size):
                baseline_values.append(observable_vector_torch(block_links(links)))
        baseline_all = torch.cat(baseline_values).cpu().numpy()
        np.savez(baseline_path, observables=baseline_all)

    model = LinkCoefficientCNN().to(device)
    checkpoint = torch.load(run / "kernel.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    if args.reuse_values:
        saved = np.load(run / "observable_distributions.npz")
        trained_all = np.empty_like(baseline_all)
        for split in indices:
            trained_all[indices[split]] = saved[f"trained_{split}"]
    else:
        trained_all = _projected_blocked_observables(
            fine_paths, model, device, requires_grad=False, feature_cache=cache, batch_size=args.batch_size
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
    test_pass = records["test"]["trained"]["loss"] < records["test"]["baseline"]["loss"]
    metrics = {
        "observables": observable_names(),
        "acceptance": {"test_trained_loss_below_baseline": test_pass, "all": test_pass},
        "splits": records,
        "best_epoch": int(checkpoint["epoch"]),
        "loss_definition": "covariance-weighted mean plus standardized log-variance mismatch",
    }
    (run / "metrics.json").write_text(json.dumps(_json_ready(metrics), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rows = _history_rows()
    with (run / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    np.savez(
        run / "observable_distributions.npz",
        reference_test=targets["test"], baseline_test=baseline["test"], trained_test=trained["test"],
        reference_all=reference, reference_train=targets["train"], reference_validation=targets["validation"],
        baseline_train=baseline["train"], baseline_validation=baseline["validation"],
        trained_train=trained["train"], trained_validation=trained["validation"],
    )
    _plot_diagnostics(run / "diagnostics.pdf", rows, {"reference": targets["test"], "baseline": baseline["test"], "downsampled": trained["test"]}, observable_names())

    from pyquda_utils import core
    core.init(None, lattice, backend="numpy", resource_path=str(run / ".quda-cache"))
    pyquda_trained, trained_errors = measure_blocked_with_pyquda(
        _split_paths(fine_paths, indices, "test"), model, lattice, device, cache
    )
    metrics["pyquda_crosscheck"] = {
        "trained_max_abs_observable_difference": float(np.max(np.abs(trained["test"] - pyquda_trained))),
        "trained_max_unitarity_error": trained_errors[0],
        "trained_max_determinant_error": trained_errors[1],
    }
    (run / "metrics.json").write_text(json.dumps(_json_ready(metrics), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    kernel = {
        "type": "SU3_APE_style_local_CNN_path_convolution",
        "architecture": {"class": "LinkCoefficientCNN", "hidden_channels": 16, "output_channels": 3},
        "path_channels": ["straight", "six_staples", "six_transverse_1x2_rectangles"],
        "projection": "polar SVD projection followed by determinant-one phase correction",
        "blocking": "all-even anchors, product of two projected links",
        "reversibility": "not exact; projection and factor-two blocking are lossy",
        "observables": observable_names(), "data_split": indices, "best_epoch": int(checkpoint["epoch"]),
        "optimizer": {"name": "Adam", "learning_rate": 0.003, "epochs": 8, "patience": 8, "seed": 1234, "batch_size": args.batch_size},
        "backend": {"requested": "compile", "used": "compile", "used_for_training": "compile", "used_for_evaluation": "eager", "compiled": True},
        "input_pipeline": "NERSC reader with one-configuration threaded prefetch and pinned host-to-CUDA transfer",
        "feature_cache": {"directory": cache.directory, "hits": cache.hits, "misses": cache.misses},
        "covariance": covariance, "variance_scale": variance_scale,
        "sources": {"fine_manifest": fine_manifest, "coarse_manifest": coarse_manifest, "fine_sha256": [_sha256(path) for path in fine_paths], "coarse_sha256": [_sha256(path) for path in coarse_paths]},
    }
    (run / "kernel.json").write_text(json.dumps(_json_ready(kernel), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Evaluation complete: test trained={records['test']['trained']['loss']:.6g}, baseline={records['test']['baseline']['loss']:.6g}, pass={test_pass}")


if __name__ == "__main__":
    main()
