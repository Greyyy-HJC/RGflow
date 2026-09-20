import json
from pathlib import Path

import numpy as np

from rgflow.cli import build_parser, main
from rgflow.phi4.kernel_cli import DEFAULT_TRAINING_PAIRS, DEFAULT_TRANSFER_PAIRS


def test_phi4_production_defaults_use_reference_critical_coupling() -> None:
    generate = build_parser().parse_args(["generate", "phi4"])
    assert generate.sizes == [8, 16, 32]
    assert generate.kappa == 0.340301

    train = build_parser().parse_args(["train", "kernel", "phi4"])
    assert train.training_pair is None
    assert train.transfer_pair is None
    assert [pair[0].name for pair in DEFAULT_TRAINING_PAIRS] == [
        "phi4_L16_k0.340301_lam1.npz",
        "phi4_L24_k0.340301_lam1.npz",
        "phi4_L32_k0.340301_lam1.npz",
    ]
    assert [pair[0].name for pair in DEFAULT_TRANSFER_PAIRS] == [
        "phi4_L48_k0.340301_lam1.npz",
        "phi4_L64_k0.340301_lam1.npz",
    ]
    assert train.output == Path(
        "artifacts/phi4/kernel/multivolume_sos2_k0.340301"
    )
    assert train.starts == 6
    assert train.max_iterations == 250
    assert train.optimization_samples_per_chain == 128


    benchmark = build_parser().parse_args(
        [
            "benchmark",
            "phi4",
            "--case",
            "checkpoint.pt",
            "coarse.npz",
            "native.npz",
        ]
    )
    assert benchmark.chains == 128
    assert benchmark.sweeps == 100
    assert benchmark.replicates == 1
    assert "detail2-wolff" in benchmark.methods
    assert "hmc-wolff" in benchmark.methods


def test_cli_writes_configurations_and_diagnostics(tmp_path) -> None:
    main(
        [
            "generate",
            "phi4",
            "--sizes",
            "4",
            "--chains",
            "2",
            "--samples-per-chain",
            "3",
            "--warmup-cycles",
            "25",
            "--sample-interval",
            "1",
            "--seed",
            "31",
            "--output",
            str(tmp_path),
        ]
    )

    data_path = next(tmp_path.glob("*.npz"))
    diagnostics_path = next(tmp_path.glob("*.json"))
    with np.load(data_path) as data:
        assert data["configurations"].shape == (2, 3, 4, 4)
        assert int(data["size"]) == 4
    diagnostics = json.loads(diagnostics_path.read_text())
    assert diagnostics["parameters"]["seed"] == 31
    assert "effective_sample_size" in diagnostics
    assert len(list(tmp_path.glob("*.svg"))) == 2


def _write_kernel_ensemble(
    path: Path,
    configurations: np.ndarray,
    size: int,
) -> None:
    np.savez_compressed(
        path,
        configurations=configurations,
        size=size,
        kappa=0.340301,
        lam=1.0,
        seed=17,
    )


def test_kernel_training_cli_writes_operational_artifacts(tmp_path) -> None:
    rng = np.random.default_rng(19)
    fine = rng.normal(size=(4, 5, 4, 4))
    coarse = fine[:, :, 0::2, 0::2] + 0.05 * rng.normal(size=(4, 5, 2, 2))
    transfer = rng.normal(size=(4, 5, 8, 8))
    fine_path = tmp_path / "fine.npz"
    coarse_path = tmp_path / "coarse.npz"
    transfer_path = tmp_path / "transfer.npz"
    output = tmp_path / "kernel"
    repeated_output = tmp_path / "kernel_repeated"
    _write_kernel_ensemble(fine_path, fine, 4)
    _write_kernel_ensemble(coarse_path, coarse, 2)
    _write_kernel_ensemble(transfer_path, transfer, 8)

    command = [
        "train",
        "kernel",
        "phi4",
        "--training-pair",
        str(fine_path),
        str(coarse_path),
        "--transfer-pair",
        str(transfer_path),
        str(fine_path),
        "--starts",
        "1",
        "--max-iterations",
        "2",
        "--bootstrap-samples",
        "20",
        "--seed",
        "23",
    ]
    main([*command, "--output", str(output)])

    kernel = json.loads((output / "kernel.json").read_text())
    metrics = json.loads((output / "metrics.json").read_text())
    assert np.asarray(kernel["matrix"]).shape == (5, 5)
    assert kernel["selected_start"] == 0
    assert kernel["parameterization"] == "multichannel_sos"
    assert kernel["training_strategy"] == "equal_weight_multivolume_one_step"
    assert kernel["optimizer"]["optimization_samples_per_chain"] == {"L4_to_L2": 5}
    assert kernel["invertibility"] == "by_construction_positive_spectrum"
    assert np.asarray(kernel["sos"]["filters"]).shape == (2, 3, 3)
    assert kernel["spectrum"]["L256"]["min_abs_K"] > 0.0
    assert list(metrics["training_pairs"]) == ["L4_to_L2"]
    assert list(metrics["transfer_pairs"]) == ["L8_to_L4"]
    assert len(
        metrics["operator_distributions"]["configuration_operators"]
    ) == 24
    assert len(metrics["operator_distributions"]["bootstrap_observables"]) == 3
    assert len(metrics["kernel_observable_distributions"]) == 10
    assert (output / "kernel_observable_histograms.pdf").stat().st_size > 0
    assert (output / "kernel_observable_histograms.png").stat().st_size > 0
    assert (output / "diagnostics.pdf").stat().st_size > 0
    assert (output / "operator_distributions.pdf").stat().st_size > 0

    main([*command, "--output", str(repeated_output)])
    repeated_kernel = json.loads((repeated_output / "kernel.json").read_text())
    np.testing.assert_allclose(repeated_kernel["matrix"], kernel["matrix"])
