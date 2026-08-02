import json
from pathlib import Path

import numpy as np

from rgflow.cli import main


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
        kappa=0.3401,
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

    main(
        [
            "train",
            "kernel",
            "phi4",
            "--fine",
            str(fine_path),
            "--coarse",
            str(coarse_path),
            "--transfer-fine",
            str(transfer_path),
            "--output",
            str(output),
            "--starts",
            "1",
            "--max-iterations",
            "2",
            "--bootstrap-samples",
            "20",
            "--seed",
            "23",
        ]
    )

    kernel = json.loads((output / "kernel.json").read_text())
    metrics = json.loads((output / "metrics.json").read_text())
    assert np.asarray(kernel["matrix"]).shape == (5, 5)
    assert kernel["selected_start"] == 0
    assert "L32_to_L16_transfer_test" in metrics
    assert len(
        metrics["operator_distributions"]["configuration_operators"]
    ) == 24
    assert len(metrics["operator_distributions"]["bootstrap_observables"]) == 3
    assert (output / "diagnostics.pdf").stat().st_size > 0
    assert (output / "operator_distributions.pdf").stat().st_size > 0

    main(
        [
            "train",
            "kernel",
            "phi4",
            "--fine",
            str(fine_path),
            "--coarse",
            str(coarse_path),
            "--transfer-fine",
            str(transfer_path),
            "--output",
            str(repeated_output),
            "--starts",
            "1",
            "--max-iterations",
            "2",
            "--bootstrap-samples",
            "20",
            "--seed",
            "23",
        ]
    )
    repeated_kernel = json.loads((repeated_output / "kernel.json").read_text())
    np.testing.assert_allclose(repeated_kernel["matrix"], kernel["matrix"])
