import json

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
