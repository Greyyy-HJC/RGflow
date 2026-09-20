import numpy as np

from rgflow.phi4.benchmark import run_native_baseline


def test_native_benchmark_is_deterministic(tmp_path) -> None:
    rng = np.random.default_rng(71)
    configurations = rng.normal(size=(4, 5, 4, 4))
    source = tmp_path / "native.npz"
    np.savez_compressed(source, configurations=configurations)
    first = run_native_baseline(
        source, tmp_path / "first", chains=3, sweeps=3, seed=19
    )
    second = run_native_baseline(
        source, tmp_path / "second", chains=3, sweeps=3, seed=19
    )
    assert first["radial_acceptance"] == second["radial_acceptance"]
    assert first["mean_cluster_fraction"] == second["mean_cluster_fraction"]
    with np.load(tmp_path / "first" / "observable_history.npz") as left:
        with np.load(tmp_path / "second" / "observable_history.npz") as right:
            for name in left:
                np.testing.assert_array_equal(left[name], right[name])
