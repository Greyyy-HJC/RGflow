import numpy as np

from rgflow.phi4.action import Phi4Action
from rgflow.phi4.operator_diagnostics import (
    KERNEL_OBSERVABLE_NAMES,
    OPERATOR_NAMES,
    bootstrap_ensemble_observables,
    distribution_metrics,
    kernel_observable_series,
    operator_series,
    save_kernel_observable_histograms,
)


def test_constant_field_operator_normalizations() -> None:
    fields = np.ones((3, 8, 8), dtype=np.float64)
    action = Phi4Action(kappa=0.2, lam=1.0)
    series = operator_series(fields, action)

    assert tuple(series) == OPERATOR_NAMES
    for values in series.values():
        assert values.shape == (3,)
    np.testing.assert_allclose(series["potential_density"], 1.0)
    np.testing.assert_allclose(series["hopping_action_density"], -0.8)
    np.testing.assert_allclose(series["action_density"], 0.2)
    np.testing.assert_allclose(series["gradient_squared"], 0.0)
    for name in (
        "phi2",
        "phi4",
        "phi6",
        "phi4_over_phi2_squared",
        "magnetization",
        "abs_magnetization",
        "magnetization_squared",
        "magnetization_fourth",
        "C10",
        "C11",
        "C20",
        "C21",
        "C22",
        "C30",
        "C40",
    ):
        np.testing.assert_allclose(series[name], 1.0)
    for name in ("G10", "G11", "G20", "G21", "G22"):
        np.testing.assert_allclose(series[name], 0.0, atol=1.0e-28)


def test_operators_are_invariant_under_translation_and_d4_rotation() -> None:
    rng = np.random.default_rng(53)
    fields = rng.normal(size=(5, 8, 8))
    shifted = np.roll(np.roll(fields, 3, axis=-2), -2, axis=-1)
    rotated = np.rot90(fields, axes=(-2, -1))
    action = Phi4Action()

    original = operator_series(fields, action)
    translated = operator_series(shifted, action)
    d4_rotated = operator_series(rotated, action)
    for name in OPERATOR_NAMES:
        np.testing.assert_allclose(original[name], translated[name])
        np.testing.assert_allclose(original[name], d4_rotated[name])
    np.testing.assert_allclose(
        original["action_density"],
        original["potential_density"] + original["hopping_action_density"],
    )


def test_block_bootstrap_is_finite_and_reproducible() -> None:
    rng = np.random.default_rng(59)
    slow_mode = rng.normal(size=(2, 20, 1, 1))
    fields = slow_mode + 0.1 * rng.normal(size=(2, 20, 8, 8))

    first = bootstrap_ensemble_observables(fields, samples=30, seed=61)
    second = bootstrap_ensemble_observables(fields, samples=30, seed=61)

    assert set(first) == {"susceptibility", "binder_cumulant", "xi_over_L"}
    for name in first:
        assert first[name].shape == (30,)
        assert np.all(np.isfinite(first[name]))
        np.testing.assert_allclose(first[name], second[name])


def test_kernel_observables_match_reference_definitions() -> None:
    fields = np.ones((3, 8, 8), dtype=np.float64)
    action = Phi4Action(kappa=0.2, lam=1.0)
    series = kernel_observable_series(fields, action)

    assert tuple(series) == KERNEL_OBSERVABLE_NAMES
    for name in ("phi2", "phi4", "local_kurtosis_ratio", "NN", "diag", "2nn", "m2", "m4"):
        np.testing.assert_allclose(series[name], 1.0)
    np.testing.assert_allclose(series["G_pmin_avg"], 0.0, atol=1.0e-28)
    np.testing.assert_allclose(series["action_density"], 0.2)


def test_distribution_metrics_include_support_and_tail_diagnostics() -> None:
    values = np.linspace(-2.0, 2.0, 101)
    metrics = distribution_metrics(values, values)

    np.testing.assert_allclose(metrics["js_divergence"], 0.0)
    np.testing.assert_allclose(metrics["total_variation"], 0.0)
    np.testing.assert_allclose(metrics["ks_statistic"], 0.0)
    np.testing.assert_allclose(metrics["wasserstein_1"], 0.0)
    assert 0.97 <= metrics["inside_target_q01_q99"] <= 0.99
    assert 0.89 <= metrics["inside_target_q05_q95"] <= 0.91


def test_kernel_histogram_figure_and_metrics(tmp_path) -> None:
    rng = np.random.default_rng(67)
    coarse = rng.normal(size=(4, 6, 8, 8))
    blocked = coarse + 0.02 * rng.normal(size=coarse.shape)
    output = tmp_path / "kernel_observable_histograms.pdf"

    metrics = save_kernel_observable_histograms(
        output,
        coarse,
        blocked,
        Phi4Action(),
    )

    assert output.exists()
    assert output.with_suffix(".png").exists()
    assert tuple(metrics) == KERNEL_OBSERVABLE_NAMES
    assert all(set(metrics[name]) == {"test", "all"} for name in metrics)
