import numpy as np

from rgflow.phi4.action import Phi4Action
from rgflow.phi4.operator_diagnostics import (
    OPERATOR_NAMES,
    bootstrap_ensemble_observables,
    operator_series,
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
