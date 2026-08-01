import numpy as np

from rgflow.diagnostics import half_drift_z_score, integrated_autocorrelation_time


def test_constant_series_has_minimal_autocorrelation_time() -> None:
    values = np.ones((3, 8))
    assert integrated_autocorrelation_time(values) == 0.5


def test_identical_halves_have_no_drift() -> None:
    half = np.array([[1.0, 2.0], [3.0, 4.0]])
    values = np.concatenate((half, half), axis=1)
    assert half_drift_z_score(values) == 0.0


def test_drift_requires_two_values_per_half() -> None:
    values = np.array([[1.0, 2.0]])
    assert half_drift_z_score(values) == 0.0
