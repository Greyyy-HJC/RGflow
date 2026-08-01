"""Observables and ensemble summaries for scalar phi-four fields."""

import numpy as np
from numpy.typing import NDArray

from rgflow.diagnostics import half_drift_z_score, integrated_autocorrelation_time

from .action import Phi4Action


FloatArray = NDArray[np.float64]


def observable_series(
    configurations: FloatArray,
    action: Phi4Action,
) -> dict[str, FloatArray]:
    volume = configurations.shape[-1] ** 2
    magnetization = np.mean(configurations, axis=(-2, -1))
    return {
        "action_density": action(configurations) / volume,
        "magnetization": magnetization,
        "abs_magnetization": np.abs(magnetization),
        "magnetization_squared": magnetization**2,
    }


def summarize_ensemble(
    configurations: FloatArray,
    action: Phi4Action,
) -> dict[str, object]:
    """Return scalar observables, autocorrelations, ESS, and drift checks."""
    volume = configurations.shape[-1] ** 2
    series = observable_series(configurations, action)
    action_density = series["action_density"]
    magnetization = series["magnetization"]
    abs_magnetization = series["abs_magnetization"]
    magnetization_sq = series["magnetization_squared"]
    magnetization_fourth = magnetization_sq**2

    means = {
        "action_density": float(np.mean(action_density)),
        "magnetization": float(np.mean(magnetization)),
        "abs_magnetization": float(np.mean(abs_magnetization)),
        "magnetization_squared": float(np.mean(magnetization_sq)),
        "susceptibility": float(
            volume * (np.mean(magnetization_sq) - np.mean(magnetization) ** 2)
        ),
        "binder_cumulant": float(
            1.0
            - np.mean(magnetization_fourth) / (3.0 * np.mean(magnetization_sq) ** 2)
        ),
    }

    diagnostic_series = {
        name: series[name]
        for name in ("action_density", "abs_magnetization", "magnetization_squared")
    }
    tau_int = {
        name: integrated_autocorrelation_time(values)
        for name, values in diagnostic_series.items()
    }
    total_samples = configurations.shape[0] * configurations.shape[1]
    ess = {
        name: float(total_samples / (2.0 * tau)) for name, tau in tau_int.items()
    }
    drift = {
        name: half_drift_z_score(values)
        for name, values in diagnostic_series.items()
    }
    return {
        "means": means,
        "tau_int": tau_int,
        "effective_sample_size": ess,
        "half_drift_z_score": drift,
    }
