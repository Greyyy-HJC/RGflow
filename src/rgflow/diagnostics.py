"""Diagnostics for scalar observables measured along multiple Markov chains."""

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def integrated_autocorrelation_time(values: FloatArray) -> float:
    """Estimate tau_int with the initial-positive-pair prescription."""
    _, samples = values.shape
    if samples < 2:
        return 0.5

    centered = values - np.mean(values, axis=1, keepdims=True)
    fft_size = 1 << (2 * samples - 1).bit_length()
    transformed = np.fft.rfft(centered, n=fft_size, axis=1)
    autocovariance = np.fft.irfft(
        transformed * transformed.conj(), n=fft_size, axis=1
    )[:, :samples]
    autocovariance /= np.arange(samples, 0, -1)
    autocovariance = np.mean(autocovariance, axis=0)
    if autocovariance[0] <= 0.0:
        return 0.5

    autocorrelation = autocovariance / autocovariance[0]
    tau = 0.5
    for lag in range(1, samples - 1, 2):
        pair = autocorrelation[lag] + autocorrelation[lag + 1]
        if pair <= 0.0:
            break
        tau += float(pair)
    return max(0.5, tau)


def half_drift_z_score(values: FloatArray) -> float:
    """Compare the first and second halves of a multi-chain scalar series."""
    midpoint = values.shape[1] // 2
    if midpoint == 0:
        return 0.0
    first = values[:, :midpoint].reshape(-1)
    second = values[:, -midpoint:].reshape(-1)
    if len(first) < 2:
        return 0.0
    variance = np.var(first, ddof=1) / len(first) + np.var(second, ddof=1) / len(
        second
    )
    if variance == 0.0:
        return 0.0
    return float((np.mean(second) - np.mean(first)) / np.sqrt(variance))
