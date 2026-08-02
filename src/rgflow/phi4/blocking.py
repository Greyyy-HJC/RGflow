"""Translationally invariant blocking kernels for two-dimensional phi-four."""

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]

ETA = 0.25
KERNEL_SUM = 2.0 ** (ETA / 2.0)
ORBIT_NAMES = ("K10", "K11", "K20", "K21", "K22")
ORBIT_MULTIPLICITIES = np.array([4.0, 4.0, 4.0, 8.0, 4.0])


def kernel_from_parameters(
    parameters: FloatArray,
    total: float = KERNEL_SUM,
) -> FloatArray:
    """Return a D4-symmetric 5x5 kernel from its five off-center orbits."""
    parameters = np.asarray(parameters, dtype=np.float64)
    k10, k11, k20, k21, k22 = parameters
    k00 = total - float(ORBIT_MULTIPLICITIES @ parameters)
    return np.array(
        [
            [k22, k21, k20, k21, k22],
            [k21, k11, k10, k11, k21],
            [k20, k10, k00, k10, k20],
            [k21, k11, k10, k11, k21],
            [k22, k21, k20, k21, k22],
        ],
        dtype=np.float64,
    )


def identity_kernel(total: float = KERNEL_SUM) -> FloatArray:
    """Return the scaled identity in the 5x5 kernel parameterization."""
    return kernel_from_parameters(np.zeros(5, dtype=np.float64), total=total)


def apply_kernel(field: FloatArray, kernel: FloatArray) -> FloatArray:
    """Apply a finite-range kernel with periodic boundary conditions."""
    field = np.asarray(field, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    radius = kernel.shape[0] // 2
    transformed = np.zeros_like(field)
    for row, column in np.ndindex(kernel.shape):
        transformed += kernel[row, column] * np.roll(
            np.roll(field, row - radius, axis=-2),
            column - radius,
            axis=-1,
        )
    return transformed


def block_field(field: FloatArray, kernel: FloatArray) -> FloatArray:
    """Apply the kernel and retain the even-even sublattice."""
    return apply_kernel(field, kernel)[..., 0::2, 0::2]


def kernel_transform(kernel: FloatArray, lattice_size: int) -> NDArray[np.complex128]:
    """Return the discrete Fourier transform of a periodic convolution kernel."""
    kernel = np.asarray(kernel, dtype=np.float64)
    weights = np.zeros((lattice_size, lattice_size), dtype=np.float64)
    radius = kernel.shape[0] // 2
    for row, column in np.ndindex(kernel.shape):
        weights[
            (row - radius) % lattice_size,
            (column - radius) % lattice_size,
        ] += kernel[row, column]
    return np.fft.fft2(weights)


def spectrum_diagnostics(kernel: FloatArray, lattice_size: int) -> dict[str, float]:
    """Summarize the magnitude and conditioning of K on a periodic lattice."""
    magnitude = np.abs(kernel_transform(kernel, lattice_size))
    minimum = float(np.min(magnitude))
    maximum = float(np.max(magnitude))
    return {
        "min_abs_K": minimum,
        "max_abs_K": maximum,
        "condition_number": maximum / minimum,
        "max_abs_inverse_K": 1.0 / minimum,
    }


def inverse_kernel(field: FloatArray, kernel: FloatArray) -> FloatArray:
    """Invert a periodic kernel in Fourier space."""
    field = np.asarray(field, dtype=np.float64)
    transform = kernel_transform(kernel, field.shape[-1])
    axes = (-2, -1)
    inverse = np.fft.ifft2(np.fft.fft2(field, axes=axes) / transform, axes=axes)
    return inverse.real
