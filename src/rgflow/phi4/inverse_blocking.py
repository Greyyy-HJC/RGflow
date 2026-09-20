"""Coordinates for factor-two stochastic inverse blocking."""

import numpy as np
from numpy.typing import NDArray

from .blocking import block_field, inverse_kernel


FloatArray = NDArray[np.float64]

DETAIL_NAMES = ("d01", "d10", "d11")


def assemble_details(coarse: FloatArray, details: FloatArray) -> FloatArray:
    """Interleave ``(coarse, d01, d10, d11)`` into a fine-size psi field."""
    coarse = np.asarray(coarse)
    details = np.asarray(details)
    lattice_size = coarse.shape[-1]
    psi = np.empty(
        (*coarse.shape[:-2], 2 * lattice_size, 2 * lattice_size),
        dtype=np.result_type(coarse, details),
    )
    psi[..., 0::2, 0::2] = coarse
    psi[..., 0::2, 1::2] = details[..., 0, :, :]
    psi[..., 1::2, 0::2] = details[..., 1, :, :]
    psi[..., 1::2, 1::2] = details[..., 2, :, :]
    return psi


def extract_details(psi: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Extract the retained coarse field and three missing detail sublattices."""
    psi = np.asarray(psi)
    coarse = psi[..., 0::2, 0::2]
    details = np.stack(
        (
            psi[..., 0::2, 1::2],
            psi[..., 1::2, 0::2],
            psi[..., 1::2, 1::2],
        ),
        axis=-3,
    )
    return coarse, details


def reconstruct_fine(
    coarse: FloatArray,
    details: FloatArray,
    kernel: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Return ``(phi_f, psi)`` for physical inverse-blocking coordinates."""
    psi = assemble_details(coarse, details)
    return inverse_kernel(psi, kernel), psi


def reblocking_error(
    fine: FloatArray,
    coarse: FloatArray,
    kernel: FloatArray,
) -> FloatArray:
    """Return one max-norm ``DK phi_f - coarse`` error per configuration."""
    difference = block_field(fine, kernel) - coarse
    return np.max(np.abs(difference), axis=(-2, -1))
