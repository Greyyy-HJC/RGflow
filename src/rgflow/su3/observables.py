"""Wilson-loop and Polyakov-loop observables for SU(3) link fields."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from .linalg import matrix_product


def _direction_axis(links: torch.Tensor) -> int:
    axis = links.ndim - 7
    if links.ndim < 7 or links.shape[axis] != 4 or links.shape[-2:] != (3, 3):
        raise ValueError("links must have shape (..., 4, L0, L1, L2, L3, 3, 3)")
    return axis


def _site_axis(links: torch.Tensor, direction: int) -> int:
    return _direction_axis(links) + 1 + (3 - direction)


def _dagger(matrix: torch.Tensor) -> torch.Tensor:
    return matrix.conj().transpose(-2, -1)


def _shift(matrix: torch.Tensor, site_axes: Sequence[int], offset: Sequence[int]) -> torch.Tensor:
    result = matrix
    for axis, shift in zip(site_axes, offset):
        if shift:
            result = torch.roll(result, shifts=-int(shift), dims=axis)
    return result


def path_product(links: torch.Tensor, path: Iterable[int]) -> torch.Tensor:
    """Multiply a path at every starting site using 0..3/+ and 4..7/- directions."""
    direction_axis = _direction_axis(links)
    site_axes = [direction_axis + 1 + (3 - direction) for direction in range(4)]
    result = None
    offset = [0, 0, 0, 0]
    for step in path:
        direction = int(step) % 4
        positive = int(step) < 4
        link = links.select(direction_axis, direction)
        if positive:
            factor = _shift(link, [axis - 1 for axis in site_axes], offset)
            offset[direction] += 1
        else:
            start = offset.copy()
            start[direction] -= 1
            factor = _dagger(_shift(link, [axis - 1 for axis in site_axes], start))
            offset[direction] -= 1
        result = factor if result is None else matrix_product(result, factor)
    if result is None:
        raise ValueError("path must not be empty")
    return result


def _normalized_trace(matrix: torch.Tensor) -> torch.Tensor:
    return torch.diagonal(matrix, dim1=-2, dim2=-1).sum(-1).real / 3.0


def _normalized_complex_trace(matrix: torch.Tensor) -> torch.Tensor:
    return torch.diagonal(matrix, dim1=-2, dim2=-1).sum(-1) / 3.0


def _loop_mean(links: torch.Tensor, path: Sequence[int], *, checkpoint_path: bool = False) -> torch.Tensor:
    matrix = checkpoint(path_product, links, path, use_reentrant=False) if checkpoint_path else path_product(links, path)
    values = _normalized_trace(matrix)
    if links.ndim == 8:
        return values.mean(dim=tuple(range(1, values.ndim)))
    return values.mean()


def _polyakov_mean(links: torch.Tensor, direction: int) -> torch.Tensor:
    # Cyclicity makes the trace independent of the origin along the winding
    # direction. Compute each line once instead of once per lattice site.
    link = links.select(_direction_axis(links), direction)
    axis = _site_axis(links, direction) - 1
    product = link.select(axis, 0)
    for position in range(1, link.shape[axis]):
        product = matrix_product(product, link.select(axis, position))
    trace = _normalized_complex_trace(product)
    return trace.mean(dim=tuple(range(1, trace.ndim))) if links.ndim == 8 else trace.mean()


# The default stays compatible with archived four-observable experiments.
LEGACY_OBSERVABLES = ("plaquette", "rectangle_1x2", "square_2x2", "polyakov_abs2")
DOWNSAMPLING_OBSERVABLES = (
    "plaquette", "rectangle_1x2", "rectangle_1x3", "square_2x2",
    "rectangle_2x3", "square_3x3", "polyakov_abs2", "polyakov_abs4",
)
TRAINING_OBSERVABLES = tuple(
    name for name in DOWNSAMPLING_OBSERVABLES
    if name not in ("rectangle_1x3", "square_2x2")
)
LOOP_SHAPES = {
    "plaquette": (1, 1), "rectangle_1x2": (1, 2),
    "rectangle_1x3": (1, 3), "square_2x2": (2, 2),
    "rectangle_2x3": (2, 3), "square_3x3": (3, 3),
}


def _wilson_paths(rows: int, columns: int) -> list[tuple[int, ...]]:
    return [
        (mu,) * columns + (nu,) * rows + (mu + 4,) * columns + (nu + 4,) * rows
        for mu in range(4) for nu in range(4)
        if mu < nu or (rows != columns and mu != nu)
    ]


def observable_vector_torch(
    links: torch.Tensor, *, names: Sequence[str] | None = None,
    checkpoint_loops: bool = False,
) -> torch.Tensor:
    """Measure selected orientation-averaged loops and global Polyakov moments."""
    names = LEGACY_OBSERVABLES if names is None else tuple(names)
    polyakov_abs2 = None
    if any(name.startswith("polyakov_") for name in names):
        means = [
            checkpoint(_polyakov_mean, links, mu, use_reentrant=False)
            if checkpoint_loops else _polyakov_mean(links, mu)
            for mu in range(4)
        ]
        polyakov_abs2 = torch.stack(means).abs().square()
    values = []
    for name in names:
        if name.startswith("polyakov_"):
            moment = polyakov_abs2 if name == "polyakov_abs2" else polyakov_abs2.square()
            values.append(moment.mean(dim=0))
            continue
        paths = _wilson_paths(*LOOP_SHAPES[name])
        values.append(torch.stack([
            _loop_mean(links, path, checkpoint_path=checkpoint_loops) for path in paths
        ]).mean(dim=0))
    return torch.stack(values, dim=-1)


def observable_names() -> tuple[str, ...]:
    return LEGACY_OBSERVABLES


def _polyakov_direction_pyquda(gauge, direction: int) -> complex:
    """Use PyQUDA's temporal Polyakov kernel after a coordinate permutation."""
    from pyquda_utils.core import LatticeGauge, LatticeInfo

    array = gauge.lexico()
    old_dirs_for_new = [value for value in range(4) if value != direction] + [direction]
    old_axis_for_direction = {old: 4 - old for old in range(4)}
    coordinate_axes = tuple(
        old_axis_for_direction[old_dirs_for_new[new_direction]]
        for new_direction in (3, 2, 1, 0)
    )
    permuted = array[old_dirs_for_new]
    permuted = permuted.transpose((0, *[axis - 0 for axis in coordinate_axes], 5, 6))
    info = LatticeInfo(gauge.latt_info.global_size)
    rotated = LatticeGauge(info, info.evenodd(np.ascontiguousarray(permuted), True))
    value = rotated.polyakovLoop()
    return complex(float(value[0]), float(value[1]))


def observable_vector_pyquda(gauge, *, names: Sequence[str] | None = None) -> np.ndarray:
    """Measure the same selected observables with native PyQUDA routines."""
    names = LEGACY_OBSERVABLES if names is None else tuple(names)
    volume = int(np.prod(gauge.latt_info.global_size))
    polyakov_abs2 = None
    if any(name.startswith("polyakov_") for name in names):
        means = np.asarray([_polyakov_direction_pyquda(gauge, mu) for mu in range(4)]) / 3.0
        polyakov_abs2 = np.abs(means) ** 2
    values = []
    for name in names:
        if name.startswith("polyakov_"):
            moment = polyakov_abs2 if name == "polyakov_abs2" else polyakov_abs2 ** 2
            values.append(float(moment.mean()))
            continue
        if name == "plaquette":
            values.append(float(gauge.plaquette()[0]))
            continue
        paths = [list(path) for path in _wilson_paths(*LOOP_SHAPES[name])]
        values.append(float((gauge.loopTrace(paths).real / (3.0 * volume)).mean()))
    return np.asarray(values, dtype=np.float64)
