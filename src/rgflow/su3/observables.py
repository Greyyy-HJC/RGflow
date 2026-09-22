"""Wilson-loop and Polyakov-loop observables for SU(3) link fields."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch


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
        result = factor if result is None else result @ factor
    if result is None:
        raise ValueError("path must not be empty")
    return result


def _normalized_trace(matrix: torch.Tensor) -> torch.Tensor:
    return torch.diagonal(matrix, dim1=-2, dim2=-1).sum(-1).real / 3.0


def _normalized_complex_trace(matrix: torch.Tensor) -> torch.Tensor:
    return torch.diagonal(matrix, dim1=-2, dim2=-1).sum(-1) / 3.0


def _loop_mean(links: torch.Tensor, path: Sequence[int]) -> torch.Tensor:
    values = _normalized_trace(path_product(links, path))
    if links.ndim == 8:
        return values.mean(dim=tuple(range(1, values.ndim)))
    return values.mean()


def observable_vector_torch(links: torch.Tensor) -> torch.Tensor:
    """Return plaquette, 1x2 rectangle, 2x2 square, and Polyakov second moment."""
    plaquettes = []
    rectangles = []
    squares = []
    polyakov = []
    for mu in range(4):
        for nu in range(mu + 1, 4):
            plaquettes.append(_loop_mean(links, (mu, nu, mu + 4, nu + 4)))
            squares.append(
                _loop_mean(links, (mu, mu, nu, nu, mu + 4, mu + 4, nu + 4, nu + 4))
            )
        polyakov_loop = _normalized_complex_trace(
            path_product(links, (mu,) * links.shape[_site_axis(links, mu)])
        )
        if links.ndim == 8:
            polyakov_mean = polyakov_loop.mean(dim=tuple(range(1, polyakov_loop.ndim)))
        else:
            polyakov_mean = polyakov_loop.mean()
        polyakov.append(torch.abs(polyakov_mean) ** 2)
        for nu in range(4):
            if mu != nu:
                rectangles.append(
                    _loop_mean(links, (mu, mu, nu, mu + 4, mu + 4, nu + 4))
                )
    if links.ndim == 8:
        return torch.stack(
            (
                torch.stack(plaquettes).mean(dim=0),
                torch.stack(rectangles).mean(dim=0),
                torch.stack(squares).mean(dim=0),
                torch.stack(polyakov).mean(dim=0),
            ),
            dim=1,
        )
    return torch.stack(
        (
            torch.stack(plaquettes).mean(),
            torch.stack(rectangles).mean(),
            torch.stack(squares).mean(),
            torch.stack(polyakov).mean(),
        )
    )


def observable_names() -> tuple[str, ...]:
    return ("plaquette", "rectangle_1x2", "square_2x2", "polyakov_abs2")


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


def observable_vector_pyquda(gauge) -> np.ndarray:
    """Measure the four observables with PyQUDA's native gauge routines."""
    lattice = tuple(int(value) for value in gauge.latt_info.global_size)
    volume = int(np.prod(lattice))
    plaquette = float(gauge.plaquette()[0])
    rectangles = []
    squares = []
    for mu in range(4):
        for nu in range(mu + 1, 4):
            squares.append([mu, mu, nu, nu, mu + 4, mu + 4, nu + 4, nu + 4])
        for nu in range(4):
            if mu != nu:
                rectangles.append([mu, mu, nu, mu + 4, mu + 4, nu + 4])
    rectangle_values = gauge.loopTrace(rectangles).real / (3.0 * volume)
    square_values = gauge.loopTrace(squares).real / (3.0 * volume)
    polyakov_values = [_polyakov_direction_pyquda(gauge, direction) for direction in range(4)]
    polyakov_abs2 = float(np.mean(np.abs(np.asarray(polyakov_values) / 3.0) ** 2))
    return np.asarray(
        [plaquette, float(rectangle_values.mean()), float(square_values.mean()), polyakov_abs2],
        dtype=np.float64,
    )
