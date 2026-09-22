"""Differentiable, gauge-covariant factor-two SU(3) blocking."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _direction_axis(links: torch.Tensor) -> int:
    if links.ndim < 7 or links.shape[-2:] != (3, 3):
        raise ValueError("links must have shape (..., 4, L0, L1, L2, L3, 3, 3)")
    axis = links.ndim - 7
    if links.shape[axis] != 4:
        raise ValueError("the link direction axis must have length four")
    if any(size % 2 for size in links.shape[axis + 1 : axis + 5]):
        raise ValueError("all lattice extents must be even for factor-two blocking")
    return axis


def _site_axis(links: torch.Tensor, direction: int) -> int:
    # PyQUDA/NERSC storage is (t, z, y, x), while direction numbers are
    # (x, y, z, t).
    return _direction_axis(links) + 1 + (3 - direction)


def _dagger(matrix: torch.Tensor) -> torch.Tensor:
    return matrix.conj().transpose(-2, -1)


def _path_product(links: torch.Tensor, path: tuple[int, ...]) -> torch.Tensor:
    """Multiply links along a path encoded as 0..3/+ and 4..7/-."""
    axis = _direction_axis(links)
    site_axes = [axis + 1 + (3 - direction) for direction in range(4)]
    result = None
    offset = [0, 0, 0, 0]
    for step in path:
        direction = int(step) % 4
        positive = int(step) < 4
        link = links.select(axis, direction)
        if positive:
            shifts = [-(offset[d]) for d in range(4)]
            factor = torch.roll(link, shifts=shifts, dims=[site_axes[d] - 1 for d in range(4)])
            offset[direction] += 1
        else:
            start = offset.copy()
            start[direction] -= 1
            shifts = [-(start[d]) for d in range(4)]
            factor = _dagger(
                torch.roll(link, shifts=shifts, dims=[site_axes[d] - 1 for d in range(4)])
            )
            offset[direction] -= 1
        result = factor if result is None else result @ factor
    if result is None:
        raise ValueError("path must not be empty")
    return result


def staple_sum(links: torch.Tensor, direction: int) -> torch.Tensor:
    """Return the six forward/backward staples for one link direction."""
    axis = _direction_axis(links)
    link = links.select(axis, direction)
    result = torch.zeros_like(link)
    link_site_axis = lambda transverse: axis + 3 - transverse
    for transverse in range(4):
        if transverse == direction:
            continue
        transverse_link = links.select(axis, transverse)
        forward = (
            transverse_link
            @ torch.roll(link, shifts=-1, dims=link_site_axis(transverse))
            @ _dagger(torch.roll(transverse_link, shifts=-1, dims=link_site_axis(direction)))
        )
        backward = (
            _dagger(torch.roll(transverse_link, shifts=1, dims=link_site_axis(transverse)))
            @ torch.roll(link, shifts=1, dims=link_site_axis(transverse))
            @ torch.roll(
                torch.roll(transverse_link, shifts=1, dims=link_site_axis(transverse)),
                shifts=-1,
                dims=link_site_axis(direction),
            )
        )
        result = result + forward + backward
    return result


def rectangle_sum(links: torch.Tensor, direction: int) -> torch.Tensor:
    """Return six transverse 1x2 rectangle paths for a link direction."""
    result = torch.zeros_like(links.select(_direction_axis(links), direction))
    for transverse in range(4):
        if transverse == direction:
            continue
        result = result + _path_product(
            links, (transverse, transverse, direction, transverse + 4, transverse + 4)
        )
        result = result + _path_product(
            links, (transverse + 4, transverse + 4, direction, transverse, transverse)
        )
    return result


def _loop_trace(matrix: torch.Tensor) -> torch.Tensor:
    return torch.diagonal(matrix, dim1=-2, dim2=-1).sum(-1) / 3.0


def gauge_invariant_features(links: torch.Tensor) -> torch.Tensor:
    """Build local scalar channels for the coefficient CNN.

    For each link direction and transverse direction, the real and imaginary
    traces of the forward/backward plaquette-like loops are returned.  The
    output is ``(channels, t, z, y, x)`` for one configuration and
    ``(batch, channels, t, z, y, x)`` for a batch.
    """
    axis = _direction_axis(links)
    batched = links.ndim == 8
    input_links = links if batched else links.unsqueeze(0)
    axis = _direction_axis(input_links)
    values = []
    for direction in range(4):
        link = input_links.select(axis, direction)
        base = _dagger(link)
        for transverse in range(4):
            if transverse == direction:
                continue
            transverse_link = input_links.select(axis, transverse)
            site_axis = axis + 3 - transverse
            direction_axis = axis + 3 - direction
            forward = (
                transverse_link
                @ torch.roll(link, shifts=-1, dims=site_axis)
                @ _dagger(torch.roll(transverse_link, shifts=-1, dims=direction_axis))
            )
            shifted_transverse = torch.roll(transverse_link, shifts=1, dims=site_axis)
            backward = (
                _dagger(shifted_transverse)
                @ torch.roll(link, shifts=1, dims=site_axis)
                @ torch.roll(shifted_transverse, shifts=-1, dims=direction_axis)
            )
            forward_trace = _loop_trace(base @ forward)
            backward_trace = _loop_trace(base @ backward)
            values.extend((forward_trace.real, backward_trace.real, forward_trace.imag, backward_trace.imag))
    return torch.stack(values, dim=1).float() if batched else torch.stack(values, dim=1).squeeze(0).float()


class PeriodicConv4d(nn.Module):
    """Factorized periodic 4D convolution using spatial Conv3d + temporal Conv1d."""

    def __init__(self, channels_in: int, channels_out: int, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.spatial = nn.Conv3d(channels_in, channels_out, kernel_size=(kernel_size, kernel_size, kernel_size), padding=0)
        self.temporal = nn.Conv1d(channels_out, channels_out, kernel_size=kernel_size, padding=0)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, channels, time, z, y, x = values.shape
        spatial = values.permute(0, 2, 1, 3, 4, 5).reshape(batch * time, channels, z, y, x)
        spatial = self.spatial(F.pad(spatial, (self.kernel_size // 2,) * 6, mode="circular"))
        spatial = spatial.reshape(batch, time, -1, z, y, x).permute(0, 2, 1, 3, 4, 5)
        temporal = spatial.permute(0, 3, 4, 5, 1, 2).reshape(batch * z * y * x, -1, time)
        temporal = self.temporal(F.pad(temporal, (self.kernel_size // 2,) * 2, mode="circular"))
        return temporal.reshape(batch, z, y, x, -1, time).permute(0, 4, 5, 1, 2, 3)


class LinkCoefficientCNN(nn.Module):
    """Translation-equivariant CNN producing three path weights per link."""

    def __init__(
        self,
        hidden_channels: int = 16,
        initial_weights: tuple[float, float, float] = (0.30, 0.65, 0.05),
        local_logit_scale: float = 0.1,
    ):
        super().__init__()
        self.base_logits = nn.Parameter(torch.log(torch.tensor(initial_weights)))
        self.local_logit_scale = local_logit_scale
        self.layers = nn.Sequential(
            PeriodicConv4d(48, hidden_channels),
            nn.GELU(),
            PeriodicConv4d(hidden_channels, hidden_channels),
            nn.GELU(),
            PeriodicConv4d(hidden_channels, 12, kernel_size=1),
        )
        # Start from a spatially constant, physically useful APE kernel while
        # retaining a gradient path into the CNN.  Zeroing both factorized
        # convolutions would make their gradients mutually zero and reduce the
        # model permanently to its output bias.
        nn.init.zeros_(self.layers[-1].spatial.weight)
        nn.init.zeros_(self.layers[-1].spatial.bias)
        nn.init.zeros_(self.layers[-1].temporal.weight)
        center = self.layers[-1].kernel_size // 2
        with torch.no_grad():
            for channel in range(12):
                self.layers[-1].temporal.weight[channel, channel, center] = 1.0
        nn.init.constant_(self.layers[-1].temporal.bias, 0.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.local_logit_scale * self.layers(features)
        batch, _, time, z, y, x = logits.shape
        local_logits = logits.reshape(batch, 4, 3, time, z, y, x)
        return local_logits + self.base_logits.reshape(1, 1, 3, 1, 1, 1, 1)


def weights_from_path_logits(logits: torch.Tensor) -> torch.Tensor:
    """Softmax three path weights along the channel dimension."""
    return torch.softmax(logits, dim=2)


def su3_polar_projection(matrix: torch.Tensor) -> torch.Tensor:
    """Project arbitrary complex 3x3 matrices to the nearest SU(3) matrix."""
    # A fixed, sub-ulp-for-links diagonal perturbation removes the undefined
    # singular-vector basis at exactly unitary inputs (notably the identity),
    # while leaving the polar factor unchanged to the requested precision.
    dtype = matrix.real.dtype
    perturbation = torch.diag(
        torch.arange(1, 4, device=matrix.device, dtype=dtype)
    ).to(matrix.dtype)
    unitary_left, _, unitary_right = torch.linalg.svd(
        matrix + 1.0e-6 * perturbation, full_matrices=False
    )
    unitary = unitary_left @ unitary_right
    determinant = torch.linalg.det(unitary)
    phase = torch.exp(-1j * torch.angle(determinant) / 3.0)
    return unitary * phase[..., None, None]


def weights_from_logits(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return straight-link and total-staple weights."""
    if logits.numel() != 2:
        raise ValueError("logits must contain exactly two values")
    weights = torch.softmax(logits.reshape(2), dim=0)
    return weights[0], weights[1]


def _mix_projected_links(links: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    axis = _direction_axis(links)
    mixed = []
    batched = links.ndim == 8
    rectangle_weights = weights[:, :, 2] if batched else weights[:, 2]
    has_rectangle = bool(torch.max(torch.abs(rectangle_weights)).detach().cpu() > 1.0e-10)
    for direction in range(4):
        link = links.select(axis, direction)
        if batched:
            direction_weights = weights[:, direction]
            straight = direction_weights[:, 0][..., None, None]
            staple = direction_weights[:, 1][..., None, None]
            rectangle = direction_weights[:, 2][..., None, None]
        else:
            direction_weights = weights[direction]
            straight = direction_weights[0]
            staple = direction_weights[1]
            rectangle = direction_weights[2]
        value = straight * link + (staple / 6.0) * staple_sum(links, direction)
        if has_rectangle:
            value = value + (rectangle / 6.0) * rectangle_sum(links, direction)
        mixed.append(value)
    return su3_polar_projection(torch.stack(mixed, dim=axis))


def smear_links(links: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Apply shared scalar path weights and SU(3) projection.

    Two-logit tensors retain the original straight/staple baseline API;
    three-logit tensors additionally enable the rectangle path channel.
    """
    if logits.numel() == 2:
        values = torch.softmax(logits.reshape(2), dim=0)
        weights = torch.zeros((4, 3), dtype=values.dtype, device=values.device)
        weights[:, :2] = values
    elif logits.numel() == 3:
        values = torch.softmax(logits.reshape(3), dim=0)
        weights = values.reshape(1, 3).expand(4, 3)
    else:
        raise ValueError("scalar logits must contain two or three values")
    if links.ndim == 8:
        weights = weights.reshape(1, 4, 3, 1, 1, 1, 1).expand(links.shape[0], 4, 3, *links.shape[2:6])
    return _mix_projected_links(links, weights)


def smear_links_cnn(
    links: torch.Tensor,
    model: LinkCoefficientCNN,
    features: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply CNN-generated per-link path weights and SU(3) projection."""
    batched = links.ndim == 8
    input_links = links if batched else links.unsqueeze(0)
    input_features = features
    if input_features is None:
        input_features = gauge_invariant_features(input_links)
    logits = model(input_features)
    weights = weights_from_path_logits(logits)
    projected = _mix_projected_links(input_links, weights)
    return projected if batched else projected.squeeze(0)


def block_links(smeared: torch.Tensor) -> torch.Tensor:
    """Keep the all-even sublattice and multiply two links along each direction."""
    axis = _direction_axis(smeared)
    blocked = []
    for direction in range(4):
        link = smeared.select(axis, direction)
        shifted = torch.roll(link, shifts=-1, dims=_site_axis(smeared, direction) - 1)
        even_index = [slice(None)] * link.ndim
        even_index[axis - 1 + 1 : axis - 1 + 5] = [slice(None, None, 2)] * 4
        first = link[tuple(even_index)]
        second = shifted[tuple(even_index)]
        blocked.append(first @ second)
    return torch.stack(blocked, dim=axis)


def smear_and_block(
    links: torch.Tensor,
    kernel: torch.Tensor | LinkCoefficientCNN,
    features: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply path mixing, SU(3) projection, and factor-two blocking."""
    smeared = smear_links_cnn(links, kernel, features) if isinstance(kernel, nn.Module) else smear_links(links, kernel)
    return block_links(smeared)


def su3_errors(links: torch.Tensor) -> tuple[float, float]:
    """Return maximum unitarity and determinant errors for diagnostics."""
    identity = torch.eye(3, dtype=links.dtype, device=links.device)
    unitary_error = torch.max(
        torch.abs(links @ links.conj().transpose(-2, -1) - identity)
    )
    determinant_error = torch.max(torch.abs(torch.linalg.det(links) - 1.0))
    return float(unitary_error.detach().cpu()), float(determinant_error.detach().cpu())
