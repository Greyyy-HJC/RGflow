"""Differentiable, gauge-covariant factor-two SU(3) blocking."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


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


def square_sum(links: torch.Tensor, direction: int) -> torch.Tensor:
    """Return six transverse 2x2 square-staple paths for one link direction."""
    result = torch.zeros_like(links.select(_direction_axis(links), direction))
    for transverse in range(4):
        if transverse == direction:
            continue
        result = result + _path_product(
            links,
            (transverse, transverse, direction, direction, transverse + 4, transverse + 4, direction + 4),
        )
        result = result + _path_product(
            links,
            (transverse + 4, transverse + 4, direction, direction, transverse, transverse, direction + 4),
        )
    return result


def hook_sum(links: torch.Tensor, direction: int) -> torch.Tensor:
    """Return the symmetry-completed longitudinal P5/P6 hook paths."""
    result = torch.zeros_like(links.select(_direction_axis(links), direction))
    for transverse in range(4):
        if transverse == direction:
            continue
        for signed_transverse in (transverse, transverse + 4):
            opposite = transverse + 4 if signed_transverse == transverse else transverse
            result = result + _path_product(
                links, (signed_transverse, direction, direction, opposite, direction + 4)
            )
            result = result + _path_product(
                links, (direction + 4, signed_transverse, direction, direction, opposite)
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


class StoutKernel(nn.Module):
    """Global isotropic path weights for differentiable stout smearing."""

    def __init__(
        self,
        initial_weights: tuple[float, float, float, float] = (0.30, 0.65, 0.049, 0.001),
    ):
        super().__init__()
        self.initial_weights = tuple(initial_weights)
        self.logits = nn.Parameter(torch.log(torch.tensor(initial_weights, dtype=torch.float32)))

    def forward(self) -> torch.Tensor:
        return self.logits


class PolynomialStoutKernel(nn.Module):
    """Polynomial covariant smoother from the perturbative perfect-blocking ansatz."""

    def __init__(
        self,
        initial_coefficients: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0),
        hook_coefficients: tuple[float, ...] = (),
        local_coefficients: tuple[float, ...] = (),
    ):
        super().__init__()
        if len(initial_coefficients) != 4:
            raise ValueError("the polynomial smoother requires four L coefficients")
        if len(hook_coefficients) not in (0, 3):
            raise ValueError("hook coefficients must be empty or contain b0, b1, b2")
        if len(local_coefficients) not in (0, 4):
            raise ValueError("local coefficients must be empty or contain four L coefficients")
        self.initial_coefficients = tuple(float(value) for value in initial_coefficients)
        self.initial_hook_coefficients = tuple(float(value) for value in hook_coefficients)
        self.initial_local_coefficients = tuple(float(value) for value in local_coefficients)
        # The perturbative ansatz is a local coordinate chart around the
        # perfect-blocking kernel.  Keep optimization in that chart: a raw
        # Adam step can otherwise change every coefficient by roughly the
        # learning rate, independently of the coefficient's natural scale.
        self.coefficient_radius = tuple(
            max(2.5e-4, 0.75 * abs(value)) for value in self.initial_coefficients
        )
        self.hook_radius = tuple(
            max(2.5e-4, 0.75 * abs(value)) for value in self.initial_hook_coefficients
        )
        self.coefficients = nn.Parameter(torch.tensor(initial_coefficients, dtype=torch.float32))
        self.hook_coefficients = nn.Parameter(
            torch.tensor(hook_coefficients, dtype=torch.float32), requires_grad=bool(hook_coefficients)
        )
        self.local_coefficients = (
            nn.Parameter(torch.tensor(local_coefficients, dtype=torch.float32))
            if local_coefficients else None
        )

    @property
    def uses_hook(self) -> bool:
        return self.hook_coefficients.numel() != 0

    def forward(self, links: torch.Tensor) -> torch.Tensor:
        return polynomial_smear_matrix(links, self.coefficients, self.hook_coefficients, self.local_coefficients)

    @torch.no_grad()
    def project_parameters(self) -> None:
        """Project the fit back into the perturbative trust region."""
        center = self.coefficients.new_tensor(self.initial_coefficients)
        radius = self.coefficients.new_tensor(self.coefficient_radius)
        self.coefficients.clamp_(center - radius, center + radius)
        if self.uses_hook:
            center = self.hook_coefficients.new_tensor(self.initial_hook_coefficients)
            radius = self.hook_coefficients.new_tensor(self.hook_radius)
            self.hook_coefficients.clamp_(center - radius, center + radius)


def _covariant_link_laplacian(links: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
    """Apply the transverse covariant link Laplacian to a link-valued field."""
    axis = _direction_axis(links)
    values = []
    for direction in range(4):
        field_link = field.select(axis, direction)
        value = torch.zeros_like(field_link)
        link_site_axis = lambda transverse: axis + 3 - transverse
        for transverse in range(4):
            if transverse == direction:
                continue
            transverse_link = links.select(axis, transverse)
            forward = (
                transverse_link
                @ torch.roll(field_link, shifts=-1, dims=link_site_axis(transverse))
                @ _dagger(torch.roll(transverse_link, shifts=-1, dims=link_site_axis(direction)))
            )
            backward_transverse = torch.roll(transverse_link, shifts=1, dims=link_site_axis(transverse))
            backward = (
                _dagger(backward_transverse)
                @ torch.roll(field_link, shifts=1, dims=link_site_axis(transverse))
                @ torch.roll(backward_transverse, shifts=-1, dims=link_site_axis(direction))
            )
            value = value + forward + backward - 2.0 * field_link
        values.append(value)
    return torch.stack(values, dim=axis)


def polynomial_smear_matrix(
    links: torch.Tensor, coefficients: torch.Tensor, hook_coefficients: torch.Tensor | None = None,
    local_coefficients: torch.Tensor | None = None,
) -> torch.Tensor:
    """Construct and project ``[1 + a1 L + ... + a4 L^4] U``."""
    field = links
    result = links
    local_result = torch.zeros_like(links) if local_coefficients is not None else None
    for order, coefficient in enumerate(coefficients):
        field = _covariant_link_laplacian(links, field)
        result = result + coefficient * field
        if local_coefficients is not None:
            if order == 0:
                # Average of the six adjacent plaquette traces, minus one.
                # This starts at O(A^2), preserving the perturbative kernel.
                local_feature = _loop_trace(_dagger(links) @ field).real / 6.0
            local_result = local_result + local_coefficients[order] * field
    if local_result is not None:
        result = result + local_feature[..., None, None] * local_result
    if hook_coefficients is not None and hook_coefficients.numel() != 0:
        field = torch.stack(
            [hook_sum(links, direction) for direction in range(4)],
            dim=_direction_axis(links),
        )
        hook_result = torch.zeros_like(field)
        hook_fields = [field]
        for _ in range(2):
            field = _covariant_link_laplacian(links, field)
            hook_fields.append(field)
        for coefficient, hook_field in zip(hook_coefficients, hook_fields):
            hook_result = hook_result + coefficient * hook_field
        result = result + hook_result
    return su3_polar_projection(result)


def polynomial_blocking_basis(links: torch.Tensor, *, hook: bool = False, local: bool = False) -> torch.Tensor:
    """Cache only the fine links used in blocking, before linear combination.

    Shape: ``(terms, 4, 2, tc, zc, yc, xc, 3, 3)`` for an unbatched field.
    Terms are U, L U / 12, ..., L^4 U / 12^4, optionally followed by
    H U / 12, L H U / 12^2, L^2 H U / 12^3. Dimensionless coefficients
    keep a least-squares fit well conditioned. Projection is site-local,
    so discarding unused sites before projection preserves the exact map.
    """
    fields = []
    field = links
    for _ in range(5):
        fields.append(field)
        if len(fields) < 5:
            field = _covariant_link_laplacian(links, field) / 12.0
    if hook:
        field = torch.stack([hook_sum(links, direction) for direction in range(4)]) / 12.0
        for order in range(3):
            fields.append(field)
            if order < 2:
                field = _covariant_link_laplacian(links, field) / 12.0
    basis = []
    for field in fields:
        directions = []
        for direction in range(4):
            first = field[direction, ::2, ::2, ::2, ::2]
            index = [slice(None, None, 2)] * 4
            index[3 - direction] = slice(1, None, 2)
            second = field[direction][tuple(index)]
            directions.append(torch.stack((first, second)))
        basis.append(torch.stack(directions))
    basis = torch.stack(basis)
    if local:
        feature = 2.0 * _loop_trace(_dagger(basis[0]) @ basis[1]).real
        basis = torch.cat((basis, feature[None, ..., None, None] * basis[1:5]))
    return basis


def block_polynomial_basis(basis: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    """Block a cached polynomial basis using its dimensionless coefficients."""
    weights = torch.cat((coefficients.new_ones(1), coefficients))
    matrix = (weights.reshape(-1, *([1] * (basis.ndim - 1))) * basis).sum(0)
    projected = su3_polar_projection(matrix)
    return projected[:, 0] @ projected[:, 1]


def _ordered_transverse_offsets(direction: int) -> tuple[tuple[int, int, int], ...]:
    transverse = [axis for axis in range(4) if axis != direction]
    return tuple(
        (first, second, sign)
        for first in transverse
        for second in transverse
        if first != second
        for sign in (-1, 1)
    )


def _plaquette(links: torch.Tensor, first: int, second: int) -> torch.Tensor:
    return _path_product(links, (first, second, first + 4, second + 4))


def _antihermitian_traceless(matrix: torch.Tensor) -> torch.Tensor:
    antihermitian = 0.5 * (matrix - _dagger(matrix))
    trace = torch.diagonal(antihermitian, dim1=-2, dim2=-1).sum(-1) / 3.0
    identity = torch.eye(3, dtype=matrix.dtype, device=matrix.device)
    return antihermitian - trace[..., None, None] * identity


def _parity_mask(links: torch.Tensor, color: int) -> torch.Tensor:
    """Return one of the sixteen site parities for batched links."""
    axis = _direction_axis(links)
    extents = links.shape[axis + 1 : axis + 5]
    values = torch.zeros(extents, dtype=torch.int64, device=links.device)
    for index, extent in enumerate(extents):
        shape = [1, 1, 1, 1]
        shape[index] = extent
        values = values + ((torch.arange(extent, device=links.device) % 2) << (3 - index)).reshape(shape)
    return (values == color).unsqueeze(0).expand(links.shape[0], *extents)


def _links_at_offset(
    links: torch.Tensor, direction: int, offset: tuple[int, int, int, int], mask: torch.Tensor,
) -> torch.Tensor:
    axis = _direction_axis(links)
    site_axes = [_site_axis(links, current) - 1 for current in range(4)]
    link = links.select(axis, direction)
    shifted = torch.roll(link, shifts=[-value for value in offset], dims=site_axes)
    return shifted[mask]


def masked_staple_plaquette_data(
    links: torch.Tensor, direction: int, mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build twelve safe local loops for one parity-masked link update.

    At an active ``U_mu(x)``, each loop is a ``mu-nu`` plaquette based at
    ``x +/- rho`` with ``rho != nu``, transported back to ``x``.  Its two
    ``mu`` links therefore belong to different parity classes from ``x``.
    A convolution on the active parity sublattice can safely use their traced
    values when updating the active parity class.
    """
    loops = []
    for transverse, offset, sign in _ordered_transverse_offsets(direction):
        at_offset = [0, 0, 0, 0]
        at_offset[offset] = sign
        at_offset_transverse = at_offset.copy()
        at_offset_transverse[transverse] += 1
        at_offset_direction = at_offset.copy()
        at_offset_direction[direction] += 1
        if sign > 0:
            transporter = _links_at_offset(links, offset, (0, 0, 0, 0), mask)
        else:
            transporter_offset = [0, 0, 0, 0]
            transporter_offset[offset] = -1
            transporter = _dagger(
                _links_at_offset(links, offset, tuple(transporter_offset), mask)
            )
        plaquette = (
            _links_at_offset(links, direction, tuple(at_offset), mask)
            @ _links_at_offset(links, transverse, tuple(at_offset_direction), mask)
            @ _dagger(_links_at_offset(links, direction, tuple(at_offset_transverse), mask))
            @ _dagger(_links_at_offset(links, transverse, tuple(at_offset), mask))
        )
        loops.append(transporter @ plaquette @ _dagger(transporter))
    loop_stack = torch.stack(loops, dim=1)
    traces = _loop_trace(loop_stack)
    features = torch.cat((traces.real, traces.imag), dim=1).float()
    return features, _antihermitian_traceless(loop_stack)


class GaugeGeneratorCNN(nn.Module):
    """Small periodic 3x3x3x3 CNN producing twelve local loop weights."""

    def __init__(self, hidden_channels: int = 16, initial_output_scale: float = 0.015):
        super().__init__()
        self.layers = nn.Sequential(
            PeriodicConv4d(24, hidden_channels, kernel_size=3),
            nn.GELU(),
            PeriodicConv4d(hidden_channels, hidden_channels, kernel_size=3),
            nn.GELU(),
            PeriodicConv4d(hidden_channels, 12, kernel_size=3),
        )
        output = self.layers[-1]
        nn.init.normal_(output.spatial.weight, std=initial_output_scale)
        nn.init.zeros_(output.spatial.bias)
        nn.init.zeros_(output.temporal.weight)
        nn.init.zeros_(output.temporal.bias)
        center = output.kernel_size // 2
        with torch.no_grad():
            for channel in range(12):
                output.temporal.weight[channel, channel, center] = 1.0

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class GaugeEquivariantFieldTransform(nn.Module):
    """Parity-masked, gauge-equivariant, exactly reversible SU(3) coupling flow."""

    def __init__(
        self, hidden_channels: int = 16, step_scale: float = 0.10,
        initial_output_scale: float = 0.015, flow_steps: int = 1,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.step_scale = step_scale
        self.initial_output_scale = initial_output_scale
        self.flow_steps = flow_steps
        self.generators = nn.ModuleList(
            [
                nn.ModuleList(
                    [GaugeGeneratorCNN(hidden_channels, initial_output_scale) for _ in range(4)]
                )
                for _ in range(flow_steps)
            ]
        )

    def _generator(
        self, links: torch.Tensor, step: int, direction: int, mask: torch.Tensor,
    ) -> torch.Tensor:
        features, transverse_loops = masked_staple_plaquette_data(links, direction, mask)
        axis = _direction_axis(links)
        sublattice_shape = tuple(size // 2 for size in links.shape[axis + 1 : axis + 5])
        batch_size = links.shape[0]
        sublattice_features = features.reshape(batch_size, *sublattice_shape, 24).permute(0, 5, 1, 2, 3, 4)
        coefficients = self.generators[step][direction](sublattice_features)
        coefficients = coefficients.permute(0, 2, 3, 4, 5, 1).reshape(-1, 12)
        coefficients = self.step_scale * torch.tanh(coefficients)
        return (coefficients[..., None, None] * transverse_loops).sum(dim=1)

    def _apply_direction(
        self, links: torch.Tensor, step: int, direction: int, color: int, sign: float,
    ) -> torch.Tensor:
        axis = _direction_axis(links)
        mask = _parity_mask(links, color)
        generator = self._generator(links, step, direction, mask)
        link = links.select(axis, direction)
        updated_link = link.clone()
        updated_link[mask] = torch.matrix_exp(sign * generator) @ link[mask]
        values = [
            updated_link if current == direction else links.select(axis, current)
            for current in range(4)
        ]
        return torch.stack(values, dim=axis)

    def _apply_direction_sweep(
        self, links: torch.Tensor, step: int, direction: int, sign: float,
    ) -> torch.Tensor:
        for color in range(16):
            links = self._apply_direction(links, step, direction, color, sign)
        return links

    def forward(self, links: torch.Tensor) -> torch.Tensor:
        batched = links.ndim == 8
        transformed = links if batched else links.unsqueeze(0)
        for step in range(self.flow_steps):
            for direction in range(4):
                if self.training and torch.is_grad_enabled():
                    transformed = checkpoint(
                        lambda current, step=step, direction=direction: self._apply_direction_sweep(current, step, direction, 1.0),
                        transformed,
                        use_reentrant=False,
                    )
                else:
                    transformed = self._apply_direction_sweep(transformed, step, direction, 1.0)
        return transformed if batched else transformed.squeeze(0)

    def inverse(self, links: torch.Tensor) -> torch.Tensor:
        batched = links.ndim == 8
        transformed = links if batched else links.unsqueeze(0)
        for step in reversed(range(self.flow_steps)):
            for direction in reversed(range(4)):
                for color in reversed(range(16)):
                    transformed = self._apply_direction(transformed, step, direction, color, -1.0)
        return transformed if batched else transformed.squeeze(0)


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
    path_axis = 2 if batched else 1
    if weights.shape[path_axis] == 3:
        padding_shape = list(weights.shape)
        padding_shape[path_axis] = 1
        weights = torch.cat((weights, torch.zeros(padding_shape, dtype=weights.dtype, device=weights.device)), dim=path_axis)
    rectangle_weights = weights[:, :, 2] if batched else weights[:, 2]
    has_rectangle = bool(torch.max(torch.abs(rectangle_weights)).detach().cpu() > 1.0e-10)
    square_weights = weights[:, :, 3] if batched else weights[:, 3]
    has_square = bool(torch.max(torch.abs(square_weights)).detach().cpu() > 1.0e-10)
    for direction in range(4):
        link = links.select(axis, direction)
        if batched:
            direction_weights = weights[:, direction]
            straight = direction_weights[:, 0][..., None, None]
            staple = direction_weights[:, 1][..., None, None]
            rectangle = direction_weights[:, 2][..., None, None]
            square = direction_weights[:, 3][..., None, None]
        else:
            direction_weights = weights[direction]
            straight = direction_weights[0]
            staple = direction_weights[1]
            rectangle = direction_weights[2]
            square = direction_weights[3]
        value = straight * link + (staple / 6.0) * staple_sum(links, direction)
        if has_rectangle:
            value = value + (rectangle / 6.0) * rectangle_sum(links, direction)
        if has_square:
            value = value + (square / 6.0) * square_sum(links, direction)
        mixed.append(value)
    return su3_polar_projection(torch.stack(mixed, dim=axis))


def smear_links(links: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Apply shared scalar path weights and SU(3) projection.

    Two-logit tensors retain the original straight/staple baseline API;
    three- and four-logit tensors additionally enable rectangle and square
    path channels.
    """
    if logits.numel() == 2:
        values = torch.softmax(logits.reshape(2), dim=0)
        weights = torch.zeros((4, 4), dtype=values.dtype, device=values.device)
        weights[:, :2] = values
    elif logits.numel() == 3:
        values = torch.softmax(logits.reshape(3), dim=0)
        weights = torch.zeros((4, 4), dtype=values.dtype, device=values.device)
        weights[:, :3] = values
    elif logits.numel() == 4:
        values = torch.softmax(logits.reshape(4), dim=0)
        weights = values.reshape(1, 4).expand(4, 4)
    else:
        raise ValueError("scalar logits must contain two, three, or four values")
    if links.ndim == 8:
        weights = weights.reshape(1, 4, 4, 1, 1, 1, 1).expand(links.shape[0], 4, 4, *links.shape[2:6])
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
    kernel: torch.Tensor | LinkCoefficientCNN | StoutKernel | PolynomialStoutKernel,
    features: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply path mixing, SU(3) projection, and factor-two blocking."""
    if isinstance(kernel, PolynomialStoutKernel):
        smeared = kernel(links)
    elif isinstance(kernel, LinkCoefficientCNN):
        smeared = smear_links_cnn(links, kernel, features)
    elif isinstance(kernel, StoutKernel):
        smeared = smear_links(links, kernel())
    else:
        smeared = smear_links(links, kernel)
    return block_links(smeared)


def transform_and_block(
    links: torch.Tensor, transform: GaugeEquivariantFieldTransform,
) -> torch.Tensor:
    """Apply the reversible field transform before lossy factor-two blocking."""
    return block_links(transform(links))


def su3_errors(links: torch.Tensor) -> tuple[float, float]:
    """Return maximum unitarity and determinant errors for diagnostics."""
    identity = torch.eye(3, dtype=links.dtype, device=links.device)
    unitary_error = torch.max(
        torch.abs(links @ links.conj().transpose(-2, -1) - identity)
    )
    determinant_error = torch.max(torch.abs(torch.linalg.det(links) - 1.0))
    return float(unitary_error.detach().cpu()), float(determinant_error.detach().cpu())
