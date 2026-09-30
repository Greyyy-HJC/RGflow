"""Two-stage SU(3) inverse blocking with invertible SU(2) fibre couplings.

The first flow acts on the 60 free links per coarse cell, conditioned on the
four retained two-link products. The second transports the smeared measure
to the fine measure. Both maps have analytic Jacobians relative to Haar.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .downsampling import _dagger
from .linalg import matrix_product as _matrix_product


def quaternion_transport(a, b, scale, raw_shape, *, inverse=False, compute_logdet=True):
    """Möbius angle shift plus a monotone cubic deformation on S^3.

    g(u)=u+alpha*u*(1-u^2), |alpha|<.45, has derivative at least .1.
    Its Haar Jacobian is g'(u)*sqrt((1-g(u)^2)/(1-u^2)). The factored
    square-root expression below stays regular at both quaternion poles.
    """
    alpha = .45 * torch.tanh(raw_shape)
    logdet = torch.zeros_like(scale) if compute_logdet else None
    if inverse:
        target, value = a.real, a.real
        for _ in range(8):
            value = value - (value + alpha*value*(1-value.square()) - target) / (1+alpha*(1-3*value.square()))
        factor = torch.sqrt(1-2*alpha*value.square()-alpha.square()*value.square()*(1-value.square()))
        a, b = torch.complex(value, a.imag/factor), b/factor
        if compute_logdet:
            logdet = -torch.log(1+alpha*(1-3*value.square())) - torch.log(factor)
        scale = -scale
    denominator = (1+a.real) + torch.exp(2*scale)*(1-a.real)
    factor = 2*torch.exp(scale)/denominator
    real = ((1+a.real)-torch.exp(2*scale)*(1-a.real))/denominator
    a, b = torch.complex(real, factor*a.imag), factor*b
    if compute_logdet:
        logdet = logdet + 3*(scale+math.log(2.)-torch.log(denominator))
    if not inverse:
        value = a.real
        factor = torch.sqrt(1-2*alpha*value.square()-alpha.square()*value.square()*(1-value.square()))
        a, b = torch.complex(value+alpha*value*(1-value.square()), factor*a.imag), factor*b
        if compute_logdet:
            logdet = logdet + torch.log(1+alpha*(1-3*value.square())) + torch.log(factor)
    return a, b, logdet


def haar_links(shape, *, device="cpu", dtype=torch.complex64, generator=None):
    """Draw independent normalized-Haar SU(3) matrices."""
    noise = torch.randn(*shape, 3, 2, dtype=dtype, device=device, generator=generator)
    first = noise[..., 0]
    first = first / torch.linalg.vector_norm(first, dim=-1, keepdim=True)
    second = noise[..., 1]
    second = second - first * (first.conj() * second).sum(-1, keepdim=True)
    second = second / torch.linalg.vector_norm(second, dim=-1, keepdim=True)
    third = torch.linalg.cross(first, second, dim=-1).conj()
    return torch.stack((first, second, third), -1)


def constrained_haar_lift(coarse, *, generator=None):
    """Haar details with block_links(lift) == coarse, with no fine input.

    At each coarse origin the first link is free Haar and the second is
    U_first^dagger C. All other links are independent Haar. The product
    substitution has unit Haar Jacobian and leaves 60 SU(3) links per cell.
    """
    batched = coarse.ndim == 8
    coarse = coarse if batched else coarse.unsqueeze(0)
    extents = [2 * size for size in coarse.shape[2:6]]
    fine = haar_links((len(coarse), 4, *extents), device=coarse.device,
                      dtype=coarse.dtype, generator=generator)
    for direction in range(4):
        second = [slice(None, None, 2)] * 4
        second[3 - direction] = slice(1, None, 2)
        first = fine[:, direction, ::2, ::2, ::2, ::2]
        fine[(slice(None), direction, *second)] = _dagger(first) @ coarse[:, direction]
    return fine if batched else fine.squeeze(0)


def coupling_mask(extents, direction, parity, *, constrained, device):
    coordinates = torch.meshgrid(
        *[torch.arange(size, device=device) for size in extents], indexing="ij"
    )
    mask = sum(coordinates) % 2 == parity
    if constrained:
        # Freeze both members of retained bonds. The four first links remain
        # independent Haar gauge degrees of freedom; the other 56 are flowed.
        retained = torch.ones(extents, dtype=torch.bool, device=device)
        for axis, coordinate in enumerate(coordinates):
            if axis != 3 - direction:
                retained = retained & (coordinate % 2 == 0)
        mask = mask & ~retained
    return mask


def transverse_staples(links, direction, mask, distance):
    """Odd-length transverse staples, safe on a two-colour checkerboard.

    The only same-direction link is three transverse sites away, hence is
    inactive. Length-two transverse staples would invalidate this Jacobian.
    """
    link = links[:, direction]
    value = torch.zeros_like(link[:, mask])
    for transverse in range(4):
        if transverse != direction:
            nu = links[:, transverse]
            nu_axis, mu_axis = 4 - transverse, 4 - direction
            transporter = nu
            for offset in range(1, distance):
                transporter = _matrix_product(transporter, torch.roll(nu, -offset, nu_axis))
            value = value + _matrix_product(
                _matrix_product(transporter[:, mask], torch.roll(link, -distance, nu_axis)[:, mask]),
                _dagger(torch.roll(transporter, -1, mu_axis)[:, mask]),
            )
            backward = torch.roll(transporter, distance, nu_axis)
            value = value + _matrix_product(
                _matrix_product(_dagger(backward[:, mask]), torch.roll(link, distance, nu_axis)[:, mask]),
                torch.roll(backward, -1, mu_axis)[:, mask],
            )
    return value / 6.


class SU3ConditionalFlow(nn.Module):
    """Checkerboard SU(2) radial couplings with nonzero Haar log-Jacobian.

    For each embedded SU(2), project U C^dagger onto its quaternion part
    r = |r| q, where C is the staple sum. Left SU(2) multiplication rotates
    q on S^3 and leaves |r| and the remaining fibre coordinates unchanged.
    The map tan(theta'/2) = exp(s) tan(theta/2) is bijective almost everywhere.
    Follow it with q0 -> q0+alpha*q0*(1-q0^2), |alpha|<.45, rescaling
    the other quaternion components to stay on S^3. Both scale and shape
    depend only on rotation-invariant features, so the Jacobian is triangular.

    SU(2) subgroups select a colour basis: this prototype is not exactly
    gauge equivariant. Training/evaluation use gauge-invariant operators.
    """

    def __init__(self, sweeps=4, hidden=8, constrained=False, initial_scale=-0.35,
                 long_staples=True):
        super().__init__()
        self.sweeps = sweeps
        self.hidden = hidden
        self.constrained = constrained
        self.long_staples = long_staples
        self.long_weights = nn.Parameter(torch.full((sweeps,), .2)) if long_staples else None
        self.scales = nn.ModuleList()
        for _ in range(sweeps):
            network = nn.Sequential(nn.Linear(9, hidden), nn.Tanh(), nn.Linear(hidden, 6))
            with torch.no_grad():
                network[0].weight[:, 2:].zero_()
            nn.init.zeros_(network[-1].weight)
            nn.init.zeros_(network[-1].bias)
            with torch.no_grad():
                network[-1].bias[:3].fill_(initial_scale)
            self.scales.append(network)

    def _direction(self, links, sweep, direction, parity, inverse, compute_logdet=True):
        mask = coupling_mask(links.shape[2:6], direction, parity,
                             constrained=self.constrained, device=links.device)
        link = links[:, direction][:, mask]
        nearest = transverse_staples(links, direction, mask, 1)
        long = torch.zeros_like(nearest)
        staples = nearest
        if self.long_staples:
            long = transverse_staples(links, direction, mask, 3)
            staples = staples + self.long_weights[sweep] * long
        staple_size = staples.abs().square().sum((-2, -1)) / 3.0
        long_size = long.abs().square().sum((-2, -1)) / 3.0
        overlap = (nearest * long.conj()).sum((-2, -1)) / 3.0
        coordinates = torch.meshgrid(*[torch.arange(n, device=links.device) % 2
                                       for n in links.shape[2:6]], indexing="ij")
        parities = torch.stack(coordinates, -1)[mask].to(links.real.dtype)
        parities = parities.unsqueeze(0).expand(links.shape[0], -1, -1)
        logdet = torch.zeros(links.shape[0], dtype=links.real.dtype, device=links.device) if compute_logdet else None
        pairs = [(0, 1), (0, 2), (1, 2)]
        if inverse:
            pairs = list(reversed(pairs))
        for first, second in pairs:
            matrix = _matrix_product(link, _dagger(staples))
            a = (matrix[..., first, first] + matrix[..., second, second].conj()) / 2.0
            b = (matrix[..., first, second] - matrix[..., second, first].conj()) / 2.0
            norm = torch.sqrt(a.abs().square() + b.abs().square()).clamp_min(1.e-12)
            a, b = a / norm, b / norm
            features = torch.cat((torch.stack((norm, staple_size, long_size,
                                              overlap.real, overlap.imag), -1), parities), -1)
            index = [(0, 1), (0, 2), (1, 2)].index((first, second))
            parameters = self.scales[sweep](features)
            scale = parameters[..., index].clamp(-2., 2.)
            new_a, new_b, change = quaternion_transport(
                a, b, scale, parameters[..., index+3], inverse=inverse, compute_logdet=compute_logdet,
            )
            # Q_new Q_old^dagger, acting on the two selected rows of U.
            rotate_a = new_a * a.conj() + new_b * b.conj()
            rotate_b = -new_a * b + new_b * a
            row_first, row_second = link[..., first, :], link[..., second, :]
            rows = [link[..., row, :] for row in range(3)]
            rows[first] = rotate_a[..., None] * row_first + rotate_b[..., None] * row_second
            rows[second] = -rotate_b.conj()[..., None] * row_first + rotate_a.conj()[..., None] * row_second
            link = torch.stack(rows, -2)
            if compute_logdet:
                logdet = logdet + change.sum(-1)
        updated = links[:, direction].clone()
        updated[:, mask] = link
        result = torch.stack([updated if mu == direction else links[:, mu] for mu in range(4)], 1)
        return result, logdet

    def forward(self, links, *, inverse=False, compute_logdet=True):
        batched = links.ndim == 8
        links = links if batched else links.unsqueeze(0)
        logdet = torch.zeros(len(links), dtype=links.real.dtype, device=links.device) if compute_logdet else None
        operations = [(sweep, direction, parity) for sweep in range(self.sweeps)
                      for direction in range(4) for parity in range(2)]
        if inverse:
            operations.reverse()
        for sweep, direction, parity in operations:
            if self.training and torch.is_grad_enabled():
                links, change = checkpoint(
                    lambda value, s=sweep, d=direction, p=parity:
                    self._direction(value, s, d, p, inverse, compute_logdet), links, use_reentrant=False
                )
            else:
                links, change = self._direction(links, sweep, direction, parity, inverse, compute_logdet)
            if compute_logdet:
                logdet = logdet + change
        return (links, logdet) if batched else (links.squeeze(0), logdet.squeeze(0) if compute_logdet else None)

    def inverse(self, links, *, compute_logdet=True):
        return self(links, inverse=True, compute_logdet=compute_logdet)

    def sample(self, coarse, *, generator=None, compute_logdet=True):
        """Conditional inverse-blocking sample; normalized-Haar base log p=0."""
        base = constrained_haar_lift(coarse, generator=generator)
        smeared, logdet = self(base, compute_logdet=compute_logdet)
        return smeared, -logdet if compute_logdet else None

    def log_prob(self, smeared):
        """Conditional density on the fixed blocking fibre (Haar measure)."""
        _, inverse_logdet = self.inverse(smeared)
        return inverse_logdet
