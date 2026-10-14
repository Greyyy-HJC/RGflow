"""Two-stage SU(3) inverse blocking with invertible SU(2) fibre couplings.

The first flow acts on the 60 free links per coarse cell, conditioned on the
four retained two-link products. The second transports the smeared measure
to the fine measure. Both maps have analytic Jacobians relative to Haar.
"""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .downsampling import _dagger
from .linalg import matrix_product as _matrix_product

EQUIVARIANT_COUPLING = "covariant_frame_mobius_monotone_cubic_v1"
STABLE_COUPLING = "covariant_frame_mobius_monotone_cubic_fp64_v2"
LEGACY_COUPLING = "mobius_monotone_cubic"


def _cpu_staple_frame(staples):
    """Inactive-context colour frame and a gauge-invariant degeneracy gate.

    Ordered eigenvectors of C C† transform as G F D at a nondegenerate
    spectrum. D contains arbitrary phases; the fibre map cancels them.
    Repeated spectra use the identity map, without differentiating undefined
    eigenvectors. Relative gaps smoothly suppress maps near this boundary.
    """
    # CUDA eigh allocates an impractically large solver workspace for the
    # L24 batch of tiny matrices. CPU copies preserve the autograd graph.
    output_device = staples.device
    context = staples.cpu() if staples.is_cuda else staples
    hermitian = context @ _dagger(context)
    with torch.no_grad():
        spectrum = torch.linalg.eigvalsh(hermitian)
        magnitude = spectrum[..., -1].clamp_min(torch.finfo(spectrum.dtype).tiny)
        regular = (spectrum[..., 1:] - spectrum[..., :-1]).amin(-1) > (
            64 * torch.finfo(spectrum.dtype).eps * magnitude
        )
    frame = torch.eye(3, device=context.device, dtype=context.dtype).expand_as(context).clone()
    gate = torch.zeros(context.shape[:-2], device=context.device, dtype=context.real.dtype)
    if regular.any():
        values, vectors = torch.linalg.eigh(hermitian[regular])
        gaps = torch.stack((values[:, 1]-values[:, 0], values[:, 2]-values[:, 1],
                            values[:, 2]-values[:, 0]), -1) / values[:, -1, None]
        frame[regular] = vectors
        gate[regular] = (gaps.square() / (gaps.square() + 1.e-6)).prod(-1)
    return frame.to(output_device), gate.to(output_device)


def covariant_staple_frame(staples):
    """Covariant eigenframe; analytic CUDA 3x3 solve with CPU collision fallback.

    The cubic spectrum and row-cross-product eigenvectors avoid CUDA's huge
    batched eigh workspace. Column phases cancel in the physical update.
    Near a repeated root, CPU eigh supplies the same ordered spectral frame.
    All operations, including CPU fallback, retain their parameter gradients.
    """
    if not staples.is_cuda:
        return _cpu_staple_frame(staples)
    context = staples.to(torch.complex128)
    matrix = _matrix_product(context, _dagger(context))
    identity = torch.eye(3, device=context.device, dtype=context.dtype)
    center = matrix.diagonal(dim1=-2, dim2=-1).real.mean(-1)
    shifted = matrix-center[..., None, None]*identity
    radius = torch.sqrt(shifted.abs().square().sum((-2, -1)).clamp_min(1.e-60)/6.)
    normalized = shifted/radius[..., None, None]
    # Explicit 3x3 determinant avoids a batched LU solver and its launches.
    determinant = (
        normalized[..., 0, 0]*(normalized[..., 1, 1]*normalized[..., 2, 2]-normalized[..., 1, 2]*normalized[..., 2, 1])
        -normalized[..., 0, 1]*(normalized[..., 1, 0]*normalized[..., 2, 2]-normalized[..., 1, 2]*normalized[..., 2, 0])
        +normalized[..., 0, 2]*(normalized[..., 1, 0]*normalized[..., 2, 1]-normalized[..., 1, 1]*normalized[..., 2, 0])
    )
    cubic = determinant.real/2.
    # The analytic root has ill-conditioned derivatives at |cubic|=1.
    # Test in detached coordinates, and never evaluate that branch there.
    regular = ((center.detach() > 0.) & (cubic.detach().abs() < .99)
               & (radius.detach() > 1.e-12*center.detach().abs()))
    frame = identity.expand_as(context).clone()
    gate = torch.zeros(context.shape[:-2], device=context.device, dtype=torch.float64)
    if regular.any():
        angle = torch.acos(cubic[regular])/3.
        middle = center[regular]
        spread = radius[regular]
        low = middle+2*spread*torch.cos(angle+2*math.pi/3.)
        high = middle+2*spread*torch.cos(angle)
        values = torch.stack((low, 3*middle-low-high, high), -1)
        vectors = []
        for value in (low, high):
            residual = matrix[regular]-value[:, None, None]*identity
            crosses = torch.stack([torch.linalg.cross(residual[:, a], residual[:, b], dim=-1)
                                   for a, b in [(0, 1), (0, 2), (1, 2)]], 1)
            choice = crosses.abs().square().sum(-1).argmax(-1)
            vector = crosses[torch.arange(len(choice), device=context.device), choice]
            if vectors:
                vector = vector-vectors[0]*(vectors[0].conj()*vector).sum(-1, keepdim=True)
            vectors.append(vector/torch.linalg.vector_norm(vector, dim=-1, keepdim=True))
        central = torch.linalg.cross(vectors[1], vectors[0], dim=-1).conj()
        frame[regular] = torch.stack((vectors[0], central, vectors[1]), -1)
        gaps = torch.stack((values[:, 1]-values[:, 0], values[:, 2]-values[:, 1],
                            values[:, 2]-values[:, 0]), -1)/values[:, -1, None]
        gate[regular] = (gaps.square()/(gaps.square()+1.e-6)).prod(-1)
    if (~regular).any():
        fallback, fallback_gate = _cpu_staple_frame(context[~regular])
        frame[~regular], gate[~regular] = fallback, fallback_gate
    return frame.to(staples.dtype), gate.to(staples.real.dtype)


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

    By default the SU(2) subgroups act in the covariant inactive-staple
    eigenframe. Eigenvector phases cancel, giving local gauge equivariance.
    The legacy fixed-colour coupling remains available for old checkpoints.
    """

    def __init__(self, sweeps=4, hidden=8, constrained=False, initial_scale=-0.35,
                 long_staples=True, coupling=EQUIVARIANT_COUPLING):
        super().__init__()
        if coupling not in (EQUIVARIANT_COUPLING, STABLE_COUPLING, LEGACY_COUPLING):
            raise ValueError(f"unknown coupling: {coupling}")
        self.coupling = coupling
        self.gauge_equivariant = coupling in (EQUIVARIANT_COUPLING, STABLE_COUPLING)
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
        if self.coupling == STABLE_COUPLING:
            self.double()

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
        if self.gauge_equivariant:
            frame, gate = covariant_staple_frame(staples)
            link = _matrix_product(_dagger(frame), link)
            staples = _matrix_product(_dagger(frame), staples)
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
            norm_square = a.abs().square() + b.abs().square()
            nonzero = norm_square > 1.e-24
            norm = torch.sqrt(norm_square.clamp_min(1.e-24))
            denominator = norm
            a = torch.where(nonzero, a / denominator, torch.ones_like(a))
            b = torch.where(nonzero, b / denominator, torch.zeros_like(b))
            features = torch.cat((torch.stack((norm, staple_size, long_size,
                                              overlap.real, overlap.imag), -1), parities), -1)
            index = [(0, 1), (0, 2), (1, 2)].index((first, second))
            parameters = self.scales[sweep](features)
            scale = parameters[..., index].clamp(-2., 2.)
            shape = parameters[..., index+3]
            if self.gauge_equivariant:
                scale, shape = scale * gate, shape * gate
            scale = torch.where(nonzero, scale, torch.zeros_like(scale))
            shape = torch.where(nonzero, shape, torch.zeros_like(shape))
            new_a, new_b, change = quaternion_transport(
                a, b, scale, shape, inverse=inverse, compute_logdet=compute_logdet,
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
        if self.gauge_equivariant:
            link = _matrix_product(frame, link)
        updated = links[:, direction].clone()
        updated[:, mask] = link
        result = torch.stack([updated if mu == direction else links[:, mu] for mu in range(4)], 1)
        return result, logdet

    def forward(self, links, *, inverse=False, compute_logdet=True):
        if self.coupling == STABLE_COUPLING:
            links = links.to(torch.complex128)
        batched = links.ndim == 8
        links = links if batched else links.unsqueeze(0)
        logdet = torch.zeros(len(links), dtype=links.real.dtype, device=links.device) if compute_logdet else None
        operations = [(sweep, direction, parity) for sweep in range(self.sweeps)
                      for direction in range(4) for parity in range(2)]
        if inverse:
            operations.reverse()
        for sweep, direction, parity in operations:
            if self.training and torch.is_grad_enabled():
                # Each L24 double-precision checkpoint input is 184 MiB;
                # 64 substeps otherwise retain over 11 GiB on the GPU.
                # Offload these exact inputs; recompute one coupling for VJP.
                storage = (torch.autograd.graph.save_on_cpu(pin_memory=True)
                           if links.is_cuda and self.coupling == STABLE_COUPLING else nullcontext())
                with storage:
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
        if self.coupling == STABLE_COUPLING:
            coarse = coarse.to(torch.complex128)
        base = constrained_haar_lift(coarse, generator=generator)
        smeared, logdet = self(base, compute_logdet=compute_logdet)
        return smeared, -logdet if compute_logdet else None

    def log_prob(self, smeared):
        """Conditional density on the fixed blocking fibre (Haar measure)."""
        _, inverse_logdet = self.inverse(smeared)
        return inverse_logdet


def flow_roundtrip_check(flow, coarse=None):
    """Check the complete trained map, not just individual subgroup inverses.

    Deep contractions can amplify floating-point errors even when every local
    inverse passes. Training and calibration use this to reject such states.
    Local generators and a forked CPU RNG leave training random draws unchanged.
    """
    devices = [("cpu", torch.complex128, 1.e-10, 1.e-8)]
    if next(flow.parameters()).is_cuda:
        devices.append(("cuda", torch.complex64, 3.e-4, 3.e-2))
    records = []
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        for device, dtype, link_tolerance, density_tolerance in devices:
            model = SU3ConditionalFlow(sweeps=flow.sweeps, hidden=flow.hidden,
                                       constrained=flow.constrained, long_staples=flow.long_staples,
                                       coupling=flow.coupling).to(device).eval()
            model = model.double() if dtype == torch.complex128 else model
            model.load_state_dict(flow.state_dict())
            generator = torch.Generator(device=device).manual_seed(381)
            small_coarse = haar_links((1, 4, 2, 2, 2, 2), device=device, dtype=dtype, generator=generator)
            links = constrained_haar_lift(small_coarse, generator=generator)
            output, forward = model(links)
            recovered, backward = model.inverse(output)
            gauge_dtype = output.dtype
            gauge = haar_links((1, 4, 4, 4, 4), device=device, dtype=gauge_dtype, generator=generator)
            current = links.to(gauge_dtype)
            gauged = torch.stack([gauge @ current[:, mu] @ _dagger(torch.roll(gauge, -1, 4-mu))
                                  for mu in range(4)], 1)
            gauged_output, gauge_density = model(gauged)
            expected = torch.stack([gauge @ output[:, mu] @ _dagger(torch.roll(gauge, -1, 4-mu))
                                    for mu in range(4)], 1)
            inverse_gauged, _ = model.inverse(expected)
            expected_inverse = torch.stack([gauge @ recovered[:, mu] @ _dagger(torch.roll(gauge, -1, 4-mu))
                                            for mu in range(4)], 1)
            if flow.coupling == STABLE_COUPLING:
                link_tolerance, density_tolerance = 1.e-10, 1.e-8
            error = float((recovered-links).abs().max())
            density_error = float((forward+backward).abs().max())
            gauge_error = float((gauged_output-expected).abs().max())
            inverse_gauge_error = float((inverse_gauged-expected_inverse).abs().max())
            density_gauge_error = float((forward-gauge_density).abs().max())
            records.append({"device": device, "roundtrip_error": error,
                            "logdet_roundtrip_error": density_error,
                            "gauge_error": gauge_error, "inverse_gauge_error": inverse_gauge_error,
                            "logdet_gauge_error": density_gauge_error,
                            "passed": (max(error, gauge_error, inverse_gauge_error) < link_tolerance
                                       and max(density_error, density_gauge_error) < density_tolerance)})
        if coarse is not None:
            generator = torch.Generator(device=device).manual_seed(8324)
            source = constrained_haar_lift(coarse.to(device=device, dtype=output.dtype), generator=generator)
            proposal, _ = model(source, compute_logdet=False)
            recovered, _ = model.inverse(proposal, compute_logdet=False)
            error = float((recovered-source).abs().max())
            records.append({"device": device, "lattice_size": list(source.shape[-6:-2]),
                            "roundtrip_error": error, "passed": error < link_tolerance})
    return {"passed": all(row["passed"] for row in records), "records": records}
