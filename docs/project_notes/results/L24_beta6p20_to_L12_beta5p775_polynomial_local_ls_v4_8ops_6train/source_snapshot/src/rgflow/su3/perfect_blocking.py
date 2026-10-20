"""Perturbative hard-blocking kernels for the Wilson gauge action."""

from __future__ import annotations

import math

import torch


def transverse_link_laplacian_kernel(momentum: torch.Tensor) -> torch.Tensor:
    """Return the linearized covariant link-Laplacian kernel ``L(k)``.

    The link field uses the left-endpoint Fourier convention from the
    perturbative blocking note.  Rows label the output link direction and
    columns label the input gauge-field direction.
    """
    momentum = torch.as_tensor(momentum)
    dtype = torch.complex128 if momentum.dtype == torch.float64 else torch.complex64
    momentum = momentum.to(dtype=dtype)
    kernel = torch.zeros((*momentum.shape[:-1], 4, 4), dtype=dtype, device=momentum.device)
    for mu in range(4):
        for nu in range(4):
            if mu == nu:
                continue
            kernel[..., mu, mu] = kernel[..., mu, mu] + 2.0 * torch.cos(momentum[..., nu]) - 2.0
            kernel[..., mu, nu] = kernel[..., mu, nu] + (
                (1.0 - torch.exp(1j * momentum[..., mu]))
                * (1.0 - torch.exp(-1j * momentum[..., nu]))
            )
    return kernel


def polynomial_kernel_matrix(momentum: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    """Return ``I + a1 L + ... + aN L**N`` for each momentum."""
    momentum = torch.as_tensor(momentum)
    coefficients = torch.as_tensor(coefficients, dtype=momentum.real.dtype, device=momentum.device)
    laplacian = transverse_link_laplacian_kernel(momentum)
    identity = torch.eye(4, dtype=laplacian.dtype, device=laplacian.device)
    result = identity.expand(*momentum.shape[:-1], 4, 4).clone()
    # L_U U has zero constant term. Further applications transport that
    # zero-background field, so they multiply each row by the scalar
    # transverse Laplacian, rather than by the full gauge-field matrix L.
    power = laplacian
    diagonal = torch.diagonal(laplacian, dim1=-2, dim2=-1)
    for coefficient in coefficients:
        result = result + coefficient * power
        power = diagonal[..., :, None] * power
    return result


def wilson_transverse_propagator(momentum: torch.Tensor) -> torch.Tensor:
    """Return the transverse Wilson propagator at the supplied momenta."""
    momentum = torch.as_tensor(momentum)
    # Match L and D: all three use fields based at the left link endpoint.
    vector = 1.0 - torch.exp(1j * momentum)
    norm = vector.abs().square().sum(dim=-1)
    identity = torch.eye(4, dtype=vector.dtype, device=momentum.device)
    projector = identity - vector[..., :, None] * vector.conj()[..., None, :] / norm.clamp_min(1.0e-30)[..., None, None]
    propagator = projector / norm.clamp_min(1.0e-30)[..., None, None]
    return torch.where(
        (norm > 1.0e-14)[..., None, None], propagator,
        torch.zeros_like(propagator),
    )


def perturbative_blocking_objective(
    coefficients: torch.Tensor,
    *,
    grid_size: int = 16,
    momenta: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate the normalized full-Brillouin-zone hard-blocking residual.

    This is Eq. (2.37) of ``Classically perfect blocking.pdf`` for the
    factorized two-link blocking ``Omega(k) = D(k) omega(k)``.
    """
    if momenta is None:
        values = torch.arange(grid_size, dtype=coefficients.dtype, device=coefficients.device)
        values = 2.0 * math.pi * values / grid_size - math.pi
        momenta = torch.cartesian_prod(values, values, values, values)
    else:
        momenta = momenta.to(device=coefficients.device, dtype=coefficients.dtype)
    aliases = torch.cartesian_prod(
        torch.tensor((0.0, 1.0), dtype=coefficients.dtype, device=coefficients.device),
        torch.tensor((0.0, 1.0), dtype=coefficients.dtype, device=coefficients.device),
        torch.tensor((0.0, 1.0), dtype=coefficients.dtype, device=coefficients.device),
        torch.tensor((0.0, 1.0), dtype=coefficients.dtype, device=coefficients.device),
    )
    coarse = wilson_transverse_propagator(momenta)
    fine_momenta = (momenta[:, None, :] / 2.0 + math.pi * aliases[None, :, :]).reshape(-1, 4)
    propagators = wilson_transverse_propagator(fine_momenta)
    omega = polynomial_kernel_matrix(fine_momenta, coefficients)
    diagonal = torch.diag_embed(1.0 + torch.exp(1j * fine_momenta))
    blocking = diagonal @ omega
    blocked = blocking @ propagators.to(blocking.dtype) @ blocking.conj().transpose(-2, -1)
    blocked = blocked.reshape(momenta.shape[0], 16, 4, 4).sum(dim=1) / 16.0
    norm = (2.0 * torch.sin(momenta / 2.0)).square().sum(dim=-1)
    projector = coarse.to(blocked.dtype) * norm[..., None, None]
    difference = projector @ (coarse - blocked) @ projector
    numerator = (difference.abs() ** 2).sum(dim=(-2, -1))
    denominator = (coarse.abs() ** 2).sum(dim=(-2, -1)).clamp_min(1.0e-30)
    valid = norm > 1.0e-14
    return (numerator[valid] / denominator[valid]).mean()


def optimize_perturbative_coefficients(
    *,
    grid_size: int = 16,
    steps: int = 250,
    learning_rate: float = 0.05,
    device: str = "cpu",
) -> tuple[torch.Tensor, float]:
    """Optimize the four polynomial coefficients on a full momentum grid."""
    parameter = torch.zeros(4, dtype=torch.float64, device=device, requires_grad=True)
    optimizer = torch.optim.Adam((parameter,), lr=learning_rate)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        value = perturbative_blocking_objective(parameter, grid_size=grid_size)
        value.backward()
        optimizer.step()
    optimizer = torch.optim.LBFGS((parameter,), max_iter=50, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        value = perturbative_blocking_objective(parameter, grid_size=grid_size)
        value.backward()
        return value

    optimizer.step(closure)
    value = perturbative_blocking_objective(parameter, grid_size=grid_size)
    return parameter.detach(), float(value.detach().cpu())
