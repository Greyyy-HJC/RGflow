"""Hybrid radial-Metropolis and embedded-Ising Wolff sampling."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .action import Phi4Action


FloatArray = NDArray[np.float64]


@dataclass
class SamplingResult:
    configurations: FloatArray
    proposal_width: float
    warmup_acceptance: float
    production_acceptance: float
    mean_cluster_fraction: float


def checkerboard_mask(size: int, parity: int) -> NDArray[np.bool_]:
    coordinates = np.indices((size, size))
    return (coordinates[0] + coordinates[1]) % 2 == parity


def radial_metropolis_sweep(
    fields: FloatArray,
    action: Phi4Action,
    proposal_width: float,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Update all field magnitudes once while preserving their signs."""
    accepted_total = 0
    proposed_total = 0
    size = fields.shape[-1]

    for parity in (0, 1):
        site_mask = checkerboard_mask(size, parity)
        old = fields.copy()
        radial_step = rng.uniform(-proposal_width, proposal_width, size=fields.shape)
        signs = np.where(old < 0.0, -1.0, 1.0)
        proposal = signs * np.abs(np.abs(old) + radial_step)
        delta_action = action.local_delta(old, proposal)
        accept = np.log(rng.random(fields.shape)) < -delta_action
        accept &= site_mask
        fields[accept] = proposal[accept]
        accepted_total += int(np.count_nonzero(accept))
        proposed_total += fields.shape[0] * int(np.count_nonzero(site_mask))

    return accepted_total, proposed_total


def wolff_cluster_update(
    field: FloatArray,
    kappa: float,
    rng: np.random.Generator,
) -> int:
    """Flip one Wolff cluster of the embedded Ising signs."""
    size = field.shape[0]
    seed = tuple(rng.integers(0, size, size=2))
    in_cluster = np.zeros((size, size), dtype=bool)
    in_cluster[seed] = True
    stack = [seed]

    while stack:
        x, y = stack.pop()
        for neighbor in (
            ((x + 1) % size, y),
            ((x - 1) % size, y),
            (x, (y + 1) % size),
            (x, (y - 1) % size),
        ):
            if in_cluster[neighbor] or field[x, y] * field[neighbor] <= 0.0:
                continue
            bond_probability = 1.0 - np.exp(
                -4.0 * kappa * abs(field[x, y] * field[neighbor])
            )
            if rng.random() < bond_probability:
                in_cluster[neighbor] = True
                stack.append(neighbor)

    field[in_cluster] *= -1.0
    return int(np.count_nonzero(in_cluster))


def _update_cycle(
    fields: FloatArray,
    action: Phi4Action,
    proposal_width: float,
    radial_sweeps: int,
    rng: np.random.Generator,
) -> tuple[int, int, int]:
    accepted = 0
    proposed = 0
    for _ in range(radial_sweeps):
        sweep_accepted, sweep_proposed = radial_metropolis_sweep(
            fields, action, proposal_width, rng
        )
        accepted += sweep_accepted
        proposed += sweep_proposed

    cluster_sites = 0
    for field in fields:
        cluster_sites += wolff_cluster_update(field, action.kappa, rng)
    return accepted, proposed, cluster_sites


def generate_ensemble(
    *,
    size: int,
    chains: int,
    samples_per_chain: int,
    warmup_cycles: int,
    sample_interval: int,
    radial_sweeps: int,
    action: Phi4Action,
    proposal_width: float = 1.0,
    target_acceptance: float = 0.5,
    seed: int = 1234,
) -> SamplingResult:
    """Generate an ensemble shaped ``(chains, samples, size, size)``."""
    rng = np.random.default_rng(seed)
    fields = rng.normal(0.0, 0.5, size=(chains, size, size))
    width = proposal_width
    warmup_accepted = 0
    warmup_proposed = 0
    adaptation_accepted = 0
    adaptation_proposed = 0

    for cycle in range(warmup_cycles):
        accepted, proposed, _ = _update_cycle(
            fields, action, width, radial_sweeps, rng
        )
        warmup_accepted += accepted
        warmup_proposed += proposed
        adaptation_accepted += accepted
        adaptation_proposed += proposed
        if (cycle + 1) % 25 == 0:
            rate = adaptation_accepted / adaptation_proposed
            width *= np.exp(0.5 * (rate - target_acceptance))
            width = float(np.clip(width, 1e-3, 10.0))
            adaptation_accepted = 0
            adaptation_proposed = 0

    configurations = np.empty(
        (chains, samples_per_chain, size, size), dtype=np.float64
    )
    production_accepted = 0
    production_proposed = 0
    cluster_sites = 0
    cluster_updates = 0

    for sample_index in range(samples_per_chain):
        for _ in range(sample_interval):
            accepted, proposed, clustered = _update_cycle(
                fields, action, width, radial_sweeps, rng
            )
            production_accepted += accepted
            production_proposed += proposed
            cluster_sites += clustered
            cluster_updates += chains
        configurations[:, sample_index] = fields

    return SamplingResult(
        configurations=configurations,
        proposal_width=width,
        warmup_acceptance=warmup_accepted / warmup_proposed,
        production_acceptance=production_accepted / production_proposed,
        mean_cluster_fraction=cluster_sites / (cluster_updates * size * size),
    )
