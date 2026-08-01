import numpy as np

from rgflow.phi4.action import Phi4Action
from rgflow.phi4.sampling import (
    checkerboard_mask,
    generate_ensemble,
    wolff_cluster_update,
)


def test_checkerboard_masks_partition_even_lattice() -> None:
    even = checkerboard_mask(8, 0)
    odd = checkerboard_mask(8, 1)

    assert not np.any(even & odd)
    assert np.all(even | odd)
    assert np.count_nonzero(even) == np.count_nonzero(odd) == 32


def test_wolff_at_zero_kappa_flips_only_seed() -> None:
    field = np.ones((6, 6))
    original = field.copy()

    cluster_size = wolff_cluster_update(field, 0.0, np.random.default_rng(3))

    assert cluster_size == 1
    assert np.count_nonzero(field != original) == 1


def test_generation_is_reproducible() -> None:
    arguments = {
        "size": 4,
        "chains": 2,
        "samples_per_chain": 3,
        "warmup_cycles": 25,
        "sample_interval": 2,
        "radial_sweeps": 1,
        "action": Phi4Action(),
        "seed": 19,
    }
    first = generate_ensemble(**arguments)
    second = generate_ensemble(**arguments)

    np.testing.assert_array_equal(first.configurations, second.configurations)
    assert first.production_acceptance == second.production_acceptance
    assert first.configurations.shape == (2, 3, 4, 4)


def test_kappa_zero_moment_matches_single_site_quadrature() -> None:
    action = Phi4Action(kappa=0.0, lam=1.0)
    result = generate_ensemble(
        size=2,
        chains=128,
        samples_per_chain=100,
        warmup_cycles=200,
        sample_interval=2,
        radial_sweeps=1,
        action=action,
        seed=23,
    )
    nodes, weights = np.polynomial.legendre.leggauss(300)
    phi = 4.0 * nodes
    quadrature_weights = 4.0 * weights
    density = np.exp(-action.potential(phi))
    expected_phi_sq = np.sum(quadrature_weights * density * phi**2) / np.sum(
        quadrature_weights * density
    )

    measured_phi_sq = float(np.mean(result.configurations**2))
    assert abs(measured_phi_sq - expected_phi_sq) < 0.03
