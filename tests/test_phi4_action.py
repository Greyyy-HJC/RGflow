import numpy as np

from rgflow.phi4.action import Phi4Action


def test_action_has_global_z2_symmetry() -> None:
    rng = np.random.default_rng(7)
    fields = rng.normal(size=(3, 6, 6))
    action = Phi4Action(kappa=0.3401, lam=1.0)

    np.testing.assert_allclose(action(fields), action(-fields))


def test_action_uses_periodic_nearest_neighbors() -> None:
    field = np.arange(16, dtype=float).reshape(4, 4) / 10.0
    action = Phi4Action(kappa=0.2, lam=0.7)
    hopping = 0.0
    potential = 0.0
    for x in range(4):
        for y in range(4):
            value = field[x, y]
            hopping += value * field[(x + 1) % 4, y]
            hopping += value * field[x, (y + 1) % 4]
            potential += value**2 + 0.7 * (value**2 - 1.0) ** 2

    np.testing.assert_allclose(action(field), potential - 0.4 * hopping)


def test_local_delta_matches_full_action_difference() -> None:
    rng = np.random.default_rng(11)
    field = rng.normal(size=(8, 8))
    action = Phi4Action(kappa=0.3401, lam=1.0)
    x, y = 3, 6
    proposal_value = -0.27
    proposal = field.copy()
    proposal[x, y] = proposal_value

    local_delta = action.local_delta(field, proposal)[x, y]
    total_delta = action(proposal) - action(field)
    np.testing.assert_allclose(local_delta, total_delta, rtol=1e-12, atol=1e-12)
