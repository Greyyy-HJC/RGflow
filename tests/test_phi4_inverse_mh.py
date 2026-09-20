import numpy as np

from rgflow.phi4.action import Phi4Action
from rgflow.phi4.blocking import identity_kernel
from rgflow.phi4.inverse_blocking import reblocking_error
from rgflow.phi4.inverse_mh import (
    build_state,
    choose_state,
    coarse_log_acceptance,
    detail_log_acceptance,
    flow_refresh_log_acceptance,
    mh_sweep,
    trajectory_diagnostics,
    wolff_state_transition,
)


def test_acceptance_ratios_and_rejected_state() -> None:
    old_fine = np.array([1.0, 4.0])
    new_fine = np.array([2.5, 2.0])
    old_coarse = np.array([0.5, 1.5])
    new_coarse = np.array([0.75, 1.0])
    np.testing.assert_allclose(
        detail_log_acceptance(old_fine, new_fine),
        -new_fine + old_fine,
    )
    np.testing.assert_allclose(
        coarse_log_acceptance(old_fine, new_fine, old_coarse, new_coarse),
        -new_fine + old_fine + new_coarse - old_coarse,
    )

    rng = np.random.default_rng(113)
    kernel = identity_kernel()
    action = Phi4Action()
    coarse = rng.normal(size=(2, 4, 4))
    details = rng.normal(size=(2, 3, 4, 4))
    old = build_state(coarse, details, kernel, action)
    proposed = build_state(coarse + 0.1, details - 0.1, kernel, action)
    selected = choose_state(old, proposed, np.array([False, True]))
    np.testing.assert_allclose(selected.fine[0], old.fine[0])
    np.testing.assert_allclose(selected.fine[1], proposed.fine[1])


def test_one_exact_mh_sweep_preserves_inverse_blocking_constraint() -> None:
    rng = np.random.default_rng(127)
    kernel = identity_kernel()
    action = Phi4Action()
    state = build_state(
        rng.normal(size=(3, 4, 4)),
        rng.normal(size=(3, 3, 4, 4)),
        kernel,
        action,
    )
    updated, diagnostics = mh_sweep(
        state,
        kernel,
        action,
        {"coarse": 0.01, "d01": 0.01, "d10": 0.01, "d11": 0.01},
        rng,
    )
    assert np.max(reblocking_error(updated.fine, updated.coarse, kernel)) < 1.0e-9
    assert diagnostics["max_reblocking_error"] < 1.0e-9
    for name in ("coarse", "d01", "d10", "d11"):
        assert diagnostics["coordinates"][name]["outer_attempts"] == 12



def test_flow_refresh_ratio_is_antisymmetric() -> None:
    old_action = np.array([1.0, 2.0])
    new_action = np.array([1.5, 1.25])
    old_logq = np.array([-3.0, -2.0])
    new_logq = np.array([-2.5, -2.25])
    forward = flow_refresh_log_acceptance(
        old_action, new_action, old_logq, new_logq
    )
    backward = flow_refresh_log_acceptance(
        new_action, old_action, new_logq, old_logq
    )
    np.testing.assert_allclose(forward, -backward)


def test_wolff_transition_rebuilds_inverse_coordinates() -> None:
    rng = np.random.default_rng(41)
    action = Phi4Action()
    kernel = identity_kernel()
    state = build_state(
        rng.normal(size=(3, 4, 4)),
        rng.normal(size=(3, 3, 4, 4)),
        kernel,
        action,
    )
    transitioned, diagnostics = wolff_state_transition(
        state, kernel, action, np.random.default_rng(42)
    )
    assert diagnostics["mean_cluster_fraction"] > 0.0
    np.testing.assert_allclose(
        transitioned.fine_action, action(transitioned.fine), atol=1.0e-12
    )
    assert np.max(
        reblocking_error(transitioned.fine, transitioned.coarse, kernel)
    ) < 1.0e-12


def test_parallel_trajectory_diagnostics_uses_time_axis() -> None:
    rng = np.random.default_rng(91)
    white = rng.normal(size=(8, 256))
    result = trajectory_diagnostics({"white": white}, 2.0)
    assert 0.5 <= result["observables"]["white"]["tau_int"] < 1.5
    assert result["observables"]["white"]["ess"] > 600
