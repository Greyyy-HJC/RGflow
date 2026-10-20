import importlib.util
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import torch

from rgflow.su3.downsampling import block_polynomial_basis, polynomial_blocking_basis
from rgflow.su3.observables import observable_vector_torch


SCRIPT = Path(__file__).parents[1] / "scripts/downsample/fit_polynomial.py"
SPEC = importlib.util.spec_from_file_location("fit_polynomial", SCRIPT)
fitting = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fitting)


def test_parallel_configuration_moments_preserve_values_and_jacobians():
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(77)
    bases = [torch.randn(12, 4, 2, 2, 2, 2, 2, 3, 3, dtype=torch.complex64,
                         generator=generator) for _ in range(4)]
    x = np.linspace(-0.03, 0.04, 11)
    measure = partial(fitting.configuration_moments, x=x, device=torch.device("cpu"))
    serial = list(map(measure, bases))
    with ThreadPoolExecutor(max_workers=4) as pool:
        parallel = list(pool.map(measure, bases))
    for expected, actual in zip(serial, parallel):
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])


def test_configuration_jacobian_matches_finite_differences_with_hooks_and_local_terms():
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(123)
    raw = 0.1 * torch.randn(4, 4, 4, 4, 4, 3, 3, dtype=torch.complex64, generator=generator)
    algebra = raw - raw.mH
    trace = torch.diagonal(algebra, dim1=-2, dim2=-1).sum(-1) / 3
    links = torch.linalg.matrix_exp(algebra - trace[..., None, None] * torch.eye(3))
    basis = polynomial_blocking_basis(links, hook=True, local=True)
    x = np.linspace(-0.03, 0.04, 11)
    _, jacobian = fitting.configuration_moments(basis, x, torch.device("cpu"))
    finite_difference = np.empty_like(jacobian)
    step = 0.003
    with torch.no_grad():
        for index in range(len(x)):
            delta = np.zeros_like(x)
            delta[index] = step
            plus = observable_vector_torch(block_polynomial_basis(basis, torch.tensor(x + delta, dtype=torch.float32)))
            minus = observable_vector_torch(block_polynomial_basis(basis, torch.tensor(x - delta, dtype=torch.float32)))
            finite_difference[:, index] = ((plus - minus) / (2 * step)).numpy()
    np.testing.assert_allclose(jacobian, finite_difference, rtol=0.01, atol=1e-4)


def test_six_operator_jacobian_and_held_out_exclusion():
    from rgflow.su3.observables import DOWNSAMPLING_OBSERVABLES, TRAINING_OBSERVABLES
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(39)
    raw = 0.1 * torch.randn(4, 4, 4, 4, 4, 3, 3, dtype=torch.complex64, generator=generator)
    algebra = raw - raw.mH
    algebra -= torch.diagonal(algebra, dim1=-2, dim2=-1).sum(-1)[..., None, None] * torch.eye(3) / 3
    basis = polynomial_blocking_basis(torch.linalg.matrix_exp(algebra), hook=True, local=True)
    x = np.linspace(-0.03, 0.04, 11)
    values, jacobian = fitting.configuration_moments(basis, x, torch.device('cpu'), names=TRAINING_OBSERVABLES)
    all_values, all_jacobian = fitting.configuration_moments(basis, x, torch.device('cpu'), names=DOWNSAMPLING_OBSERVABLES)
    columns = [DOWNSAMPLING_OBSERVABLES.index(n) for n in TRAINING_OBSERVABLES]
    np.testing.assert_array_equal(values, all_values[columns])
    np.testing.assert_array_equal(jacobian, all_jacobian[columns])
    finite_difference = np.empty_like(jacobian)
    step = 0.003
    with torch.no_grad():
        for i in range(len(x)):
            delta = np.eye(len(x))[i] * step
            plus = observable_vector_torch(block_polynomial_basis(basis, torch.tensor(x + delta, dtype=torch.float32)), names=TRAINING_OBSERVABLES)
            minus = observable_vector_torch(block_polynomial_basis(basis, torch.tensor(x - delta, dtype=torch.float32)), names=TRAINING_OBSERVABLES)
            finite_difference[:, i] = ((plus - minus) / (2 * step)).numpy()
    np.testing.assert_allclose(jacobian, finite_difference, rtol=0.02, atol=1e-4)
    ensemble = np.stack([all_values * factor for factor in (0.9, 1., 1.1)])
    jacobians = np.stack([all_jacobian] * 3)
    targets = ensemble * 1.05
    before = fitting.moment_residuals(ensemble[:, columns], jacobians[:, columns], targets[:, columns])
    ensemble[:, [2, 3]] += 1e3
    targets[:, [2, 3]] -= 1e3
    jacobians[:, [2, 3]] *= 1e6
    after = fitting.moment_residuals(ensemble[:, columns], jacobians[:, columns], targets[:, columns])
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)


def test_moment_residuals_and_metrics_preserve_tiny_fourth_moment_variance():
    from rgflow.su3.training import _covariance_inverse, _metric_record
    rng = np.random.default_rng(13)
    targets = rng.normal(size=(20, 6))
    values = 2 * targets
    jacobians = rng.normal(size=(20, 6, 11))
    expected = fitting.moment_residuals(values, jacobians, targets)
    scale = np.array([1., 1., 1., 1., 1e-4, 1e-8])
    actual = fitting.moment_residuals(values * scale, jacobians * scale[None, :, None], targets * scale)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    assert abs(actual[0][-1]) > 1
    _, inverse = _covariance_inverse(targets * scale, values * scale, .1, observable_scale=targets.std(0, ddof=1) * scale)
    record = _metric_record(values * scale, targets * scale, inverse, variance_epsilon=1e-12 * (targets * scale).var(0, ddof=1))
    np.testing.assert_allclose(record['std_ratio'], 2.)
    np.testing.assert_allclose(record['standardized_log_variance_shift'], np.log(4.), rtol=1e-10)
