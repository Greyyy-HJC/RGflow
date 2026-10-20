import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


SCRIPT = Path(__file__).parents[1] / "scripts/upsample/train.py"
SPEC = importlib.util.spec_from_file_location("upsample_training", SCRIPT)
training = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(training)


def test_six_operator_loss_gradient_and_tiny_variance_are_scale_invariant():
    rng = np.random.default_rng(27)
    target = rng.normal(size=(18, 8))
    values = torch.tensor(rng.normal(size=(12, 6)), dtype=torch.float64)
    args = SimpleNamespace(mean_metric="diagonal", variance_weight=10.)
    coefficients, report = training.moment_loss(values, target, args)
    scale = np.array([1., 1., 1., 1., 1., 1., 1.e-5, 1.e-9])
    selected_scale = torch.tensor(scale[training.TRAINING_COLUMNS])
    small = (values * selected_scale).requires_grad_()
    scaled_coefficients, scaled_report = training.moment_loss(small, target * scale, args)
    np.testing.assert_allclose(report["loss"], scaled_report["loss"], rtol=1.e-12)
    torch.testing.assert_close(scaled_coefficients * selected_scale, coefficients)

    reference = torch.tensor(target[:, training.TRAINING_COLUMNS] * scale[training.TRAINING_COLUMNS])
    target_variance = reference.var(0)
    epsilon = 1.e-12 * target_variance
    loss = ((small.mean(0)-reference.mean(0)).square()/target_variance).sum()
    loss = loss + 10.*(torch.log(small.var(0)+epsilon)-torch.log(target_variance+epsilon)).square().sum()
    gradient = torch.autograd.grad(loss, small)[0]
    torch.testing.assert_close(scaled_coefficients, gradient)
    assert abs(float(scaled_coefficients[:, -1].sum().detach())) > 1.e7


def test_held_out_targets_do_not_affect_training_loss_or_gradient():
    rng = np.random.default_rng(1234)
    target = rng.normal(size=(18, 8))
    values = torch.tensor(rng.normal(size=(12, 6)), dtype=torch.float64)
    args = SimpleNamespace(mean_metric="covariance", variance_weight=10.)
    before, report = training.moment_loss(values, target, args)
    target[:, [2, 3]] = 1.e8 * rng.normal(size=(18, 2))
    after, changed = training.moment_loss(values, target, args)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert report == changed
