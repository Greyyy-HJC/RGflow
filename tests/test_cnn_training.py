import numpy as np
import torch

from rgflow.su3.training import _covariance_inverse, _loss_coefficients, _stats


def test_mean_variance_surrogate_gradient_matches_full_loss() -> None:
    values = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [3.0, 5.0, 7.0, 11.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    target = np.asarray([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0]])
    _, inverse = _covariance_inverse(target, target, 0.1)
    target_mean, _, target_log_variance = _stats(target)
    inverse_tensor = torch.as_tensor(inverse, dtype=torch.float64)
    target_mean_tensor = torch.as_tensor(target_mean, dtype=torch.float64)
    target_log_variance_tensor = torch.as_tensor(target_log_variance, dtype=torch.float64)
    variance_scale = torch.ones(4, dtype=torch.float64)

    coefficients, _ = _loss_coefficients(
        values,
        target_mean_tensor,
        target_log_variance_tensor,
        inverse_tensor,
        variance_scale,
    )
    mean = values.mean(dim=0)
    variance = values.var(dim=0, unbiased=True)
    full_loss = (mean - target_mean_tensor) @ inverse_tensor @ (mean - target_mean_tensor)
    full_loss = full_loss + (torch.log(variance + 1.0e-12) - target_log_variance_tensor).square().sum()
    gradient = torch.autograd.grad(full_loss, values)[0]
    torch.testing.assert_close(coefficients, gradient, rtol=1.0e-10, atol=1.0e-10)
