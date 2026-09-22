import numpy as np
import torch

from rgflow.su3.downsampling import (
    LinkCoefficientCNN,
    gauge_invariant_features,
    rectangle_sum,
    smear_and_block,
    smear_links,
    su3_errors,
    weights_from_path_logits,
)
from rgflow.su3.observables import observable_vector_torch


def _random_su3(lattice: int = 4) -> torch.Tensor:
    random = torch.randn(4, lattice, lattice, lattice, lattice, 3, 3, dtype=torch.complex64)
    unitary, _ = torch.linalg.qr(random)
    determinant = torch.linalg.det(unitary)
    return unitary * torch.exp(-1j * torch.angle(determinant) / 3.0)[..., None, None]


def test_identity_field_is_preserved_and_observables_are_one() -> None:
    identity = torch.eye(3, dtype=torch.complex64).reshape(1, 1, 1, 1, 1, 3, 3)
    links = identity.expand(4, 4, 4, 4, 4, 3, 3).clone()
    blocked = smear_and_block(links, torch.tensor([2.0, -1.0]))

    assert blocked.shape == (4, 2, 2, 2, 2, 3, 3)
    assert su3_errors(blocked)[0] < 1.0e-5
    np.testing.assert_allclose(observable_vector_torch(blocked).numpy(), np.ones(4), atol=1.0e-5)


def test_smearing_and_blocking_are_gauge_covariant() -> None:
    links = _random_su3()
    gauge = _random_su3().select(0, 0)
    transformed = []
    for direction in range(4):
        site_axis = 3 - direction
        gauge_forward = torch.roll(gauge, shifts=-1, dims=site_axis)
        transformed.append(gauge @ links[direction] @ gauge_forward.conj().transpose(-2, -1))
    transformed = torch.stack(transformed)
    logits = torch.tensor([1.3, -0.4])

    smeared = smear_links(links, logits)
    transformed_smeared = smear_links(transformed, logits)
    expected = []
    for direction in range(4):
        gauge_forward = torch.roll(gauge, shifts=-1, dims=3 - direction)
        expected.append(gauge @ smeared[direction] @ gauge_forward.conj().transpose(-2, -1))
    np.testing.assert_allclose(
        transformed_smeared.numpy(), torch.stack(expected).numpy(), rtol=2.0e-4, atol=2.0e-4
    )
    np.testing.assert_allclose(
        observable_vector_torch(smear_and_block(links, logits)).detach().numpy(),
        observable_vector_torch(smear_and_block(transformed, logits)).detach().numpy(),
        rtol=2.0e-4,
        atol=2.0e-4,
    )


def test_projection_gradient_is_finite() -> None:
    links = _random_su3()
    logits = torch.tensor([1.0, -1.0], requires_grad=True)
    value = observable_vector_torch(smear_and_block(links, logits)).sum()
    gradient = torch.autograd.grad(value, logits)[0]
    assert torch.isfinite(gradient).all()


def test_rectangle_path_and_cnn_weights_are_well_formed() -> None:
    links = _random_su3()
    rectangles = rectangle_sum(links, 0)
    assert rectangles.shape == links.shape[1:]
    model = LinkCoefficientCNN(hidden_channels=4)
    features = gauge_invariant_features(links.unsqueeze(0))
    weights = weights_from_path_logits(model(features))
    assert weights.shape == (1, 4, 3, 4, 4, 4, 4)
    np.testing.assert_allclose(weights.sum(dim=2).detach().numpy(), 1.0, atol=1.0e-6)
    assert float(weights.min().detach()) >= 0.0


def test_cnn_constant_initialization_keeps_a_live_local_gradient() -> None:
    links = _random_su3()
    model = LinkCoefficientCNN(hidden_channels=4)
    features = gauge_invariant_features(links.unsqueeze(0))
    weights = weights_from_path_logits(model(features))
    loss = (weights[:, :, 0] * features[:, :4]).mean()
    loss.backward()

    final_spatial = model.layers[-1].spatial.weight.grad
    assert final_spatial is not None
    assert torch.isfinite(final_spatial).all()
    assert float(final_spatial.abs().max()) > 0.0


def test_cnn_downsampling_is_gauge_covariant() -> None:
    links = _random_su3()
    gauge = _random_su3().select(0, 0)
    transformed = []
    for direction in range(4):
        gauge_forward = torch.roll(gauge, shifts=-1, dims=3 - direction)
        transformed.append(gauge @ links[direction] @ gauge_forward.conj().transpose(-2, -1))
    transformed = torch.stack(transformed)
    model = LinkCoefficientCNN(hidden_channels=4)
    features = gauge_invariant_features(links.unsqueeze(0))
    transformed_features = gauge_invariant_features(transformed.unsqueeze(0))
    np.testing.assert_allclose(features.detach().numpy(), transformed_features.detach().numpy(), atol=2.0e-4)
    blocked = smear_and_block(links, model)
    transformed_blocked = smear_and_block(transformed, model)
    expected = []
    coarse_gauge = gauge[::2, ::2, ::2, ::2]
    for direction in range(4):
        gauge_forward = torch.roll(coarse_gauge, shifts=-1, dims=3 - direction)
        expected.append(coarse_gauge @ blocked[direction] @ gauge_forward.conj().transpose(-2, -1))
    np.testing.assert_allclose(
        transformed_blocked.detach().numpy(), torch.stack(expected).detach().numpy(), rtol=4.0e-4, atol=4.0e-4
    )
