import numpy as np
import torch

from rgflow.su3.downsampling import (
    GaugeEquivariantFieldTransform,
    LinkCoefficientCNN,
    PolynomialStoutKernel,
    StoutKernel,
    block_links,
    block_polynomial_basis,
    gauge_invariant_features,
    polynomial_blocking_basis,
    rectangle_sum,
    smear_and_block,
    smear_links,
    square_sum,
    su3_errors,
    transform_and_block,
    weights_from_path_logits,
)
from rgflow.su3.perfect_blocking import (
    optimize_perturbative_coefficients,
    perturbative_blocking_objective,
    polynomial_kernel_matrix,
    transverse_link_laplacian_kernel,
    wilson_transverse_propagator,
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


def test_square_staples_and_global_stout_kernel_are_well_formed() -> None:
    links = _random_su3()
    squares = square_sum(links, 0)
    assert squares.shape == links.shape[1:]

    kernel = StoutKernel()
    weights = torch.softmax(kernel(), dim=0)
    np.testing.assert_allclose(weights.detach().numpy().sum(), 1.0, atol=1.0e-6)
    blocked = smear_and_block(links, kernel)
    assert su3_errors(blocked)[0] < 2.0e-5
    value = observable_vector_torch(blocked).sum()
    value.backward()
    assert kernel.logits.grad is not None
    assert torch.isfinite(kernel.logits.grad).all()


def test_square_staples_are_gauge_covariant() -> None:
    links = _random_su3()
    gauge = _random_su3().select(0, 0)
    transformed = _gauge_transform(links, gauge)
    square = square_sum(links, 0)
    transformed_square = square_sum(transformed, 0)
    gauge_forward = torch.roll(gauge, shifts=-1, dims=3)
    expected = gauge @ square @ gauge_forward.conj().transpose(-2, -1)
    np.testing.assert_allclose(
        transformed_square.detach().numpy(), expected.detach().numpy(), rtol=5.0e-4, atol=5.0e-4
    )


def test_polynomial_stout_is_gauge_covariant_and_has_finite_gradients() -> None:
    links = _random_su3()
    gauge = _random_su3().select(0, 0)
    transformed = _gauge_transform(links, gauge)
    kernel = PolynomialStoutKernel((0.02, -0.03, 0.004, -0.0002), (0.0, 0.0, 0.0))
    blocked = smear_and_block(links, kernel)
    transformed_blocked = smear_and_block(transformed, kernel)
    expected = _gauge_transform(blocked, gauge[::2, ::2, ::2, ::2])
    np.testing.assert_allclose(
        transformed_blocked.detach().numpy(), expected.detach().numpy(), rtol=8.0e-4, atol=8.0e-4
    )
    assert su3_errors(blocked)[0] < 3.0e-5
    observable_vector_torch(blocked).sum().backward()
    gradients = [parameter.grad for parameter in kernel.parameters() if parameter.requires_grad]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)


def test_perturbative_polynomial_objective_and_initialization_are_finite() -> None:
    coefficients = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    objective = perturbative_blocking_objective(coefficients, grid_size=4)
    objective.backward()
    assert torch.isfinite(objective)
    assert torch.isfinite(coefficients.grad).all()
    fitted, residual = optimize_perturbative_coefficients(grid_size=4, steps=1)
    assert fitted.shape == (4,)
    assert torch.isfinite(fitted).all()
    assert np.isfinite(residual)


def test_perturbative_laplacian_annihilates_a_pure_gauge_mode() -> None:
    momentum = torch.tensor([0.3, 0.5, 0.7, 1.1], dtype=torch.float64)
    kernel = transverse_link_laplacian_kernel(momentum)
    pure_gauge = 1.0 - torch.exp(1j * momentum)
    np.testing.assert_allclose((kernel @ pure_gauge).numpy(), np.zeros(4), atol=1.0e-12)


def test_wilson_propagator_uses_the_endpoint_gauge_mode() -> None:
    momentum = torch.tensor([0.3, 0.5, 0.7, 1.1], dtype=torch.float64)
    gauge = 1.0 - torch.exp(1j * momentum)
    propagator = wilson_transverse_propagator(momentum)
    torch.testing.assert_close(propagator @ gauge, torch.zeros_like(gauge), atol=1.0e-14, rtol=0)


def test_polynomial_momentum_kernel_matches_nonlinear_weak_field() -> None:
    momentum = torch.tensor([torch.pi / 2, torch.pi, 0, torch.pi / 2], dtype=torch.float64)
    coordinates = torch.stack(torch.meshgrid(*[torch.arange(4)] * 4, indexing="ij"), dim=-1).flip(-1)
    phase = torch.exp(1j * (coordinates * momentum).sum(-1))
    amplitudes = torch.tensor([0.2, -0.3, 0.5, 0.7], dtype=torch.complex128)
    generator = torch.diag(torch.tensor([1.0, -1.0, 0.0], dtype=torch.complex128))
    field = (amplitudes[:, None, None, None, None] * phase).real
    epsilon = 1.0e-5
    links = torch.matrix_exp(1j * epsilon * field[..., None, None] * generator)
    coefficients = (0.1, 0.005, -0.0002, 0.00001)
    kernel = PolynomialStoutKernel(coefficients).double()
    actual = kernel(links)[..., 0, 0].imag / epsilon
    response = polynomial_kernel_matrix(momentum, torch.tensor(coefficients, dtype=torch.float64)) @ amplitudes
    expected = (response[:, None, None, None, None] * phase).real
    torch.testing.assert_close(actual, expected, atol=2.0e-6, rtol=2.0e-6)


def test_cached_polynomial_blocking_matches_full_map_and_gradient() -> None:
    links = _random_su3()
    kernel = PolynomialStoutKernel(
        (0.1, 0.002, -0.0001, 0.00001), (0.02, 0.001, -0.0001),
        (0.01, -0.002, 0.0001, -0.00001),
    )
    scales = torch.tensor([12., 12.**2, 12.**3, 12.**4, 12., 12.**2, 12.**3, 12., 12.**2, 12.**3, 12.**4])
    parameter = (torch.cat((kernel.coefficients, kernel.hook_coefficients, kernel.local_coefficients)).detach() * scales).requires_grad_()
    basis = polynomial_blocking_basis(links, hook=True, local=True)
    cached = block_polynomial_basis(basis, parameter)
    full = smear_and_block(links, kernel)
    torch.testing.assert_close(cached, full, atol=3.0e-6, rtol=3.0e-6)
    observable_vector_torch(cached).sum().backward()
    observable_vector_torch(full).sum().backward()
    expected_gradient = torch.cat((kernel.coefficients.grad, kernel.hook_coefficients.grad, kernel.local_coefficients.grad)) / scales
    torch.testing.assert_close(parameter.grad, expected_gradient, atol=3.0e-5, rtol=3.0e-4)


def test_local_polynomial_is_gauge_covariant() -> None:
    links = _random_su3()
    gauge = _random_su3().select(0, 0)
    kernel = PolynomialStoutKernel((0.1, 0.002, -0.0001, 0.00001), local_coefficients=(0.05, -0.002, 0.0002, 0.00001))
    expected = _gauge_transform(kernel(links), gauge)
    actual = kernel(_gauge_transform(links, gauge))
    torch.testing.assert_close(actual, expected, atol=5.0e-4, rtol=5.0e-4)


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


def _gauge_transform(links: torch.Tensor, gauge: torch.Tensor) -> torch.Tensor:
    transformed = []
    for direction in range(4):
        gauge_forward = torch.roll(gauge, shifts=-1, dims=3 - direction)
        transformed.append(gauge @ links[direction] @ gauge_forward.conj().transpose(-2, -1))
    return torch.stack(transformed)


def _nontrivial_transform() -> GaugeEquivariantFieldTransform:
    transform = GaugeEquivariantFieldTransform(hidden_channels=4, step_scale=0.08)
    with torch.no_grad():
        for block in transform.generators:
            for generator in block:
                torch.nn.init.normal_(generator.layers[-1].spatial.weight, std=0.03)
    return transform


def test_field_transform_identity_initialization_matches_naive_blocking() -> None:
    links = _random_su3()
    transform = GaugeEquivariantFieldTransform(hidden_channels=4, initial_output_scale=0.0)

    np.testing.assert_allclose(transform(links).detach().numpy(), links.numpy(), atol=1.0e-6)
    np.testing.assert_allclose(
        transform_and_block(links, transform).detach().numpy(),
        block_links(links).detach().numpy(),
        atol=1.0e-6,
    )


def test_field_transform_is_su3_and_exactly_reversible() -> None:
    links = _random_su3()
    transform = _nontrivial_transform()
    transformed = transform(links)
    recovered = transform.inverse(transformed)

    assert su3_errors(transformed)[0] < 2.0e-5
    assert su3_errors(transformed)[1] < 2.0e-5
    np.testing.assert_allclose(recovered.detach().numpy(), links.numpy(), rtol=2.0e-5, atol=2.0e-5)
    np.testing.assert_allclose(
        transform(transform.inverse(links)).detach().numpy(), links.numpy(), rtol=2.0e-5, atol=2.0e-5
    )


def test_field_transform_is_gauge_covariant_and_has_finite_gradients() -> None:
    links = _random_su3()
    gauge = _random_su3().select(0, 0)
    transform = _nontrivial_transform()
    transformed = transform(links)
    expected = _gauge_transform(transformed, gauge)

    np.testing.assert_allclose(
        transform(_gauge_transform(links, gauge)).detach().numpy(),
        expected.detach().numpy(),
        rtol=3.0e-5,
        atol=3.0e-5,
    )
    loss = observable_vector_torch(transform_and_block(links, transform)).sum()
    loss.backward()
    gradients = [parameter.grad for parameter in transform.parameters() if parameter.requires_grad]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
