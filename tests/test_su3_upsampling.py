import torch
import pytest

from rgflow.su3.downsampling import block_links, su3_errors
from rgflow.su3.upsampling import SU3ConditionalFlow, constrained_haar_lift, haar_links, quaternion_transport
from rgflow.su3.downsampling import PolynomialStoutKernel
from rgflow.su3.observables import observable_vector_torch, path_product


def test_conditional_lift_retains_coarse_products_and_noise():
    coarse = haar_links((4, 2, 2, 2, 2), dtype=torch.complex128)
    first, second = constrained_haar_lift(coarse), constrained_haar_lift(coarse)
    torch.testing.assert_close(block_links(first), coarse)
    assert not torch.allclose(first, second)
    assert max(su3_errors(first)) < 1.e-12


def test_flow_roundtrip_logdet_constraint_and_gradient():
    torch.manual_seed(123)
    coarse = haar_links((4, 2, 2, 2, 2), dtype=torch.complex128)
    source = constrained_haar_lift(coarse)
    flow = SU3ConditionalFlow(sweeps=2, constrained=True).double().eval()
    target, forward_logdet = flow(source)
    recovered, inverse_logdet = flow.inverse(target)
    torch.testing.assert_close(recovered, source, rtol=1.e-10, atol=1.e-10)
    torch.testing.assert_close(forward_logdet, -inverse_logdet, rtol=1.e-10, atol=1.e-10)
    torch.testing.assert_close(block_links(target), coarse)
    assert max(su3_errors(target)) < 1.e-10
    assert abs(float(forward_logdet.detach())) > 1.
    (-flow.log_prob(target.detach())).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in flow.parameters())


def test_haar_jacobian_against_sphere_tangent_determinant():
    # Stereographic radial map acts on a three-dimensional SU(2) fibre.
    value = torch.tensor([.2, -.4, .6], dtype=torch.float64, requires_grad=True)
    scale = torch.tensor(-.7, dtype=torch.float64)

    def sphere(coordinate):
        radius2 = coordinate.square().sum()
        return torch.cat(((1-radius2).reshape(1), 2*coordinate)) / (1+radius2)

    before = torch.autograd.functional.jacobian(sphere, value)
    after = torch.autograd.functional.jacobian(lambda x: sphere(x * scale.exp()), value)
    numeric = .5 * (torch.linalg.slogdet(after.T @ after)[1]
                    - torch.linalg.slogdet(before.T @ before)[1])
    q0 = sphere(value)[0]
    analytic = 3 * (scale + torch.log(torch.tensor(2.))
                    - torch.log((1+q0)+torch.exp(2*scale)*(1-q0)))
    torch.testing.assert_close(numeric, analytic)


def test_cubic_quaternion_transport_roundtrip_and_volume():
    torch.manual_seed(13)
    q = torch.randn(100, 4, dtype=torch.float64)
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    a, b = torch.complex(q[:, 0], q[:, 1]), torch.complex(q[:, 2], q[:, 3])
    scale = torch.linspace(-1., 1., 100, dtype=torch.float64)
    shape = torch.linspace(-3., 3., 100, dtype=torch.float64)
    na, nb, forward = quaternion_transport(a, b, scale, shape)
    ra, rb, backward = quaternion_transport(na, nb, scale, shape, inverse=True)
    torch.testing.assert_close(ra, a, atol=1.e-10, rtol=1.e-10)
    torch.testing.assert_close(rb, b, atol=1.e-10, rtol=1.e-10)
    torch.testing.assert_close(forward, -backward, atol=1.e-10, rtol=1.e-10)
    value = torch.tensor([.2, -.4, .6], dtype=torch.float64)

    def mapped(x, transform):
        radius2 = x.square().sum()
        current = torch.cat(((1-radius2).reshape(1), 2*x))/(1+radius2)
        if not transform:
            return current
        aa, bb, _ = quaternion_transport(torch.complex(current[0], current[1]),
                                         torch.complex(current[2], current[3]), scale[0], shape[-1])
        return torch.stack((aa.real, aa.imag, bb.real, bb.imag))

    before = torch.autograd.functional.jacobian(lambda x: mapped(x, False), value)
    after = torch.autograd.functional.jacobian(lambda x: mapped(x, True), value)
    numeric = .5*(torch.linalg.slogdet(after.T@after)[1]-torch.linalg.slogdet(before.T@before)[1])
    current = mapped(value, False)
    _, _, analytic = quaternion_transport(torch.complex(current[0], current[1]),
                                         torch.complex(current[2], current[3]), scale[0], shape[-1])
    torch.testing.assert_close(numeric, analytic, atol=1.e-10, rtol=1.e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA projection")
def test_cpu_projection_preserves_cuda_smoothed_field():
    links = haar_links((4, 4, 4, 4, 4), device="cuda")
    kernel = PolynomialStoutKernel((.08, -.02, -.003, -.00005), (.04, .02, .003)).cuda()
    with torch.no_grad():
        cpu_projection = kernel(links, projection_device="cpu")
        cuda_projection = kernel(links)
    assert cpu_projection.device == links.device
    torch.testing.assert_close(cpu_projection, cuda_projection, rtol=3.e-5, atol=3.e-5)


def test_full_su3_coupling_jacobian_matches_tangent_volume(monkeypatch):
    # Check the implementation on all eight SU(3) tangent directions, including
    # the neural conditioner and all three overlapping SU(2) fibre maps.
    torch.manual_seed(41)
    links = haar_links((1, 4, 4, 4, 4, 4), dtype=torch.complex128)
    flow = SU3ConditionalFlow(sweeps=1, initial_scale=-.2).double().eval()
    torch.nn.init.normal_(flow.scales[0][-1].weight, std=.1)
    mask = torch.zeros((4, 4, 4, 4), dtype=torch.bool)
    mask[0, 0, 0, 0] = True
    monkeypatch.setattr('rgflow.su3.upsampling.coupling_mask', lambda *args, **kwargs: mask)
    generators = []
    for first, second in [(0, 1), (0, 2), (1, 2)]:
        real = torch.zeros((3, 3), dtype=torch.complex128)
        real[first, second] = real[second, first] = 1.
        imaginary = torch.zeros_like(real)
        imaginary[first, second], imaginary[second, first] = -1j, 1j
        generators.extend([real, imaginary])
    generators.extend([torch.diag(torch.tensor([1., -1., 0.], dtype=torch.complex128)),
                       torch.diag(torch.tensor([1., 1., -2.], dtype=torch.complex128)) / 3.**.5])
    generators = torch.stack(generators)
    coordinate = torch.zeros(8, dtype=torch.float64, requires_grad=True)

    def mapped(value, transform):
        current = links.clone()
        current[0, 0, 0, 0, 0, 0] = torch.matrix_exp(
            1j * (value[:, None, None] * generators).sum(0)
        ) @ links[0, 0, 0, 0, 0, 0]
        if transform:
            current, _ = flow._direction(current, 0, 0, 0, False)
        return torch.view_as_real(current[0, 0, 0, 0, 0, 0]).reshape(-1)

    before = torch.autograd.functional.jacobian(lambda x: mapped(x, False), coordinate)
    after = torch.autograd.functional.jacobian(lambda x: mapped(x, True), coordinate)
    measured = .5 * (torch.linalg.slogdet(after.T @ after)[1]
                     - torch.linalg.slogdet(before.T @ before)[1])
    _, logdet = flow._direction(links, 0, 0, 0, False)
    torch.testing.assert_close(measured, logdet[0], rtol=1.e-8, atol=1.e-8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires fused CUDA products")
def test_cuda_coupling_roundtrip_and_backpropagation():
    coarse = haar_links((4, 2, 2, 2, 2), device="cuda", dtype=torch.complex128)
    base = constrained_haar_lift(coarse)
    flow = SU3ConditionalFlow(sweeps=2, constrained=True).cuda().double().eval()
    target, forward = flow(base)
    with torch.no_grad():
        fused_target, fused_logdet = flow(base)
    torch.testing.assert_close(fused_target, target, rtol=1.e-9, atol=1.e-9)
    torch.testing.assert_close(fused_logdet, forward, rtol=1.e-9, atol=1.e-9)
    recovered, backward = flow.inverse(target)
    torch.testing.assert_close(recovered, base, rtol=1.e-9, atol=1.e-9)
    torch.testing.assert_close(forward, -backward, rtol=1.e-9, atol=1.e-9)
    (-flow.log_prob(target.detach())).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in flow.parameters())


def test_detached_operator_vjp_matches_end_to_end_gradient():
    base = haar_links((4, 4, 4, 4, 4), dtype=torch.complex128)
    flow = SU3ConditionalFlow(sweeps=2).double().eval()
    coefficients = torch.tensor([.1, -.4, .8, .5], dtype=torch.float64)
    field, _ = flow(base)
    loss = (observable_vector_torch(field)*coefficients).sum()
    direct = torch.autograd.grad(loss, tuple(flow.parameters()))
    with torch.no_grad():
        field, _ = flow(base)
    field.requires_grad_(True)
    gradient = torch.autograd.grad((observable_vector_torch(field)*coefficients).sum(), field)[0]
    replay, density = flow(base, compute_logdet=False)
    assert density is None
    streamed = torch.autograd.grad(replay, tuple(flow.parameters()), grad_outputs=gradient)
    for left, right in zip(direct, streamed):
        torch.testing.assert_close(left, right, rtol=1.e-9, atol=1.e-9)


def test_checkpointed_loop_measurements_and_gradients_match():
    field = haar_links((4, 4, 4, 4, 4), dtype=torch.complex128).requires_grad_(True)
    plain = observable_vector_torch(field)
    checkpointed = observable_vector_torch(field, checkpoint_loops=True)
    torch.testing.assert_close(checkpointed, plain)
    coefficients = torch.tensor([.2, .3, .4, .5])
    direct = torch.autograd.grad((plain*coefficients).sum(), field)[0]
    streamed = torch.autograd.grad((checkpointed*coefficients).sum(), field)[0]
    torch.testing.assert_close(streamed, direct, rtol=1.e-10, atol=1.e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires fused CUDA measurements")
def test_fused_cuda_observables_match_native_products():
    field = haar_links((2, 4, 4, 4, 4, 4), device='cuda', dtype=torch.complex128)
    native = observable_vector_torch(field)
    with torch.no_grad():
        fused = observable_vector_torch(field)
    torch.testing.assert_close(fused, native, rtol=1.e-10, atol=1.e-12)


def test_polyakov_fixed_origin_matches_all_origins_and_gradient():
    field = haar_links((2, 4, 4, 4, 4, 4), dtype=torch.complex128).requires_grad_(True)
    moments = []
    for direction in range(4):
        product = path_product(field, (direction,)*4)
        trace = torch.diagonal(product, dim1=-2, dim2=-1).sum(-1)/3.
        moments.append(trace.mean((1, 2, 3, 4)).abs().square())
    all_origins = torch.stack(moments).mean(0)
    fixed_origin = observable_vector_torch(field)[:, -1]
    torch.testing.assert_close(fixed_origin, all_origins, rtol=1.e-10, atol=1.e-12)
    direct = torch.autograd.grad(all_origins.sum(), field)[0]
    fixed = torch.autograd.grad(fixed_origin.sum(), field)[0]
    torch.testing.assert_close(fixed, direct, rtol=1.e-10, atol=1.e-12)
