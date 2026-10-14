import torch
import pytest

from rgflow.su3.downsampling import block_links, su3_errors
from rgflow.su3.upsampling import (
    LEGACY_COUPLING, STABLE_COUPLING, SU3ConditionalFlow, constrained_haar_lift,
    _cpu_staple_frame, covariant_staple_frame, coupling_mask, haar_links, quaternion_transport,
    transverse_staples, flow_roundtrip_check,
)
from rgflow.su3.downsampling import PolynomialStoutKernel
from rgflow.su3.observables import observable_vector_torch, path_product


def _gauge_transform(links, gauge):
    return torch.stack([
        gauge @ links[:, mu] @ torch.roll(gauge, -1, 4-mu).conj().transpose(-2, -1)
        for mu in range(4)
    ], 1)


@pytest.mark.parametrize("constrained", [False, True])
@pytest.mark.parametrize("device,dtype,tolerance", [
    ("cpu", torch.complex128, 1.e-10),
    pytest.param("cuda", torch.complex64, 2.e-4,
                 marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")),
])
def test_flow_is_gauge_equivariant_with_invariant_density(constrained, device, dtype, tolerance):
    torch.manual_seed(381)
    links = haar_links((2, 4, 4, 4, 4, 4), device=device, dtype=dtype)
    gauge = haar_links((2, 4, 4, 4, 4), device=device, dtype=dtype)
    flow = SU3ConditionalFlow(sweeps=2, constrained=constrained).to(device)
    flow = flow.double() if dtype == torch.complex128 else flow
    for network in flow.scales:
        torch.nn.init.normal_(network[-1].weight, std=.04)
    flow.eval()
    with torch.no_grad():
        output, density = flow(links)
        transformed, transformed_density = flow(_gauge_transform(links, gauge))
        recovered, inverse_density = flow.inverse(output)
        recovered_gauge, inverse_density_gauge = flow.inverse(_gauge_transform(output, gauge))
    torch.testing.assert_close(transformed, _gauge_transform(output, gauge), rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(density, transformed_density, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(recovered, links, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(recovered_gauge, _gauge_transform(recovered, gauge), rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(inverse_density, inverse_density_gauge, rtol=tolerance, atol=tolerance)
    if constrained:
        torch.testing.assert_close(block_links(output), block_links(links), rtol=tolerance, atol=tolerance)


def test_frame_phases_cancel_in_coupling(monkeypatch):
    torch.manual_seed(625)
    links = haar_links((1, 4, 4, 4, 4, 4), dtype=torch.complex128)
    flow = SU3ConditionalFlow(sweeps=1).double().eval()
    torch.nn.init.normal_(flow.scales[0][-1].weight, std=.1)
    expected, expected_density = flow._direction(links, 0, 0, 0, False)

    def phased(staples):
        frame, gate = covariant_staple_frame(staples)
        phases = torch.exp(1j*torch.randn(*gate.shape, 3, dtype=torch.float64))
        return frame*phases[..., None, :], gate

    monkeypatch.setattr('rgflow.su3.upsampling.covariant_staple_frame', phased)
    actual, actual_density = flow._direction(links, 0, 0, 0, False)
    torch.testing.assert_close(actual, expected, atol=1.e-12, rtol=1.e-12)
    torch.testing.assert_close(actual_density, expected_density, atol=1.e-12, rtol=1.e-12)


def test_staple_context_does_not_depend_on_active_links():
    links = haar_links((1, 4, 4, 4, 4, 4), dtype=torch.complex128)
    mask = coupling_mask((4,)*4, 0, 0, constrained=False, device='cpu')
    changed = links.clone()
    changed[:, 0, mask] = haar_links((1, int(mask.sum())), dtype=torch.complex128)
    for distance in (1, 3):
        torch.testing.assert_close(transverse_staples(changed, 0, mask, distance),
                                   transverse_staples(links, 0, mask, distance))


@pytest.mark.parametrize("gap", [0., 1.e-8, 1.e-3])
@pytest.mark.parametrize("device,dtype", [
    ("cpu", torch.complex128),
    pytest.param("cuda", torch.complex64,
                 marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")),
])
def test_degenerate_frame_is_safe_and_near_degeneracy_is_suppressed(gap, device, dtype):
    context = torch.diag(torch.tensor([1., 1.+gap, 2.], dtype=dtype, device=device)).requires_grad_()
    frame, gate = covariant_staple_frame(context)
    assert torch.isfinite(frame).all() and torch.isfinite(gate).all()
    if context[1, 1].real == context[0, 0].real:
        assert gate == 0.
    else:
        assert 0. < gate < 1.
        (gate + (frame.conj().transpose(-2, -1) @ context @ frame).real.sum()).backward()
        assert torch.isfinite(context.grad).all()
    zero_frame, zero_gate = covariant_staple_frame(torch.zeros_like(context))
    assert zero_gate == 0.
    torch.testing.assert_close(zero_frame, torch.eye(3, dtype=context.dtype, device=device))


def test_identity_field_has_identity_coupling_and_finite_gradients():
    links = torch.eye(3, dtype=torch.complex128).expand(1, 4, 4, 4, 4, 4, 3, 3).clone().requires_grad_()
    flow = SU3ConditionalFlow(sweeps=1).double().eval()
    output, density = flow(links)
    torch.testing.assert_close(output, links)
    torch.testing.assert_close(density, torch.zeros_like(density))
    (output.real.sum()+density.sum()).backward()
    assert torch.isfinite(links.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in flow.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires analytic CUDA frames")
def test_analytic_cuda_frame_matches_cpu_spectral_map_and_gradient():
    torch.manual_seed(193)
    context = torch.randn(20, 3, 3, dtype=torch.complex128, device='cuda').requires_grad_()
    probe = torch.randn_like(context)
    phases = torch.diag(torch.exp(1j*torch.tensor([.2, -.5, .3], dtype=torch.float64, device='cuda')))
    frame, gate = covariant_staple_frame(context)
    reference, reference_gate = _cpu_staple_frame(context)
    mapped = frame @ phases @ frame.conj().transpose(-2, -1)
    expected = reference @ phases @ reference.conj().transpose(-2, -1)
    torch.testing.assert_close(mapped, expected, rtol=1.e-10, atol=1.e-10)
    torch.testing.assert_close(gate, reference_gate, rtol=1.e-10, atol=1.e-10)
    actual_gradient = torch.autograd.grad((mapped*probe).real.sum()+gate.sum(), context)[0]
    reference_gradient = torch.autograd.grad((expected*probe).real.sum()+reference_gate.sum(), context)[0]
    torch.testing.assert_close(actual_gradient, reference_gradient, rtol=1.e-8, atol=1.e-8)


def test_legacy_checkpoint_keeps_fixed_colour_semantics():
    flow = SU3ConditionalFlow(sweeps=1, coupling=LEGACY_COUPLING)
    assert not flow.gauge_equivariant
    links = haar_links((4, 4, 4, 4, 4))
    output, density = flow(links)
    recovered, inverse_density = flow.inverse(output)
    torch.testing.assert_close(recovered, links, atol=2.e-5, rtol=2.e-5)
    torch.testing.assert_close(density, -inverse_density, atol=2.e-4, rtol=2.e-5)


def test_conditional_lift_retains_coarse_products_and_noise():
    coarse = haar_links((4, 2, 2, 2, 2), dtype=torch.complex128)
    first, second = constrained_haar_lift(coarse), constrained_haar_lift(coarse)
    torch.testing.assert_close(block_links(first), coarse)
    assert not torch.allclose(first, second)
    assert max(su3_errors(first)) < 1.e-12


def test_two_flow_generation_with_synchronously_transformed_haar_details(monkeypatch):
    torch.manual_seed(902)
    coarse = haar_links((1, 4, 2, 2, 2, 2), dtype=torch.complex128)
    noise = haar_links((1, 4, 4, 4, 4, 4), dtype=torch.complex128)
    gauge = haar_links((1, 4, 4, 4, 4), dtype=torch.complex128)
    coarse_gauge = gauge[:, ::2, ::2, ::2, ::2]
    flows = [SU3ConditionalFlow(sweeps=2, constrained=True).double().eval(),
             SU3ConditionalFlow(sweeps=1).double().eval()]
    monkeypatch.setattr('rgflow.su3.upsampling.haar_links', lambda *a, **kw: noise.clone())
    base = constrained_haar_lift(coarse)
    transformed_noise = _gauge_transform(noise, gauge)
    monkeypatch.setattr('rgflow.su3.upsampling.haar_links', lambda *a, **kw: transformed_noise.clone())
    transformed_base = constrained_haar_lift(_gauge_transform(coarse, coarse_gauge))
    torch.testing.assert_close(transformed_base, _gauge_transform(base, gauge), atol=1.e-12, rtol=1.e-12)
    with torch.no_grad():
        output, transformed = base, transformed_base
        for flow in flows:
            output, density = flow(output)
            transformed, transformed_density = flow(transformed)
            torch.testing.assert_close(transformed, _gauge_transform(output, gauge), atol=1.e-10, rtol=1.e-10)
            torch.testing.assert_close(transformed_density, density, atol=1.e-10, rtol=1.e-10)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CPU-offloaded CUDA checkpoints")
@pytest.mark.parametrize("constrained", [False, True])
@pytest.mark.parametrize("input_gradient", [False, True])
def test_offloaded_checkpoints_preserve_outputs_and_exact_gradients(constrained, input_gradient):
    torch.manual_seed(134)
    coarse = haar_links((4, 2, 2, 2, 2), device="cuda", dtype=torch.complex128)
    source = constrained_haar_lift(coarse)
    probe = torch.randn_like(source)
    flow = SU3ConditionalFlow(sweeps=2, constrained=constrained, coupling=STABLE_COUPLING).cuda()
    with torch.no_grad():
        for network in flow.scales:
            network[-1].weight.normal_(std=.03)
    outputs, gradients = [], []
    for training in [False, True]:
        flow.train(training)
        flow.zero_grad(set_to_none=True)
        links = source.detach().clone().requires_grad_(input_gradient)
        output, density = flow(links)
        ((output*probe).real.mean()+.003*density.sum()).backward()
        outputs.append((output.detach(), density.detach()))
        gradients.append([p.grad.clone() for p in flow.parameters()]
                         + ([links.grad.clone()] if input_gradient else []))
    for expected, actual in zip(outputs[0], outputs[1]):
        torch.testing.assert_close(actual, expected, rtol=1.e-11, atol=1.e-11)
    for expected, actual in zip(gradients[0], gradients[1]):
        torch.testing.assert_close(actual, expected, rtol=1.e-9, atol=1.e-10)


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


def test_full_flow_stability_check_rejects_deep_contractions_without_changing_rng():
    flow = SU3ConditionalFlow(sweeps=8, constrained=True, initial_scale=-.18).eval()
    state = torch.random.get_rng_state().clone()
    report = flow_roundtrip_check(flow)
    assert report["passed"], report
    torch.testing.assert_close(torch.random.get_rng_state(), state, rtol=0, atol=0)
    with torch.no_grad():
        for network in flow.scales:
            network[-1].bias[:3].fill_(-2.)
            network[-1].bias[3:].fill_(2.)
    report = flow_roundtrip_check(flow)
    assert not report["passed"], report


def test_cubic_transport_remains_invertible_at_parameter_bounds_and_poles():
    real = torch.linspace(-1., 1., 2001, dtype=torch.float64)
    a = torch.complex(real, torch.sqrt(1-real.square()))
    b = torch.zeros_like(a)
    for shift in [-2., 2.]:
        for deformation in [-20., 20.]:
            scale, shape = torch.full_like(real, shift), torch.full_like(real, deformation)
            aa, bb, forward = quaternion_transport(a, b, scale, shape)
            recovered_a, recovered_b, inverse = quaternion_transport(aa, bb, scale, shape, inverse=True)
            torch.testing.assert_close(recovered_a, a, rtol=1.e-10, atol=1.e-10)
            torch.testing.assert_close(recovered_b, b, rtol=1.e-10, atol=1.e-10)
            torch.testing.assert_close(forward, -inverse, rtol=1.e-10, atol=1.e-10)


def test_stability_check_uses_supplied_coarse_lattice():
    flow = SU3ConditionalFlow(sweeps=1, constrained=True, coupling=STABLE_COUPLING).eval()
    coarse = haar_links((1, 4, 3, 3, 3, 3), dtype=torch.complex128)
    report = flow_roundtrip_check(flow, coarse)
    assert report["passed"], report
    assert report["records"][-1]["lattice_size"] == [6, 6, 6, 6]


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"))])
def test_stable_coupling_retains_double_precision_for_float_inputs(device):
    flow = SU3ConditionalFlow(sweeps=2, constrained=True, coupling=STABLE_COUPLING).to(device).eval()
    coarse = haar_links((1, 4, 2, 2, 2, 2), device=device)
    with torch.no_grad():
        output, density = flow.sample(coarse)
        recovered, inverse_density = flow.inverse(output)
        regenerated, forward = flow(recovered)
    assert output.dtype == torch.complex128
    assert density.dtype == torch.float64
    torch.testing.assert_close(block_links(output), coarse.to(torch.complex128), rtol=2.e-6, atol=2.e-6)
    torch.testing.assert_close(regenerated, output, rtol=1.e-10, atol=1.e-10)
    torch.testing.assert_close(forward, -inverse_density, rtol=1.e-10, atol=1.e-10)
    report = flow_roundtrip_check(flow)
    assert report["passed"], report


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
