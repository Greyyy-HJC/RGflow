import torch

from rgflow.su3.observables import (
    DOWNSAMPLING_OBSERVABLES, TRAINING_OBSERVABLES, _polyakov_mean,
    observable_names, observable_vector_torch, path_product,
)


def random_links():
    generator = torch.Generator().manual_seed(95)
    raw = torch.randn(4, 4, 4, 4, 4, 3, 3, dtype=torch.complex128, generator=generator)
    unitary, _ = torch.linalg.qr(raw)
    return unitary * torch.exp(-1j * torch.angle(torch.linalg.det(unitary)) / 3)[..., None, None]


def test_eight_observables_identity_batch_and_legacy_selection():
    torch.set_num_threads(2)
    field = torch.eye(3, dtype=torch.complex128).expand(4, 4, 4, 4, 4, 3, 3).clone()
    names = DOWNSAMPLING_OBSERVABLES
    torch.testing.assert_close(observable_vector_torch(field, names=names), torch.ones(8, dtype=torch.float64))
    field = random_links()
    single = observable_vector_torch(field, names=names)
    other = field @ field
    batched = observable_vector_torch(torch.stack((field, other)), names=names)
    torch.testing.assert_close(batched, torch.stack((single, observable_vector_torch(other, names=names))))
    torch.testing.assert_close(observable_vector_torch(field), single[[names.index(n) for n in observable_names()]])


def test_loop_orientation_average_and_polyakov_fourth_moment():
    field = random_links()
    observed = observable_vector_torch(field, names=DOWNSAMPLING_OBSERVABLES)
    expected = []
    for mu in range(4):
        for nu in range(4):
            if mu == nu:
                continue
            path = [mu] * 3 + [nu] * 2 + [mu + 4] * 3 + [nu + 4] * 2
            expected.append(path_product(field, path).diagonal(dim1=-2, dim2=-1).sum(-1).real.mean() / 3)
    torch.testing.assert_close(observed[4], torch.stack(expected).mean())
    moments = torch.stack([_polyakov_mean(field, mu).abs().square() for mu in range(4)])
    torch.testing.assert_close(observed[-2], moments.mean())
    torch.testing.assert_close(observed[-1], moments.square().mean())
    assert not torch.isclose(observed[-1], moments.mean().square(), atol=1e-12, rtol=1e-3)


def test_eight_observables_gauge_invariance_checkpointing_and_training_gradient():
    field = random_links().requires_grad_(True)
    gauge = random_links()[0]
    transformed = torch.stack([
        gauge @ field[mu] @ torch.roll(gauge, -1, dims=3 - mu).mH for mu in range(4)
    ])
    all_values = observable_vector_torch(field, names=DOWNSAMPLING_OBSERVABLES)
    torch.testing.assert_close(observable_vector_torch(transformed, names=DOWNSAMPLING_OBSERVABLES), all_values, atol=1e-12, rtol=1e-10)
    checkpointed = observable_vector_torch(field, names=DOWNSAMPLING_OBSERVABLES, checkpoint_loops=True)
    torch.testing.assert_close(checkpointed, all_values)
    selected = observable_vector_torch(field, names=TRAINING_OBSERVABLES)
    indices = [DOWNSAMPLING_OBSERVABLES.index(name) for name in TRAINING_OBSERVABLES]
    expected_gradient = torch.autograd.grad(all_values[indices].sum(), field, retain_graph=True)[0]
    gradient = torch.autograd.grad(selected.sum(), field)[0]
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient, expected_gradient)
    torch.testing.assert_close(torch.autograd.grad(checkpointed.sum(), field)[0], torch.autograd.grad(all_values.sum(), field)[0])
