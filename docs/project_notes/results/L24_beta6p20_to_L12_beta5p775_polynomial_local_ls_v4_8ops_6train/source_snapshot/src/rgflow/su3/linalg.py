"""Small complex matrix products used in group maps and Wilson paths."""

import torch


@torch.compile(dynamic=True)
def _fused_matrix_product(left, right):
    real = (left[..., 0].unsqueeze(-1)*right[..., 0].unsqueeze(-3)
            - left[..., 1].unsqueeze(-1)*right[..., 1].unsqueeze(-3)).sum(-2)
    imaginary = (left[..., 0].unsqueeze(-1)*right[..., 1].unsqueeze(-3)
                 + left[..., 1].unsqueeze(-1)*right[..., 0].unsqueeze(-3)).sum(-2)
    return torch.stack((real, imaginary), -1)


def matrix_product(left, right):
    # Native autograd composes with nested checkpoint recomputation. Fuse
    # no-grad CUDA proposals and measurements, where tiny cuBLAS batches
    # are much slower. Canonical layouts avoid compiler specializations.
    if not left.is_cuda or torch.is_grad_enabled():
        return left @ right
    # Base-free real tensors also avoid Dynamo's fragile complex-view guards.
    left_flat = torch.view_as_real(left.resolve_conj()).reshape(-1, 3, 3, 2).clone(memory_format=torch.contiguous_format)
    right_flat = torch.view_as_real(right.resolve_conj()).reshape(-1, 3, 3, 2).clone(memory_format=torch.contiguous_format)
    return torch.view_as_complex(_fused_matrix_product(left_flat, right_flat)).reshape(left.shape)
