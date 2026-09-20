"""Conditional affine and residual-spline inverse-blocking flows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class AffineFlowConfig:
    lattice_size: int
    coupling_layers: int = 6
    hidden_channels: int = 32
    kernel_size: int = 3
    log_scale_bound: float = 2.0
    model_type: str = "affine"
    spline_layers: int = 4
    spline_bins: int = 8
    spline_tail_bound: float = 4.0


class CircularConditioner(nn.Module):
    def __init__(
        self,
        input_channels: int,
        config: AffineFlowConfig,
        output_channels: int = 2,
    ):
        super().__init__()
        padding = config.kernel_size // 2
        hidden = config.hidden_channels
        self.net = nn.Sequential(
            nn.Conv2d(
                input_channels,
                hidden,
                config.kernel_size,
                padding=padding,
                padding_mode="circular",
            ),
            nn.SiLU(),
            nn.Conv2d(
                hidden,
                hidden,
                config.kernel_size,
                padding=padding,
                padding_mode="circular",
            ),
            nn.SiLU(),
            nn.Conv2d(
                hidden,
                output_channels,
                config.kernel_size,
                padding=padding,
                padding_mode="circular",
            ),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, masked: Tensor, condition: Tensor) -> tuple[Tensor, Tensor]:
        shift, raw_scale = self.net(
            torch.cat((masked[:, None], condition), dim=1)
        ).unbind(dim=1)
        return shift, raw_scale


class CheckerboardAffineCoupling(nn.Module):
    def __init__(
        self,
        condition_channels: int,
        parity: int,
        config: AffineFlowConfig,
    ):
        super().__init__()
        yy, xx = torch.meshgrid(
            torch.arange(config.lattice_size),
            torch.arange(config.lattice_size),
            indexing="ij",
        )
        mask = ((xx + yy) % 2 == parity).to(torch.float32)
        self.register_buffer("mask", mask, persistent=False)
        self.log_scale_bound = config.log_scale_bound
        self.conditioner = CircularConditioner(condition_channels + 1, config)

    def _affine_parameters(
        self, field: Tensor, condition: Tensor
    ) -> tuple[Tensor, Tensor]:
        masked = field * self.mask
        shift, raw_scale = self.conditioner(masked, condition)
        active = 1.0 - self.mask
        log_scale = self.log_scale_bound * torch.tanh(
            raw_scale / self.log_scale_bound
        )
        return shift * active, log_scale * active

    def forward(self, field: Tensor, condition: Tensor) -> tuple[Tensor, Tensor]:
        shift, log_scale = self._affine_parameters(field, condition)
        active = 1.0 - self.mask
        transformed = field * self.mask + active * (
            field * torch.exp(log_scale) + shift
        )
        return transformed, log_scale.sum(dim=(-2, -1))

    def inverse(self, field: Tensor, condition: Tensor) -> tuple[Tensor, Tensor]:
        shift, log_scale = self._affine_parameters(field, condition)
        active = 1.0 - self.mask
        transformed = field * self.mask + active * (
            (field - shift) * torch.exp(-log_scale)
        )
        return transformed, -log_scale.sum(dim=(-2, -1))


def _select(values: Tensor, indices: Tensor) -> Tensor:
    return torch.gather(values, -1, indices[..., None]).squeeze(-1)


def rational_quadratic_spline(
    inputs: Tensor,
    parameters: Tensor,
    *,
    bins: int,
    tail_bound: float,
    inverse: bool = False,
) -> tuple[Tensor, Tensor]:
    """Elementwise monotone rational-quadratic spline with linear tails."""
    minimum_bin = 1.0e-3
    minimum_derivative = 1.0e-3
    widths_raw = parameters[..., :bins]
    heights_raw = parameters[..., bins : 2 * bins]
    derivatives_raw = parameters[..., 2 * bins :]
    widths = minimum_bin + (2.0 * tail_bound - bins * minimum_bin) * torch.softmax(
        widths_raw, dim=-1
    )
    heights = minimum_bin + (2.0 * tail_bound - bins * minimum_bin) * torch.softmax(
        heights_raw, dim=-1
    )
    boundary = inputs.new_ones((*inputs.shape, 1))
    derivatives = torch.cat(
        (
            boundary,
            minimum_derivative + torch.nn.functional.softplus(derivatives_raw),
            boundary,
        ),
        dim=-1,
    )
    zero = inputs.new_zeros((*inputs.shape, 1))
    cumulative_widths = torch.cat((zero, widths), dim=-1).cumsum(dim=-1)
    cumulative_heights = torch.cat((zero, heights), dim=-1).cumsum(dim=-1)
    cumulative_widths = cumulative_widths - tail_bound
    cumulative_heights = cumulative_heights - tail_bound

    inside = (inputs >= -tail_bound) & (inputs <= tail_bound)
    lookup = cumulative_heights if inverse else cumulative_widths
    indices = torch.sum(inputs[..., None] >= lookup[..., 1:], dim=-1).clamp(
        max=bins - 1
    )
    x0 = _select(cumulative_widths[..., :-1], indices)
    y0 = _select(cumulative_heights[..., :-1], indices)
    width = _select(widths, indices)
    height = _select(heights, indices)
    delta = height / width
    derivative_left = _select(derivatives[..., :-1], indices)
    derivative_right = _select(derivatives[..., 1:], indices)

    if inverse:
        shifted = inputs - y0
        a = shifted * (derivative_left + derivative_right - 2.0 * delta)
        a += height * (delta - derivative_left)
        b = height * derivative_left
        b -= shifted * (derivative_left + derivative_right - 2.0 * delta)
        c = -delta * shifted
        discriminant = (b.square() - 4.0 * a * c).clamp_min(0.0)
        theta = (2.0 * c) / (-b - torch.sqrt(discriminant))
        theta = torch.where(torch.abs(a) < 1.0e-8, -c / b, theta).clamp(0.0, 1.0)
    else:
        theta = ((inputs - x0) / width).clamp(0.0, 1.0)

    theta_one_minus = theta * (1.0 - theta)
    denominator = delta + (
        derivative_left + derivative_right - 2.0 * delta
    ) * theta_one_minus
    numerator = height * (
        delta * theta.square() + derivative_left * theta_one_minus
    )
    outputs = y0 + numerator / denominator
    derivative_numerator = delta.square() * (
        derivative_right * theta.square()
        + 2.0 * delta * theta_one_minus
        + derivative_left * (1.0 - theta).square()
    )
    logabsdet = torch.log(derivative_numerator) - 2.0 * torch.log(denominator)
    if inverse:
        outputs = x0 + theta * width
        logabsdet = -logabsdet
    return (
        torch.where(inside, outputs, inputs),
        torch.where(inside, logabsdet, torch.zeros_like(logabsdet)),
    )


class CheckerboardSplineCoupling(nn.Module):
    def __init__(self, condition_channels: int, parity: int, config: AffineFlowConfig):
        super().__init__()
        yy, xx = torch.meshgrid(
            torch.arange(config.lattice_size),
            torch.arange(config.lattice_size),
            indexing="ij",
        )
        self.register_buffer(
            "mask", ((xx + yy) % 2 == parity).to(torch.float32), persistent=False
        )
        self.bins = config.spline_bins
        self.tail_bound = config.spline_tail_bound
        self.conditioner = CircularConditioner(
            condition_channels + 1, config, 3 * config.spline_bins - 1
        )
        derivative_bias = math.log(math.expm1(1.0 - 1.0e-3))
        with torch.no_grad():
            self.conditioner.net[-1].bias[2 * self.bins :].fill_(derivative_bias)

    def _spline_parameters(self, field: Tensor, condition: Tensor) -> Tensor:
        values = self.conditioner.net(
            torch.cat(((field * self.mask)[:, None], condition), dim=1)
        )
        return values.movedim(1, -1)

    def _transform(
        self, field: Tensor, condition: Tensor, inverse: bool
    ) -> tuple[Tensor, Tensor]:
        transformed, logdet = rational_quadratic_spline(
            field,
            self._spline_parameters(field, condition),
            bins=self.bins,
            tail_bound=self.tail_bound,
            inverse=inverse,
        )
        active = 1.0 - self.mask
        output = field * self.mask + active * transformed
        return output, (active * logdet).sum(dim=(-2, -1))

    def forward(self, field: Tensor, condition: Tensor) -> tuple[Tensor, Tensor]:
        return self._transform(field, condition, False)

    def inverse(self, field: Tensor, condition: Tensor) -> tuple[Tensor, Tensor]:
        return self._transform(field, condition, True)


class DetailStage(nn.Module):
    def __init__(self, condition_channels: int, config: AffineFlowConfig):
        super().__init__()
        layers: list[nn.Module] = [
            CheckerboardAffineCoupling(condition_channels, layer % 2, config)
            for layer in range(config.coupling_layers)
        ]
        if config.model_type == "residual-spline":
            layers.extend(
                CheckerboardSplineCoupling(condition_channels, layer % 2, config)
                for layer in range(config.spline_layers)
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, latent: Tensor, condition: Tensor) -> tuple[Tensor, Tensor]:
        field = latent
        logdet = latent.new_zeros(latent.shape[0])
        for layer in self.layers:
            field, contribution = layer(field, condition)
            logdet += contribution
        return field, logdet

    def inverse(self, detail: Tensor, condition: Tensor) -> tuple[Tensor, Tensor]:
        field = detail
        logdet = detail.new_zeros(detail.shape[0])
        for layer in reversed(self.layers):
            field, contribution = layer.inverse(field, condition)
            logdet += contribution
        return field, logdet


class ConditionalAffineFlow(nn.Module):
    """Autoregressively generate d01, d10, and d11 conditioned on coarse."""

    def __init__(self, config: AffineFlowConfig):
        super().__init__()
        self.config = config
        self.stages = nn.ModuleList(
            DetailStage(condition_channels, config)
            for condition_channels in (1, 2, 3)
        )

    @staticmethod
    def _condition(coarse: Tensor, details: Tensor, stage: int) -> Tensor:
        if stage == 0:
            return coarse[:, None]
        return torch.cat((coarse[:, None], details[:, :stage]), dim=1)

    def forward(self, coarse: Tensor, latent: Tensor) -> tuple[Tensor, Tensor]:
        details = coarse.new_zeros((coarse.shape[0], 3, *coarse.shape[-2:]))
        logdet = coarse.new_zeros(coarse.shape[0])
        for stage, flow in enumerate(self.stages):
            detail, contribution = flow(
                latent[:, stage],
                self._condition(coarse, details, stage),
            )
            details[:, stage] = detail
            logdet += contribution
        return details, logdet

    def inverse(self, coarse: Tensor, details: Tensor) -> tuple[Tensor, Tensor]:
        latent = torch.empty_like(details)
        logdet = coarse.new_zeros(coarse.shape[0])
        for stage, flow in enumerate(self.stages):
            value, contribution = flow.inverse(
                details[:, stage],
                self._condition(coarse, details, stage),
            )
            latent[:, stage] = value
            logdet += contribution
        return latent, logdet

    def log_prob(self, coarse: Tensor, details: Tensor) -> Tensor:
        latent, inverse_logdet = self.inverse(coarse, details)
        log_base = -0.5 * (
            latent.square() + math.log(2.0 * math.pi)
        ).sum(dim=(1, 2, 3))
        return log_base + inverse_logdet

    def sample(
        self,
        coarse: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        latent = torch.randn(
            (coarse.shape[0], 3, *coarse.shape[-2:]),
            dtype=coarse.dtype,
            device=coarse.device,
            generator=generator,
        )
        details, logdet = self.forward(coarse, latent)
        log_base = -0.5 * (
            latent.square() + math.log(2.0 * math.pi)
        ).sum(dim=(1, 2, 3))
        return details, log_base - logdet, latent

    def configuration(self) -> dict[str, Any]:
        return asdict(self.config)


def model_from_checkpoint(
    checkpoint: dict[str, Any],
    device: torch.device | str = "cpu",
) -> ConditionalAffineFlow:
    values = dict(checkpoint["config"]["model"])
    values.setdefault("model_type", checkpoint.get("model_type", "affine"))
    config = AffineFlowConfig(**values)
    model = ConditionalAffineFlow(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model
