#!/usr/bin/env python3
"""Plot the static potential and its Coulomb-subtracted linear behavior."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

from plot_settings import (
    BLUE,
    ERRORBAR_CIRCLE_STYLE,
    FIG_WIDTH,
    RED,
    SILVER,
    default_sub_plot,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "artifacts" / "4dsu3" / "L16_beta5p95"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input.resolve()
    result = json.loads((input_dir / "static_potential.json").read_text())
    data = np.load(input_dir / "static_potential.npz")

    potential = data["potential"]
    potential_bootstrap = data["potential_bootstrap"]
    parameter_bootstrap = data["cornell_parameters_bootstrap"]
    potential_error = potential_bootstrap.std(axis=0, ddof=1)
    offset, sigma, alpha, _ = data["cornell_parameters"]

    radii = np.arange(1, len(potential) + 1, dtype=float)
    fit_radii = np.linspace(0.85, radii[-1] + 0.15, 400)
    cornell_curve = offset + sigma * fit_radii - alpha / fit_radii
    cornell_bootstrap = (
        parameter_bootstrap[:, 0, None]
        + parameter_bootstrap[:, 1, None] * fit_radii
        - parameter_bootstrap[:, 2, None] / fit_radii
    )
    cornell_low, cornell_high = np.percentile(cornell_bootstrap, [16, 84], axis=0)

    linear_curve = offset + sigma * fit_radii
    linear_bootstrap = parameter_bootstrap[:, 0, None] + parameter_bootstrap[:, 1, None] * fit_radii
    linear_low, linear_high = np.percentile(linear_bootstrap, [16, 84], axis=0)

    corrected_potential = potential + alpha / radii
    corrected_bootstrap = potential_bootstrap + parameter_bootstrap[:, 2, None] / radii
    corrected_error = corrected_bootstrap.std(axis=0, ddof=1)

    fig, (ax, linear_ax) = default_sub_plot(height_ratio=2)
    fig.set_size_inches(FIG_WIDTH, 6.5)

    ax.fill_between(fit_radii, cornell_low, cornell_high, color=BLUE, alpha=0.18, linewidth=0)
    ax.plot(fit_radii, cornell_curve, color=BLUE, linewidth=1.8, label="Cornell fit")
    ax.plot(
        fit_radii,
        linear_curve,
        color=RED,
        linewidth=1.5,
        linestyle="--",
        label=r"linear asymptote $V_0+\sigma r$",
    )
    ax.errorbar(
        radii[1:],
        potential[1:],
        yerr=potential_error[1:],
        color=BLUE,
        label="Wilson loops",
        **ERRORBAR_CIRCLE_STYLE,
    )
    ax.errorbar(
        radii[:1],
        potential[:1],
        yerr=potential_error[:1],
        color=SILVER,
        label="excluded from Cornell fit",
        **ERRORBAR_CIRCLE_STYLE,
    )
    ax.set_ylabel(r"$aV(r)$", fontsize=18)
    ax.legend(frameon=False, fontsize=12, loc="lower right")
    ax.text(
        0.04,
        0.95,
        rf"$\beta={result['ensemble']['beta']}$, $L^4={result['ensemble']['lattice_size'][0]}^4$"
        "\n"
        rf"$\sigma a^2={sigma:.4f}$, $r_0/a={result['scale']['r0_over_a']:.3f}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
    )

    linear_ax.fill_between(fit_radii, linear_low, linear_high, color=RED, alpha=0.18, linewidth=0)
    linear_ax.plot(fit_radii, linear_curve, color=RED, linewidth=1.8)
    linear_ax.errorbar(
        radii[1:],
        corrected_potential[1:],
        yerr=corrected_error[1:],
        color=RED,
        **ERRORBAR_CIRCLE_STYLE,
    )
    linear_ax.errorbar(
        radii[:1],
        corrected_potential[:1],
        yerr=corrected_error[:1],
        color=SILVER,
        **ERRORBAR_CIRCLE_STYLE,
    )
    linear_ax.set_xlabel(r"$r/a$", fontsize=18)
    linear_ax.set_ylabel(r"$aV(r)+\alpha/(r/a)$", fontsize=15)
    linear_ax.set_xlim(0.7, radii[-1] + 0.3)

    output = (args.output or input_dir / "static_potential.png").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    pdf_output = output.with_suffix(".pdf")
    fig.savefig(pdf_output, bbox_inches="tight")
    print(f"saved {output}")
    print(f"saved {pdf_output}")


if __name__ == "__main__":
    main()
