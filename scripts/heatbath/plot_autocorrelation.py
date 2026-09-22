#!/usr/bin/env python3
"""Plot Monte Carlo histories and autocorrelations of saved gauge configurations."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from plot_settings import BLUE, FIG_WIDTH, GREEN, RED, VIOLET, apply_plot_style


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "artifacts" / "4dsu3"
ENSEMBLES = ("L12_beta5p80", "L16_beta5p95", "L24_beta6p20")
COLORS = (GREEN, BLUE, RED)


def autocorrelation(values: np.ndarray, max_lag: int) -> np.ndarray:
    centered = values - values.mean()
    normalization = centered @ centered
    return np.array(
        [1.0]
        + [centered[:-lag] @ centered[lag:] / normalization for lag in range(1, max_lag + 1)]
    )


def integrated_time(acf: np.ndarray) -> float:
    """Estimate tau_int with Geyer's initial-positive-pair truncation."""
    correlation_sum = 0.0
    for lag in range(1, len(acf), 2):
        pair_sum = acf[lag : lag + 2].sum()
        if pair_sum <= 0:
            break
        correlation_sum += pair_sum
    return max(0.5, 0.5 + correlation_sum)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--ensemble", choices=ENSEMBLES, help="plot one ensemble into its own directory")
    parser.add_argument("--max-lag", type=int, default=12)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def observable_summary(values: np.ndarray, acf: np.ndarray, separation: int, name: str) -> dict:
    tau_int = integrated_time(acf)
    inefficiency = 2.0 * tau_int
    return {
        "observable": name,
        "saved_configurations": len(values),
        "compound_steps_between_saved_configurations": separation,
        "lag_1_autocorrelation": float(acf[1]),
        "integrated_autocorrelation_time_saved_lags": float(tau_int),
        "statistical_inefficiency": float(inefficiency),
        "effective_sample_size": float(len(values) / inefficiency),
    }


def plot_ensemble(root: Path, name: str, color: str, max_lag: int, dpi: int, output: Path | None) -> None:
    directory = root / name
    manifest = json.loads((directory / "ensemble.json").read_text())
    values = np.asarray([record["plaquette"] for record in manifest["configurations"]])
    steps = np.asarray([record["step"] for record in manifest["configurations"]])
    acf = autocorrelation(values, max_lag)
    separation = manifest["separation_steps"]
    label = rf"$L={manifest['lattice_size'][0]},\ \beta={manifest['beta']:.2f}$"
    summaries = {"plaquette": observable_summary(values, acf, separation, "plaquette")}

    wilson_path = directory / "static_potential.npz"
    w22_acf = None
    if wilson_path.exists():
        w22 = np.load(wilson_path)["wilson_loops"][:, 1, 1]
        w22_acf = autocorrelation(w22, max_lag)
        summaries["W22"] = observable_summary(w22, w22_acf, separation, "20-step APE-smeared W(2,2)")

    fig, (history_ax, acf_ax) = plt.subplots(2, 1, figsize=(FIG_WIDTH * 1.15, 7.0))
    for axis in (history_ax, acf_ax):
        axis.tick_params(direction="in", top=True, right=True, labelsize=16)
        axis.grid(linestyle=":")

    standardized = (values - values.mean()) / values.std(ddof=1)
    history_ax.plot(steps, standardized, marker="o", markersize=3.5, linewidth=1.0, color=color, label=label)
    history_ax.axhline(0.0, color="black", linewidth=0.8)
    history_ax.set_ylabel(r"$(P-\langle P\rangle)/s_P$", fontsize=17)
    history_ax.set_xlabel("compound update step", fontsize=17)
    history_ax.legend(frameon=False, fontsize=13, loc="lower center", bbox_to_anchor=(0.5, 1.0))

    lags = np.arange(1, max_lag + 1)
    sample_count = len(values)
    pointwise_confidence = 1.96 / np.sqrt(sample_count)
    series_count = 2 if w22_acf is not None else 1
    comparisons = series_count * max_lag
    simultaneous_confidence = norm.ppf(1.0 - 0.05 / (2 * comparisons)) / np.sqrt(sample_count)
    acf_ax.axhspan(
        -simultaneous_confidence,
        simultaneous_confidence,
        color="#E0E0E0",
        alpha=0.35,
        label=r"simultaneous $95\%$ band",
    )
    acf_ax.axhspan(
        -pointwise_confidence,
        pointwise_confidence,
        color="#AFAFAF",
        alpha=0.35,
        label=r"pointwise $95\%$ band",
    )
    inefficiency = summaries["plaquette"]["statistical_inefficiency"]
    acf_ax.plot(
        lags,
        acf[1:],
        marker="o",
        markersize=4,
        linewidth=1.2,
        color=color,
        label=rf"{label}: $P$, $g={inefficiency:.2f}$",
    )
    if w22_acf is not None:
        w22_inefficiency = summaries["W22"]["statistical_inefficiency"]
        acf_ax.plot(
            lags,
            w22_acf[1:],
            marker="s",
            markersize=4,
            linewidth=1.2,
            linestyle="--",
            color=VIOLET,
            label=rf"smeared $W(2,2)$, $g={w22_inefficiency:.2f}$",
        )
    acf_ax.axhline(0.0, color="black", linewidth=0.8)
    acf_ax.set_xlim(0.7, max_lag + 0.3)
    acf_ax.set_ylim(-0.52, 0.52)
    acf_ax.set_xlabel(f"lag in saved configurations (1 lag = {separation} compound steps)", fontsize=15)
    acf_ax.set_ylabel(r"$\rho(k)$", fontsize=17)
    acf_ax.legend(frameon=False, fontsize=11, loc="upper center")

    fig.tight_layout()
    output = (output or directory / "autocorrelation.png").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    pdf_output = output.with_suffix(".pdf")
    fig.savefig(pdf_output, bbox_inches="tight")
    plt.close(fig)
    (output.with_suffix(".json")).write_text(
        json.dumps(
            {
                "ensemble": name,
                "method": "normalized sample ACF with Geyer initial-positive-pair tau_int",
                "pointwise_95_white_noise_limit": float(pointwise_confidence),
                "simultaneous_95_white_noise_limit": float(simultaneous_confidence),
                "simultaneous_comparisons": comparisons,
                "confidence_sample_count": sample_count,
                "interpretation": (
                    f"These data test correlations at multiples of {separation} compound steps; "
                    "they cannot resolve the autocorrelation time below that spacing."
                ),
                "observables": summaries,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"saved {output}")
    print(f"saved {pdf_output}")
    print(f"saved {output.with_suffix('.json')}")


def main() -> None:
    args = parse_args()
    if args.output is not None and args.ensemble is None:
        raise SystemExit("--output applies to a single --ensemble")
    root = args.root.resolve()
    apply_plot_style()
    colors = dict(zip(ENSEMBLES, COLORS))
    ensemble_names = (args.ensemble,) if args.ensemble else ENSEMBLES
    for name in ensemble_names:
        plot_ensemble(root, name, colors[name], args.max_lag, args.dpi, args.output)


if __name__ == "__main__":
    main()
