"""Distribution-level diagnostics for blocked and native coarse ensembles."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from numpy.typing import NDArray
from scipy.stats import wasserstein_distance

from .action import Phi4Action


FloatArray = NDArray[np.float64]

OPERATOR_NAMES = (
    "action_density",
    "potential_density",
    "hopping_action_density",
    "gradient_squared",
    "phi2",
    "phi4",
    "phi6",
    "phi4_over_phi2_squared",
    "magnetization",
    "abs_magnetization",
    "magnetization_squared",
    "magnetization_fourth",
    "C10",
    "C11",
    "C20",
    "C21",
    "C22",
    "C30",
    "C40",
    "G10",
    "G11",
    "G20",
    "G21",
    "G22",
)

OPERATOR_TITLES = {
    "action_density": r"$S/V$",
    "potential_density": "potential density",
    "hopping_action_density": "hopping-action density",
    "gradient_squared": r"$\frac{1}{2}\sum_\mu(\nabla_\mu\phi)^2$",
    "phi2": r"$\langle\phi^2\rangle$",
    "phi4": r"$\langle\phi^4\rangle$",
    "phi6": r"$\langle\phi^6\rangle$",
    "phi4_over_phi2_squared": r"$\langle\phi^4\rangle/\langle\phi^2\rangle^2$",
    "magnetization": r"$m$",
    "abs_magnetization": r"$|m|$",
    "magnetization_squared": r"$m^2$",
    "magnetization_fourth": r"$m^4$",
    **{f"C{a}{b}": rf"$C_{{{a}{b}}}$" for a, b in ((1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (3, 0), (4, 0))},
    **{f"G{a}{b}": rf"$G_{{{a}{b}}}$" for a, b in ((1, 0), (1, 1), (2, 0), (2, 1), (2, 2))},
}


def _d4_orbit(a: int, b: int) -> tuple[tuple[int, int], ...]:
    offsets = {
        (sign_a * a, sign_b * b)
        for sign_a in (-1, 1)
        for sign_b in (-1, 1)
    }
    offsets.update(
        {
            (sign_a * b, sign_b * a)
            for sign_a in (-1, 1)
            for sign_b in (-1, 1)
        }
    )
    return tuple(sorted(offsets))


def operator_series(
    configurations: FloatArray,
    action: Phi4Action,
) -> dict[str, FloatArray]:
    """Return 24 per-configuration local, global, and two-point operators."""
    fields = np.asarray(configurations, dtype=np.float64)
    axes = (-2, -1)
    volume = fields.shape[-1] ** 2
    phi2 = np.mean(fields**2, axis=axes)
    phi4 = np.mean(fields**4, axis=axes)
    nearest_y = fields * np.roll(fields, -1, axis=-2)
    nearest_x = fields * np.roll(fields, -1, axis=-1)
    magnetization = np.mean(fields, axis=axes)

    series = {
        "action_density": action(fields) / volume,
        "potential_density": np.mean(action.potential(fields), axis=axes),
        "hopping_action_density": -2.0
        * action.kappa
        * (
            np.mean(nearest_y, axis=axes)
            + np.mean(nearest_x, axis=axes)
        ),
        "gradient_squared": 0.5
        * (
            np.mean((np.roll(fields, -1, axis=-2) - fields) ** 2, axis=axes)
            + np.mean((np.roll(fields, -1, axis=-1) - fields) ** 2, axis=axes)
        ),
        "phi2": phi2,
        "phi4": phi4,
        "phi6": np.mean(fields**6, axis=axes),
        "phi4_over_phi2_squared": phi4 / phi2**2,
        "magnetization": magnetization,
        "abs_magnetization": np.abs(magnetization),
        "magnetization_squared": magnetization**2,
        "magnetization_fourth": magnetization**4,
    }

    for a, b in ((1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (3, 0), (4, 0)):
        correlators = [
            np.mean(
                fields
                * np.roll(np.roll(fields, -dy, axis=-2), -dx, axis=-1),
                axis=axes,
            )
            for dy, dx in _d4_orbit(a, b)
        ]
        series[f"C{a}{b}"] = np.mean(correlators, axis=0)

    transformed = np.fft.fft2(fields, axes=axes)
    power = np.abs(transformed) ** 2 / volume
    lattice_size = fields.shape[-1]
    for a, b in ((1, 0), (1, 1), (2, 0), (2, 1), (2, 2)):
        modes = [
            power[..., ky % lattice_size, kx % lattice_size]
            for ky, kx in _d4_orbit(a, b)
        ]
        series[f"G{a}{b}"] = np.mean(modes, axis=0)
    return series


def distribution_metrics(
    blocked: FloatArray,
    target: FloatArray,
) -> dict[str, float]:
    """Compare two scalar distributions with scale and shape diagnostics."""
    blocked = np.asarray(blocked, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    blocked = blocked[np.isfinite(blocked)]
    target = target[np.isfinite(target)]
    blocked_std = float(np.std(blocked, ddof=1 if blocked.size > 1 else 0))
    target_std = float(np.std(target, ddof=1 if target.size > 1 else 0))
    pooled_std = np.sqrt(0.5 * (blocked_std**2 + target_std**2))
    pooled = np.sort(np.concatenate((blocked, target)))
    blocked_cdf = np.searchsorted(np.sort(blocked), pooled, side="right") / len(
        blocked
    )
    target_cdf = np.searchsorted(np.sort(target), pooled, side="right") / len(
        target
    )
    return {
        "blocked_mean": float(np.mean(blocked)),
        "target_mean": float(np.mean(target)),
        "standardized_mean_shift": (
            float((np.mean(blocked) - np.mean(target)) / pooled_std)
            if pooled_std > 0.0
            else 0.0
        ),
        "blocked_std": blocked_std,
        "target_std": target_std,
        "std_ratio": blocked_std / target_std if target_std > 0.0 else float("nan"),
        "ks_statistic": float(np.max(np.abs(blocked_cdf - target_cdf))),
        "wasserstein_1": float(wasserstein_distance(blocked, target)),
    }


def bootstrap_ensemble_observables(
    configurations: FloatArray,
    *,
    samples: int = 2000,
    block_length: int = 2,
    seed: int = 1234,
) -> dict[str, FloatArray]:
    """Circular moving-block bootstrap for susceptibility, Binder U4, and xi/L."""
    fields = np.asarray(configurations, dtype=np.float64)
    if fields.ndim == 3:
        fields = fields[None, ...]
    chains, configurations_per_chain = fields.shape[:2]
    lattice_size = fields.shape[-1]
    volume = lattice_size**2
    magnetization = np.mean(fields, axis=(-2, -1))
    transformed = np.fft.fft2(fields, axes=(-2, -1))
    minimum_momentum = 0.5 * (
        np.abs(transformed[..., 1, 0]) ** 2
        + np.abs(transformed[..., 0, 1]) ** 2
    ) / volume

    rng = np.random.default_rng(seed)
    blocks_per_chain = int(np.ceil(configurations_per_chain / block_length))
    susceptibility = np.empty(samples, dtype=np.float64)
    binder = np.empty(samples, dtype=np.float64)
    xi_over_l = np.empty(samples, dtype=np.float64)
    offsets = np.arange(block_length)
    for sample in range(samples):
        indices = (
            rng.integers(
                0,
                configurations_per_chain,
                size=(chains, blocks_per_chain, 1),
            )
            + offsets
        ) % configurations_per_chain
        indices = indices.reshape(chains, -1)[:, :configurations_per_chain]
        selected_m = np.take_along_axis(magnetization, indices, axis=1).reshape(-1)
        selected_gp = np.take_along_axis(
            minimum_momentum, indices, axis=1
        ).reshape(-1)
        mean_m = float(np.mean(selected_m))
        mean_m2 = float(np.mean(selected_m**2))
        susceptibility[sample] = volume * (mean_m2 - mean_m**2)
        binder[sample] = 1.0 - np.mean(selected_m**4) / (3.0 * mean_m2**2)
        ratio = susceptibility[sample] / np.mean(selected_gp)
        xi_over_l[sample] = (
            np.sqrt(ratio - 1.0)
            / (2.0 * np.sin(np.pi / lattice_size))
            / lattice_size
            if ratio > 1.0
            else np.nan
        )
    return {
        "susceptibility": susceptibility,
        "binder_cumulant": binder,
        "xi_over_L": xi_over_l,
    }


def save_operator_distribution_figure(
    path: Path,
    coarse_chains: FloatArray,
    blocked_chains: FloatArray,
    action: Phi4Action,
    *,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 1234,
) -> dict[str, object]:
    """Save the single-page 27-panel comparison and return its metrics."""
    coarse_all = coarse_chains.reshape(
        -1, coarse_chains.shape[-1], coarse_chains.shape[-1]
    )
    blocked_all = blocked_chains.reshape(
        -1, blocked_chains.shape[-1], blocked_chains.shape[-1]
    )
    coarse_test = coarse_chains[3]
    blocked_test = blocked_chains[3]
    coarse_series = {
        "all": operator_series(coarse_all, action),
        "test": operator_series(coarse_test, action),
    }
    blocked_series = {
        "all": operator_series(blocked_all, action),
        "test": operator_series(blocked_test, action),
    }

    bootstrap_coarse = {
        "all": bootstrap_ensemble_observables(
            coarse_chains,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        ),
        "test": bootstrap_ensemble_observables(
            coarse_test,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        ),
    }
    bootstrap_blocked = {
        "all": bootstrap_ensemble_observables(
            blocked_chains,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        ),
        "test": bootstrap_ensemble_observables(
            blocked_test,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        ),
    }

    metrics: dict[str, object] = {
        "configuration_operators": {},
        "bootstrap_observables": {},
        "bootstrap": {
            "samples": bootstrap_samples,
            "block_length": 2,
            "seed": bootstrap_seed,
        },
    }
    for name in OPERATOR_NAMES:
        metrics["configuration_operators"][name] = {
            split: distribution_metrics(
                blocked_series[split][name],
                coarse_series[split][name],
            )
            for split in ("test", "all")
        }
    for name in ("susceptibility", "binder_cumulant", "xi_over_L"):
        metrics["bootstrap_observables"][name] = {
            split: distribution_metrics(
                bootstrap_blocked[split][name],
                bootstrap_coarse[split][name],
            )
            for split in ("test", "all")
        }

    figure, axes = plt.subplots(5, 6, figsize=(30.0, 22.0))
    axes_flat = axes.reshape(-1)
    plot_names = list(OPERATOR_NAMES) + [
        "susceptibility",
        "binder_cumulant",
        "xi_over_L",
    ]
    for axis, name in zip(axes_flat, plot_names):
        if name in OPERATOR_NAMES:
            coarse_values = {split: coarse_series[split][name] for split in ("test", "all")}
            blocked_values = {split: blocked_series[split][name] for split in ("test", "all")}
            comparison = metrics["configuration_operators"][name]["test"]
            title = OPERATOR_TITLES[name]
        else:
            coarse_values = {split: bootstrap_coarse[split][name] for split in ("test", "all")}
            blocked_values = {split: bootstrap_blocked[split][name] for split in ("test", "all")}
            comparison = metrics["bootstrap_observables"][name]["test"]
            title = {
                "susceptibility": r"$\chi$ (bootstrap)",
                "binder_cumulant": r"$U_4$ (bootstrap)",
                "xi_over_L": r"$\xi_{\rm 2nd}/L$ (bootstrap)",
            }[name]
        coarse_values = {
            split: values[np.isfinite(values)]
            for split, values in coarse_values.items()
        }
        blocked_values = {
            split: values[np.isfinite(values)]
            for split, values in blocked_values.items()
        }
        pooled = np.concatenate(
            (
                coarse_values["all"],
                blocked_values["all"],
                coarse_values["test"],
                blocked_values["test"],
            )
        )
        bins = np.histogram_bin_edges(pooled, bins=35)
        for values, color, linestyle in (
            (coarse_values["all"], "tab:blue", "--"),
            (blocked_values["all"], "tab:orange", "--"),
            (coarse_values["test"], "tab:blue", "-"),
            (blocked_values["test"], "tab:orange", "-"),
        ):
            axis.hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                color=color,
                linestyle=linestyle,
                linewidth=1.35,
            )
        axis.set_title(title, fontsize=12)
        axis.tick_params(labelsize=8)
        axis.text(
            0.03,
            0.96,
            (
                rf"test: $\Delta\mu/\sigma={comparison['standardized_mean_shift']:.3f}$"
                + "\n"
                + rf"$KS={comparison['ks_statistic']:.3f}$"
            ),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )

    axes_flat[27].axis("off")
    axes_flat[27].legend(
        handles=[
            Line2D([0], [0], color="tab:blue", linestyle="-", label=r"$\phi_c$ test"),
            Line2D([0], [0], color="tab:orange", linestyle="-", label=r"$\phi_b$ test"),
            Line2D([0], [0], color="tab:blue", linestyle="--", label=r"$\phi_c$ all"),
            Line2D([0], [0], color="tab:orange", linestyle="--", label=r"$\phi_b$ all"),
        ],
        loc="center",
        frameon=False,
        fontsize=14,
    )
    axes_flat[28].axis("off")
    axes_flat[28].text(
        0.5,
        0.5,
        (
            r"$\phi_b=D_{00}K\phi_f$"
            "\n"
            "solid: held-out chain 3\n"
            "dashed: all four chains\n"
            f"bootstrap: {bootstrap_samples} circular moving-block resamples"
        ),
        ha="center",
        va="center",
        fontsize=14,
        linespacing=1.5,
    )
    axes_flat[29].axis("off")
    figure.suptitle(
        r"Native coarse $\phi_c$ vs. blocked $\phi_b$ operator distributions",
        fontsize=20,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    figure.savefig(path)
    plt.close(figure)
    return metrics
