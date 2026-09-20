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


KERNEL_OBSERVABLE_NAMES = (
    "phi2",
    "phi4",
    "local_kurtosis_ratio",
    "NN",
    "diag",
    "2nn",
    "m2",
    "m4",
    "G_pmin_avg",
    "action_density",
)

KERNEL_OBSERVABLE_TITLES = {
    "phi2": r"$\langle\phi^2\rangle$",
    "phi4": r"$\langle\phi^4\rangle$",
    "local_kurtosis_ratio": r"$\langle\phi^4\rangle/\langle\phi^2\rangle^2$",
    "NN": r"$C(1,0)$",
    "diag": r"$C(1,1)$",
    "2nn": r"$C(2,0)$",
    "m2": r"$m^2$",
    "m4": r"$m^4$",
    "G_pmin_avg": r"$G(p_{\min})$",
    "action_density": r"$S/V$",
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


def kernel_observable_series(
    configurations: FloatArray,
    action: Phi4Action,
) -> dict[str, FloatArray]:
    """Return the per-configuration observables used to select the kernel."""
    fields = np.asarray(configurations, dtype=np.float64)
    axes = (-2, -1)
    volume = fields.shape[-1] ** 2
    phi2 = np.mean(fields**2, axis=axes)
    phi4 = np.mean(fields**4, axis=axes)
    magnetization = np.mean(fields, axis=axes)
    transformed = np.fft.fft2(fields, axes=axes)
    minimum_momentum = 0.5 * (
        np.abs(transformed[..., 1, 0]) ** 2
        + np.abs(transformed[..., 0, 1]) ** 2
    ) / volume
    return {
        "phi2": phi2,
        "phi4": phi4,
        "local_kurtosis_ratio": phi4 / phi2**2,
        "NN": 0.5
        * (
            np.mean(fields * np.roll(fields, -1, axis=-2), axis=axes)
            + np.mean(fields * np.roll(fields, -1, axis=-1), axis=axes)
        ),
        "diag": np.mean(
            fields * np.roll(np.roll(fields, -1, axis=-2), -1, axis=-1),
            axis=axes,
        ),
        "2nn": 0.5
        * (
            np.mean(fields * np.roll(fields, -2, axis=-2), axis=axes)
            + np.mean(fields * np.roll(fields, -2, axis=-1), axis=axes)
        ),
        "m2": magnetization**2,
        "m4": magnetization**4,
        "G_pmin_avg": minimum_momentum,
        "action_density": action(fields) / volume,
    }


def distribution_metrics(
    blocked: FloatArray,
    target: FloatArray,
) -> dict[str, float]:
    """Compare two scalar distributions with scale and shape diagnostics."""
    blocked = np.asarray(blocked, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    blocked = blocked[np.isfinite(blocked)]
    target = target[np.isfinite(target)]
    if blocked.size == 0 or target.size == 0:
        names = (
            "blocked_mean",
            "target_mean",
            "standardized_mean_shift",
            "blocked_std",
            "target_std",
            "std_ratio",
            "ks_statistic",
            "wasserstein_1",
            "js_divergence",
            "total_variation",
            "below_target_q01",
            "below_target_q05",
            "below_target_q10",
            "above_target_q90",
            "above_target_q95",
            "above_target_q99",
            "inside_target_q01_q99",
            "inside_target_q05_q95",
        )
        return {name: float("nan") for name in names}
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

    edges = np.histogram_bin_edges(pooled, bins=35)
    if len(edges) == 2 and edges[0] == edges[1]:
        edges = np.array([edges[0] - 0.5, edges[1] + 0.5])
    blocked_hist = np.histogram(blocked, bins=edges)[0].astype(np.float64)
    target_hist = np.histogram(target, bins=edges)[0].astype(np.float64)
    blocked_probability = blocked_hist / np.sum(blocked_hist)
    target_probability = target_hist / np.sum(target_hist)
    midpoint = 0.5 * (blocked_probability + target_probability)
    blocked_nonzero = blocked_probability > 0.0
    target_nonzero = target_probability > 0.0
    js_divergence = 0.5 * (
        np.sum(
            blocked_probability[blocked_nonzero]
            * np.log(
                blocked_probability[blocked_nonzero] / midpoint[blocked_nonzero]
            )
        )
        + np.sum(
            target_probability[target_nonzero]
            * np.log(target_probability[target_nonzero] / midpoint[target_nonzero])
        )
    )
    quantiles = {
        label: float(np.quantile(target, probability))
        for label, probability in (
            ("q01", 0.01),
            ("q05", 0.05),
            ("q10", 0.10),
            ("q90", 0.90),
            ("q95", 0.95),
            ("q99", 0.99),
        )
    }
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
        "js_divergence": float(js_divergence),
        "total_variation": float(
            0.5 * np.sum(np.abs(blocked_probability - target_probability))
        ),
        "below_target_q01": float(np.mean(blocked < quantiles["q01"])),
        "below_target_q05": float(np.mean(blocked < quantiles["q05"])),
        "below_target_q10": float(np.mean(blocked < quantiles["q10"])),
        "above_target_q90": float(np.mean(blocked > quantiles["q90"])),
        "above_target_q95": float(np.mean(blocked > quantiles["q95"])),
        "above_target_q99": float(np.mean(blocked > quantiles["q99"])),
        "inside_target_q01_q99": float(
            np.mean((blocked >= quantiles["q01"]) & (blocked <= quantiles["q99"]))
        ),
        "inside_target_q05_q95": float(
            np.mean((blocked >= quantiles["q05"]) & (blocked <= quantiles["q95"]))
        ),
    }


def save_kernel_observable_histograms(
    path: Path,
    coarse_chains: FloatArray,
    blocked_chains: FloatArray,
    action: Phi4Action,
) -> dict[str, dict[str, dict[str, float]]]:
    """Plot the ten support observables used for distribution-aware selection."""
    coarse = {
        "test": kernel_observable_series(coarse_chains[3], action),
        "all": kernel_observable_series(
            coarse_chains.reshape(-1, *coarse_chains.shape[-2:]), action
        ),
    }
    blocked = {
        "test": kernel_observable_series(blocked_chains[3], action),
        "all": kernel_observable_series(
            blocked_chains.reshape(-1, *blocked_chains.shape[-2:]), action
        ),
    }
    metrics = {
        name: {
            split: distribution_metrics(
                blocked[split][name],
                coarse[split][name],
            )
            for split in ("test", "all")
        }
        for name in KERNEL_OBSERVABLE_NAMES
    }

    figure, axes = plt.subplots(2, 5, figsize=(22.0, 8.5))
    for axis, name in zip(axes.reshape(-1), KERNEL_OBSERVABLE_NAMES):
        coarse_values = coarse["test"][name]
        blocked_values = blocked["test"][name]
        bins = np.histogram_bin_edges(
            np.concatenate((coarse_values, blocked_values)),
            bins=30,
        )
        axis.hist(
            coarse_values,
            bins=bins,
            density=True,
            alpha=0.55,
            color="tab:blue",
            label=r"native coarse $\phi_c$",
        )
        axis.hist(
            blocked_values,
            bins=bins,
            density=True,
            alpha=0.55,
            color="tab:orange",
            label=r"blocked $DK\phi_f$",
        )
        comparison = metrics[name]["test"]
        axis.set_title(KERNEL_OBSERVABLE_TITLES[name], fontsize=13)
        axis.set_ylabel("density")
        axis.tick_params(labelsize=8)
        axis.text(
            0.03,
            0.96,
            (
                rf"$\Delta\mu/\sigma={comparison['standardized_mean_shift']:.3f}$"
                + "\n"
                + rf"$KS={comparison['ks_statistic']:.3f}$, "
                + rf"$\sigma_b/\sigma_c={comparison['std_ratio']:.3f}$"
            ),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )
    axes[0, 0].legend(frameon=False, fontsize=9)
    figure.suptitle(
        r"Held-out ensemble: native coarse $\phi_c$ vs. blocked $DK\phi_f$",
        fontsize=18,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig(path)
    figure.savefig(path.with_suffix(".png"), dpi=180)
    plt.close(figure)
    return metrics


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
        bins = (
            np.histogram_bin_edges(pooled, bins=35)
            if pooled.size
            else np.array([0.0, 1.0])
        )
        for values, color, linestyle in (
            (coarse_values["all"], "tab:blue", "--"),
            (blocked_values["all"], "tab:orange", "--"),
            (coarse_values["test"], "tab:blue", "-"),
            (blocked_values["test"], "tab:orange", "-"),
        ):
            if values.size == 0:
                continue
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
