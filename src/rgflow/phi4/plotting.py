"""SVG presentation figures for generated ensembles."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .action import Phi4Action
from .observables import observable_series


def save_production_figures(
    configurations: np.ndarray,
    action: Phi4Action,
    saved_cycles: np.ndarray,
    output: Path,
    stem: str,
) -> list[Path]:
    """Save configuration and diagnostic figures as SVG files."""
    snapshot_path = output / f"{stem}_configurations.svg"
    diagnostics_path = output / f"{stem}_diagnostics.svg"

    shown_chains = min(4, configurations.shape[0])
    snapshots = configurations[:shown_chains, -1]
    limit = float(np.max(np.abs(snapshots)))
    figure, axes = plt.subplots(
        1, shown_chains, figsize=(3.2 * shown_chains, 3.1), squeeze=False
    )
    image = None
    for chain, axis in enumerate(axes[0]):
        image = axis.imshow(
            snapshots[chain],
            origin="lower",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        axis.set_title(f"Chain {chain + 1}, final sample")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    figure.colorbar(image, ax=axes.ravel().tolist(), label="phi")
    figure.suptitle(
        f"2D phi4 configurations, L={configurations.shape[-1]}, "
        f"kappa={action.kappa:g}, lambda={action.lam:g}"
    )
    figure.savefig(snapshot_path, format="svg", bbox_inches="tight")
    plt.close(figure)

    series = observable_series(configurations, action)
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for chain in range(configurations.shape[0]):
        axes[0, 0].plot(
            saved_cycles,
            series["action_density"][chain],
            linewidth=0.8,
            alpha=0.8,
            label=f"chain {chain + 1}",
        )
        axes[0, 1].plot(
            saved_cycles,
            series["abs_magnetization"][chain],
            linewidth=0.8,
            alpha=0.8,
        )
    axes[0, 0].set_title("Action-density history")
    axes[0, 0].set_xlabel("production cycle")
    axes[0, 0].set_ylabel("S / V")
    axes[0, 0].legend(fontsize="small")
    axes[0, 1].set_title("Absolute-magnetization history")
    axes[0, 1].set_xlabel("production cycle")
    axes[0, 1].set_ylabel("|m|")
    axes[1, 0].hist(series["action_density"].ravel(), bins=30)
    axes[1, 0].set_title("Action-density distribution")
    axes[1, 0].set_xlabel("S / V")
    axes[1, 0].set_ylabel("stored configurations")
    axes[1, 1].hist(series["magnetization"].ravel(), bins=30)
    axes[1, 1].set_title("Magnetization distribution")
    axes[1, 1].set_xlabel("m")
    axes[1, 1].set_ylabel("stored configurations")
    figure.suptitle(f"Production diagnostics, L={configurations.shape[-1]}")
    figure.savefig(diagnostics_path, format="svg")
    plt.close(figure)
    return [snapshot_path, diagnostics_path]
