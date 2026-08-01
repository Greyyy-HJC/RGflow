"""Command-line interface for native 2D phi-four ensemble generation."""

import argparse
import json
from pathlib import Path

import numpy as np

from .action import Phi4Action
from .observables import summarize_ensemble
from .plotting import save_production_figures
from .sampling import generate_ensemble


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """Add phi-four generation options to a command parser."""
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--kappa", type=float, default=0.3401)
    parser.add_argument("--lambda", dest="lam", type=float, default=1.0)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--samples-per-chain", type=int, default=250)
    parser.add_argument("--warmup-cycles", type=int, default=1000)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--radial-sweeps", type=int, default=1)
    parser.add_argument("--proposal-width", type=float, default=1.0)
    parser.add_argument("--target-acceptance", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, default=Path("artifacts/phi4"))


def _validate_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if any(size <= 0 or size % 2 for size in args.sizes):
        parser.error("all lattice sizes must be positive and even")
    if args.kappa < 0.0 or args.lam < 0.0:
        parser.error("kappa and lambda must be non-negative")
    if min(
        args.chains,
        args.samples_per_chain,
        args.warmup_cycles,
        args.sample_interval,
        args.radial_sweeps,
    ) <= 0:
        parser.error("chain, sample, warmup, interval, and sweep counts must be positive")
    if args.proposal_width <= 0.0:
        parser.error("proposal width must be positive")
    if not 0.0 < args.target_acceptance < 1.0:
        parser.error("target acceptance must lie between zero and one")


def _generate_size(args: argparse.Namespace, size: int, seed: int) -> None:
    action = Phi4Action(kappa=args.kappa, lam=args.lam)
    result = generate_ensemble(
        size=size,
        chains=args.chains,
        samples_per_chain=args.samples_per_chain,
        warmup_cycles=args.warmup_cycles,
        sample_interval=args.sample_interval,
        radial_sweeps=args.radial_sweeps,
        action=action,
        proposal_width=args.proposal_width,
        target_acceptance=args.target_acceptance,
        seed=seed,
    )
    diagnostics = summarize_ensemble(result.configurations, action)
    diagnostics["sampler"] = {
        "warmup_acceptance": result.warmup_acceptance,
        "production_acceptance": result.production_acceptance,
        "proposal_width": result.proposal_width,
        "mean_cluster_fraction": result.mean_cluster_fraction,
    }
    diagnostics["parameters"] = {
        "size": size,
        "kappa": args.kappa,
        "lambda": args.lam,
        "chains": args.chains,
        "samples_per_chain": args.samples_per_chain,
        "warmup_cycles": args.warmup_cycles,
        "sample_interval": args.sample_interval,
        "radial_sweeps": args.radial_sweeps,
        "seed": seed,
    }

    stem = f"phi4_L{size}_k{args.kappa:g}_lam{args.lam:g}"
    saved_cycles = (
        np.arange(1, args.samples_per_chain + 1) * args.sample_interval
    )
    np.savez_compressed(
        args.output / f"{stem}.npz",
        configurations=result.configurations,
        saved_cycles=saved_cycles,
        size=size,
        kappa=args.kappa,
        lam=args.lam,
        seed=seed,
    )
    with (args.output / f"{stem}.json").open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)
        handle.write("\n")
    save_production_figures(
        result.configurations,
        action,
        saved_cycles,
        args.output,
        stem,
    )

    means = diagnostics["means"]
    print(
        f"L={size}: acceptance={result.production_acceptance:.3f}, "
        f"<S/V>={means['action_density']:.6f}, "
        f"Binder={means['binder_cumulant']:.4f}"
    )


def run(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Generate phi-four ensembles from parsed command-line arguments."""
    _validate_arguments(parser, args)
    args.output.mkdir(parents=True, exist_ok=True)
    for index, size in enumerate(args.sizes):
        _generate_size(args, size, args.seed + index)
