"""Command-line interface for phi-four blocking-kernel training."""

import argparse
from pathlib import Path

from .kernel_training import run_training


DEFAULT_ROOT = Path("artifacts/2dphi4")


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--fine",
        type=Path,
        default=DEFAULT_ROOT / "phi4_L16_k0.3401_lam1.npz",
    )
    parser.add_argument(
        "--coarse",
        type=Path,
        default=DEFAULT_ROOT / "phi4_L8_k0.3401_lam1.npz",
    )
    parser.add_argument(
        "--transfer-fine",
        type=Path,
        default=DEFAULT_ROOT / "phi4_L32_k0.3401_lam1.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/phi4/kernel/L16_to_L8"),
    )
    parser.add_argument("--starts", type=int, default=12)
    parser.add_argument("--max-iterations", type=int, default=400)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1234)


def run(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if min(args.starts, args.max_iterations, args.bootstrap_samples) <= 0:
        parser.error("starts, max-iterations, and bootstrap-samples must be positive")
    kernel, metrics = run_training(
        args.fine,
        args.coarse,
        args.transfer_fine,
        args.output,
        starts=args.starts,
        max_iterations=args.max_iterations,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    test = metrics["L16_to_L8"]["trained"]["test"]["summary"]
    transfer = metrics["L32_to_L16_transfer_test"]["trained"]["summary"]
    spectrum = kernel["spectrum"]["L256"]
    status = "PASS" if metrics["acceptance"]["all"] else "FAIL"
    print(
        f"Kernel training {status}: "
        f"L16->L8 test RMS shift={test['rms_standardized_shift']:.4f}, "
        f"L32->L16 transfer RMS shift={transfer['rms_standardized_shift']:.4f}, "
        f"min|K(p)|={spectrum['min_abs_K']:.4f}, "
        f"condition={spectrum['condition_number']:.4f}"
    )
