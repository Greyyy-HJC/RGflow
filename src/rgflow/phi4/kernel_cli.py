"""Command-line interface for phi-four blocking-kernel training."""

import argparse
from pathlib import Path

from .kernel_training import run_multivolume_training


DEFAULT_ROOT = Path("artifacts/2dphi4")
DEFAULT_TRAINING_PAIRS = [
    (
        DEFAULT_ROOT / "phi4_L16_k0.340301_lam1.npz",
        DEFAULT_ROOT / "phi4_L8_k0.340301_lam1.npz",
    ),
    (
        DEFAULT_ROOT / "phi4_L24_k0.340301_lam1.npz",
        DEFAULT_ROOT / "phi4_L12_k0.340301_lam1.npz",
    ),
    (
        DEFAULT_ROOT / "phi4_L32_k0.340301_lam1.npz",
        DEFAULT_ROOT / "phi4_L16_k0.340301_lam1.npz",
    ),
]
DEFAULT_TRANSFER_PAIRS = [
    (
        DEFAULT_ROOT / "phi4_L48_k0.340301_lam1.npz",
        DEFAULT_ROOT / "phi4_L24_k0.340301_lam1.npz",
    ),
    (
        DEFAULT_ROOT / "phi4_L64_k0.340301_lam1.npz",
        DEFAULT_ROOT / "phi4_L32_k0.340301_lam1.npz",
    ),
]


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--training-pair",
        type=Path,
        nargs=2,
        action="append",
        metavar=("FINE", "COARSE"),
    )
    parser.add_argument(
        "--transfer-pair",
        type=Path,
        nargs=2,
        action="append",
        metavar=("FINE", "COARSE"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/phi4/kernel/multivolume_sos2_k0.340301"),
    )
    parser.add_argument("--starts", type=int, default=6)
    parser.add_argument("--max-iterations", type=int, default=250)
    parser.add_argument("--optimization-samples-per-chain", type=int, default=128)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--support-weight", type=float, default=1.0)
    parser.add_argument("--physical-weight", type=float, default=1.0)
    parser.add_argument("--conditioning-weight", type=float, default=0.05)
    parser.add_argument("--sos-channels", type=int, default=2)
    parser.add_argument("--sos-floor-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1234)


def run(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if min(
        args.starts,
        args.max_iterations,
        args.optimization_samples_per_chain,
        args.bootstrap_samples,
    ) <= 0:
        parser.error("optimization budgets and bootstrap-samples must be positive")
    if min(
        args.support_weight,
        args.physical_weight,
        args.conditioning_weight,
    ) < 0.0:
        parser.error("loss weights must be nonnegative")
    if args.sos_channels <= 0:
        parser.error("sos-channels must be positive")
    if not 0.0 < args.sos_floor_fraction < 1.0:
        parser.error("sos-floor-fraction must lie between zero and one")

    training_pairs = args.training_pair or DEFAULT_TRAINING_PAIRS
    transfer_pairs = args.transfer_pair or DEFAULT_TRANSFER_PAIRS
    kernel, metrics = run_multivolume_training(
        [tuple(pair) for pair in training_pairs],
        [tuple(pair) for pair in transfer_pairs],
        args.output,
        starts=args.starts,
        max_iterations=args.max_iterations,
        optimization_samples_per_chain=args.optimization_samples_per_chain,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
        support_weight=args.support_weight,
        physical_weight=args.physical_weight,
        conditioning_weight=args.conditioning_weight,
        sos_channels=args.sos_channels,
        sos_floor_fraction=args.sos_floor_fraction,
    )
    training_rms = [
        pair["trained"]["test"]["summary"]["rms_standardized_shift"]
        for pair in metrics["training_pairs"].values()
    ]
    transfer_rms = [
        pair["trained"]["summary"]["rms_standardized_shift"]
        for pair in metrics["transfer_pairs"].values()
    ]
    spectrum = kernel["spectrum"]["L256"]
    status = "PASS" if metrics["acceptance"]["all"] else "FAIL"
    print(
        f"Kernel training {status} (SOS x{args.sos_channels}, "
        f"{len(training_rms)} training volumes): "
        f"mean training-pair test RMS={sum(training_rms) / len(training_rms):.4f}, "
        f"max transfer-pair RMS={max(transfer_rms):.4f}, "
        f"min|K(p)|={spectrum['min_abs_K']:.4f}, "
        f"condition={spectrum['condition_number']:.4f}"
    )
