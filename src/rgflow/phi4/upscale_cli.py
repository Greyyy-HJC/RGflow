"""CLI for flow initialization followed by exact inverse-blocking MH."""

import argparse
from pathlib import Path


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--coarse", type=Path, required=True)
    parser.add_argument("--native-fine", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chains", type=int, default=128)
    parser.add_argument("--sweeps", type=int, default=400)
    parser.add_argument("--calibration-chains", type=int, default=32)
    parser.add_argument("--calibration-sweeps", type=int, default=50)
    parser.add_argument("--divide", type=int, choices=(2,), default=2)
    parser.add_argument(
        "--correction",
        choices=(
            "coordinate",
            "coordinate-wolff",
            "detail-wolff",
            "flow-detail-wolff",
            "hmc-wolff",
        ),
        default="coordinate",
    )
    parser.add_argument("--detail-passes", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--initializer", choices=("flow", "gaussian"), default="flow")
    parser.add_argument("--evaluation-interval", type=int, default=5)
    parser.add_argument("--initial-width", type=float, default=0.04)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--save-sweeps",
        default="0,1,2,5,10,20,50,100,200,400",
    )


def run(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if min(
        args.chains,
        args.sweeps,
        args.calibration_chains,
        args.calibration_sweeps,
        args.divide,
        args.batch_size,
        args.detail_passes,
        args.evaluation_interval,
    ) <= 0:
        parser.error("chain, sweep, calibration, divide, and batch values must be positive")
    if not 0.0 < args.initial_width <= 0.2:
        parser.error("initial-width must lie in (0, 0.2]")
    try:
        from .inverse_mh import run_inverse_mh
    except ModuleNotFoundError as error:
        if error.name == "torch":
            parser.error("inverse upscaling requires the project 'flow' extra")
        raise
    save_sweeps = tuple(
        int(value) for value in args.save_sweeps.split(",") if value.strip()
    )
    metrics = run_inverse_mh(
        args.checkpoint,
        args.coarse,
        args.output,
        native_fine_path=args.native_fine,
        chains=args.chains,
        sweeps=args.sweeps,
        calibration_chains=args.calibration_chains,
        calibration_sweeps=args.calibration_sweeps,
        divide=args.divide,
        initial_width=args.initial_width,
        batch_size=args.batch_size,
        seed=args.seed,
        device_name=args.device,
        save_sweeps=save_sweeps,
        correction=args.correction,
        detail_passes=args.detail_passes,
        initializer=args.initializer,
        evaluation_interval=args.evaluation_interval,
    )
    convergence = metrics["convergence"]
    print(
        f"Inverse MH {'CONVERGED' if convergence['converged'] else 'NEEDS MORE SWEEPS'}: "
        f"RMS shift={convergence['rms_standardized_mean_shift']:.4f}, "
        f"max shift={convergence['max_abs_standardized_mean_shift']:.4f}"
    )
