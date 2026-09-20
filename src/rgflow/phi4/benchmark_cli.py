"""CLI for reproducible 2D phi-four sampler benchmarks."""

import argparse
from pathlib import Path

METHOD_CHOICES = (
    "coordinate",
    "coordinate-wolff",
    "detail2-wolff",
    "detail4-wolff",
    "flow-detail2-wolff",
    "flow-detail4-wolff",
    "detail2-wolff-gaussian",
    "detail4-wolff-gaussian",
    "hmc-wolff",
)


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--case",
        type=Path,
        nargs=3,
        action="append",
        required=True,
        metavar=("CHECKPOINT", "COARSE", "NATIVE_FINE"),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHOD_CHOICES,
        default=(
            "coordinate",
            "coordinate-wolff",
            "detail2-wolff",
            "detail4-wolff",
            "flow-detail2-wolff",
            "flow-detail4-wolff",
            "detail2-wolff-gaussian",
            "detail4-wolff-gaussian",
            "hmc-wolff",
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/phi4/benchmark"))
    parser.add_argument("--chains", type=int, default=128)
    parser.add_argument("--sweeps", type=int, default=100)
    parser.add_argument("--calibration-sweeps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--coarse-sampling-seconds", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--device", default="auto")


def run(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if min(
        args.chains,
        args.sweeps,
        args.calibration_sweeps,
        args.batch_size,
        args.bootstrap_samples,
        args.replicates,
    ) <= 0:
        parser.error("benchmark counts must be positive")
    if args.coarse_sampling_seconds < 0.0:
        parser.error("coarse-sampling-seconds must be nonnegative")
    try:
        from .benchmark import run_benchmark, run_benchmark_replicates
    except ModuleNotFoundError as error:
        if error.name == "torch":
            parser.error("inverse benchmarking requires the project 'flow' extra")
        raise
    arguments = {
        "methods": tuple(args.methods),
        "chains": args.chains,
        "sweeps": args.sweeps,
        "calibration_sweeps": args.calibration_sweeps,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device_name": args.device,
        "bootstrap_samples": args.bootstrap_samples,
        "coarse_sampling_seconds": args.coarse_sampling_seconds,
    }
    cases = [tuple(case) for case in args.case]
    if args.replicates == 1:
        report = run_benchmark(cases, args.output, **arguments)
        print(f"Benchmark {report['run_id']} written under {args.output}")
    else:
        suite = run_benchmark_replicates(
            cases,
            args.output,
            replicates=args.replicates,
            **arguments,
        )
        print(f"Confirmation suite {suite['suite_id']} written under {args.output}")
