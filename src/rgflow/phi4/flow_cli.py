"""CLI for conditional affine-flow training."""

import argparse
from pathlib import Path


DEFAULT_DATA_ROOT = Path("artifacts/2dphi4")
DEFAULT_FLOW_PAIRS = [
    (
        DEFAULT_DATA_ROOT / "phi4_L16_k0.340301_lam1.npz",
        DEFAULT_DATA_ROOT / "phi4_L8_k0.340301_lam1.npz",
    ),
    (
        DEFAULT_DATA_ROOT / "phi4_L32_k0.340301_lam1.npz",
        DEFAULT_DATA_ROOT / "phi4_L16_k0.340301_lam1.npz",
    ),
]
DEFAULT_KERNEL = Path(
    "artifacts/phi4/kernel/multivolume_sos2_k0.340301/kernel.json"
)


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pair",
        type=Path,
        nargs=2,
        action="append",
        metavar=("FINE", "COARSE"),
        help="Factor-two fine/coarse ensemble pair; repeat for multiple mappings.",
    )
    parser.add_argument("--kernel", type=Path, default=DEFAULT_KERNEL)
    parser.add_argument("--output", type=Path, default=Path("artifacts/phi4/flow"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--warmup-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument(
        "--model", choices=("affine", "strong-affine", "residual-spline"),
        default="affine",
    )
    parser.add_argument("--coupling-layers", type=int)
    parser.add_argument("--hidden-channels", type=int)
    parser.add_argument("--kernel-size", type=int)
    parser.add_argument("--spline-layers", type=int, default=4)
    parser.add_argument("--spline-bins", type=int, default=8)
    parser.add_argument("--spline-tail-bound", type=float, default=4.0)
    parser.add_argument(
        "--reverse-kl-weight", type=float, choices=(0.0, 0.01, 0.05, 0.1), default=0.0
    )
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto")


def run(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if min(
        args.epochs,
        args.batch_size,
        args.patience,
        args.coupling_layers or 1,
        args.hidden_channels or 1,
        args.kernel_size or 1,
        args.spline_layers,
        args.spline_bins,
    ) <= 0:
        parser.error("training budgets and model sizes must be positive")
    if args.warmup_epochs < 0 or args.warmup_epochs >= args.epochs:
        parser.error("warmup-epochs must be nonnegative and smaller than epochs")
    if min(args.learning_rate, args.weight_decay, args.reverse_kl_weight) < 0.0:
        parser.error("optimizer parameters must be nonnegative")
    if args.kernel_size is not None and args.kernel_size % 2 == 0:
        parser.error("kernel-size must be odd")
    if args.spline_bins < 2 or args.spline_tail_bound <= 0.0:
        parser.error("spline bins and tail bound must define a positive spline")
    try:
        from .flow_training import load_configurations, train_flow_pair
    except ModuleNotFoundError as error:
        if error.name == "torch":
            parser.error("flow training requires the project 'flow' extra")
        raise

    defaults = {
        "affine": (6, 32, 3, "affine"),
        "strong-affine": (10, 64, 5, "affine"),
        "residual-spline": (10, 64, 5, "residual-spline"),
    }
    default_layers, default_hidden, default_kernel, model_type = defaults[args.model]
    coupling_layers = args.coupling_layers or default_layers
    hidden_channels = args.hidden_channels or default_hidden
    kernel_size = args.kernel_size or default_kernel
    pairs = args.pair or DEFAULT_FLOW_PAIRS
    for index, (fine, coarse) in enumerate(pairs):
        fine_fields = load_configurations(fine)
        coarse_fields = load_configurations(coarse)
        fine_size = fine_fields.shape[-1]
        coarse_size = coarse_fields.shape[-1]
        suffix = args.model.replace("-", "_")
        if args.reverse_kl_weight:
            suffix += f"_rkl{args.reverse_kl_weight:g}"
        output = args.output / f"L{coarse_size}_to_L{fine_size}_{suffix}"
        metrics = train_flow_pair(
            fine,
            coarse,
            args.kernel,
            output,
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            coupling_layers=coupling_layers,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            model_type=model_type,
            spline_layers=args.spline_layers,
            spline_bins=args.spline_bins,
            spline_tail_bound=args.spline_tail_bound,
            reverse_kl_weight=args.reverse_kl_weight,
            initialize_from=args.initialize_from,
            seed=args.seed + index,
            device_name=args.device,
        )
        print(
            f"Flow L{coarse_size}->L{fine_size}: best epoch "
            f"{metrics['best_epoch']}, test RMS shift "
            f"{metrics['test']['rms_standardized_shift']:.4f}"
        )
