"""Top-level command-line interface for RGflow."""

import argparse
from typing import Sequence

from .phi4 import cli as phi4_cli
from .phi4 import kernel_cli as phi4_kernel_cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rgflow",
        description="Renormalization-group accelerated lattice-field sampling.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    generate_parser = commands.add_parser(
        "generate", help="Generate native lattice configurations."
    )
    theories = generate_parser.add_subparsers(dest="theory", required=True)
    phi4_parser = theories.add_parser(
        "phi4",
        help="Generate 2D scalar phi-four configurations.",
        description="Generate 2D phi-four configurations with hybrid MCMC.",
    )
    phi4_cli.configure_parser(phi4_parser)
    phi4_parser.set_defaults(handler=phi4_cli.run, command_parser=phi4_parser)

    train_parser = commands.add_parser(
        "train", help="Train RG transformations and generative models."
    )
    train_components = train_parser.add_subparsers(
        dest="train_component", required=True
    )
    kernel_parser = train_components.add_parser(
        "kernel", help="Train an approximate blocking kernel."
    )
    kernel_theories = kernel_parser.add_subparsers(
        dest="kernel_theory", required=True
    )
    kernel_phi4_parser = kernel_theories.add_parser(
        "phi4",
        help="Train the 2D scalar phi-four blocking kernel.",
        description="Train a D4-symmetric 5x5 phi-four blocking kernel.",
    )
    phi4_kernel_cli.configure_parser(kernel_phi4_parser)
    kernel_phi4_parser.set_defaults(
        handler=phi4_kernel_cli.run,
        command_parser=kernel_phi4_parser,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args.command_parser, args)
