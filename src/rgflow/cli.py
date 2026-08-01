"""Top-level command-line interface for RGflow."""

import argparse
from typing import Sequence

from .phi4 import cli as phi4_cli


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
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args.command_parser, args)
