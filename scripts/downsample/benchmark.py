#!/usr/bin/env python3
"""Benchmark eager versus torch.compile for the reversible field transform."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from rgflow.su3.downsampling import GaugeEquivariantFieldTransform, transform_and_block
from rgflow.su3.io import read_nersc_gauge


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = next((ROOT / "artifacts" / "4dsu3" / "L24_beta6p20").glob("*.nersc"))


def _measure(model, links, device, warmup: int, repeats: int) -> tuple[float, int | None]:
    with torch.no_grad():
        for _ in range(warmup):
            transform_and_block(links, model)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(repeats):
        with torch.no_grad():
            transform_and_block(links, model)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats, torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--backend", choices=("eager", "compile", "both"), default="both")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    device = torch.device(args.device)
    links, _ = read_nersc_gauge(args.input)
    links = torch.from_numpy(links).to(device)
    model = GaugeEquivariantFieldTransform().to(device)
    results = []
    if args.backend in ("eager", "both"):
        elapsed, memory = _measure(model, links, device, args.warmup, args.repeats)
        results.append({"backend": "eager", "average_seconds": elapsed, "peak_memory_bytes": memory})
    if args.backend in ("compile", "both"):
        compiled = torch.compile(model, dynamic=False, fullgraph=False, mode="default")
        with torch.no_grad():
            compiled(links.unsqueeze(0))
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed, memory = _measure(compiled, links, device, args.warmup, args.repeats)
        results.append({"backend": "torch.compile", "average_seconds": elapsed, "peak_memory_bytes": memory})
    result = {"device": str(device), "input": str(args.input), "results": results}
    print(json.dumps(result, indent=2))
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
