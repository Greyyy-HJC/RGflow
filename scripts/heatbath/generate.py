#!/usr/bin/env python3
"""Generate a Wilson pure-gauge ensemble with QUDA heatbath updates."""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from pyquda_utils import core, io
from pyquda_utils.core import LatticeGauge, LatticeInfo


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "4dsu3"
DEFAULT_QUDA_PATH = Path("/home/jinchen/package/quda/install")
DEFAULT_MPICXX = Path("/home/jinchen/software/openmpi/bin/mpicxx")


def build_bridge(output: Path, quda_path: Path, mpicxx: Path) -> Path:
    source = Path(__file__).with_name("heatbath_bridge.cpp")
    build_dir = output / ".build"
    build_dir.mkdir(parents=True, exist_ok=True)
    library = build_dir / "libheatbath_bridge.so"
    if library.exists() and library.stat().st_mtime >= source.stat().st_mtime:
        return library

    subprocess.run(
        [
            str(mpicxx),
            "-O3",
            "-std=c++20",
            "-shared",
            "-fPIC",
            f"-I{quda_path / 'include'}",
            str(source),
            f"-L{quda_path / 'lib'}",
            f"-Wl,-rpath,{quda_path / 'lib'}",
            "-lquda",
            "-o",
            str(library),
        ],
        check=True,
    )
    return library


def load_bridge(library: Path):
    bridge = ctypes.CDLL(str(library))
    bridge.rgflow_heatbath_create.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_uint64,
        ctypes.c_int,
    ]
    bridge.rgflow_heatbath_create.restype = ctypes.c_void_p
    bridge.rgflow_heatbath_update.argtypes = [
        ctypes.c_void_p,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    bridge.rgflow_heatbath_update.restype = ctypes.c_int
    bridge.rgflow_heatbath_copy_to_host.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    bridge.rgflow_heatbath_copy_to_host.restype = ctypes.c_int
    bridge.rgflow_heatbath_destroy.argtypes = [ctypes.c_void_p]
    bridge.rgflow_heatbath_last_error.restype = ctypes.c_char_p
    return bridge


def check_bridge(bridge, status: int) -> None:
    if status:
        message = bridge.rgflow_heatbath_last_error().decode()
        raise RuntimeError(message or "heatbath bridge failed")


def copy_to_gauge(bridge, context, gauge: LatticeGauge) -> None:
    pointers = (ctypes.c_void_p * 4)(*[gauge.data[direction].ctypes.data for direction in range(4)])
    check_bridge(bridge, bridge.rgflow_heatbath_copy_to_host(context, pointers))


def gauge_errors(gauge: LatticeGauge) -> tuple[float, float]:
    links = gauge.lexico().reshape(-1, 3, 3)
    identity = np.eye(3)
    unitarity = np.max(np.abs(links @ links.conj().transpose(0, 2, 1) - identity))
    determinant = np.max(np.abs(np.linalg.det(links) - 1.0))
    return float(unitarity), float(determinant)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lattice-size", type=int, default=16)
    parser.add_argument("--beta", type=float, default=5.95)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--separation", type=int, default=100)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--heatbath-hits", type=int, default=1)
    parser.add_argument("--overrelaxation-hits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--append", action="store_true", help="append configurations to an existing ensemble")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quda-path", type=Path, default=Path(os.environ.get("QUDA_PATH", DEFAULT_QUDA_PATH)))
    parser.add_argument("--mpicxx", type=Path, default=Path(os.environ.get("MPICXX", DEFAULT_MPICXX)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    beta_tag = f"{args.beta:.2f}".replace(".", "p")
    output = (args.output or DEFAULT_OUTPUT_ROOT / f"L{args.lattice_size}_beta{beta_tag}").resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "ensemble.json"
    if args.append and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["lattice_size"] != [args.lattice_size] * 4 or manifest["beta"] != args.beta:
            raise RuntimeError("existing ensemble parameters do not match the requested ensemble")
        records = manifest["configurations"]
        if records and records[-1]["index"] != len(records) - 1:
            raise RuntimeError("existing ensemble indices are not contiguous")
        generation_seeds = manifest.setdefault("generation_seeds", [manifest["seed"]])
        if args.seed not in generation_seeds:
            generation_seeds.append(args.seed)
        start_index = len(records)
        start_step = records[-1]["step"] if records else args.warmup
    else:
        manifest = None
        records = []
        start_index = 0
        start_step = args.warmup
    tuning_cache = output / ".quda-cache"
    tuning_cache.mkdir(exist_ok=True)
    library = build_bridge(output, args.quda_path.resolve(), args.mpicxx)

    lattice = [args.lattice_size] * 4
    core.init(None, lattice, backend="numpy", resource_path=str(tuning_cache))
    bridge = load_bridge(library)
    dims = (ctypes.c_int * 4)(*lattice)
    context = bridge.rgflow_heatbath_create(dims, args.seed, 0)
    if not context:
        check_bridge(bridge, 1)

    gauge = LatticeGauge(LatticeInfo(lattice))
    started = time.perf_counter()

    try:
        print(f"Thermalizing {lattice} at beta={args.beta} for {args.warmup} compound steps")
        remaining = args.warmup
        while remaining:
            chunk = min(100, remaining)
            check_bridge(
                bridge,
                bridge.rgflow_heatbath_update(
                    context, args.beta, args.heatbath_hits, args.overrelaxation_hits, chunk
                ),
            )
            remaining -= chunk
            print(f"  warmup {args.warmup - remaining}/{args.warmup}", flush=True)

        for index in range(args.samples):
            check_bridge(
                bridge,
                bridge.rgflow_heatbath_update(
                    context,
                    args.beta,
                    args.heatbath_hits,
                    args.overrelaxation_hits,
                    args.separation,
                ),
            )
            copy_to_gauge(bridge, context, gauge)
            plaquette = float(gauge.plaquette()[0])
            unitarity_error, determinant_error = gauge_errors(gauge)
            record_index = start_index + index
            filename = f"wilson_beta{beta_tag}_L{args.lattice_size}_cfg_{record_index:04d}.nersc"
            path = output / filename
            if path.exists():
                raise RuntimeError(f"refusing to overwrite existing configuration {path}")
            io.writeNERSCGauge(
                str(path),
                gauge,
                use_fp32=True,
                ensemble_id=f"RGflow-Wilson-beta{args.beta}-L{args.lattice_size}",
                ensemble_label="4D SU(3) Wilson pure gauge heatbath",
                sequence_number=record_index,
            )
            record = {
                "index": record_index,
                "step": start_step + (index + 1) * args.separation,
                "filename": filename,
                "seed": args.seed,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
                "plaquette": plaquette,
                "max_unitarity_error": unitarity_error,
                "max_determinant_error": determinant_error,
            }
            records.append(record)
            if manifest is None:
                manifest = {
                    "action": "Wilson pure gauge",
                    "gauge_group": "SU(3)",
                    "lattice_size": lattice,
                    "beta": args.beta,
                    "seed": args.seed,
                    "generation_seeds": [args.seed],
                    "start": "hot",
                    "warmup_steps": args.warmup,
                    "separation_steps": args.separation,
                    "heatbath_hits_per_step": args.heatbath_hits,
                    "overrelaxation_hits_per_step": args.overrelaxation_hits,
                    "storage": "NERSC IEEE32 little-endian 3x3",
                    "configurations": records,
                }
            manifest["configurations"] = records
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(
                f"  saved {record_index + 1:03d}/{start_index + args.samples}: plaquette={plaquette:.8f}, "
                f"unitarity={unitarity_error:.2e}",
                flush=True,
            )
    finally:
        bridge.rgflow_heatbath_destroy(context)

    hashes = [record["sha256"] for record in records]
    if len(set(hashes)) != len(hashes):
        raise RuntimeError("duplicate gauge configurations detected")
    elapsed = time.perf_counter() - started
    print(f"Generated {len(records)} configurations in {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
