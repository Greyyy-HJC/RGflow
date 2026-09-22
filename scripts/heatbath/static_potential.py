#!/usr/bin/env python3
"""Measure the static potential and set the lattice spacing."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from pyquda_utils import core, io
from pyquda_utils.core import T, X, Y, Z


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "artifacts" / "4dsu3" / "L16_beta5p95"
DEFAULT_QUDA_PATH = Path("/home/jinchen/package/quda/install")
DEFAULT_MPICXX = Path("/home/jinchen/software/openmpi/bin/mpicxx")
HBARC_MEV_FM = 197.3269804


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
    bridge.rgflow_gauge_loop_trace.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    bridge.rgflow_gauge_loop_trace.restype = ctypes.c_int
    bridge.rgflow_heatbath_last_error.restype = ctypes.c_char_p
    return bridge


def loop_trace(bridge, gauge, lattice: list[int], loops: list[list[int]]) -> np.ndarray:
    lengths = np.asarray([len(path) for path in loops], dtype=np.int32)
    max_length = int(lengths.max())
    paths = np.full((len(loops), max_length), -1, dtype=np.int32)
    for index, path in enumerate(loops):
        paths[index, : len(path)] = [direction if direction < 4 else 7 - (direction - 4) for direction in path]
    real = np.empty(len(loops), dtype=np.float64)
    imag = np.empty(len(loops), dtype=np.float64)
    dims = (ctypes.c_int * 4)(*lattice)
    pointers = (ctypes.c_void_p * 4)(*[gauge.data[direction].ctypes.data for direction in range(4)])
    status = bridge.rgflow_gauge_loop_trace(
        dims,
        pointers,
        paths.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        lengths.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        len(loops),
        max_length,
        real.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        imag.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if status:
        raise RuntimeError(bridge.rgflow_heatbath_last_error().decode() or "gauge loop trace failed")
    return real + 1j * imag


def rectangular_loops(r_max: int, t_max: int):
    loops = []
    for radius in range(1, r_max + 1):
        for extent in range(1, t_max + 1):
            for direction in (X, Y, Z):
                loops.append(
                    [direction] * radius
                    + [T] * extent
                    + [direction + 4] * radius
                    + [T + 4] * extent
                )
    return loops


def extract_fit(wilson_loops: np.ndarray, t_min: int, t_max: int, r_min: int, r_max: int):
    times = np.arange(t_min, t_max + 1, dtype=float)
    potential = np.empty(wilson_loops.shape[0])
    for index, values in enumerate(wilson_loops[:, t_min - 1 : t_max]):
        if np.any(values <= 0) or not np.all(np.isfinite(values)):
            return None
        slope, _ = np.polyfit(times, np.log(values), 1)
        potential[index] = -slope

    radii = np.arange(r_min, r_max + 1, dtype=float)
    design = np.column_stack([np.ones_like(radii), radii, -1.0 / radii])
    offset, sigma, alpha = np.linalg.lstsq(design, potential[r_min - 1 : r_max], rcond=None)[0]
    scale_argument = (1.65 - alpha) / sigma
    if sigma <= 0 or scale_argument <= 0 or not np.isfinite(scale_argument):
        return None
    r0_over_a = np.sqrt(scale_argument)
    return potential, np.array([offset, sigma, alpha, r0_over_a])


def uncertainty(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--ape-steps", type=int, default=20)
    parser.add_argument("--ape-alpha", type=float, default=0.5)
    parser.add_argument("--expected-configurations", type=int, default=50)
    parser.add_argument("--r-max", type=int, default=6)
    parser.add_argument("--t-max", type=int, default=6)
    parser.add_argument("--potential-t-min", type=int, default=2)
    parser.add_argument("--potential-t-max", type=int, default=5)
    parser.add_argument("--cornell-r-min", type=int, default=2)
    parser.add_argument("--cornell-r-max", type=int, default=6)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=12345)
    parser.add_argument("--max-invalid-fraction", type=float, default=0.2)
    parser.add_argument("--r0-fm", type=float, default=0.5)
    parser.add_argument("--sqrt-sigma-mev", type=float, default=440.0)
    parser.add_argument("--quda-path", type=Path, default=Path(os.environ.get("QUDA_PATH", DEFAULT_QUDA_PATH)))
    parser.add_argument("--mpicxx", type=Path, default=Path(os.environ.get("MPICXX", DEFAULT_MPICXX)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input.resolve()
    manifest = json.loads((input_dir / "ensemble.json").read_text())
    lattice = manifest["lattice_size"]
    records = manifest["configurations"]
    if len(records) != args.expected_configurations:
        raise RuntimeError(f"expected {args.expected_configurations} configurations, found {len(records)}")
    if len({record["sha256"] for record in records}) != len(records):
        raise RuntimeError("duplicate gauge configurations in ensemble manifest")

    bridge = load_bridge(build_bridge(input_dir, args.quda_path.resolve(), args.mpicxx))
    tuning_cache = input_dir / ".quda-cache"
    tuning_cache.mkdir(exist_ok=True)
    core.init(None, lattice, backend="numpy", resource_path=str(tuning_cache))
    volume = int(np.prod(lattice))
    measured = np.empty((len(records), args.r_max, args.t_max), dtype=float)
    plaquettes = np.asarray([record["plaquette"] for record in records], dtype=float)

    loop_paths = rectangular_loops(args.r_max, args.t_max)
    for index, record in enumerate(records):
        gauge = io.readNERSCGauge(
            str(input_dir / record["filename"]),
            checksum=True,
            plaquette=False,
            link_trace=False,
            reunitarize_sigma=0,
        )
        if gauge.latt_info.global_size != lattice:
            raise RuntimeError(f"wrong lattice size in {record['filename']}")
        plaquette = float(gauge.plaquette()[0])
        if not np.isclose(plaquette, record["plaquette"], atol=2e-7, rtol=0):
            raise RuntimeError(f"plaquette mismatch in {record['filename']}")
        gauge.apeSmear(args.ape_steps, args.ape_alpha, T)
        traces = loop_trace(bridge, gauge, lattice, loop_paths).real.reshape(args.r_max, args.t_max, 3)
        measured[index] = traces.mean(axis=2) / (3.0 * volume)
        print(f"  measured {index + 1:02d}/{len(records)}", flush=True)

    mean_loops = measured.mean(axis=0)
    loop_errors = measured.std(axis=0, ddof=1) / np.sqrt(len(records))
    central = extract_fit(
        mean_loops,
        args.potential_t_min,
        args.potential_t_max,
        args.cornell_r_min,
        args.cornell_r_max,
    )
    if central is None:
        raise RuntimeError("central Wilson-loop data do not give a physical Cornell fit")
    potential, parameters = central

    rng = np.random.default_rng(args.bootstrap_seed)
    bootstrap_potentials = []
    bootstrap_parameters = []
    bootstrap_effective = []
    nonpositive_bootstrap = 0
    nonphysical_fit_bootstrap = 0
    for _ in range(args.bootstrap):
        sample = measured[rng.integers(0, len(records), len(records))].mean(axis=0)
        if np.any(sample <= 0) or not np.all(np.isfinite(sample)):
            nonpositive_bootstrap += 1
            continue
        fit = extract_fit(
            sample,
            args.potential_t_min,
            args.potential_t_max,
            args.cornell_r_min,
            args.cornell_r_max,
        )
        if fit is None:
            nonphysical_fit_bootstrap += 1
            continue
        sample_potential, sample_parameters = fit
        bootstrap_potentials.append(sample_potential)
        bootstrap_parameters.append(sample_parameters)
        with np.errstate(divide="ignore", invalid="ignore"):
            bootstrap_effective.append(np.log(sample[:, :-1] / sample[:, 1:]))

    invalid = args.bootstrap - len(bootstrap_parameters)
    if invalid / args.bootstrap > args.max_invalid_fraction:
        raise RuntimeError(
            f"{invalid}/{args.bootstrap} bootstrap samples were invalid "
            f"({nonpositive_bootstrap} nonpositive loops, {nonphysical_fit_bootstrap} nonphysical fits)"
        )
    bootstrap_potentials = np.asarray(bootstrap_potentials)
    bootstrap_parameters = np.asarray(bootstrap_parameters)
    bootstrap_effective = np.asarray(bootstrap_effective)

    offset, sigma, alpha, r0_over_a = parameters
    a_r0_fm = args.r0_fm / r0_over_a
    a_sigma_fm = np.sqrt(sigma) * HBARC_MEV_FM / args.sqrt_sigma_mev
    bootstrap_a_r0 = args.r0_fm / bootstrap_parameters[:, 3]
    bootstrap_a_sigma = np.sqrt(bootstrap_parameters[:, 1]) * HBARC_MEV_FM / args.sqrt_sigma_mev
    with np.errstate(divide="ignore", invalid="ignore"):
        effective = np.log(mean_loops[:, :-1] / mean_loops[:, 1:])
    effective_errors = np.std(bootstrap_effective, axis=0, ddof=1)

    result = {
        "ensemble": {
            "lattice_size": lattice,
            "beta": manifest["beta"],
            "configurations": len(records),
            "mean_plaquette": float(plaquettes.mean()),
            "plaquette_standard_error": float(plaquettes.std(ddof=1) / np.sqrt(len(plaquettes))),
        },
        "measurement": {
            "ape_steps": args.ape_steps,
            "ape_alpha": args.ape_alpha,
            "radii": list(range(1, args.r_max + 1)),
            "times": list(range(1, args.t_max + 1)),
            "potential_time_fit": [args.potential_t_min, args.potential_t_max],
            "cornell_radius_fit": [args.cornell_r_min, args.cornell_r_max],
            "bootstrap_samples": args.bootstrap,
            "valid_bootstrap_samples": len(bootstrap_parameters),
            "nonpositive_bootstrap_samples": nonpositive_bootstrap,
            "nonphysical_fit_bootstrap_samples": nonphysical_fit_bootstrap,
        },
        "wilson_loops": [
            {
                "r_over_a": radius,
                "values": [
                    {
                        "t_over_a": extent,
                        "mean": float(mean_loops[radius - 1, extent - 1]),
                        "standard_error": float(loop_errors[radius - 1, extent - 1]),
                    }
                    for extent in range(1, args.t_max + 1)
                ],
            }
            for radius in range(1, args.r_max + 1)
        ],
        "effective_potential": [
            {
                "r_over_a": radius,
                "values": [
                    {
                        "t_over_a": extent,
                        "aV_effective": float(effective[radius - 1, extent - 1]),
                        "error": float(effective_errors[radius - 1, extent - 1]),
                    }
                    for extent in range(1, args.t_max)
                ],
            }
            for radius in range(1, args.r_max + 1)
        ],
        "cornell_fit": {
            "offset": float(offset),
            "offset_error": uncertainty(bootstrap_parameters[:, 0]),
            "sigma_a2": float(sigma),
            "sigma_a2_error": uncertainty(bootstrap_parameters[:, 1]),
            "alpha": float(alpha),
            "alpha_error": uncertainty(bootstrap_parameters[:, 2]),
        },
        "scale": {
            "r0_physical_fm": args.r0_fm,
            "r0_over_a": float(r0_over_a),
            "r0_over_a_error": uncertainty(bootstrap_parameters[:, 3]),
            "a_from_r0_fm": float(a_r0_fm),
            "a_from_r0_fm_error": uncertainty(bootstrap_a_r0),
            "sqrt_sigma_physical_mev": args.sqrt_sigma_mev,
            "a_from_string_tension_fm": float(a_sigma_fm),
            "a_from_string_tension_fm_error": uncertainty(bootstrap_a_sigma),
        },
        "potential": [
            {
                "r_over_a": radius,
                "aV": float(potential[radius - 1]),
                "error": uncertainty(bootstrap_potentials[:, radius - 1]),
            }
            for radius in range(1, args.r_max + 1)
        ],
    }
    (input_dir / "static_potential.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(
        input_dir / "static_potential.npz",
        wilson_loops=measured,
        mean_wilson_loops=mean_loops,
        wilson_loop_standard_error=loop_errors,
        effective_potential=effective,
        effective_potential_bootstrap_standard_error=effective_errors,
        potential=potential,
        potential_bootstrap=bootstrap_potentials,
        cornell_parameters=parameters,
        cornell_parameters_bootstrap=bootstrap_parameters,
        plaquettes=plaquettes,
    )

    print(f"r0/a = {r0_over_a:.5f} +/- {uncertainty(bootstrap_parameters[:, 3]):.5f}")
    print(f"a(r0=0.5 fm) = {a_r0_fm:.5f} +/- {uncertainty(bootstrap_a_r0):.5f} fm")
    print(f"a(sqrt(sigma)=440 MeV) = {a_sigma_fm:.5f} +/- {uncertainty(bootstrap_a_sigma):.5f} fm")
    print(f"valid bootstrap fits = {len(bootstrap_parameters)}/{args.bootstrap}")


if __name__ == "__main__":
    main()
