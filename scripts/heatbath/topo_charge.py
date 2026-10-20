#!/usr/bin/env python3
"""Measure clover topological charge along Wilson flow on saved gauge fields."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
from pyquda.enum_quda import QudaVerbosity
from pyquda.quda import setVerbosityQuda
from pyquda_utils import core, io


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "artifacts/4dsu3/L24_beta6p20"
DEFAULT_OUTPUT = ROOT / "presentation/results/topo_charge"


def measure_flow(gauge, epsilon: float, n_steps: int, measure_every: int):
    """Keep the flowed field on the GPU and measure at chunk endpoints."""
    steps = np.unique(np.r_[np.arange(0, n_steps + 1, measure_every), n_steps])
    flow_times = steps * epsilon
    charge = np.empty(len(steps))
    energy = np.empty((len(steps), 3))

    with gauge.use() as dirac:
        dirac.smear_param.rk_order = 3
        charge[0] = dirac.qcharge()
        energy[0] = dirac.obs_param.energy
        try:
            for point in range(1, len(steps)):
                # wilsonFlow's obs_param holds the chunk's *initial* values.
                # Measure explicitly after advancing to obtain Q(t_endpoint).
                dirac.wilsonFlow(
                    int(steps[point] - steps[point - 1]),
                    epsilon,
                    float(flow_times[point - 1]),
                    restart=point > 1,
                    compute_qcharge=False,
                )
                charge[point] = dirac.qcharge()
                energy[point] = dirac.obs_param.energy
        finally:
            dirac.freeSmearedGauge()
    return flow_times, charge, energy


def save_measurements(output, settings, indices, flow_times, charge, energy, plaquette,
                      elapsed_seconds, reused):
    """Checkpoint scalar measurements after seeding or measuring a configuration."""
    temporary = output / "measurements.tmp.npz"
    np.savez_compressed(
        temporary, config_indices=indices, flow_times=flow_times,
        topological_charge=charge, energy=energy, initial_plaquette=plaquette,
        elapsed_seconds=elapsed_seconds,
    )
    completed = np.isfinite(charge).all(axis=1)
    metadata = {
        "settings": settings,
        "completed_configurations": int(completed.sum()),
        "complete": bool(completed.all()),
        "total_measurement_seconds": float(np.nansum(elapsed_seconds)),
    }
    if reused:
        metadata["reused_measurements"] = reused
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.replace(output / "measurements.npz")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", type=int, default=0, help="first position in the sorted ensemble")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--epsilon", type=float, default=0.02, help="flow integration step in units of a^2")
    parser.add_argument("--t-max", type=float, default=5.0, help="maximum flow time in units of a^2")
    parser.add_argument("--measure-every", type=int, default=5, help="integration steps between measurements")
    parser.add_argument("--smearing-run", type=Path, help="apply this downsampling polynomial smearing before flow")
    parser.add_argument("--reuse-from", type=Path, help="reuse matching configurations from a previous result bundle")
    args = parser.parse_args()
    if args.start < 0 or min(args.stride, args.samples, args.measure_every) <= 0:
        parser.error("start must be nonnegative; stride, samples and measure-every must be positive")
    if args.epsilon <= 0 or args.t_max <= 0:
        parser.error("epsilon and t-max must be positive")
    n_steps = round(args.t_max / args.epsilon)
    if not np.isclose(n_steps * args.epsilon, args.t_max, rtol=0, atol=1e-12):
        parser.error("t-max must be an integer multiple of epsilon")

    input_dir = args.input.resolve()
    output = args.output.resolve()
    manifest = json.loads((input_dir / "ensemble.json").read_text())
    records = sorted(manifest["configurations"], key=lambda record: record["index"])
    selected = records[args.start :: args.stride][: args.samples]
    if len(selected) != args.samples:
        parser.error(f"requested {args.samples} configurations, but the selection contains {len(selected)}")

    indices = np.asarray([record["index"] for record in selected])
    steps = np.unique(np.r_[np.arange(0, n_steps + 1, args.measure_every), n_steps])
    flow_times = steps * args.epsilon
    settings = {
        "input": str(input_dir),
        "lattice_size": manifest["lattice_size"],
        "beta": manifest["beta"],
        "ensemble_size": len(records),
        "start": args.start,
        "stride": args.stride,
        "samples": args.samples,
        "epsilon": args.epsilon,
        "t_max": args.t_max,
        "measure_every": args.measure_every,
        "flow": "4D Wilson gradient flow",
        "integrator": "QUDA third-order Runge-Kutta (RK3)",
        "precision": "double",
        "charge_operator": "QUDA clover field strength, traceless SU(3), unrounded total Q",
        "time_units": "t/a^2",
        "reader": "PyQUDA NERSC reader: checksum checked; FP32 links reunitarized on input",
        "pyquda_version": version("pyquda"),
        "configurations": [
            {"index": record["index"], "filename": record["filename"], "sha256": record["sha256"],
             "seed": record.get("seed", manifest["seed"])}
            for record in selected
        ],
    }
    if manifest.get("generation_method") == "HMC":
        settings["hmc_settings"] = manifest["hmc_settings"]
        settings["input_storage"] = manifest["storage"]
        settings["reader"] = "PyQUDA NERSC reader: checksum checked; double-precision HMC links"
    if manifest.get("generation_method") == "two_normalizing_flows":
        settings["upsampling"] = {"stage": manifest["stage"], **manifest["upsampling_settings"]}
        settings["input_storage"] = manifest["storage"]
        settings["reader"] = "PyQUDA NERSC reader: checksum checked; double-precision flow outputs, reunitarized before export"
    model = None
    if args.smearing_run is not None:
        smearing_run = args.smearing_run.resolve()
        kernel_path = smearing_run / "kernel.json"
        checkpoint_path = smearing_run / "kernel.pt"
        kernel = json.loads(kernel_path.read_text())
        architecture = kernel["architecture"]
        if architecture["class"] != "PolynomialStoutKernel":
            parser.error("--smearing-run must contain a PolynomialStoutKernel checkpoint")
        settings["smearing"] = {
            "run": smearing_run.name,
            "architecture": architecture,
            "kernel_json_sha256": hashlib.sha256(kernel_path.read_bytes()).hexdigest(),
            "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "operation": "one trained polynomial/path smearing and SU(3) polar projection, without blocking",
            "input": "same reunitarized L24 reference as the unsmeared measurement",
            "precision": "single, matching downsampling; reunitarized in double before Wilson flow",
        }
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / "measurements.npz"
    metadata_path = output / "run.json"
    charge = np.full((len(selected), len(flow_times)), np.nan)
    energy = np.full((*charge.shape, 3), np.nan)
    plaquette = np.full(len(selected), np.nan)
    elapsed_seconds = np.full(len(selected), np.nan)
    reused = {}
    if data_path.exists():
        previous = json.loads(metadata_path.read_text())
        if previous["settings"] != settings:
            parser.error("output contains a different measurement; choose a new --output directory")
        with np.load(data_path) as saved:
            np.testing.assert_array_equal(saved["config_indices"], indices)
            np.testing.assert_array_equal(saved["flow_times"], flow_times)
            charge[:] = saved["topological_charge"]
            energy[:] = saved["energy"]
            plaquette[:] = saved["initial_plaquette"]
            elapsed_seconds[:] = saved["elapsed_seconds"]
        reused = previous.get("reused_measurements", {})
    if args.reuse_from is not None:
        source = args.reuse_from.resolve()
        source_settings = json.loads((source / "run.json").read_text())["settings"]
        selection_keys = {"start", "stride", "samples", "configurations"}
        common_settings = {key: value for key, value in settings.items() if key not in selection_keys}
        previous_settings = {key: value for key, value in source_settings.items() if key not in selection_keys}
        if common_settings != previous_settings:
            parser.error("--reuse-from has different input, flow or smearing settings")
        source_records = {record["index"]: record for record in source_settings["configurations"]}
        copied = []
        with np.load(source / "measurements.npz") as saved:
            np.testing.assert_array_equal(saved["flow_times"], flow_times)
            np.testing.assert_array_equal(saved["config_indices"], list(source_records))
            source_rows = {index: row for row, index in enumerate(saved["config_indices"])}
            for row, record in enumerate(settings["configurations"]):
                index = record["index"]
                if index not in source_rows or np.isfinite(charge[row]).all():
                    continue
                if record != source_records[index]:
                    parser.error(f"--reuse-from has a different source configuration {index}")
                source_row = source_rows[index]
                if not np.isfinite(saved["topological_charge"][source_row]).all():
                    continue
                charge[row] = saved["topological_charge"][source_row]
                energy[row] = saved["energy"][source_row]
                plaquette[row] = saved["initial_plaquette"][source_row]
                elapsed_seconds[row] = saved["elapsed_seconds"][source_row]
                copied.append(index)
        if copied:
            reused = {"source": str(source), "config_indices": copied}
            save_measurements(output, settings, indices, flow_times, charge, energy,
                              plaquette, elapsed_seconds, reused)
            print(f"Reused {len(copied)} configurations from {source}", flush=True)
    completed = np.isfinite(charge).all(axis=1)
    if completed.all():
        print(f"All {len(selected)} configurations are already measured: {data_path}")
        return

    if args.smearing_run is not None:
        import torch
        from pyquda_io.io_utils import gaugeReunitarize
        from rgflow.su3.downsampling import PolynomialStoutKernel

        model = PolynomialStoutKernel(
            tuple(architecture["initial_coefficients"]),
            tuple(architecture.get("initial_hook_coefficients", ())),
            tuple(architecture.get("initial_local_coefficients", ())),
        ).to(device="cuda")
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True)["state_dict"])
        model.eval()

    cache = ROOT / "artifacts/4dsu3/topo_charge/.quda-cache"
    cache.mkdir(parents=True, exist_ok=True)
    core.init(None, manifest["lattice_size"], backend="numpy", resource_path=str(cache))
    setVerbosityQuda(QudaVerbosity.QUDA_SILENT, b"")
    print(f"Selected IDs: {indices.tolist()}", flush=True)
    print(f"Wilson flow: dt={args.epsilon:g}, t/a^2=0..{args.t_max:g}, {len(flow_times)} measurements", flush=True)

    for row, record in enumerate(selected):
        if completed[row]:
            continue
        started = time.perf_counter()
        # Match the existing static-potential and PyQUDA cross-check readers.
        gauge = io.readNERSCGauge(
            str(input_dir / record["filename"]),
            checksum=True, plaquette=False, link_trace=False, reunitarize_sigma=0,
        )
        if gauge.latt_info.global_size != manifest["lattice_size"]:
            raise RuntimeError(f"wrong lattice size in {record['filename']}")
        plaquette[row] = gauge.plaquette()[0]
        if not np.isclose(plaquette[row], record["plaquette"], atol=2e-7, rtol=0):
            raise RuntimeError(f"plaquette mismatch in {record['filename']}")
        if model is not None:
            with torch.no_grad():
                links = torch.as_tensor(gauge.lexico(), dtype=torch.complex64, device="cuda")
                smeared = model(links, projection_device="cpu").cpu().numpy()
            smeared = gaugeReunitarize(smeared.astype(np.complex128), reunitarize_sigma=0)
            gauge.data = gauge.latt_info.evenodd(np.ascontiguousarray(smeared), True)
            del links, smeared
            torch.cuda.empty_cache()
        _, charge[row], energy[row] = measure_flow(gauge, args.epsilon, n_steps, args.measure_every)
        elapsed_seconds[row] = time.perf_counter() - started
        completed[row] = True
        # Save scalar measurements after each configuration so a rerun resumes.
        save_measurements(output, settings, indices, flow_times, charge, energy,
                          plaquette, elapsed_seconds, reused)
        print(
            f"  {completed.sum():02d}/{len(selected)} cfg {record['index']:04d}: "
            f"Q(0)={charge[row, 0]:+.6f}, Q({args.t_max:g})={charge[row, -1]:+.6f}, "
            f"{elapsed_seconds[row]:.1f} s", flush=True,
        )
        del gauge
    print(f"Saved {data_path}", flush=True)


if __name__ == "__main__":
    main()
