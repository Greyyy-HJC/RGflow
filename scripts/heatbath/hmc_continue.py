#!/usr/bin/env python3
"""Continue the last reference field with Wilson pure-gauge HMC."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
from pyquda.action import GaugeAction, LoopParam
from pyquda.hmc import HMC, O2Nf2Ng0V
from pyquda.enum_quda import QudaVerbosity
from pyquda.quda import setVerbosityQuda
from pyquda_utils import core, io


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "artifacts/4dsu3/L24_beta6p20")
    parser.add_argument("--output", type=Path, default=ROOT / "presentation/results/topo_charge/hmc")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "temp/topo_charge_hmc_L24")
    parser.add_argument("--trajectories", type=int, default=300)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--trajectory-length", type=float, default=1.0)
    parser.add_argument("--md-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261015)
    args = parser.parse_args()
    if min(args.trajectories, args.save_every, args.md_steps, args.trajectory_length) <= 0:
        parser.error("trajectory count, save interval, MD steps and trajectory length must be positive")
    if args.trajectories % args.save_every:
        parser.error("trajectories must be a multiple of save-every")

    input_dir, output, config_dir = args.input.resolve(), args.output.resolve(), args.config_dir.resolve()
    manifest = json.loads((input_dir / "ensemble.json").read_text())
    source = max(manifest["configurations"], key=lambda record: record["index"])
    source_path = input_dir / source["filename"]
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != source["sha256"]:
        parser.error("the last reference configuration does not match its manifest hash")
    settings = {
        "source": {"input": str(input_dir), **source},
        "lattice_size": manifest["lattice_size"], "beta": manifest["beta"],
        "action": "Wilson pure gauge SU(3), periodic in all four directions",
        "trajectories": args.trajectories, "save_every": args.save_every,
        "trajectory_length": args.trajectory_length, "md_steps": args.md_steps,
        "md_step_size": args.trajectory_length / args.md_steps,
        "integrator": "second-order Omelyan O2Nf2Ng0V", "lambda": O2Nf2Ng0V.lambda_,
        "projection_tolerance": 2e-15,
        "seed": args.seed, "precision": "double", "pyquda_version": version("pyquda"),
        "metropolis": True, "warmup": 0, "config_dir": str(config_dir),
        "saved_state": "post-Metropolis chain state; rejected trajectories count as HMC steps",
        "storage": "NERSC IEEE64 little-endian 3x3",
    }
    output.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "run.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if previous["settings"] != settings:
            parser.error("output contains a different HMC run; choose a new --output and --config-dir")
        if previous["complete"]:
            print(f"All {args.trajectories} HMC trajectories are already complete: {output}")
            return

    cache = ROOT / "artifacts/4dsu3/topo_charge/.quda-cache"
    core.init(None, manifest["lattice_size"], backend="numpy", resource_path=str(cache))
    setVerbosityQuda(QudaVerbosity.QUDA_SILENT, b"")
    gauge = io.readNERSCGauge(str(source_path), checksum=True, plaquette=False,
                             link_trace=False, reunitarize_sigma=0)
    loops = LoopParam([[mu, nu, mu + 4, nu + 4]
                       for mu in range(4) for nu in range(mu + 1, 4)], [1.0] * 6)
    hmc = HMC(gauge.latt_info, [GaugeAction(gauge.latt_info, loops, manifest["beta"])],
              O2Nf2Ng0V(args.md_steps))
    hmc.initialize(args.seed, gauge)
    plaquette = np.full(args.trajectories + 1, np.nan)
    delta_h = np.full(args.trajectories, np.nan)
    acceptance_probability = np.full(args.trajectories, np.nan)
    accepted = np.zeros(args.trajectories, dtype=bool)
    elapsed_seconds = np.full(args.trajectories, np.nan)
    plaquette[0] = hmc.plaquette()[0]
    np.testing.assert_allclose(plaquette[0], source["plaquette"], rtol=0, atol=2e-7)
    start = 0
    records = []
    checkpoint = config_dir / "checkpoint.npz"
    if checkpoint.exists():
        with np.load(checkpoint) as saved:
            if json.loads(str(saved["settings"])) != settings:
                parser.error("the temporary HMC checkpoint has different settings")
            start = int(saved["trajectory"])
            gauge.data = saved["gauge"].copy()
            plaquette[:start + 1] = saved["plaquette"]
            delta_h[:start] = saved["delta_h"]
            acceptance_probability[:start] = saved["acceptance_probability"]
            accepted[:start] = saved["accepted"]
            elapsed_seconds[:start] = saved["elapsed_seconds"]
            records = json.loads(str(saved["records"]))
            rng_state = json.loads(str(saved["rng_state"]))
        hmc.loadGauge(gauge)
        hmc.random.setstate((rng_state[0], tuple(rng_state[1]), rng_state[2]))
        print(f"Resuming from trajectory {start}", flush=True)

    print(f"Starting from reference ID {source['index']}: {source['filename']}", flush=True)
    print(f"HMC: {args.trajectories} trajectories, length={args.trajectory_length:g}, "
          f"{args.md_steps} Omelyan MD steps, seed={args.seed}", flush=True)
    for trajectory in range(start + 1, args.trajectories + 1):
        row = trajectory - 1
        started = time.perf_counter()
        hmc.gaussMom()
        initial_hamiltonian = hmc.gaugeAction() + hmc.momAction()
        hmc.integrate(args.trajectory_length, project_tol=settings["projection_tolerance"])
        delta_h[row] = hmc.gaugeAction() + hmc.momAction() - initial_hamiltonian
        if not np.isfinite(delta_h[row]):
            raise RuntimeError(f"nonfinite Hamiltonian at trajectory {trajectory}")
        acceptance_probability[row] = np.exp(min(0.0, -delta_h[row]))
        accepted[row] = hmc.accept(delta_h[row])
        if accepted[row]:
            hmc.saveGauge(gauge)
        else:
            hmc.loadGauge(gauge)
        plaquette[trajectory] = hmc.plaquette()[0]
        elapsed_seconds[row] = time.perf_counter() - started
        if trajectory % args.save_every == 0:
            filename = f"hmc_L24_beta6p20_traj_{trajectory:04d}.nersc"
            path = config_dir / filename
            io.writeNERSCGauge(str(path), gauge, use_fp32=False,
                               ensemble_label="L24 Wilson HMC from reference cfg 0299",
                               sequence_number=trajectory)
            records.append({"index": trajectory, "step": trajectory, "filename": filename,
                            "seed": args.seed, "plaquette": float(plaquette[trajectory]),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "bytes": path.stat().st_size})
            temporary = config_dir / "checkpoint.tmp.npz"
            np.savez(temporary, trajectory=trajectory, gauge=gauge.data,
                     plaquette=plaquette[:trajectory + 1], delta_h=delta_h[:trajectory],
                     acceptance_probability=acceptance_probability[:trajectory],
                     accepted=accepted[:trajectory], elapsed_seconds=elapsed_seconds[:trajectory],
                     rng_state=json.dumps(hmc.random.getstate()), records=json.dumps(records),
                     settings=json.dumps(settings))
            temporary.replace(checkpoint)
            (config_dir / "ensemble.json").write_text(json.dumps({
                "action": "Wilson pure gauge", "gauge_group": "SU(3)",
                "lattice_size": manifest["lattice_size"], "beta": manifest["beta"],
                "seed": args.seed, "generation_method": "HMC", "hmc_settings": settings,
                "storage": settings["storage"], "configurations": records,
            }, indent=2) + "\n")
        temporary = output / "chain.tmp.npz"
        np.savez_compressed(temporary, trajectories=np.arange(trajectory + 1),
                            plaquette=plaquette[:trajectory + 1], delta_h=delta_h[:trajectory],
                            acceptance_probability=acceptance_probability[:trajectory],
                            accepted=accepted[:trajectory], elapsed_seconds=elapsed_seconds[:trajectory])
        temporary.replace(output / "chain.npz")
        metadata_path.write_text(json.dumps({
            "settings": settings, "completed_trajectories": trajectory,
            "checkpoint_trajectory": trajectory // args.save_every * args.save_every,
            "complete": trajectory == args.trajectories,
            "accepted_trajectories": int(accepted[:trajectory].sum()),
            "acceptance_rate": float(accepted[:trajectory].mean()),
            "saved_configurations": len(records),
        }, indent=2) + "\n")
        print(f"{trajectory:03d}/{args.trajectories}: P={plaquette[trajectory]:.10f}, "
              f"dH={delta_h[row]:+.6f}, accept={bool(accepted[row])}, "
              f"{elapsed_seconds[row]:.1f} s", flush=True)
    print(f"Saved {output / 'chain.npz'} and {len(records)} temporary configurations", flush=True)


if __name__ == "__main__":
    main()
