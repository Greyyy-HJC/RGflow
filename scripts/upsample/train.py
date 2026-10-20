#!/usr/bin/env python3
"""Train and evaluate two SU(3) flows on the existing heatbath ensembles."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ks_2samp, wasserstein_distance

from rgflow.su3.downsampling import PolynomialStoutKernel, block_links, su3_errors
from rgflow.su3.io import read_nersc_gauge
from rgflow.su3.observables import (
    DOWNSAMPLING_OBSERVABLES, TRAINING_OBSERVABLES, observable_vector_torch,
)
from rgflow.su3.training import (
    _covariance_inverse, _loss_coefficients, _metric_record, _stats, _write_json,
    distribution_acceptance, ensemble_paths, split_indices,
)
from rgflow.su3.upsampling import (
    EQUIVARIANT_COUPLING, STABLE_COUPLING, LEGACY_COUPLING, SU3ConditionalFlow, constrained_haar_lift, flow_roundtrip_check,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts" / "4dsu3"
DEFAULT_DOWN = ARTIFACTS / "downsample" / "L24_beta6p20_to_L12_beta5p775_polynomial_local_ls_v4_8ops_6train"
DEFAULT_RUN = ARTIFACTS / "upsample" / "L12_to_L24_2flow_v6_8ops_6train"
TRAINING_COLUMNS = [DOWNSAMPLING_OBSERVABLES.index(name) for name in TRAINING_OBSERVABLES]


def load_links(path, device):
    return torch.from_numpy(read_nersc_gauge(path)[0]).to(device)


def moment_loss(values, reference, args):
    # Per-target-sigma mean loss + log-width loss. Float64 reductions resolve
    # the narrow L24 distributions without subtractive float32 cancellation.
    reference = reference[:, TRAINING_COLUMNS].astype(np.float64)
    variance = reference.var(0, ddof=1)
    epsilon = 1.e-12 * variance
    if args.mean_metric == "diagonal":
        inverse = np.diag(1./variance)
    else:
        _, inverse = _covariance_inverse(reference, reference, .1, observable_scale=np.sqrt(variance))
    mean, _, log_variance = _stats(reference, epsilon)
    return _loss_coefficients(
        values.double(), torch.as_tensor(mean, device=values.device),
        torch.as_tensor(log_variance, device=values.device),
        torch.as_tensor(inverse, device=values.device),
        torch.full((len(TRAINING_OBSERVABLES),), args.variance_weight**-.5,
                   dtype=torch.float64, device=values.device),
        variance_epsilon=torch.as_tensor(epsilon, device=values.device),
    )


def check_training_update(flow, optimizer, best_state, args):
    coarse = None
    if args.stability_coarse_file is not None:
        coarse = torch.load(args.stability_coarse_file, map_location="cpu", weights_only=True)["coarse"]
    report = flow_roundtrip_check(flow, coarse) if args.require_stable_inverse else {"passed": True}
    if not report["passed"]:
        flow.load_state_dict(best_state)
        optimizer.state.clear()
        for group in optimizer.param_groups:
            group["lr"] *= .5
    return report


def fit_conditional_likelihood(flow1, train_ids, val_ids, smeared_field, args, history, save):
    if args.epochs == 0:
        return
    # Flow 1: exact normalized conditional Haar likelihood on blocking fibre.
    flow1.train()
    optimizer = torch.optim.Adam(flow1.parameters(), lr=args.learning_rate)
    previous = [row for row in history if row["phase"] == "conditional_nll"]
    epoch_offset = max([row["epoch"] for row in previous], default=0)
    def validation_nll():
        flow1.eval()
        with torch.no_grad():
            values = []
            for index in val_ids:
                smeared = smeared_field(index)
                values.append(float(-flow1.log_prob(smeared)/(smeared[..., 0, 0].numel()*15/16)))
        return float(np.mean(values))

    best_loss = validation_nll()
    best_state = copy.deepcopy(flow1.state_dict())
    for epoch in range(args.epochs):
        flow1.train()
        started, losses = time.time(), []
        for index in np.random.permutation(train_ids):
            with torch.no_grad():
                smeared = smeared_field(int(index))
            optimizer.zero_grad(set_to_none=True)
            nll = -flow1.log_prob(smeared) / (smeared[..., 0, 0].numel() * 15 / 16)
            nll.backward()
            torch.nn.utils.clip_grad_norm_(flow1.parameters(), 10.)
            optimizer.step()
            losses.append(float(nll.detach()))
            print(f"flow1 epoch={epoch+1} cfg={index} nll/link={losses[-1]:.6g}", flush=True)
        stability = check_training_update(flow1, optimizer, best_state, args)
        record = {"phase": "conditional_nll", "epoch": epoch_offset+epoch+1,
                  "stability": stability,
                  "train_loss": np.mean(losses), "validation_loss": validation_nll(),
                  "seconds": time.time()-started}
        if record["validation_loss"] < best_loss:
            best_loss = record["validation_loss"]
            best_state = copy.deepcopy(flow1.state_dict())
        history.append(record)
        save()
        print(record, flush=True)
        flow1.train()
    flow1.load_state_dict(best_state)
    save()



def fit_smeared_operators(flow1, train_ids, val_ids, smeared_field, reference, device, args, history, save):
    if args.smeared_moment_epochs == 0:
        return
    flow1.eval()
    # The conditional likelihood family is deliberately small. Directly
    # constrain the generated smeared operators as well. Compute operator
    # VJPs on detached fields before replaying the flow; this avoids keeping
    # the large operator graph and eight-sweep flow graph in memory together.
    optimizer = torch.optim.Adam([
        {"params": flow1.scales.parameters(), "lr": args.learning_rate*.3},
        {"params": [flow1.long_weights], "lr": args.learning_rate*3.},
    ])
    epoch_offset = max([row["epoch"] for row in history if row["phase"] == "smeared_moments"], default=0)
    def validation_loss():
        flow1.eval()
        with torch.no_grad():
            values = []
            for index in val_ids:
                coarse = block_links(smeared_field(index))
                generator = torch.Generator(device=device).manual_seed(args.seed+50000+index)
                field, _ = flow1.sample(coarse, generator=generator, compute_logdet=False)
                values.append(observable_vector_torch(field, names=TRAINING_OBSERVABLES))
        return moment_loss(torch.stack(values), reference["validation"]["smeared"], args)[1]["loss"]

    best_loss = validation_loss()
    best_state = copy.deepcopy(flow1.state_dict())
    print(f"Flow 1: {args.smeared_moment_epochs} moment epochs; initial validation loss={best_loss:.6g}", flush=True)
    for epoch in range(args.smeared_moment_epochs):
        started = time.time()
        ids = np.random.choice(train_ids, min(args.moment_configs, len(train_ids)), replace=False)
        seeds = [args.seed+40000+(epoch+epoch_offset)*100+int(i) for i in ids]
        values, inputs, fields = [], [], []
        with torch.no_grad():
            for index, seed in zip(ids, seeds):
                coarse = block_links(smeared_field(int(index)))
                generator = torch.Generator(device=device).manual_seed(seed)
                base = constrained_haar_lift(coarse, generator=generator)
                field, _ = flow1(base, compute_logdet=False)
                inputs.append(base.cpu())
                fields.append(field.cpu())
                values.append(observable_vector_torch(field, names=TRAINING_OBSERVABLES))
        del base, field
        coefficients, losses = moment_loss(torch.stack(values), reference["train"]["smeared"], args)
        optimizer.zero_grad(set_to_none=True)
        for row, (base, source) in enumerate(zip(inputs, fields)):
            field = source.to(device)
            field.requires_grad_(True)
            gradient = torch.autograd.grad(
                (observable_vector_torch(field, names=TRAINING_OBSERVABLES, checkpoint_loops=True).double()*coefficients[row].detach()).sum(), field
            )[0]
            del field
            flow1.train()
            field, _ = flow1(base.to(device), compute_logdet=False)
            field.backward(gradient)
            del field, gradient
            flow1.eval()
            if epoch == 0:
                print(f"Flow 1: epoch {epoch_offset+1} gradient {row+1}/{len(inputs)} complete", flush=True)
        torch.nn.utils.clip_grad_norm_(flow1.parameters(), 100.)
        optimizer.step()
        stability = check_training_update(flow1, optimizer, best_state, args)
        val_loss = validation_loss()
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(flow1.state_dict())
        record = {"phase": "smeared_moments", "epoch": epoch_offset+epoch+1,
                  "stability": stability,
                  "mean_metric": args.mean_metric, "variance_weight": args.variance_weight,
                  "train_loss": losses["loss"], "validation_loss": val_loss,
                  "mean": torch.stack(values).mean(0).cpu().numpy(),
                  "seconds": time.time()-started}
        history.append(record)
        save()
        print(record, flush=True)
    flow1.load_state_dict(best_state)
    save()



def fit_paired_transport(flow2, train_ids, val_ids, paths, smeared_field, device, args, history, save):
    if args.transport_epochs == 0:
        return
    # Flow 2: paired transport initialization; preserves all fine DOFs.
    flow2.train()
    optimizer = torch.optim.Adam(flow2.parameters(), lr=args.learning_rate)
    epoch_offset = max([row["epoch"] for row in history if row["phase"] == "paired_transport"], default=0)

    def validation_loss():
        flow2.eval()
        with torch.no_grad():
            values = []
            for index in val_ids:
                fine = load_links(paths[index], device)
                reconstructed, _ = flow2(smeared_field(int(index)), compute_logdet=False)
                values.append(float((reconstructed-fine).abs().square().mean()))
        return float(np.mean(values))

    best_loss = validation_loss()
    best_state = copy.deepcopy(flow2.state_dict())
    for epoch in range(args.transport_epochs):
        flow2.train()
        started = time.time()
        losses = []
        for index in np.random.permutation(train_ids):
            with torch.no_grad():
                fine = load_links(paths[index], device)
                smeared = smeared_field(int(index))
            optimizer.zero_grad(set_to_none=True)
            reconstructed, _ = flow2(smeared, compute_logdet=False)
            loss = (reconstructed-fine).abs().square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow2.parameters(), 10.)
            optimizer.step()
            losses.append(float(loss.detach()))
            print(f"flow2 paired epoch={epoch+1} cfg={index} loss={losses[-1]:.6g}", flush=True)
        stability = check_training_update(flow2, optimizer, best_state, args)
        val_loss = validation_loss()
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(flow2.state_dict())
        record = {"phase": "paired_transport", "epoch": epoch_offset+epoch+1,
                  "stability": stability,
                  "train_loss": np.mean(losses), "validation_loss": val_loss,
                  "seconds": time.time()-started}
        history.append(record)
        save()
        print(record, flush=True)
    flow2.load_state_dict(best_state)
    save()




def fit_fine_operators(flow1, flow2, train_ids, val_ids, smeared_field, reference, device, args, history, save):
    if args.moment_epochs == 0:
        return
    # Fit actual stochastic two-flow proposals, not teacher-forced smeared
    # inputs. Replay fixed latent seeds for exact streaming moment gradients.
    optimizer = torch.optim.Adam([
        {"params": flow2.scales.parameters(), "lr": args.learning_rate*.2},
        {"params": [flow2.long_weights], "lr": args.learning_rate},
    ])
    best_state = copy.deepcopy(flow2.state_dict())
    epoch_offset = max([row["epoch"] for row in history if row["phase"] == "generated_moments"], default=0)
    fingerprint = hashlib.sha256()
    for name, value in flow1.state_dict().items():
        fingerprint.update(name.encode())
        fingerprint.update(value.detach().cpu().numpy().tobytes())
    input_cache = args.output / "flow2_inputs" / fingerprint.hexdigest()[:12]
    input_cache.mkdir(parents=True, exist_ok=True)

    def generated_input(index, seed):
        path = input_cache / f"cfg_{index:04d}_seed_{seed}.pt"
        if path.exists():
            return torch.load(path, map_location=device, weights_only=True)
        coarse = block_links(smeared_field(int(index)))
        generator = torch.Generator(device=device).manual_seed(seed)
        field, _ = flow1.sample(coarse, generator=generator, compute_logdet=False)
        temporary = path.with_suffix(".tmp")
        torch.save(field.cpu(), temporary)
        temporary.replace(path)
        return field

    def validation_loss():
        flow2.eval()
        with torch.no_grad():
            values = []
            for index in val_ids:
                smeared = generated_input(index, args.seed+20000+index)
                fine, _ = flow2(smeared, compute_logdet=False)
                values.append(observable_vector_torch(fine, names=TRAINING_OBSERVABLES))
        return moment_loss(torch.stack(values), reference["validation"]["fine"], args)[1]["loss"]

    best_loss = validation_loss()

    print(f"Flow 2: {args.moment_epochs} moment epochs; initial validation loss={best_loss:.6g}", flush=True)
    for epoch in range(args.moment_epochs):
        started = time.time()
        ids = np.random.choice(train_ids, min(args.fine_moment_configs, len(train_ids)), replace=False)
        # Four latent draws per training source form a reusable input pool.
        # Validation seeds remain fixed and separate from the training pool.
        seeds = [args.seed + 10000 + ((epoch+epoch_offset) % 4)*100 + int(i) for i in ids]
        values, inputs, fields = [], [], []
        flow2.eval()
        with torch.no_grad():
            for index, seed in zip(ids, seeds):
                smeared = generated_input(int(index), seed)
                fine, _ = flow2(smeared, compute_logdet=False)
                inputs.append(smeared.cpu())
                fields.append(fine.cpu())
                values.append(observable_vector_torch(fine, names=TRAINING_OBSERVABLES))
        del smeared, fine
        coefficients, losses = moment_loss(torch.stack(values), reference["train"]["fine"], args)
        optimizer.zero_grad(set_to_none=True)
        flow2.train()
        for row, (source, field) in enumerate(zip(inputs, fields)):
            fine = field.to(device).requires_grad_(True)
            gradient = torch.autograd.grad(
                (observable_vector_torch(fine, names=TRAINING_OBSERVABLES, checkpoint_loops=True).double()*coefficients[row].detach()).sum(), fine
            )[0]
            del fine
            fine, _ = flow2(source.to(device), compute_logdet=False)
            fine.backward(gradient)
            del fine, gradient
            if epoch == 0:
                print(f"Flow 2: epoch {epoch_offset+1} gradient {row+1}/{len(inputs)} complete", flush=True)
        torch.nn.utils.clip_grad_norm_(flow2.parameters(), 100.)
        optimizer.step()
        stability = check_training_update(flow2, optimizer, best_state, args)
        # Independent validation latent seeds, fixed across epochs.
        val_loss = validation_loss()
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(flow2.state_dict())
        record = {"phase": "generated_moments", "epoch": epoch_offset+epoch+1,
                  "stability": stability,
                  "mean_metric": args.mean_metric, "variance_weight": args.variance_weight,
                  "train_loss": losses["loss"], "validation_loss": val_loss,
                  "mean": torch.stack(values).mean(0).cpu().numpy(),
                  "seconds": time.time()-started}
        history.append(record)
        save()
        print(record, flush=True)
    flow2.load_state_dict(best_state)
    save()



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-dir", type=Path, default=ARTIFACTS / "L24_beta6p20")
    parser.add_argument("--downsampling-run", type=Path, default=DEFAULT_DOWN)
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--coupling", choices=(EQUIVARIANT_COUPLING, STABLE_COUPLING, LEGACY_COUPLING),
                        default=EQUIVARIANT_COUPLING)
    parser.add_argument("--sweeps", type=int, default=8)
    parser.add_argument("--fine-sweeps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--transport-epochs", type=int, default=3)
    parser.add_argument("--smeared-moment-epochs", type=int, default=20)
    parser.add_argument("--moment-epochs", type=int, default=20)
    parser.add_argument("--train-configs", type=int, default=12)
    parser.add_argument("--validation-configs", type=int, default=8)
    parser.add_argument("--moment-configs", type=int, default=6)
    parser.add_argument("--fine-moment-configs", type=int, default=12)
    parser.add_argument("--test-configs", type=int, default=0, help="0 means full held-out test")
    parser.add_argument("--learning-rate", type=float, default=.01)
    parser.add_argument("--mean-metric", choices=("covariance", "diagonal"), default="covariance")
    parser.add_argument("--variance-weight", type=float, default=1.)
    parser.add_argument("--require-stable-inverse", action="store_true",
                        help="reject numerically unstable updates and reduce their learning rates")
    parser.add_argument("--stability-coarse-file", type=Path,
                        help="add an actual-volume roundtrip check using this validation coarse field")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--skip-evaluation", action="store_true", help="train before calibration; evaluate afterwards")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--pyquda-check", action="store_true")
    parser.add_argument("--export-fields", action="store_true",
                        help="save both generated stages as FP64 NERSC fields for Wilson-flow measurements")
    args = parser.parse_args()
    if min(args.sweeps, args.fine_sweeps, args.train_configs, args.validation_configs,
           args.moment_configs) < 2:
        raise SystemExit("sweeps and ensemble subsets must each be at least two")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths, manifest = ensemble_paths(args.fine_dir.resolve())
    down = json.loads((args.downsampling_run / "kernel.json").read_text())
    down_hash = hashlib.sha256((args.downsampling_run / "kernel.pt").read_bytes()).hexdigest()
    splits = down["data_split"]
    if splits != split_indices(len(paths)):
        raise SystemExit("downsampling split does not match fine ensemble")
    architecture = down["architecture"]
    smoother = PolynomialStoutKernel(
        architecture["coefficients"], architecture["hook_coefficients"],
        architecture["local_coefficients"],
    ).to(device).eval()
    for parameter in smoother.parameters():
        parameter.requires_grad_(False)
    train_ids = splits["train"][:args.train_configs]
    val_ids = splits["validation"][:args.validation_configs]
    test_ids = splits["test"][:args.test_configs or None]
    selected = sorted(set(train_ids + val_ids + test_ids))
    field_cache = output / "smeared_fields"
    field_cache.mkdir(exist_ok=True)
    cache_metadata = {
        "fine_dir": str(args.fine_dir.resolve()),
        "downsampling_checkpoint_sha256": down_hash,
        "observable_names": DOWNSAMPLING_OBSERVABLES,
        "training_observables": TRAINING_OBSERVABLES,
        "held_out_observables": [name for name in DOWNSAMPLING_OBSERVABLES if name not in TRAINING_OBSERVABLES],
        "source_sha256": [record["sha256"] for record in manifest["configurations"]],
    }
    metadata_path = output / "cache_metadata.json"
    if metadata_path.exists():
        if json.loads(metadata_path.read_text()) != json.loads(json.dumps(cache_metadata)):
            raise SystemExit("cached fields use a different source, checkpoint or observable set; choose a new output")
    elif (output / "targets.npz").exists() or any(field_cache.glob("*.pt")):
        raise SystemExit("old caches have no source/observable metadata; choose a new output")
    _write_json(metadata_path, cache_metadata)

    def smeared_field(index):
        field_path = field_cache / f"cfg_{index:04d}.pt"
        if field_path.exists():
            field = torch.load(field_path, map_location=device, weights_only=True)
            return field.to(torch.complex128) if args.coupling == STABLE_COUPLING else field
        with torch.no_grad():
            field = smoother(load_links(paths[index], device), projection_device="cpu")
            temporary = field_path.with_suffix(".tmp")
            torch.save(field.cpu(), temporary)
            temporary.replace(field_path)
        return field.to(torch.complex128) if args.coupling == STABLE_COUPLING else field

    cache = output / "targets.npz"
    if not cache.exists():
        measurements = {"fine": [], "smeared": []}
        with torch.no_grad():
            for index in selected:
                fine = load_links(paths[index], device)
                smeared = smeared_field(index)
                measurements["fine"].append(observable_vector_torch(fine, names=DOWNSAMPLING_OBSERVABLES).cpu().numpy())
                measurements["smeared"].append(observable_vector_torch(smeared, names=DOWNSAMPLING_OBSERVABLES).cpu().numpy())
                print(f"target {index}: fine={measurements['fine'][-1]} smeared={measurements['smeared'][-1]}", flush=True)
        np.savez(cache, indices=selected, observable_names=DOWNSAMPLING_OBSERVABLES,
                 training_observables=TRAINING_OBSERVABLES, **measurements)
    targets = np.load(cache)
    np.testing.assert_array_equal(targets["observable_names"], DOWNSAMPLING_OBSERVABLES)
    target_map = {int(i): n for n, i in enumerate(targets["indices"])}
    reference = {split: {kind: targets[kind][[target_map[i] for i in ids]]
                        for kind in ("fine", "smeared")}
                 for split, ids in [("train", train_ids), ("validation", val_ids), ("test", test_ids)]}
    numerical_sample = output / "numerical_sample.pt"
    if not numerical_sample.exists():
        with torch.no_grad():
            torch.save({"coarse": block_links(smeared_field(val_ids[0])).cpu(),
                        "source_index": val_ids[0]}, numerical_sample)
    checkpoint_path = output / "flows.pt"
    saved = None
    if args.resume or args.evaluate_only:
        saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
        args.coupling = saved.get("coupling", LEGACY_COUPLING)
        args.sweeps, args.fine_sweeps = saved["sweeps"], saved["fine_sweeps"]
    flow1 = SU3ConditionalFlow(sweeps=args.sweeps, constrained=True, initial_scale=-.18,
                               coupling=args.coupling).to(device)
    flow2 = SU3ConditionalFlow(sweeps=args.fine_sweeps, initial_scale=.04,
                               coupling=args.coupling).to(device)
    history = []
    if saved is not None:
        flow1.load_state_dict(saved["flow1"])
        flow2.load_state_dict(saved["flow2"])
        history = json.loads((output / "history.json").read_text())
        if saved["downsampling_checkpoint_sha256"] != down_hash:
            raise SystemExit("flow checkpoint was trained with a different downsampling checkpoint")
    if args.require_stable_inverse:
        coarse = (torch.load(args.stability_coarse_file, map_location="cpu", weights_only=True)["coarse"]
                  if args.stability_coarse_file is not None else None)
        for stage, flow in enumerate((flow1, flow2), 1):
            print(f"Checking initial Flow {stage} stability", flush=True)
            report = flow_roundtrip_check(flow, coarse)
            assert report["passed"], report
            print(f"Initial Flow {stage} stability passed", flush=True)

    def save():
        temporary = checkpoint_path.with_suffix(".tmp")
        torch.save({"flow1": flow1.state_dict(), "flow2": flow2.state_dict(),
                    "sweeps": args.sweeps, "fine_sweeps": args.fine_sweeps,
                    "coupling": args.coupling, **cache_metadata}, temporary)
        temporary.replace(checkpoint_path)
        _write_json(output / "history.json", history)

    if not args.evaluate_only:
        if not (output / "training_arguments.json").exists():
            _write_json(output / "training_arguments.json", vars(args))
        invocation_path = output / "training_invocations.json"
        invocations = json.loads(invocation_path.read_text()) if invocation_path.exists() else []
        invocations.append(vars(args).copy())
        _write_json(invocation_path, invocations)
        fit_conditional_likelihood(flow1, train_ids, val_ids, smeared_field, args, history, save)
        fit_smeared_operators(flow1, train_ids, val_ids, smeared_field, reference,
                              device, args, history, save)
        flow1.eval()
        for parameter in flow1.parameters():
            parameter.requires_grad_(False)
        fit_paired_transport(flow2, train_ids, val_ids, paths, smeared_field, device, args, history, save)
        fit_fine_operators(flow1, flow2, train_ids, val_ids, smeared_field, reference,
                           device, args, history, save)
        save()

    if args.skip_evaluation:
        return

    flow1.eval()
    flow2.eval()
    # One stochastic draw per held-out source configuration. The generated
    # lattice receives only C=block(smear(U)) and fresh Haar detail noise.
    values = {"flow1_smeared": [], "flow2_fine": [], "teacher_fine": []}
    max_errors = np.zeros(2)
    block_error = 0.
    crosscheck = []
    stage1_crosscheck = []
    export_records = {stage: [] for stage in ("flow1", "flow2", "smeared_reference")}
    if args.pyquda_check or args.export_fields:
        from pyquda_utils import core
        from rgflow.su3.training import _pyquda_gauge_from_links
        from rgflow.su3.observables import observable_vector_pyquda
        (output / ".quda-cache").mkdir(exist_ok=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        core.init(None, manifest["lattice_size"], backend="numpy", resource_path=str(output / ".quda-cache"))
    if args.export_fields:
        from pyquda_utils import io
        from pyquda_io.io_utils import gaugeReunitarize
        for stage in export_records:
            (output / "generated_fields" / stage).mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for index in test_ids:
            fine = load_links(paths[index], device)
            smeared_target = smeared_field(index)
            coarse = block_links(smeared_target)
            generator = torch.Generator(device=device).manual_seed(args.seed+30000+index)
            smeared, _ = flow1.sample(coarse, generator=generator, compute_logdet=False)
            generated, _ = flow2(smeared, compute_logdet=False)
            if index == test_ids[0]:
                torch.save({"links": generated.cpu(), "smeared": smeared.cpu(),
                            "coarse": coarse.cpu(), "source_index": index,
                            "seed": args.seed+30000+index}, output / "sample.pt")
            if args.export_fields:
                for stage, field in (("flow1", smeared), ("flow2", generated),
                                     ("smeared_reference", smeared_target)):
                    links = field.cpu().numpy().astype(np.complex128)
                    projected = gaugeReunitarize(links.copy(), reunitarize_sigma=0)
                    projection_error = float(np.max(np.abs(projected-links)))
                    gauge = _pyquda_gauge_from_links(projected, manifest["lattice_size"])
                    filename = f"{stage}_cfg_{index:04d}.nersc"
                    path = output / "generated_fields" / stage / filename
                    io.writeNERSCGauge(str(path), gauge, use_fp32=False,
                                       ensemble_label=f"Two-flow {stage}, no rethermalization",
                                       sequence_number=index)
                    export_records[stage].append({
                        "index": index, "filename": filename, "seed": args.seed+30000+index,
                        "source_index": index, "plaquette": float(gauge.plaquette()[0]),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "bytes": path.stat().st_size,
                        "double_reunitarization_max_abs_change": projection_error,
                        "raw_su3_errors": su3_errors(field),
                    })
            teacher, _ = flow2(smeared_target, compute_logdet=False)
            for name, field in [("flow1_smeared", smeared), ("flow2_fine", generated), ("teacher_fine", teacher)]:
                values[name].append(observable_vector_torch(field, names=DOWNSAMPLING_OBSERVABLES).cpu().numpy())
            max_errors = np.maximum(max_errors, np.maximum(su3_errors(smeared), su3_errors(generated)))
            block_error = max(block_error, float((block_links(smeared)-coarse).abs().max()))
            if args.pyquda_check and len(crosscheck) < 3:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gauge = _pyquda_gauge_from_links(generated.cpu().numpy(), manifest["lattice_size"])
                crosscheck.append(observable_vector_pyquda(gauge, names=DOWNSAMPLING_OBSERVABLES) - values["flow2_fine"][-1])
                gauge = _pyquda_gauge_from_links(smeared.cpu().numpy(), manifest["lattice_size"])
                stage1_crosscheck.append(observable_vector_pyquda(gauge, names=DOWNSAMPLING_OBSERVABLES) - values["flow1_smeared"][-1])
            print(f"test {index}: generated={values['flow2_fine'][-1]}", flush=True)
    values = {key: np.asarray(value) for key, value in values.items()}
    np.savez(output / "observable_distributions.npz", test_indices=test_ids,
             observable_names=DOWNSAMPLING_OBSERVABLES, training_observables=TRAINING_OBSERVABLES,
             fine_reference=reference["test"]["fine"], smeared_reference=reference["test"]["smeared"], **values)
    metrics = {}
    for name, kind in [("flow1_smeared", "smeared"), ("flow2_fine", "fine"), ("teacher_fine", "fine")]:
        target = reference["test"][kind].astype(np.float64)
        _, inverse = _covariance_inverse(target, target, .1, observable_scale=target.std(0, ddof=1))
        record = _metric_record(values[name], target, inverse,
                                variance_epsilon=1.e-12 * target.var(0, ddof=1))
        record["acceptance"] = distribution_acceptance(record)
        record["group_acceptance"] = {}
        for group, columns in (("training", TRAINING_COLUMNS),
                               ("held_out", [2, 3]), ("all", list(range(len(DOWNSAMPLING_OBSERVABLES))))):
            record["group_acceptance"][group] = distribution_acceptance({
                key: np.asarray(record[key])[columns]
                for key in ("standardized_shift", "mean_shift_in_target_sigma", "std_ratio")
            })
        record["ks_statistic"] = [ks_2samp(values[name][:, k], target[:, k]).statistic for k in range(len(DOWNSAMPLING_OBSERVABLES))]
        record["wasserstein_in_target_sigma"] = [wasserstein_distance(values[name][:, k], target[:, k])/target[:, k].std(ddof=1) for k in range(len(DOWNSAMPLING_OBSERVABLES))]
        metrics[name] = record
    metrics.update({"observable_names": DOWNSAMPLING_OBSERVABLES,
                    "training_observables": TRAINING_OBSERVABLES,
                    "held_out_observables": cache_metadata["held_out_observables"],
                    "source": "held-out downsampled fine ensemble",
                    "test_configurations": len(test_ids), "max_su3_errors": max_errors,
                    "max_blocking_error": block_error, "pyquda_differences": crosscheck,
                    "pyquda_stage1_differences": stage1_crosscheck,
                    "rethermalization_sweeps": 0})
    _write_json(output / "metrics.json", metrics)
    _write_json(output / "run.json", {"arguments": vars(args), "data_split": splits,
                "train_indices": train_ids, "validation_indices": val_ids, "test_indices": test_ids,
                "fine_manifest": manifest, "downsampling_architecture": architecture, **cache_metadata,
                "method": "conditional Haar-fibre likelihood + paired and operator transport",
                "coupling": args.coupling,
                "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                "gauge_equivariant": flow1.gauge_equivariant, "base": "independent normalized Haar SU(3) details",
                "stage1_free_links_per_cell": 60, "stage1_flowed_links_per_cell": 56})
    if args.export_fields:
        provenance = {
            "run": str(output), "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "downsampling_run": str(args.downsampling_run.resolve()), **cache_metadata,
            "test_indices": test_ids, "rethermalization_sweeps": 0,
            "latent_seed_formula": "seed + 30000 + source_index",
            "flow_precision": "double", "pre_wilson_flow_projection": "SU(3) reunitarization in double",
        }
        for stage, records in export_records.items():
            _write_json(output / "generated_fields" / stage / "ensemble.json", {
                "action": "Wilson pure gauge reference; generated fields are unthermalized flow proposals",
                "gauge_group": "SU(3)", "lattice_size": manifest["lattice_size"], "beta": manifest["beta"],
                "seed": args.seed, "generation_method": "two_normalizing_flows",
                "stage": stage, "upsampling_settings": provenance,
                "storage": "NERSC IEEE64 little-endian 3x3", "configurations": records,
            })
    print("FINAL", json.dumps({key: metrics[key]["acceptance"] for key in values}), flush=True)


if __name__ == "__main__":
    main()
