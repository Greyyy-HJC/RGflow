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
from rgflow.su3.observables import observable_names, observable_vector_torch
from rgflow.su3.training import (
    _covariance_inverse, _loss_coefficients, _metric_record, _stats, _write_json,
    distribution_acceptance, ensemble_paths, split_indices,
)
from rgflow.su3.upsampling import SU3ConditionalFlow, constrained_haar_lift


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts" / "4dsu3"
DEFAULT_DOWN = ARTIFACTS / "downsample" / "L24_beta6p20_to_L12_beta5p80_polynomial_local_ls_v2"
DEFAULT_RUN = ARTIFACTS / "upsample" / "L12_to_L24_2flow_v1"


def load_links(path, device):
    return torch.from_numpy(read_nersc_gauge(path)[0]).to(device)


def moment_loss(values, reference):
    # Per-target-sigma mean loss + log-width loss. Float64 reductions resolve
    # the narrow L24 distributions without subtractive float32 cancellation.
    _, inverse = _covariance_inverse(reference, reference, .1)
    mean, _, log_variance = _stats(reference)
    return _loss_coefficients(
        values.double(), torch.as_tensor(mean, device=values.device),
        torch.as_tensor(log_variance, device=values.device),
        torch.as_tensor(inverse, device=values.device),
        torch.ones(4, dtype=torch.float64, device=values.device),
    )


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
        record = {"phase": "conditional_nll", "epoch": epoch_offset+epoch+1,
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
                values.append(observable_vector_torch(field))
        return moment_loss(torch.stack(values), reference["validation"]["smeared"])[1]["loss"]

    best_loss = validation_loss()
    best_state = copy.deepcopy(flow1.state_dict())
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
                values.append(observable_vector_torch(field))
        del base, field
        coefficients, losses = moment_loss(torch.stack(values), reference["train"]["smeared"])
        optimizer.zero_grad(set_to_none=True)
        for row, (base, source) in enumerate(zip(inputs, fields)):
            field = source.to(device)
            field.requires_grad_(True)
            gradient = torch.autograd.grad(
                (observable_vector_torch(field, checkpoint_loops=True).double()*coefficients[row].detach()).sum(), field
            )[0]
            del field
            flow1.train()
            field, _ = flow1(base.to(device), compute_logdet=False)
            field.backward(gradient)
            del field, gradient
            flow1.eval()
        torch.nn.utils.clip_grad_norm_(flow1.parameters(), 100.)
        optimizer.step()
        val_loss = validation_loss()
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(flow1.state_dict())
        record = {"phase": "smeared_moments", "epoch": epoch_offset+epoch+1,
                  "train_loss": losses["loss"], "validation_loss": val_loss,
                  "mean": torch.stack(values).mean(0).cpu().numpy(),
                  "seconds": time.time()-started}
        history.append(record)
        save()
        print(record, flush=True)
    flow1.load_state_dict(best_state)
    save()



def fit_paired_transport(flow2, train_ids, paths, smeared_field, device, args, history, save):
    if args.transport_epochs == 0:
        return
    # Flow 2: paired transport initialization; preserves all fine DOFs.
    flow2.train()
    optimizer = torch.optim.Adam(flow2.parameters(), lr=args.learning_rate)
    epoch_offset = max([row["epoch"] for row in history if row["phase"] == "paired_transport"], default=0)
    for epoch in range(args.transport_epochs):
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
        history.append({"phase": "paired_transport", "epoch": epoch_offset+epoch+1, "train_loss": np.mean(losses)})
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
                values.append(observable_vector_torch(fine))
        return moment_loss(torch.stack(values), reference["validation"]["fine"])[1]["loss"]

    best_loss = validation_loss()

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
                values.append(observable_vector_torch(fine))
        del smeared, fine
        coefficients, losses = moment_loss(torch.stack(values), reference["train"]["fine"])
        optimizer.zero_grad(set_to_none=True)
        flow2.train()
        for row, (source, field) in enumerate(zip(inputs, fields)):
            fine = field.to(device).requires_grad_(True)
            gradient = torch.autograd.grad(
                (observable_vector_torch(fine, checkpoint_loops=True).double()*coefficients[row].detach()).sum(), fine
            )[0]
            del fine
            fine, _ = flow2(source.to(device), compute_logdet=False)
            fine.backward(gradient)
            del fine, gradient
        torch.nn.utils.clip_grad_norm_(flow2.parameters(), 100.)
        optimizer.step()
        # Independent validation latent seeds, fixed across epochs.
        val_loss = validation_loss()
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(flow2.state_dict())
        record = {"phase": "generated_moments", "epoch": epoch_offset+epoch+1,
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
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--pyquda-check", action="store_true")
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

    def smeared_field(index):
        field_path = field_cache / f"cfg_{index:04d}.pt"
        if field_path.exists():
            return torch.load(field_path, map_location=device, weights_only=True)
        with torch.no_grad():
            field = smoother(load_links(paths[index], device), projection_device="cpu")
            temporary = field_path.with_suffix(".tmp")
            torch.save(field.cpu(), temporary)
            temporary.replace(field_path)
        return field

    cache = output / "targets.npz"
    if not cache.exists():
        measurements = {"fine": [], "smeared": []}
        with torch.no_grad():
            for index in selected:
                fine = load_links(paths[index], device)
                smeared = smeared_field(index)
                measurements["fine"].append(observable_vector_torch(fine).cpu().numpy())
                measurements["smeared"].append(observable_vector_torch(smeared).cpu().numpy())
                print(f"target {index}: fine={measurements['fine'][-1]} smeared={measurements['smeared'][-1]}", flush=True)
        np.savez(cache, indices=selected, **measurements)
    targets = np.load(cache)
    target_map = {int(i): n for n, i in enumerate(targets["indices"])}
    reference = {split: {kind: targets[kind][[target_map[i] for i in ids]]
                        for kind in ("fine", "smeared")}
                 for split, ids in [("train", train_ids), ("validation", val_ids), ("test", test_ids)]}
    flow1 = SU3ConditionalFlow(sweeps=args.sweeps, constrained=True, initial_scale=-.18).to(device)
    flow2 = SU3ConditionalFlow(sweeps=args.fine_sweeps, initial_scale=.04).to(device)
    history = []
    checkpoint_path = output / "flows.pt"
    if args.resume or args.evaluate_only:
        saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
        flow1.load_state_dict(saved["flow1"])
        flow2.load_state_dict(saved["flow2"])
        history = json.loads((output / "history.json").read_text())

    def save():
        temporary = checkpoint_path.with_suffix(".tmp")
        torch.save({"flow1": flow1.state_dict(), "flow2": flow2.state_dict(),
                    "sweeps": args.sweeps, "fine_sweeps": args.fine_sweeps,
                    "coupling": "mobius_monotone_cubic"}, temporary)
        temporary.replace(checkpoint_path)
        _write_json(output / "history.json", history)

    if not args.evaluate_only:
        fit_conditional_likelihood(flow1, train_ids, val_ids, smeared_field, args, history, save)
        fit_smeared_operators(flow1, train_ids, val_ids, smeared_field, reference,
                              device, args, history, save)
        flow1.eval()
        for parameter in flow1.parameters():
            parameter.requires_grad_(False)
        fit_paired_transport(flow2, train_ids, paths, smeared_field, device, args, history, save)
        fit_fine_operators(flow1, flow2, train_ids, val_ids, smeared_field, reference,
                           device, args, history, save)
        save()

    flow1.eval()
    flow2.eval()
    # One stochastic draw per held-out source configuration. The generated
    # lattice receives only C=block(smear(U)) and fresh Haar detail noise.
    values = {"flow1_smeared": [], "flow2_fine": [], "teacher_fine": []}
    max_errors = np.zeros(2)
    block_error = 0.
    crosscheck = []
    if args.pyquda_check:
        from pyquda_utils import core
        from rgflow.su3.training import _pyquda_gauge_from_links
        from rgflow.su3.observables import observable_vector_pyquda
        (output / ".quda-cache").mkdir(exist_ok=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        core.init(None, manifest["lattice_size"], backend="numpy", resource_path=str(output / ".quda-cache"))
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
            teacher, _ = flow2(smeared_target, compute_logdet=False)
            for name, field in [("flow1_smeared", smeared), ("flow2_fine", generated), ("teacher_fine", teacher)]:
                values[name].append(observable_vector_torch(field).cpu().numpy())
            max_errors = np.maximum(max_errors, su3_errors(generated))
            block_error = max(block_error, float((block_links(smeared)-coarse).abs().max()))
            if args.pyquda_check and len(crosscheck) < 3:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gauge = _pyquda_gauge_from_links(generated.cpu().numpy(), manifest["lattice_size"])
                crosscheck.append(observable_vector_pyquda(gauge) - values["flow2_fine"][-1])
            print(f"test {index}: generated={values['flow2_fine'][-1]}", flush=True)
    values = {key: np.asarray(value) for key, value in values.items()}
    np.savez(output / "observable_distributions.npz", test_indices=test_ids,
             fine_reference=reference["test"]["fine"], smeared_reference=reference["test"]["smeared"], **values)
    metrics = {}
    for name, kind in [("flow1_smeared", "smeared"), ("flow2_fine", "fine"), ("teacher_fine", "fine")]:
        target = reference["test"][kind]
        _, inverse = _covariance_inverse(target, target, .1)
        record = _metric_record(values[name], target, inverse)
        record["acceptance"] = distribution_acceptance(record)
        record["ks_statistic"] = [ks_2samp(values[name][:, k], target[:, k]).statistic for k in range(4)]
        record["wasserstein_in_target_sigma"] = [wasserstein_distance(values[name][:, k], target[:, k])/target[:, k].std(ddof=1) for k in range(4)]
        metrics[name] = record
    metrics.update({"observable_names": observable_names(), "source": "held-out downsampled fine ensemble",
                    "test_configurations": len(test_ids), "max_su3_errors": max_errors,
                    "max_blocking_error": block_error, "pyquda_differences": crosscheck,
                    "rethermalization_sweeps": 0})
    _write_json(output / "metrics.json", metrics)
    _write_json(output / "run.json", {"arguments": vars(args), "data_split": splits,
                "train_indices": train_ids, "validation_indices": val_ids, "test_indices": test_ids,
                "fine_manifest": manifest, "downsampling_architecture": architecture,
                "method": "conditional Haar-fibre likelihood + paired and operator transport",
                "coupling": "mobius_monotone_cubic",
                "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                "gauge_equivariant": False, "base": "independent normalized Haar SU(3) details",
                "stage1_free_links_per_cell": 60, "stage1_flowed_links_per_cell": 56})
    print("FINAL", json.dumps({key: metrics[key]["acceptance"] for key in values}), flush=True)


if __name__ == "__main__":
    main()
