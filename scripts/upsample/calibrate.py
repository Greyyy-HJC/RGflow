#!/usr/bin/env python3
"""Refine global flow parameters by deterministic training-set moment fitting."""

import argparse
import copy
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares

from rgflow.su3.downsampling import block_links
from rgflow.su3.observables import observable_vector_torch
from rgflow.su3.training import _covariance_inverse, _stats, _write_json, VARIANCE_EPS
from rgflow.su3.upsampling import SU3ConditionalFlow, constrained_haar_lift

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run', type=Path, default=ROOT/'artifacts/4dsu3/upsample/L12_to_L24_2flow_v1')
parser.add_argument('--stage', type=int, choices=[1, 2], required=True)
parser.add_argument('--train-configs', type=int, default=12)
parser.add_argument('--validation-configs', type=int, default=8)
parser.add_argument('--max-evaluations', type=int, default=30)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--bound', type=float, default=1.5)
parser.add_argument('--anneal-init', action='store_true', help='expand details in first half and contract in second')
parser.add_argument('--mean-metric', choices=['diagonal', 'covariance'], default='diagonal')
args = parser.parse_args()
torch.set_num_threads(2)
device = torch.device('cuda')
saved = torch.load(args.run/'flows.pt', map_location=device, weights_only=True)
baseline_hash = hashlib.sha256((args.run/'flows.pt').read_bytes()).hexdigest()[:12]
baseline_directory = args.run/'calibration_baselines'
baseline_directory.mkdir(exist_ok=True)
shutil.copy2(args.run/'flows.pt', baseline_directory/f'{baseline_hash}.pt')
flow1 = SU3ConditionalFlow(sweeps=saved['sweeps'], constrained=True).to(device).eval()
flow2 = SU3ConditionalFlow(sweeps=saved['fine_sweeps']).to(device).eval()
flow1.load_state_dict(saved['flow1'])
flow2.load_state_dict(saved['flow2'])
model = flow1 if args.stage == 1 else flow2
baseline = copy.deepcopy(model.state_dict())
history = json.loads((args.run/'history.json').read_text())
phase = 'smeared_moments' if args.stage == 1 else 'generated_moments'
epoch_offset = max([row['epoch'] for row in history if row['phase'] == phase], default=0)
targets = np.load(args.run/'targets.npz')
indices = targets['indices'].tolist()
train_ids = [i for i in indices if i % 5 in [0, 1, 2]][:args.train_configs]
val_ids = [i for i in indices if i % 5 == 3][:args.validation_configs]
kind = 'smeared' if args.stage == 1 else 'fine'
references = {split: targets[kind][[indices.index(i) for i in ids]]
              for split, ids in [('train', train_ids), ('validation', val_ids)]}
inputs = {'train': [], 'validation': []}
with torch.no_grad():
    for split, ids in [('train', train_ids), ('validation', val_ids)]:
        for index in ids:
            smeared = torch.load(args.run/'smeared_fields'/f'cfg_{index:04d}.pt', map_location=device, weights_only=True)
            generator = torch.Generator(device=device).manual_seed(args.seed+60000+index)
            base = constrained_haar_lift(block_links(smeared), generator=generator)
            value = base if args.stage == 1 else flow1(base, compute_logdet=False)[0]
            inputs[split].append(value.cpu())
            print('input', split, index, flush=True)


def set_parameters(x):
    # Six offsets: shift, cubic shape and long-staple weight in the first
    # and second halves of the flow. All feature-dependent weights stay fixed.
    model.load_state_dict(baseline)
    with torch.no_grad():
        for sweep, network in enumerate(model.scales):
            half = int(sweep >= model.sweeps//2)
            network[-1].bias[:3].add_(float(x[3*half]))
            network[-1].bias[3:].add_(float(x[3*half+1]))
            model.long_weights[sweep].add_(float(x[3*half+2]))


def measure(split):
    with torch.no_grad():
        return np.array([observable_vector_torch(model(value.to(device), compute_logdet=False)[0]).cpu().numpy()
                         for value in inputs[split]], dtype=np.float64)


def residual(values, split):
    target = references[split].astype(np.float64)
    mean, _, logvar = _stats(target)
    mean_difference = values.mean(0)-mean
    if args.mean_metric == 'diagonal':
        mean_residual = mean_difference/target.std(0, ddof=1)
    else:
        _, inverse = _covariance_inverse(target, target, .1)
        mean_residual = np.linalg.cholesky(inverse).T @ mean_difference
    widths = np.log(values.var(0, ddof=1)+VARIANCE_EPS)-logvar
    return np.r_[mean_residual, widths]


set_parameters(np.zeros(6))
best_loss = float((residual(measure('validation'), 'validation')**2).sum())
best_state = copy.deepcopy(model.state_dict())
best_offsets = np.zeros(6)
evaluations = 0


def objective(x):
    global evaluations, best_loss, best_state, best_offsets
    started = time.time()
    set_parameters(x)
    values = measure('train')
    result = residual(values, 'train')
    validation = residual(measure('validation'), 'validation')
    loss = float(validation@validation)
    evaluations += 1
    if loss < best_loss:
        best_loss = loss
        best_state = copy.deepcopy(model.state_dict())
        best_offsets = np.array(x, copy=True)
        saved[f'flow{args.stage}'] = best_state
        temporary = (args.run/'flows.pt').with_suffix('.tmp')
        torch.save(saved, temporary)
        temporary.replace(args.run/'flows.pt')
    record = {'phase': phase, 'epoch': epoch_offset+evaluations, 'optimizer': 'global_least_squares',
              'mean_metric': args.mean_metric,
              'baseline_checkpoint': baseline_hash,
              'train_loss': float(result@result), 'validation_loss': loss, 'mean': values.mean(0),
              'train_configurations': len(train_ids), 'validation_configurations': len(val_ids),
              'offsets': x.tolist(), 'seconds': time.time()-started}
    history.append(record)
    _write_json(args.run/'history.json', history)
    print(record, flush=True)
    return result


def jacobian(x):
    # Fixed absolute differences resolve single-precision lattice reductions;
    # SciPy's usual ~1e-8 differences would disappear in the SU(3) fields.
    step = .002
    set_parameters(x)
    center = residual(measure('train'), 'train')
    columns = []
    for axis in range(6):
        trial = np.array(x, copy=True)
        trial[axis] += step
        set_parameters(trial)
        columns.append((residual(measure('train'), 'train')-center)/step)
    return np.stack(columns, 1)


initial = np.zeros(6)
if args.anneal_init:
    initial[[0, 3]] = [1.2, -.8]
    for half in range(2):
        sweeps = range(half*model.sweeps//2, (half+1)*model.sweeps//2)
        initial[3*half+1] = -np.mean([float(model.scales[s][-1].bias[3:].detach().mean()) for s in sweeps])
        initial[3*half+2] = -float(model.long_weights[list(sweeps)].detach().mean())
    initial = np.clip(initial, -args.bound+.001, args.bound-.001)
result = least_squares(objective, initial, jac=jacobian, bounds=(-args.bound, args.bound),
                       max_nfev=args.max_evaluations, ftol=1.e-5, xtol=1.e-5, gtol=1.e-5)
model.load_state_dict(best_state)
saved.update(flow1=flow1.state_dict(), flow2=flow2.state_dict(), coupling='mobius_monotone_cubic')
temporary = (args.run/'flows.pt').with_suffix('.tmp')
torch.save(saved, temporary)
temporary.replace(args.run/'flows.pt')
_write_json(args.run/f'calibration_stage{args.stage}.json',
            {'train_ids': train_ids, 'validation_ids': val_ids, 'seed': args.seed,
             'optimizer': 'bounded least_squares', 'best_validation_loss': best_loss,
             'mean_metric': args.mean_metric, 'initial_offsets': initial.tolist(), 'bound': args.bound,
             'baseline_checkpoint': baseline_hash,
             'evaluations': evaluations, 'terminal_offsets': result.x.tolist(),
             'best_offsets': best_offsets.tolist(),
             'message': result.message})
print('BEST VALIDATION', best_loss, flush=True)
