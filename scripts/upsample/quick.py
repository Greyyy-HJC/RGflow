#!/usr/bin/env python3
"""Short forward-only scalar calibration of the existing covariant flows."""

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from rgflow.su3.downsampling import block_links
from rgflow.su3.observables import observable_vector_torch
from rgflow.su3.training import _write_json
from rgflow.su3.upsampling import SU3ConditionalFlow, constrained_haar_lift, flow_roundtrip_check

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, default=ROOT/'artifacts/4dsu3/upsample/L12_to_L24_2flow_stable_v3')
parser.add_argument('--output', type=Path, default=ROOT/'artifacts/4dsu3/upsample/L12_to_L24_2flow_stable_quick_v4')
parser.add_argument('--minutes', type=float, default=16.)
args = parser.parse_args()
started = time.monotonic()
deadline = started+args.minutes*60
run = args.output.resolve()
run.mkdir()
(run/'pipeline.pid').write_text(str(os.getpid())+'\n')
(run/'pipeline.log').symlink_to(ROOT/'temp/quick_v4.log')
torch.set_num_threads(2)
source = args.source.resolve()
for name in ['flows.pt', 'targets.npz', 'history.json', 'training_arguments.json', 'environment.json']:
    shutil.copy2(source/name, run/name)
shutil.copy2(source/'flows.pt', run/'initial_flows.pt')
(run/'smeared_fields').symlink_to((source/'smeared_fields').resolve(), target_is_directory=True)
tune_cache = ROOT/'artifacts/4dsu3/upsample/L12_to_L24_2flow_equivariant_v2/.quda-cache/tunecache.tsv'
if tune_cache.exists():
    (run/'.quda-cache').mkdir()
    shutil.copy2(tune_cache, run/'.quda-cache/tunecache.tsv')
notes = {'mode': 'quick_scalar', 'source': str(source),
         'source_checkpoint_sha256': hashlib.sha256((source/'flows.pt').read_bytes()).hexdigest(),
         'new_gradient_epochs': 0, 'parameters_per_flow': 1, 'seed': 1234,
         'train_configurations': 4, 'validation_configurations': 4, 'test_configurations': 12,
         'minutes_budget': args.minutes, 'method': 'fixed scalar grid, forward measurements, validation selection',
         'pyquda_tune_cache': str(tune_cache),
         'test_targets_used_in_fit': False}
_write_json(run/'execution_notes.json', notes)
saved = torch.load(run/'flows.pt', map_location='cuda', weights_only=True)
flows = [SU3ConditionalFlow(sweeps=saved['sweeps'], constrained=True, coupling=saved['coupling']).cuda().eval(),
         SU3ConditionalFlow(sweeps=saved['fine_sweeps'], coupling=saved['coupling']).cuda().eval()]
for stage, flow in enumerate(flows, 1):
    flow.load_state_dict(saved[f'flow{stage}'])
targets = np.load(run/'targets.npz')
indices = targets['indices'].tolist()
ids = {'train': [i for i in indices if i % 5 in [0, 1, 2]][:4],
       'validation': [i for i in indices if i % 5 == 3][:4]}
coarse_check = torch.load(source/'numerical_sample.pt', map_location='cpu', weights_only=True)['coarse']
shutil.copy2(source/'numerical_sample.pt', run/'numerical_sample.pt')
history = json.loads((run/'history.json').read_text())
inputs = {'train': [], 'validation': []}
with torch.no_grad():
    for split, current_ids in ids.items():
        for index in current_ids:
            field = torch.load(run/'smeared_fields'/f'cfg_{index:04d}.pt', map_location='cuda', weights_only=True).to(torch.complex128)
            generator = torch.Generator(device='cuda').manual_seed(1234+60000+index)
            base = constrained_haar_lift(block_links(field), generator=generator)
            inputs[split].append(base.cpu())
            print('prepared', split, index, flush=True)


def score(values, reference):
    means = (values.mean(0)-reference.mean(0))/reference.std(0, ddof=1)
    widths = np.log((values.var(0, ddof=1)+1.e-12)/(reference.var(0, ddof=1)+1.e-12))
    return float(means@means+10*widths@widths)


for stage, model in enumerate(flows, 1):
    (run/'pipeline_status.txt').write_text(f'flow{stage}_scalar_scan\n')
    stage_started = time.monotonic()
    # Reserve time for the second scan, final checks and held-out measurement.
    scan_deadline = min(stage_started+180, deadline-240)
    baseline = copy.deepcopy(model.state_dict())
    baseline_hash = hashlib.sha256((run/'flows.pt').read_bytes()).hexdigest()[:12]
    baselines = run/'calibration_baselines'
    baselines.mkdir(exist_ok=True)
    shutil.copy2(run/'flows.pt', baselines/f'{baseline_hash}.pt')
    kind = 'smeared' if stage == 1 else 'fine'
    reference = {split: targets[kind][[indices.index(i) for i in current_ids]].astype(np.float64)
                 for split, current_ids in ids.items()}
    offsets = [0., -.025, -.05, .025] if stage == 1 else [0., .025, .05, -.025]
    trials = []
    phase = 'smeared_moments' if stage == 1 else 'generated_moments'
    epoch = max(row['epoch'] for row in history if row['phase'] == phase)
    for offset in offsets:
        if trials and time.monotonic() > scan_deadline:
            break
        model.load_state_dict(baseline)
        with torch.no_grad():
            for network in model.scales:
                network[-1].bias[:3].add_(offset)
            values = {split: np.array([observable_vector_torch(model(value.to('cuda'), compute_logdet=False)[0]).cpu().numpy()
                                      for value in current], dtype=np.float64)
                      for split, current in inputs.items()}
        trial = {'offset': offset, 'train_loss': score(values['train'], reference['train']),
                 'validation_loss': score(values['validation'], reference['validation'])}
        trials.append(trial)
        history.append({'phase': phase, 'epoch': epoch+len(trials), 'optimizer': 'scalar_grid',
                        'mean_metric': 'diagonal', 'variance_weight': 10., 'mean': values['train'].mean(0),
                        'train_configurations': 4, 'validation_configurations': 4, **trial})
        _write_json(run/'history.json', history)
        print('trial', stage, trial, 'elapsed', round(time.monotonic()-started), flush=True)
    selected = trials[0]
    checks = []
    for trial in sorted(trials, key=lambda row: row['validation_loss']):
        model.load_state_dict(baseline)
        with torch.no_grad():
            for network in model.scales:
                network[-1].bias[:3].add_(trial['offset'])
        report = flow_roundtrip_check(model, coarse_check)
        checks.append({'offset': trial['offset'], **report})
        if report['passed']:
            selected = trial
            break
    else:
        raise RuntimeError(f'No numerically stable Flow-{stage} candidate: {checks}')
    saved[f'flow{stage}'] = model.state_dict()
    torch.save(saved, run/'flows.pt')
    _write_json(run/f'calibration_stage{stage}.json',
                {'optimizer': 'scalar_grid', 'parameters': 1, 'train_ids': ids['train'],
                 'validation_ids': ids['validation'], 'seed': 1234, 'evaluations': len(trials),
                 'best_offsets': [selected['offset']], 'best_validation_loss': selected['validation_loss'],
                 'baseline_validation_loss': trials[0]['validation_loss'], 'bound': max(abs(x) for x in offsets),
                 'baseline_checkpoint': baseline_hash, 'stability_checks': checks,
                 'message': 'fixed scalar scan', 'seconds': time.monotonic()-stage_started})
    print('selected', stage, selected, flush=True)
    if stage == 1:
        with torch.no_grad():
            inputs = {split: [model(value.to('cuda'), compute_logdet=False)[0].cpu() for value in current]
                      for split, current in inputs.items()}

for phase, command in [
    ('validation', ['scripts/upsample/validate.py', '--run', str(run), '--full-lattice-sample', str(run/'numerical_sample.pt')]),
    ('evaluation', ['scripts/upsample/train.py', '--output', str(run), '--evaluate-only', '--train-configs', '4',
                    '--validation-configs', '4', '--test-configs', '12', '--pyquda-check']),
    ('notebook', ['scripts/upsample/make_notebook.py', '--run', str(run)]),
    ('figures', ['docs/project_notes/make_figures.py', '--refresh-data', '--upsampling-run', str(run)]),
    ('document', ['docs/project_notes/update_results.py', '--upsampling-run', str(run)]),
]:
    (run/'pipeline_status.txt').write_text(phase+'\n')
    with (run/f'{phase}.log').open('w') as log:
        subprocess.run([str(ROOT/'.venv/bin/python'), *command], cwd=ROOT, stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=max(1., deadline-time.monotonic()))
with (run/'compile.log').open('w') as log:
    subprocess.run([str(ROOT/'temp/latex-toolchain/tectonic'), '--synctex', '--keep-logs', 'jinchen_progress.tex'],
                   cwd=ROOT/'docs/project_notes', stdout=log, stderr=subprocess.STDOUT, check=True,
                   timeout=max(1., deadline-time.monotonic()))
notes['elapsed_seconds'] = time.monotonic()-started
_write_json(run/'execution_notes.json', notes)
(run/'pipeline_status.txt').write_text('complete\n')
archive = ROOT/'docs/project_notes/results'/run.name
shutil.copy2(run/'execution_notes.json', archive/'execution_notes.json')
shutil.copy2(run/'pipeline_status.txt', archive/'pipeline_status.txt')
print('COMPLETE', notes['elapsed_seconds'], 'seconds', flush=True)
