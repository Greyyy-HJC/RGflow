#!/usr/bin/env python3
"""Refit six fine-flow offsets without new gradient epochs or test-target fitting."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import torch

from rgflow.su3.training import _write_json
from rgflow.su3.upsampling import EQUIVARIANT_COUPLING, STABLE_COUPLING

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--flow1-source', type=Path, default=ROOT/'artifacts/4dsu3/upsample/L12_to_L24_2flow_stable_v3')
parser.add_argument('--flow2-source', type=Path, default=ROOT/'artifacts/4dsu3/upsample/L12_to_L24_2flow_equivariant_v2')
parser.add_argument('--output', type=Path, default=ROOT/'artifacts/4dsu3/upsample/L12_to_L24_2flow_overlap_v5')
parser.add_argument('--calibration-minutes', type=float, default=65.)
parser.add_argument('--patience', type=int, default=6)
parser.add_argument('--train-configs', type=int, default=8)
parser.add_argument('--validation-configs', type=int, default=8)
parser.add_argument('--test-configs', type=int, default=60)
args = parser.parse_args()
started = time.monotonic()
run = args.output.resolve()
run.mkdir()
(run/'pipeline.pid').write_text(str(os.getpid())+'\n')
source1, source2 = args.flow1_source.resolve(), args.flow2_source.resolve()
first = torch.load(source1/'flows.pt', map_location='cpu', weights_only=True)
second = torch.load(source2/'flows.pt', map_location='cpu', weights_only=True)
assert first['coupling'] == STABLE_COUPLING
assert second['coupling'] in (EQUIVARIANT_COUPLING, STABLE_COUPLING)
saved = dict(flow1={k: v.double() for k, v in first['flow1'].items()},
             flow2={k: v.double() for k, v in second['flow2'].items()},
             sweeps=first['sweeps'], fine_sweeps=second['fine_sweeps'], coupling=STABLE_COUPLING)
torch.save(saved, run/'flows.pt')
shutil.copy2(run/'flows.pt', run/'initial_flows.pt')
for name in ['targets.npz', 'training_arguments.json', 'environment.json', 'numerical_sample.pt']:
    shutil.copy2(source1/name, run/name)
history = json.loads((source1/'history.json').read_text())
_write_json(run/'history.json', [row for row in history if row.get('optimizer', 'Adam') == 'Adam'])
(run/'smeared_fields').symlink_to((source1/'smeared_fields').resolve(), target_is_directory=True)
(run/'.quda-cache').mkdir()
shutil.copy2(source2/'.quda-cache/tunecache.tsv', run/'.quda-cache/tunecache.tsv')
notes = dict(mode='six_offset_fine', flow1_source=str(source1), flow2_source=str(source2),
             flow1_source_sha256=hashlib.sha256((source1/'flows.pt').read_bytes()).hexdigest(),
             flow2_source_sha256=hashlib.sha256((source2/'flows.pt').read_bytes()).hexdigest(),
             new_gradient_epochs=0, parameters_per_flow=[0, 6], flow1_frozen=True,
             train_configurations=args.train_configs, validation_configurations=args.validation_configs,
             test_configurations=args.test_configs, seed=1234, test_targets_used_in_fit=False,
             method='six-offset fine-flow least squares; frozen stable Flow 1; validation selection',
             calibration_minutes_budget=args.calibration_minutes, patience=args.patience)
_write_json(run/'execution_notes.json', notes)
_write_json(run/'calibration_stage1.json', dict(optimizer='frozen', parameters=0, evaluations=0,
             best_offsets=[], best_validation_loss=None, bound=0., message='Flow 1 frozen at stable warm start'))
commands = [
    ('calibration', ['scripts/upsample/calibrate.py', '--run', str(run), '--stage', '2',
        '--train-configs', str(args.train_configs), '--validation-configs', str(args.validation_configs),
        '--max-evaluations', '30', '--minutes', str(args.calibration_minutes), '--bound', '1.5',
        '--patience', str(args.patience),
        '--variance-weight', '10', '--require-stable-inverse', '--stability-coarse-file', str(run/'numerical_sample.pt')]),
    ('selection', ['scripts/upsample/select_overlap.py', '--run', str(run)]),
    ('validation', ['scripts/upsample/validate.py', '--run', str(run), '--full-lattice-sample', str(run/'numerical_sample.pt')]),
    ('evaluation', ['scripts/upsample/train.py', '--output', str(run), '--evaluate-only',
        '--train-configs', str(args.train_configs), '--validation-configs', str(args.validation_configs),
        '--test-configs', str(args.test_configs), '--pyquda-check']),
]
for phase, command in commands:
    (run/'pipeline_status.txt').write_text(phase+'\n')
    print(phase, 'elapsed', round(time.monotonic()-started), flush=True)
    with (run/f'{phase}.log').open('w') as log:
        subprocess.run([str(ROOT/'.venv/bin/python'), *command], cwd=ROOT, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
notes['elapsed_seconds'] = time.monotonic()-started
_write_json(run/'execution_notes.json', notes)
(run/'pipeline_status.txt').write_text('measurements_complete\n')
print('MEASUREMENTS COMPLETE', notes['elapsed_seconds'], flush=True)
