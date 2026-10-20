#!/usr/bin/env python3
"""Train fresh two-flow maps for the selected downsampling run and measure both stages."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from rgflow.su3.observables import DOWNSAMPLING_OBSERVABLES, TRAINING_OBSERVABLES
from rgflow.su3.training import _write_json
from rgflow.su3.upsampling import EQUIVARIANT_COUPLING


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--downsampling-run', type=Path)
    parser.add_argument('--output', type=Path,
                        default=ROOT/'artifacts/4dsu3/upsample/L12_to_L24_2flow_v6_8ops_6train')
    parser.add_argument('--train-configs', type=int, default=12)
    parser.add_argument('--validation-configs', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--transport-epochs', type=int, default=3)
    parser.add_argument('--max-evaluations', type=int, default=30)
    parser.add_argument('--calibration-minutes', type=float, default=65.)
    parser.add_argument('--patience', type=int, default=6)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()
    if args.downsampling_run is None:
        selected = json.loads((ROOT/'docs/project_notes/results/selected_downsampling_run.json').read_text())
        args.downsampling_run = ROOT/'artifacts/4dsu3/downsample'/selected['run']
    down = args.downsampling_run.resolve()
    run = args.output.resolve()
    run.mkdir(parents=True, exist_ok=True)
    if (run/'flows.pt').exists():
        raise SystemExit('Choose a new output directory for fresh training.')
    started = time.monotonic()
    notes = dict(arguments=vars(args), initialization='fresh seed; no pretrained flow weights',
                 observable_names=DOWNSAMPLING_OBSERVABLES, training_observables=TRAINING_OBSERVABLES,
                 held_out_observables=[name for name in DOWNSAMPLING_OBSERVABLES if name not in TRAINING_OBSERVABLES],
                 test_targets_used_in_fit=False, rethermalization_sweeps=0,
                 warmup_precision='single', calibration_and_generation_precision='double',
                 downsampling_checkpoint_sha256=hashlib.sha256((down/'kernel.pt').read_bytes()).hexdigest(),
                 method='fresh conditional likelihood and paired transport; six-offset calibration of each stage')
    _write_json(run/'execution_notes.json', notes)
    snapshot = run/'source_snapshot'
    source_hashes = {}
    for relative in ['scripts/upsample/rerun.py', 'scripts/upsample/train.py',
                     'scripts/upsample/calibrate.py', 'scripts/upsample/validate.py',
                     'scripts/upsample/promote.py',
                     'scripts/heatbath/topo_charge.py', 'src/rgflow/su3/upsampling.py',
                     'src/rgflow/su3/training.py', 'src/rgflow/su3/observables.py',
                     'src/rgflow/su3/downsampling.py', 'src/rgflow/su3/linalg.py', 'src/rgflow/su3/io.py']:
        destination = snapshot/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/relative, destination)
        source_hashes[relative] = hashlib.sha256(destination.read_bytes()).hexdigest()
    _write_json(run/'source_hashes.json', source_hashes)
    (run/'pipeline.pid').write_text(str(os.getpid())+'\n')

    common = ['--output', str(run), '--downsampling-run', str(down), '--coupling', EQUIVARIANT_COUPLING,
              '--train-configs', str(args.train_configs), '--validation-configs', str(args.validation_configs),
              '--test-configs', '60', '--seed', str(args.seed), '--learning-rate', '.003',
              '--mean-metric', 'diagonal', '--variance-weight', '10', '--require-stable-inverse']
    calibration = ['--run', str(run), '--train-configs', str(args.train_configs),
                   '--validation-configs', str(args.validation_configs), '--seed', str(args.seed),
                   '--max-evaluations', str(args.max_evaluations), '--minutes', str(args.calibration_minutes),
                   '--patience', str(args.patience), '--bound', '1.5', '--variance-weight', '10',
                   '--require-stable-inverse', '--stability-coarse-file', str(run/'numerical_sample.pt')]
    phases = [
        ('warmup_training', ['scripts/upsample/train.py', *common, '--epochs', str(args.epochs),
                           '--smeared-moment-epochs', '0', '--transport-epochs', str(args.transport_epochs),
                           '--moment-epochs', '0', '--skip-evaluation']),
        ('promotion', ['scripts/upsample/promote.py', '--run', str(run)]),
        ('flow1_calibration', ['scripts/upsample/calibrate.py', *calibration, '--stage', '1']),
        ('flow2_calibration', ['scripts/upsample/calibrate.py', *calibration, '--stage', '2']),
        ('validation', ['scripts/upsample/validate.py', '--run', str(run),
                        '--full-lattice-sample', str(run/'numerical_sample.pt')]),
        ('evaluation', ['scripts/upsample/train.py', *common, '--evaluate-only', '--pyquda-check', '--export-fields']),
    ]
    for phase, command in phases:
        (run/'pipeline_status.txt').write_text(phase+'\n')
        print(phase, 'elapsed', round(time.monotonic()-started), flush=True)
        with (run/f'{phase}.log').open('w') as log:
            subprocess.run([str(ROOT/'.venv/bin/python'), '-u', *command], cwd=ROOT,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    notes['elapsed_seconds'] = time.monotonic()-started
    notes['checkpoint_sha256'] = hashlib.sha256((run/'flows.pt').read_bytes()).hexdigest()
    notes['calibration_stopping_reasons'] = {
        f'flow{stage}': json.loads((run/f'calibration_stage{stage}.json').read_text())['stopping_reason']
        for stage in (1, 2)
    }
    _write_json(run/'execution_notes.json', notes)
    (run/'pipeline_status.txt').write_text('measurements_complete\n')
    print('COMPLETE', notes['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()
