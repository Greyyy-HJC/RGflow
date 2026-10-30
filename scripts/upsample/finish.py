#!/usr/bin/env python3
"""Validate and evaluate an already calibrated checkpoint, then refresh its notebook."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    notes = json.loads((run/'execution_notes.json').read_text())
    training = json.loads((run/'training_arguments.json').read_text())
    calibration = [json.loads((run/f'calibration_stage{stage}.json').read_text()) for stage in (1, 2)]
    checkpoint_hash = hashlib.sha256((run/'flows.pt').read_bytes()).hexdigest()
    started = time.monotonic()
    stamp = time.strftime('%Y%m%d-%H%M%S')
    continuation = run/'continuations'/stamp
    continuation.mkdir(parents=True)
    with (continuation/'environment.txt').open('w') as environment:
        subprocess.run([str(ROOT/'.venv/bin/python'), '-m', 'pip', 'freeze'],
                       stdout=environment, check=True)
    for relative in ['scripts/upsample/finish.py', 'scripts/upsample/validate.py',
                     'scripts/upsample/train.py', 'scripts/upsample/make_notebook.py']:
        shutil.copy2(ROOT/relative, continuation/Path(relative).name)
    evaluate = ['scripts/upsample/train.py', '--output', str(run),
                '--evaluate-only', '--pyquda-check', '--export-fields', '--require-stable-inverse']
    for name in ['fine_dir', 'downsampling_run', 'train_configs', 'validation_configs',
                 'test_configs', 'seed', 'sweeps', 'fine_sweeps', 'mean_metric', 'variance_weight']:
        evaluate.extend(['--'+name.replace('_', '-'), str(training[name])])
    phases = [
        ('validation', ['scripts/upsample/validate.py', '--run', str(run),
                        '--full-lattice-sample', str(run/'numerical_sample.pt')]),
        ('evaluation', evaluate),
    ]
    for phase, command in phases:
        (run/'pipeline_status.txt').write_text(phase+'\n')
        print(phase, flush=True)
        with (continuation/f'{phase}.log').open('w') as log:
            subprocess.run([str(ROOT/'.venv/bin/python'), '-u', *command], cwd=ROOT,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    assert hashlib.sha256((run/'flows.pt').read_bytes()).hexdigest() == checkpoint_hash
    notes['checkpoint_sha256'] = checkpoint_hash
    notes['calibration_stopping_reasons'] = {
        f'flow{stage}': value['stopping_reason'] for stage, value in enumerate(calibration, 1)
    }
    notes['continuation'] = dict(directory=str(continuation), checkpoint_unchanged=True,
                                 elapsed_seconds=time.monotonic()-started,
                                 reason='Use FP64 full-lattice validation for the promoted checkpoint')
    (run/'execution_notes.json').write_text(json.dumps(notes, indent=2)+'\n')
    (run/'pipeline_status.txt').write_text('measurements_complete\n')
    with (continuation/'notebook.log').open('w') as log:
        subprocess.run([str(ROOT/'.venv/bin/python'), 'scripts/upsample/make_notebook.py',
                        '--run', str(run)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    (run/'report_status.txt').write_text('notebook_complete\n')
    print('Evaluation and notebook complete', flush=True)


if __name__ == '__main__':
    main()
