#!/usr/bin/env python3
"""Refresh Wilson-flow measurements for the current smoother and both upsampling stages."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path,
                        default=ROOT/'artifacts/4dsu3/upsample/L12_to_L24_2flow_v6_8ops_6train')
    parser.add_argument('--output', type=Path, default=ROOT/'presentation/results/topo_charge')
    args = parser.parse_args()
    run = args.run.resolve()
    metadata = json.loads((run/'run.json').read_text())
    down = Path(metadata['arguments']['downsampling_run'])
    down_hash = hashlib.sha256((down/'kernel.pt').read_bytes()).hexdigest()
    flow_hash = hashlib.sha256((run/'flows.pt').read_bytes()).hexdigest()
    assert metadata['downsampling_checkpoint_sha256'] == down_hash
    assert metadata['checkpoint_sha256'] == flow_hash
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    validation_path = output/'validation.json'
    validation = json.loads(validation_path.read_text())
    if validation.get('smearing_precision_comparison', {}).get('checkpoint_sha256') != down_hash:
        validation.pop('smearing_precision_comparison', None)
        validation_path.write_text(json.dumps(validation, indent=2)+'\n')
    script = str(ROOT/'scripts/heatbath/topo_charge.py')
    common = ['--epsilon', '.02', '--t-max', '5', '--measure-every', '5']
    # The new stage comparison uses the same 60 held-out sources throughout.
    for stale in (output/'smeared', output/'full/smeared'):
        if stale.exists():
            shutil.rmtree(stale)
    for stale in (output/'topo_charge_smeared_flow.png', output/'topo_charge_smeared_flow.pdf',
                  output/'full/topo_charge_smeared_flow_300.png', output/'full/topo_charge_smeared_flow_300.pdf',
                  output/'smearing_precision_check.npz'):
        stale.unlink(missing_ok=True)
    phases = []
    for stage in ('flow1', 'flow2', 'smeared_reference'):
        phases.append((f'upsampling/{stage}', [
            '--input', str(run/'generated_fields'/stage), '--output', str(output/'upsampling'/stage),
            '--stride', '1', '--samples', str(len(metadata['test_indices'])),
        ]))
    for phase, command in phases:
        destination = output/phase
        previous = destination/'run.json'
        if previous.exists():
            settings = json.loads(previous.read_text())['settings']
            manifest = json.loads((run/'generated_fields'/phase.split('/')[-1]/'ensemble.json').read_text())
            sources_changed = ([r['sha256'] for r in settings['configurations']] !=
                               [r['sha256'] for r in manifest['configurations']])
            if settings['upsampling']['checkpoint_sha256'] != flow_hash or sources_changed:
                shutil.rmtree(destination)
        print('Measuring', phase, flush=True)
        with (output/(phase.replace('/', '_')+'_measurement.log')).open('w') as log:
            subprocess.run([str(ROOT/'.venv/bin/python'), '-u', script, *common, *command],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    print('All 60-source smeared-reference and upsampling Wilson-flow measurements complete', flush=True)


if __name__ == '__main__':
    main()
