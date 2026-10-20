#!/usr/bin/env python3
"""Check Wilson-flow integration on the first held-out output of each trained stage."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path,
                        default=ROOT/'artifacts/4dsu3/upsample/L12_to_L24_2flow_v6_8ops_6train')
    parser.add_argument('--measurements', type=Path, default=ROOT/'presentation/results/topo_charge/upsampling')
    args = parser.parse_args()
    checkpoint_hash = hashlib.sha256((args.run/'flows.pt').read_bytes()).hexdigest()
    result = {'checkpoint_sha256': checkpoint_hash, 'scope': 'first held-out source of each stage', 'stages': {}}
    for stage in ('flow1', 'flow2'):
        check = ROOT/'temp'/f'topo_charge_{stage}_halfstep'
        if check.exists():
            shutil.rmtree(check)
        command = [str(ROOT/'.venv/bin/python'), str(ROOT/'scripts/heatbath/topo_charge.py'),
                   '--input', str(args.run/'generated_fields'/stage), '--output', str(check),
                   '--stride', '1', '--samples', '1', '--epsilon', '.01', '--t-max', '5', '--measure-every', '10']
        with (args.measurements/f'{stage}_halfstep.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        metadata = json.loads((check/'run.json').read_text())
        assert metadata['complete']
        assert metadata['settings']['upsampling']['checkpoint_sha256'] == checkpoint_hash
        with np.load(check/'measurements.npz') as small, np.load(args.measurements/stage/'measurements.npz') as production:
            np.testing.assert_allclose(small['flow_times'], production['flow_times'], atol=1.e-12, rtol=0)
            assert small['config_indices'][0] == production['config_indices'][0]
            difference = small['topological_charge'][0]-production['topological_charge'][0]
            record = {'source_index': int(small['config_indices'][0]), 'production_epsilon': .02,
                      'check_epsilon': .01, 't_max': 5., 'maximum_absolute_charge_difference': float(np.abs(difference).max()),
                      'final_charge_difference': float(difference[-1]),
                      'maximum_absolute_energy_difference': float(np.abs(small['energy'][0]-production['energy'][0]).max())}
            np.savez_compressed(args.measurements/f'{stage}_halfstep_check.npz',
                                flow_times=small['flow_times'], source_index=record['source_index'],
                                topological_charge=small['topological_charge'][0], charge_difference=difference)
        result['stages'][stage] = record
        print(stage, record, flush=True)
    (args.measurements/'validation.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
