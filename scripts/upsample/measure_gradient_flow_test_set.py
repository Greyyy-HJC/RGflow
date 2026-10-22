#!/usr/bin/env python3
"""Extend the six-source Wilson-flow measurements to all 60 test sources."""

import hashlib
import json
from pathlib import Path

import numpy as np
from pyquda.enum_quda import QudaVerbosity
from pyquda.quda import setVerbosityQuda
from pyquda_utils import core

from diagnose_gradient_flow import ROOT, DEFAULT_RUN, DEFAULT_OUTPUT, FLOW_TIMES, measure_set


def main():
    run = json.loads((DEFAULT_RUN/'run.json').read_text())
    checkpoint_hash = hashlib.sha256((DEFAULT_RUN/'flows.pt').read_bytes()).hexdigest()
    assert checkpoint_hash == run['checkpoint_sha256']
    indices = run['test_indices']
    np.testing.assert_array_equal(indices, np.arange(4, 300, 5))
    output = DEFAULT_OUTPUT/'full_test'
    output.mkdir(parents=True, exist_ok=True)
    core.init(None, [24]*4, backend='numpy',
              resource_path=str(ROOT/'artifacts/4dsu3/topo_charge/.quda-cache'))
    setVerbosityQuda(QudaVerbosity.QUDA_SILENT, b'')
    report = dict(checkpoint_sha256=checkpoint_hash, source_indices=indices,
                  measurement_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  complete=False, stages={})
    (output/'run.json').write_text(json.dumps(report, indent=2)+'\n')
    for stage in ('reference', 'flow1', 'flow2'):
        subset_dir = DEFAULT_OUTPUT/stage
        subset_run = json.loads((subset_dir/'run.json').read_text())
        assert subset_run['complete'] and subset_run['settings']['checkpoint_sha256'] == checkpoint_hash
        input_dir = ROOT/'artifacts/4dsu3/L24_beta6p20' if stage == 'reference' else DEFAULT_RUN/'generated_fields'/stage
        manifest = json.loads((input_dir/'ensemble.json').read_text())
        by_id = {record['index']: record for record in manifest['configurations']}
        records = [by_id[index] for index in indices]
        settings = dict(subset_run['settings'], configurations=records,
                        source_indices=indices, selection='all 60 held-out test sources')
        destination = output/stage
        destination.mkdir(parents=True, exist_ok=True)
        if not (destination/'measurements.npz').exists():
            shape = (len(indices), len(FLOW_TIMES))
            arrays = dict(plaquette=np.full((*shape, 3), np.nan), energy=np.full((*shape, 3), np.nan),
                          topological_charge=np.full(shape, np.nan), elapsed_seconds=np.full(len(indices), np.nan))
            with np.load(subset_dir/'measurements.npz') as subset:
                np.testing.assert_array_equal(subset['flow_times'], FLOW_TIMES)
                for source_row, index in enumerate(subset['config_indices']):
                    record = subset_run['settings']['configurations'][source_row]
                    assert record['sha256'] == by_id[int(index)]['sha256']
                    target_row = indices.index(int(index))
                    for key in arrays:
                        arrays[key][target_row] = subset[key][source_row]
            np.savez_compressed(destination/'measurements.npz', config_indices=indices,
                                flow_times=FLOW_TIMES, **arrays)
            seeded = dict(settings=settings, complete=False, completed_configurations=6,
                          endpoint_checks=subset_run['endpoint_checks'],
                          reused_source_indices=subset_run['settings']['source_indices'])
            (destination/'run.json').write_text(json.dumps(seeded, indent=2)+'\n')
        measured = measure_set(input_dir, records, .02, destination, settings)
        baseline = (ROOT/'presentation/results/topo_charge/full' if stage == 'reference'
                    else ROOT/'presentation/results/topo_charge/upsampling'/stage)
        with np.load(baseline/'measurements.npz') as old:
            rows = [old['config_indices'].tolist().index(i) for i in indices]
            points = [int(np.argmin(abs(FLOW_TIMES-t))) for t in old['flow_times']]
            dq = float(np.max(np.abs(measured['topological_charge'][:, points]-old['topological_charge'][rows])))
            de = float(np.max(np.abs(measured['energy'][:, points]-old['energy'][rows])))
        assert dq < 1e-9 and de < 1e-11, (stage, dq, de)
        report['stages'][stage] = dict(configurations=60, existing_charge_max_difference=dq,
                                       existing_energy_max_difference=de,
                                       decreasing_plaquette_intervals=int((np.diff(measured['plaquette'][:, :, 0], axis=1) < -1e-12).sum()))
        (output/'run.json').write_text(json.dumps(report, indent=2)+'\n')
    report['complete'] = True
    (output/'run.json').write_text(json.dumps(report, indent=2)+'\n')
    print('All 180 test-field histories complete', flush=True)


if __name__ == '__main__':
    main()
