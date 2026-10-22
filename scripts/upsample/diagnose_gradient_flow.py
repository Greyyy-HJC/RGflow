#!/usr/bin/env python3
"""Measure plaquette, clover energy and charge along Wilson flow on paired test fields."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
from pyquda.enum_quda import QudaVerbosity
from pyquda.quda import setVerbosityQuda
from pyquda_utils import core, io


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT/'artifacts/4dsu3/upsample/L12_to_L24_2flow_v6_8ops_6train'
DEFAULT_OUTPUT = ROOT/'presentation/results/gradient_flow_diagnostics'
STAGES = ('reference', 'smeared_reference', 'flow1', 'flow2')
# Six fixed, evenly spaced positions in the ordered 60-source test set.
SOURCE_IDS = (4, 59, 119, 179, 239, 299)
FLOW_TIMES = np.unique(np.r_[np.arange(51)*.1, np.arange(11)*.02].round(8))


def measure(gauge, epsilon, times, *, readback=False):
    """Measure the current resident field explicitly after each flow chunk."""
    steps = np.rint(times/epsilon).astype(int)
    np.testing.assert_allclose(steps*epsilon, times, atol=1e-12, rtol=0)
    plaquette = np.empty((len(times), 3))
    energy = np.empty((len(times), 3))
    charge = np.empty(len(times))
    endpoint = gauge.copy() if readback else None
    with gauge.use() as dirac:
        dirac.smear_param.rk_order = 3
        try:
            for point in range(len(times)):
                if point:
                    dirac.wilsonFlow(int(steps[point]-steps[point-1]), epsilon,
                                     float(times[point-1]), restart=point > 1,
                                     compute_plaquette=False, compute_qcharge=False)
                plaquette[point] = dirac.plaquette()
                charge[point] = dirac.qcharge()
                energy[point] = dirac.obs_param.energy
            if readback:
                dirac.saveSmearedGauge(endpoint)
        finally:
            dirac.freeSmearedGauge()
    result = dict(plaquette=plaquette, energy=energy, topological_charge=charge)
    checks = {}
    if readback:
        # Reload the downloaded endpoint as an ordinary gauge field. This tests
        # whether the resident-field observables really used the flowed links.
        checks['endpoint_reload_plaquette_max_difference'] = float(np.max(np.abs(endpoint.plaquette()-plaquette[-1])))
        checks['endpoint_reload_energy_max_difference'] = float(np.max(np.abs(endpoint.energy()-energy[-1])))
        checks['endpoint_reload_charge_difference'] = float(endpoint.qcharge()-charge[-1])
        import torch
        from rgflow.su3.observables import observable_vector_torch

        with torch.no_grad():
            for label, field, expected in [('initial', gauge, plaquette[0, 0]),
                                           ('endpoint', endpoint, plaquette[-1, 0])]:
                links = torch.as_tensor(field.lexico(), dtype=torch.complex128, device='cuda')
                independent = observable_vector_torch(links, names=('plaquette',)).item()
                checks[f'{label}_torch_plaquette_difference'] = independent-float(expected)
                unitary = links.mH @ links-torch.eye(3, dtype=links.dtype, device=links.device)
                checks[f'{label}_max_unitarity_error'] = float(unitary.abs().max())
                checks[f'{label}_max_determinant_error'] = float((torch.linalg.det(links)-1).abs().max())
                del links, unitary
        torch.cuda.empty_cache()
        assert max(abs(value) for key, value in checks.items() if 'difference' in key) < 1e-9, checks
    return result, checks


def measure_set(input_dir, records, epsilon, destination, settings, *, readback=False):
    """Resume a fixed set, saving scalar histories after each configuration."""
    destination.mkdir(parents=True, exist_ok=True)
    shape = (len(records), len(FLOW_TIMES))
    arrays = dict(plaquette=np.full((*shape, 3), np.nan),
                  energy=np.full((*shape, 3), np.nan),
                  topological_charge=np.full(shape, np.nan),
                  elapsed_seconds=np.full(len(records), np.nan))
    metadata = dict(settings=settings, complete=False, endpoint_checks={})
    if (destination/'measurements.npz').exists():
        metadata = json.loads((destination/'run.json').read_text())
        assert metadata['settings'] == settings, 'Diagnostic cache settings changed'
        with np.load(destination/'measurements.npz') as saved:
            np.testing.assert_array_equal(saved['config_indices'], [r['index'] for r in records])
            np.testing.assert_array_equal(saved['flow_times'], FLOW_TIMES)
            for key in arrays:
                arrays[key][:] = saved[key]
    for row, record in enumerate(records):
        if np.isfinite(arrays['plaquette'][row]).all():
            continue
        started = time.perf_counter()
        path = input_dir/record['filename']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
        gauge = io.readNERSCGauge(str(path), checksum=True, plaquette=False,
                                 link_trace=False, reunitarize_sigma=0)
        values, checks = measure(gauge, epsilon, FLOW_TIMES, readback=readback and row == 0)
        assert all(np.isfinite(value).all() for value in values.values()), (destination, record['index'])
        np.testing.assert_allclose(values['plaquette'][0, 0], record['plaquette'], atol=2e-7, rtol=0)
        for key, value in values.items():
            arrays[key][row] = value
        arrays['elapsed_seconds'][row] = time.perf_counter()-started
        if checks:
            metadata['endpoint_checks'][str(record['index'])] = checks
        metadata['complete'] = bool(np.isfinite(arrays['plaquette']).all())
        metadata['completed_configurations'] = int(np.isfinite(arrays['plaquette']).all(axis=(1, 2)).sum())
        temporary = destination/'measurements.tmp.npz'
        np.savez_compressed(temporary, config_indices=[r['index'] for r in records],
                            flow_times=FLOW_TIMES, **arrays)
        temporary.replace(destination/'measurements.npz')
        (destination/'run.json').write_text(json.dumps(metadata, indent=2)+'\n')
        print(f"{destination.relative_to(ROOT)}: {row+1}/{len(records)}, ID {record['index']}, "
              f"P {values['plaquette'][0, 0]:.8f} → {values['plaquette'][-1, 0]:.8f}, "
              f"E {values['energy'][0, 0]:.6f} → {values['energy'][-1, 0]:.6f}, "
              f"Q {values['topological_charge'][0]:+.4f} → {values['topological_charge'][-1]:+.4f}, "
              f"{arrays['elapsed_seconds'][row]:.1f} s", flush=True)
        del gauge
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    run = json.loads((args.run/'run.json').read_text())
    checkpoint_hash = hashlib.sha256((args.run/'flows.pt').read_bytes()).hexdigest()
    assert checkpoint_hash == run['checkpoint_sha256']
    assert set(SOURCE_IDS) <= set(run['test_indices'])
    cache = ROOT/'artifacts/4dsu3/topo_charge/.quda-cache'
    cache.mkdir(parents=True, exist_ok=True)
    core.init(None, [24]*4, backend='numpy', resource_path=str(cache))
    setVerbosityQuda(QudaVerbosity.QUDA_SILENT, b'')
    from pyquda.field import LatticeGauge, LatticeInfo
    identity, _ = measure(LatticeGauge(LatticeInfo([24]*4)), .02, np.array([0., .1, .2]))
    np.savez_compressed(output/'identity_check.npz', flow_times=[0., .1, .2], **identity)
    identity_finite = all(np.isfinite(value).all() for value in identity.values())
    validation = dict(identity_field_check=dict(finite=identity_finite), stages={})
    if identity_finite:
        np.testing.assert_allclose(identity['plaquette'], 1., atol=1e-13, rtol=0)
        np.testing.assert_allclose(identity['energy'], 0., atol=1e-13, rtol=0)
        np.testing.assert_allclose(identity['topological_charge'], 0., atol=1e-13, rtol=0)
    # A separate nonzero-force control distinguishes the exact-vacuum edge
    # case from the ensemble measurements.
    near_identity = LatticeGauge(LatticeInfo([24]*4))
    near_identity.gauss(1234, .01)
    near, _ = measure(near_identity, .02, np.array([0., .1, .2]))
    np.savez_compressed(output/'near_identity_check.npz', flow_times=[0., .1, .2], **near)
    assert all(np.isfinite(value).all() for value in near.values())
    assert np.all(np.diff(near['plaquette'][:, 0]) >= -1e-12)
    validation['near_identity_field_check'] = dict(finite=True, plaquette_monotonic=True,
                                                  seed=1234, sigma=.01)
    (output/'validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    for stage in STAGES:
        input_dir = (ROOT/'artifacts/4dsu3/L24_beta6p20' if stage == 'reference'
                     else args.run/'generated_fields'/stage)
        manifest = json.loads((input_dir/'ensemble.json').read_text())
        by_id = {r['index']: r for r in manifest['configurations']}
        records = [by_id[index] for index in SOURCE_IDS]
        settings = dict(stage=stage, input=str(input_dir.resolve()), checkpoint_sha256=checkpoint_hash,
                        downsampling_checkpoint_sha256=run['downsampling_checkpoint_sha256'],
                        configurations=records, selection='six fixed evenly spaced test-source positions',
                        source_indices=list(SOURCE_IDS), lattice_size=[24]*4, epsilon=.02, t_max=5.,
                        flow_times=FLOW_TIMES.tolist(), flow='4D Wilson gradient flow',
                        integrator='QUDA RK3', precision='double', pyquda_version=version('pyquda'),
                        measurement='explicit endpoint resident-field plaquette, clover energy and unrounded Q',
                        plaquette_components=['total', 'spatial', 'temporal'],
                        energy_components=['total', 'spatial', 'temporal'],
                        plaquette_energy_definition='E_plaq = 36 * (1 - P_total), in lattice units',
                        reader='NERSC checksum checked; same reader and exported fields as topology notebook')
        production = measure_set(input_dir, records, .02, output/stage, settings, readback=True)
        cached_dir = (ROOT/'presentation/results/topo_charge/full' if stage == 'reference'
                      else ROOT/'presentation/results/topo_charge/upsampling'/stage)
        with np.load(cached_dir/'measurements.npz') as previous:
            rows = [previous['config_indices'].tolist().index(i) for i in SOURCE_IDS]
            points = [int(np.argmin(abs(FLOW_TIMES-t))) for t in previous['flow_times']]
            old_q = previous['topological_charge'][rows]
            old_e = previous['energy'][rows]
            delta_q = production['topological_charge'][:, points]-old_q
            delta_e = production['energy'][:, points]-old_e
        # Refine both the first source and the largest-charge-drift source in
        # the fixed six-source subset. This is a stress check, not a random sample.
        drift = np.abs(production['topological_charge'][:, -1]-production['topological_charge'][:, 0])
        check_rows = sorted({0, int(np.argmax(drift))})
        check_records = [records[row] for row in check_rows]
        check_settings = dict(settings, configurations=check_records, epsilon=.01,
                              source_indices=[r['index'] for r in check_records],
                              selection='first source and largest |Q(5)-Q(0)| in the fixed six-source subset')
        refined = measure_set(input_dir, check_records, .01, output/stage/'halfstep', check_settings)
        differences = {key: float(np.max(np.abs(refined[key]-production[key][check_rows])))
                       for key in ('plaquette', 'energy', 'topological_charge')}
        validation['stages'][stage] = dict(
            existing_charge_max_difference=float(np.max(np.abs(delta_q))),
            existing_energy_max_difference=float(np.max(np.abs(delta_e))),
            halfstep_source_indices=[r['index'] for r in check_records],
            halfstep_max_absolute_differences=differences,
            production_minimum_plaquette_increment=float(np.diff(production['plaquette'][:, :, 0], axis=1).min()),
            production_plaquette_decreasing_intervals=int((np.diff(production['plaquette'][:, :, 0], axis=1) < -1e-12).sum()),
            halfstep_plaquette_decreasing_intervals=int((np.diff(refined['plaquette'][:, :, 0], axis=1) < -1e-12).sum()),
        )
        (output/'validation.json').write_text(json.dumps(validation, indent=2)+'\n')
        print(stage, validation['stages'][stage], flush=True)
    provenance = dict(checkpoint_sha256=checkpoint_hash, source_indices=list(SOURCE_IDS),
                      measurement_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      complete=True, validation='validation.json')
    (output/'run.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print('Gradient-flow diagnostics complete', flush=True)


if __name__ == '__main__':
    main()
