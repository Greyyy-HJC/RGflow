#!/usr/bin/env python3
"""Independently check QUDA's early-time clover energy and charge with Torch loops."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from pyquda.enum_quda import QudaVerbosity
from pyquda.quda import setVerbosityQuda
from pyquda_utils import core, io

from rgflow.su3.observables import path_product


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'artifacts/4dsu3/upsample/L12_to_L24_2flow_v6_8ops_6train'
OUTPUT = ROOT/'presentation/results/gradient_flow_diagnostics'


@torch.no_grad()
def clover_observables(links):
    """Four oriented plaquettes per plane, anti-Hermitian traceless field tensor."""
    strength = {}
    energy = [0., 0.]
    identity = torch.eye(3, dtype=links.dtype, device=links.device)
    for mu in range(4):
        for nu in range(mu+1, 4):
            paths = [(mu, nu, mu+4, nu+4), (nu, mu+4, nu+4, mu),
                     (mu+4, nu+4, mu, nu), (nu+4, mu, nu, mu+4)]
            clover = sum(path_product(links, path) for path in paths)
            field = (clover-clover.mH)/8
            trace = field.diagonal(dim1=-2, dim2=-1).sum(-1)
            field = field-trace[..., None, None]*identity/3
            strength[mu, nu] = field
            energy[int(nu == 3)] += float(field.abs().square().sum((-2, -1)).mean())
    products = (strength[0, 1] @ strength[2, 3]
                - strength[0, 2] @ strength[1, 3]
                + strength[0, 3] @ strength[1, 2])
    charge = -float(products.diagonal(dim1=-2, dim2=-1).sum(-1).real.sum())/(4*np.pi**2)
    return np.array([sum(energy), *energy]), charge


def main():
    torch.set_num_threads(2)
    unit = torch.eye(3, dtype=torch.complex128).expand(4, 4, 4, 4, 4, 3, 3).clone()
    energy, charge = clover_observables(unit)
    np.testing.assert_array_equal(energy, 0.)
    assert charge == 0.
    generator = torch.Generator().manual_seed(1234)
    links, _ = torch.linalg.qr(torch.randn(unit.shape, dtype=unit.dtype, generator=generator))
    links = links*torch.exp(-1j*torch.angle(torch.linalg.det(links))/3)[..., None, None]
    gauge = links[0]
    transformed = torch.stack([gauge @ links[mu] @ torch.roll(gauge, -1, dims=3-mu).mH for mu in range(4)])
    energy, charge = clover_observables(links)
    other_energy, other_charge = clover_observables(transformed)
    np.testing.assert_allclose(energy, other_energy, atol=1e-12, rtol=0)
    np.testing.assert_allclose(charge, other_charge, atol=1e-12, rtol=0)
    core.init(None, [24]*4, backend='numpy',
              resource_path=str(ROOT/'artifacts/4dsu3/topo_charge/.quda-cache'))
    setVerbosityQuda(QudaVerbosity.QUDA_SILENT, b'')
    output = dict(source_index=4, times=[0., .04, .1], precision='complex128',
                  torch_identity_and_local_gauge_invariance_checks='passed on 4^4 with seed 1234',
                  definition='traceless anti-Hermitian four-leaf clover; E=-sum Tr(F^2)/V; Q=-sum Tr(F01F23-F02F13+F03F12)/(4 pi^2)',
                  measurement_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), stages={})
    for stage in ('reference', 'smeared_reference', 'flow1', 'flow2'):
        directory = ROOT/'artifacts/4dsu3/L24_beta6p20' if stage == 'reference' else RUN/'generated_fields'/stage
        manifest = json.loads((directory/'ensemble.json').read_text())
        record = next(record for record in manifest['configurations'] if record['index'] == 4)
        path = directory/record['filename']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
        initial = io.readNERSCGauge(str(path), checksum=True, plaquette=False,
                                   link_trace=False, reunitarize_sigma=0)
        fields = [initial, initial.copy(), initial.copy()]
        with initial.use() as dirac:
            dirac.smear_param.rk_order = 3
            dirac.wilsonFlow(2, .02, 0., restart=False, compute_qcharge=True)
            dirac.saveSmearedGauge(fields[1])
            dirac.wilsonFlow(3, .02, .04, restart=True, compute_qcharge=True)
            dirac.saveSmearedGauge(fields[2])
            dirac.freeSmearedGauge()
        checks = []
        for t, field in zip(output['times'], fields):
            energy_quda = np.asarray(field.energy())
            charge_quda = float(field.qcharge())
            links = torch.as_tensor(field.lexico(), dtype=torch.complex128, device='cuda')
            energy_torch, charge_torch = clover_observables(links)
            check = dict(time=t, energy_quda=energy_quda.tolist(), energy_torch=energy_torch.tolist(),
                         charge_quda=charge_quda, charge_torch=charge_torch,
                         maximum_energy_difference=float(np.max(np.abs(energy_torch-energy_quda))),
                         charge_difference=charge_torch-charge_quda)
            checks.append(check)
            np.testing.assert_allclose(energy_torch, energy_quda, atol=1e-11, rtol=0)
            np.testing.assert_allclose(charge_torch, charge_quda, atol=1e-10, rtol=0)
            del links
            torch.cuda.empty_cache()
        output['stages'][stage] = dict(input_sha256=record['sha256'], checks=checks)
        (OUTPUT/'independent_clover.json').write_text(json.dumps(output, indent=2)+'\n')
        print(stage, checks, flush=True)
    print('Independent Torch clover checks passed', flush=True)


if __name__ == '__main__':
    main()
