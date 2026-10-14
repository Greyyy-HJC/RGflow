#!/usr/bin/env python3
"""Measure trained-flow covariance, reversibility and local Haar Jacobians."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import torch

import rgflow.su3.upsampling as up
from rgflow.su3.downsampling import block_links, su3_errors

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run', type=Path, default=ROOT/'artifacts/4dsu3/upsample/L12_to_L24_2flow_equivariant_v2')
parser.add_argument('--full-lattice-sample', type=Path, help='check synchronized-noise generation on its coarse field')
args = parser.parse_args()
torch.set_num_threads(2)
checkpoint_bytes = (args.run/'flows.pt').read_bytes()
saved = torch.load(io.BytesIO(checkpoint_bytes), map_location='cpu', weights_only=True)


def gauge_transform(links, gauge):
    dtype = torch.promote_types(links.dtype, gauge.dtype)
    links, gauge = links.to(dtype), gauge.to(dtype)
    return torch.stack([gauge @ links[:, mu] @ torch.roll(gauge, -1, 4-mu).conj().transpose(-2, -1)
                        for mu in range(4)], 1)


generators = []
for first, second in [(0, 1), (0, 2), (1, 2)]:
    real = torch.zeros(3, 3, dtype=torch.complex128)
    real[first, second] = real[second, first] = 1.
    imaginary = torch.zeros_like(real)
    imaginary[first, second], imaginary[second, first] = -1j, 1j
    generators.extend([real, imaginary])
generators.extend([torch.diag(torch.tensor([1., -1., 0.], dtype=torch.complex128)),
                   torch.diag(torch.tensor([1., 1., -2.], dtype=torch.complex128))/3.**.5])
generators = torch.stack(generators)
records, jacobians = [], []
for device, dtype, tolerance in [('cpu', torch.complex128, 1.e-10),
                                  ('cuda', torch.complex64, 3.e-4)]:
    if device == 'cuda' and not torch.cuda.is_available():
        continue
    for stage, sweeps in [(1, saved['sweeps']), (2, saved['fine_sweeps'])]:
        flow = up.SU3ConditionalFlow(sweeps=sweeps, constrained=stage == 1,
                                     coupling=saved['coupling']).to(device).eval()
        flow = flow.double() if dtype == torch.complex128 else flow
        flow.load_state_dict(saved[f'flow{stage}'])
        if saved['coupling'] == up.STABLE_COUPLING:
            tolerance = 1.e-10
        for seed in [381, 625, 902]:
            torch.manual_seed(seed)
            coarse = up.haar_links((1, 4, 2, 2, 2, 2), dtype=dtype, device=device)
            links = up.constrained_haar_lift(coarse)
            gauge_dtype = torch.complex128 if saved['coupling'] == up.STABLE_COUPLING else dtype
            gauge = up.haar_links((1, 4, 4, 4, 4), dtype=gauge_dtype, device=device)
            with torch.no_grad():
                output, density = flow(links)
                transformed, gauge_density = flow(gauge_transform(links, gauge))
                recovered, inverse_density = flow.inverse(output)
                inverse_gauge, _ = flow.inverse(gauge_transform(output, gauge))
            row = dict(stage=stage, device=device, dtype=str(dtype), output_dtype=str(output.dtype),
                       gauge_dtype=str(gauge_dtype), seed=seed,
                       gauge_error=float((transformed-gauge_transform(output, gauge)).abs().max()),
                       inverse_gauge_error=float((inverse_gauge-gauge_transform(recovered, gauge)).abs().max()),
                       roundtrip_error=float((recovered-links).abs().max()),
                       logdet_gauge_error=float((gauge_density-density).abs().max()),
                       logdet_roundtrip_error=float((density+inverse_density).abs().max()),
                       su3_errors=su3_errors(output))
            if stage == 1:
                row['blocking_error'] = float((block_links(output)-coarse).abs().max())
            assert row['gauge_error'] < tolerance, row
            assert row['inverse_gauge_error'] < tolerance, row
            assert row['roundtrip_error'] < tolerance, row
            if output.dtype == torch.complex128:
                assert row['logdet_gauge_error'] < 1.e-8, row
                assert row['logdet_roundtrip_error'] < 1.e-8, row
            records.append(row)
            print(f"checked stage={stage} device={device} seed={seed}: "
                  f"gauge={row['gauge_error']:.3e}, roundtrip={row['roundtrip_error']:.3e}", flush=True)
        if device != 'cpu':
            continue
        # One active link: compare the analytic determinant to the full
        # eight-dimensional tangent-volume change, including the conditioner.
        mask = torch.zeros((4,)*4, dtype=torch.bool)
        mask[1, 0, 0, 0] = True
        original_mask = up.coupling_mask
        up.coupling_mask = lambda *a, **kw: mask
        position = (0, 0, 1, 0, 0, 0)

        def mapped(coordinate, transform):
            current = links.clone()
            current[position] = torch.matrix_exp(1j*(coordinate[:, None, None]*generators).sum(0)) @ links[position]
            if transform:
                current, _ = flow._direction(current, 0, 0, 1, False)
            return torch.view_as_real(current[position]).reshape(-1)

        coordinate = torch.zeros(8, dtype=torch.float64)
        before = torch.autograd.functional.jacobian(lambda x: mapped(x, False), coordinate)
        after = torch.autograd.functional.jacobian(lambda x: mapped(x, True), coordinate)
        numeric = .5*(torch.linalg.slogdet(after.T@after)[1]-torch.linalg.slogdet(before.T@before)[1])
        _, analytic = flow._direction(links, 0, 0, 1, False)
        error = float((numeric-analytic[0]).abs().detach())
        assert error < 1.e-8, error
        jacobians.append(dict(stage=stage, numeric=float(numeric), analytic=float(analytic[0].detach()), error=error))
        print(f'checked stage={stage} full 8D Haar Jacobian: {error:.3e}', flush=True)
        up.coupling_mask = original_mask

result = dict(coupling=saved['coupling'], checkpoint_sha256=hashlib.sha256(checkpoint_bytes).hexdigest(),
              lattice_size=4, records=records, haar_jacobians=jacobians)
if args.full_lattice_sample:
    coarse = torch.load(args.full_lattice_sample, map_location='cuda', weights_only=True)['coarse']
    torch.manual_seed(8324)
    links = up.constrained_haar_lift(coarse).unsqueeze(0)
    gauge = up.haar_links((1, *links.shape[2:6]), device='cuda', dtype=links.dtype)
    transformed = gauge_transform(links, gauge)
    coarse_gauge = gauge[:, ::2, ::2, ::2, ::2]
    transformed_coarse = gauge_transform(coarse.unsqueeze(0), coarse_gauge)
    full = dict(lattice_size=list(links.shape[2:6]), seed=8324, device='cuda',
                synchronized_lift_error=float((block_links(transformed)-transformed_coarse).abs().max()),
                stage_gauge_errors=[], stage_roundtrip_errors=[])
    with torch.no_grad():
        for stage, sweeps in [(1, saved['sweeps']), (2, saved['fine_sweeps'])]:
            flow = up.SU3ConditionalFlow(sweeps=sweeps, constrained=stage == 1,
                                          coupling=saved['coupling']).cuda().eval()
            flow.load_state_dict(saved[f'flow{stage}'])
            source = links
            links, _ = flow(links, compute_logdet=False)
            transformed, _ = flow(transformed, compute_logdet=False)
            error = float((transformed-gauge_transform(links, gauge)).abs().max())
            tolerance = 1.e-10 if saved['coupling'] == up.STABLE_COUPLING else 3.e-4
            assert error < tolerance, error
            full['stage_gauge_errors'].append(error)
            recovered, _ = flow.inverse(links, compute_logdet=False)
            roundtrip = float((recovered-source).abs().max())
            assert roundtrip < tolerance, roundtrip
            full['stage_roundtrip_errors'].append(roundtrip)
            print(f'checked full-lattice stage={stage}: gauge={error:.3e}, roundtrip={roundtrip:.3e}', flush=True)
            if stage == 1:
                full['blocking_error'] = float((block_links(links)-coarse.unsqueeze(0)).abs().max())
    result['full_lattice_generation'] = full
(args.run/'numerics.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
