"""Exercise a full-volume Flow-1 moment VJP before resuming training."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from rgflow.su3.observables import observable_vector_torch
from rgflow.su3.upsampling import SU3ConditionalFlow, constrained_haar_lift

torch.set_num_threads(2)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run', type=Path,
                    default=Path('artifacts/4dsu3/upsample/L12_to_L24_2flow_stable_v3'))
run = parser.parse_args().run
saved = torch.load(run/'flows.pt', map_location='cuda', weights_only=True)
coarse = torch.load(run/'numerical_sample.pt', map_location='cuda', weights_only=True)['coarse']
flow = SU3ConditionalFlow(sweeps=saved['sweeps'], constrained=True, coupling=saved['coupling']).cuda()
flow.load_state_dict(saved['flow1'])
with torch.no_grad():
    flow.eval()
    base = constrained_haar_lift(coarse, generator=torch.Generator(device='cuda').manual_seed(8324))
    output, _ = flow(base, compute_logdet=False)
source = output.detach().requires_grad_()
probe = observable_vector_torch(source, checkpoint_loops=True)[0]
gradient = torch.autograd.grad(probe, source)[0]
del source, probe, output
flow.train()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
started = time.time()
output, _ = flow(base, compute_logdet=False)
print('L24 forward complete', time.time()-started, 'seconds', flush=True)
output.backward(gradient)
record = {
    'seconds': time.time()-started,
    'peak_gpu_allocated_gib': torch.cuda.max_memory_allocated()/1024**3,
    'checkpoint_sha256': hashlib.sha256((run/'flows.pt').read_bytes()).hexdigest(),
    'finite_gradients': all(p.grad is not None and torch.isfinite(p.grad).all().item()
                            for p in flow.parameters()),
    'lattice_size': list(output.shape[-6:-2]),
    'sweeps': saved['sweeps'], 'dtype': str(output.dtype),
}
print(record, flush=True)
assert record['finite_gradients']
assert record['peak_gpu_allocated_gib'] < 10
(run/'offload_validation.json').write_text(json.dumps(record, indent=2)+'\n')
