#!/usr/bin/env python3
"""Promote freshly trained covariant maps for double-precision calibration and generation."""

import argparse
import hashlib
from pathlib import Path

import torch

from rgflow.su3.training import _write_json
from rgflow.su3.upsampling import EQUIVARIANT_COUPLING, STABLE_COUPLING

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run', type=Path, required=True)
args = parser.parse_args()
path = args.run/'flows.pt'
before = hashlib.sha256(path.read_bytes()).hexdigest()
saved = torch.load(path, map_location='cpu', weights_only=True)
assert saved['coupling'] == EQUIVARIANT_COUPLING
for stage in ('flow1', 'flow2'):
    saved[stage] = {key: value.double() for key, value in saved[stage].items()}
saved['coupling'] = STABLE_COUPLING
temporary = path.with_suffix('.tmp')
torch.save(saved, temporary)
temporary.replace(path)
_write_json(args.run/'promotion.json', {'training_precision': 'single', 'evaluation_precision': 'double',
            'warmup_checkpoint_sha256': before, 'promoted_checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'operation': 'exact float32-to-float64 casting of all trained parameters; no parameter refit'})
print('Promoted both flows to double precision', flush=True)
