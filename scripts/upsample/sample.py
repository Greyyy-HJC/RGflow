#!/usr/bin/env python3
"""Generate a fine SU(3) configuration from coarse links and fresh detail noise."""

import argparse
from pathlib import Path

import torch

from rgflow.su3.downsampling import block_links, su3_errors
from rgflow.su3.io import read_nersc_gauge
from rgflow.su3.observables import observable_names, observable_vector_torch
from rgflow.su3.upsampling import SU3ConditionalFlow

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--coarse-file", type=Path, required=True, help="NERSC file or a Torch coarse-link tensor")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--device", default="cuda")
parser.add_argument("--seed", type=int, default=1234)
args = parser.parse_args()
torch.set_num_threads(2)
device = torch.device(args.device)
checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
flow1 = SU3ConditionalFlow(sweeps=checkpoint["sweeps"], constrained=True).to(device).eval()
flow2 = SU3ConditionalFlow(sweeps=checkpoint["fine_sweeps"]).to(device).eval()
flow1.load_state_dict(checkpoint["flow1"])
flow2.load_state_dict(checkpoint["flow2"])
if args.coarse_file.suffix in (".pt", ".pth"):
    coarse = torch.load(args.coarse_file, map_location=device, weights_only=True)
    if isinstance(coarse, dict):
        coarse = coarse["coarse"]
else:
    coarse = torch.from_numpy(read_nersc_gauge(args.coarse_file)[0]).to(device)
generator = torch.Generator(device=device).manual_seed(args.seed)
with torch.no_grad():
    smeared, _ = flow1.sample(coarse, generator=generator, compute_logdet=False)
    fine, _ = flow2(smeared, compute_logdet=False)
    operators = observable_vector_torch(fine).cpu()
    blocking_error = float((block_links(smeared)-coarse).abs().max())
    errors = su3_errors(fine)
args.output.parent.mkdir(parents=True, exist_ok=True)
torch.save({"links": fine.cpu(), "smeared": smeared.cpu(), "coarse": coarse.cpu(),
            "seed": args.seed, "checkpoint": str(args.checkpoint.resolve()),
            "observable_names": observable_names(), "observables": operators,
            "su3_errors": errors, "blocking_error": blocking_error}, args.output)
print(dict(zip(observable_names(), operators.tolist())))
print("SU(3) errors:", errors, "blocking error:", blocking_error)
print(args.output)
