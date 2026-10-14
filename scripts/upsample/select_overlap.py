#!/usr/bin/env python3
"""Select a stable calibration candidate with validation mean shifts below two widths."""
import argparse
import json
from pathlib import Path

import torch

from rgflow.su3.training import _write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--mean-bound', type=float, default=2.)
args = parser.parse_args()
run = args.run.resolve()
calibration = json.loads((run/'calibration_stage2.json').read_text())
rows = [row for row in json.loads((run/'history.json').read_text())
        if row.get('optimizer') == 'global_least_squares'
        and row['baseline_checkpoint'] == calibration['baseline_checkpoint']
        and row['stability']['passed']]
eligible = [row for row in rows if max(abs(value) for value in row['validation_mean_shift']) <= args.mean_bound]
if not eligible:
    raise SystemExit('No stable candidate satisfies the validation mean bound; keep the calibration for further fitting.')
selected = min(eligible, key=lambda row: row['validation_loss'])
saved = torch.load(run/'calibration_baselines'/(calibration['baseline_checkpoint']+'.pt'), map_location='cpu', weights_only=True)
state = saved['flow2']
for sweep in range(saved['fine_sweeps']):
    half = int(sweep >= saved['fine_sweeps']//2)
    offset = selected['offsets'][3*half:3*half+3]
    state[f'scales.{sweep}.2.bias'][:3].add_(offset[0])
    state[f'scales.{sweep}.2.bias'][3:].add_(offset[1])
    state['long_weights'][sweep].add_(offset[2])
temporary = run/'flows.tmp'
torch.save(saved, temporary)
temporary.replace(run/'flows.pt')
calibration.update(unconstrained_best_validation_loss=calibration['best_validation_loss'],
                   best_validation_loss=selected['validation_loss'], best_offsets=selected['offsets'],
                   validation_mean_bound=args.mean_bound, selected_epoch=selected['epoch'])
_write_json(run/'calibration_stage2.json', calibration)
_write_json(run/'overlap_selection.json', dict(mean_bound=args.mean_bound, candidates=len(rows),
            eligible=len(eligible), selected=selected, test_targets_used=False))
print('SELECTED', selected['epoch'], selected['validation_mean_shift'], selected['validation_std_ratio'])
