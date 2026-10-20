#!/usr/bin/env python3
"""Update the topological-charge notebook to the current smoother and two-flow results."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
path = ROOT/'presentation/topo_charge.ipynb'
notebook = nbf.read(path, as_version=4)
notebook.cells = [cell for cell in notebook.cells if not cell.id.startswith('twoflowtopo_')]
removed_headings = ('## Load or measure smeared reference L24', '## Smeared reference L24:',
                    '## Paired charge values')
removed_code = ('smeared_data_path = SMEARED_RESULTS', 'plot_charge_curves(smeared_charge,',
                'plot_charge_curves(full_smeared_charge,', '| Reference Q(0) | Smeared Q(0) |')
notebook.cells = [cell for cell in notebook.cells
                  if not (cell.cell_type == 'markdown' and cell.source.startswith(removed_headings))
                  and not (cell.cell_type == 'code' and any(text in cell.source for text in removed_code))]
for cell in notebook.cells:
    cell.source = cell.source.replace('L24_beta6p20_to_L12_beta5p775_polynomial_local_ls_v3',
                                      'L24_beta6p20_to_L12_beta5p775_polynomial_local_ls_v4_8ops_6train')
    cell.source = cell.source.replace('Both branches explicitly measure', 'All branches explicitly measure')
    if cell.cell_type == 'code':
        cell.outputs = []
        cell.execution_count = None
notebook.cells[0].source = r'''# Topological charge: reference, two-flow upsampling and HMC on L24

Wilson pure-gauge SU(3), $24^4$, $\beta=6.20$. Every branch uses the same
**Wilson gradient flow, double-precision RK3, $\epsilon=0.02$, $0\le t/a^2\le5$,
measurements every 0.1**, and **unrounded QUDA traceless-clover topological charge**.

The reference plots retain the 30-configuration subset (IDs 0, 10, …, 290)
and all 300 reference fields (IDs 0–299), along with the existing HMC
continuation and fixed-flow-time histories.

The current upsampling comparison uses **all 60 held-out test sources**
(IDs 4, 9, …, 299) and the **v4_8ops_6train** smoother fitted against the
lattice-spacing-matched **βcoarse = 5.775** reference. It compares original
reference fields, the exact full-L24 smeared reference inputs, and the
outputs of **Flow 1 and Flow 2**. This replaces the previous smoother's plots.
All four branches are paired by source ID and start their gradient flow at
zero separately; the learned maps are not assigned gradient-flow time offsets.

The reference FP32 NERSC links are reunitarized on input. Generated and
smeared-reference outputs are exported in FP64 after double-precision SU(3)
reunitarization; its field change is recorded. Neither learned flow output
receives heatbath/HMC rethermalization. The separate HMC experiment runs
300 trajectories from reference ID 299 and measures 30 saved chain states.
'''
notebook.cells[1].source = '\n'.join(line for line in notebook.cells[1].source.splitlines()
                                     if not line.startswith(('SMEARED_RESULTS =', 'FULL_SMEARED_RESULTS =')))+'\n'
if 'UPSAMPLING_RUN =' not in notebook.cells[1].source:
    notebook.cells[1].source += "\nUPSAMPLING_RUN = ROOT / 'artifacts/4dsu3/upsample/L12_to_L24_2flow_v6_8ops_6train'\nUPSAMPLING_RESULTS = RESULTS / 'upsampling'\n"
for cell in notebook.cells:
    if cell.cell_type == 'code' and 'def plot_charge_curves(' in cell.source:
        cell.source = '''y_limits = (np.floor(charge.min())-.25, np.ceil(charge.max())+.25)

'''+cell.source[cell.source.index('def plot_charge_curves('):]
        old_guides = "for integer in range(int(np.ceil(limits[0])), int(np.floor(limits[1])) + 1):"
        cell.source = cell.source.replace(old_guides,
            "guide_step = 1 if limits[1]-limits[0] <= 20 else 10\n"
            "    for integer in range(int(np.ceil(limits[0]/guide_step))*guide_step, "
            "int(np.floor(limits[1])) + 1, guide_step):")
    if cell.cell_type == 'markdown' and cell.source.startswith('## Complete ensemble:'):
        cell.source = '''## Complete reference ensemble: all 300 configurations

Reuse the same 30 reference measurements and measure any remaining fields
with identical flow settings. The total plot contains all 300 charge curves;
the two-flow comparison below selects its 60 held-out source IDs from this array.
'''
    if cell.cell_type == 'code' and (cell.source.startswith('def load_full_charge(')
                                      or cell.source.startswith('full_metadata_path =')):
        cell.source = '''full_metadata_path = FULL_RESULTS/'run.json'
full_complete = full_metadata_path.exists() and json.loads(full_metadata_path.read_text())['complete']
if not full_complete:
    subprocess.run([str(ROOT/'.venv/bin/python'), str(ROOT/'scripts/heatbath/topo_charge.py'),
                    '--input', str(INPUT), '--output', str(FULL_RESULTS),
                    '--stride', '1', '--samples', '300', '--epsilon', str(EPSILON),
                    '--t-max', str(T_MAX), '--measure-every', str(MEASURE_EVERY),
                    '--reuse-from', str(RESULTS)], cwd=ROOT, check=True)
full_run = json.loads(full_metadata_path.read_text())
expected = dict(requested, start=0, stride=1, samples=300)
assert all(full_run['settings'][key] == value for key, value in expected.items())
with np.load(FULL_RESULTS/'measurements.npz') as saved:
    full_indices = saved['config_indices']
    np.testing.assert_array_equal(full_indices, np.arange(300))
    np.testing.assert_array_equal(saved['flow_times'], flow_times)
    full_charge = saved['topological_charge']
np.testing.assert_array_equal(full_charge[config_indices], charge)
assert full_run['complete'] and full_charge.shape == (300, len(flow_times)) and np.isfinite(full_charge).all()
full_y_limits = (np.floor(full_charge.min())-.25, np.ceil(full_charge.max())+.25)
print(f'{len(full_indices)} reference fields × {len(flow_times)} Wilson-flow times')
'''
new_cells = [
nbf.v4.new_markdown_cell(r'''## Two-flow outputs: the same 60 held-out sources

The fresh two-flow experiment uses the current **v4_8ops_6train** smoother,
fitted against the lattice-spacing-matched **βcoarse = 5.775** reference.
For each held-out fine source $U$, form $C=B(S(U))$, sample one conditional
Flow-1 output $\widetilde S=F_1(C,z)$ and then apply
$\widetilde U=F_2(\widetilde S)$. The latent seed is **1234 + 30000 + source ID**.
No Monte Carlo updates are applied to either output.

The following two figures use all **60 test sources (IDs 4, 9, …, 299)**,
the exact outputs evaluated in `upsampling_2flow.ipynb`, and the same colors
and vertical limits. Wilson gradient flow starts at zero separately for each
stage: **$t/a^2=0$–5, double-precision RK3, ε = 0.02, measurements every 0.1**.
Charges remain unrounded. Flow-1 and Flow-2 fields are saved in FP64 NERSC
after double-precision SU(3) reunitarization, with the maximum change recorded
below. This is preparation for the common Wilson-flow measurement.

The paired smeared reference is measured from the exact cached training-map
output; the raw fine reference reuses the existing 300-field measurement.
Source IDs pair conditional samples, and do not form a continuous Monte Carlo
trajectory through the generated fields.
'''),
nbf.v4.new_code_cell('''import hashlib
upsampling_run = json.loads((UPSAMPLING_RUN/'run.json').read_text())
upsampling_hash = hashlib.sha256((UPSAMPLING_RUN/'flows.pt').read_bytes()).hexdigest()
assert upsampling_hash == upsampling_run['checkpoint_sha256']
assert upsampling_run['downsampling_checkpoint_sha256'] == hashlib.sha256((SMEARING_RUN/'kernel.pt').read_bytes()).hexdigest()
upsampling_ids = np.asarray(upsampling_run['test_indices'])
np.testing.assert_array_equal(upsampling_ids, np.arange(4, 300, 5))
reference_hashes = {r['index']: r['sha256'] for r in full_run['settings']['configurations']}
fine_hashes = {r['index']: r['sha256'] for r in upsampling_run['fine_manifest']['configurations']}
assert all(reference_hashes[int(index)] == fine_hashes[int(index)] for index in upsampling_ids)
upsampling_charges = {'reference': full_charge[upsampling_ids]}
projection_changes = {}
for stage in ('flow1', 'flow2', 'smeared_reference'):
    directory = UPSAMPLING_RESULTS/stage
    metadata_path = directory/'run.json'
    complete = metadata_path.exists() and json.loads(metadata_path.read_text())['complete']
    if not complete:
        subprocess.run([str(ROOT/'.venv/bin/python'), str(ROOT/'scripts/heatbath/topo_charge.py'),
                        '--input', str(UPSAMPLING_RUN/'generated_fields'/stage), '--output', str(directory),
                        '--stride', '1', '--samples', str(len(upsampling_ids)), '--epsilon', str(EPSILON),
                        '--t-max', str(T_MAX), '--measure-every', str(MEASURE_EVERY)], cwd=ROOT, check=True)
    stage_run = json.loads(metadata_path.read_text())
    stage_settings = stage_run['settings']
    assert stage_settings['upsampling']['checkpoint_sha256'] == upsampling_hash
    assert stage_settings['upsampling']['downsampling_checkpoint_sha256'] == upsampling_run['downsampling_checkpoint_sha256']
    assert stage_settings['upsampling']['stage'] == stage
    assert stage_settings['epsilon'] == EPSILON and stage_settings['t_max'] == T_MAX
    assert stage_settings['measure_every'] == MEASURE_EVERY and stage_settings['precision'] == 'double'
    with np.load(directory/'measurements.npz') as saved:
        np.testing.assert_array_equal(saved['config_indices'], upsampling_ids)
        np.testing.assert_array_equal(saved['flow_times'], flow_times)
        upsampling_charges[stage] = saved['topological_charge']
    assert upsampling_charges[stage].shape == (60, len(flow_times))
    assert np.isfinite(upsampling_charges[stage]).all()
    manifest = json.loads((UPSAMPLING_RUN/'generated_fields'/stage/'ensemble.json').read_text())
    projection_changes[stage] = max(r['double_reunitarization_max_abs_change'] for r in manifest['configurations'])
upsampling_limits = (np.floor(min(q.min() for q in upsampling_charges.values()))-.25,
                     np.ceil(max(q.max() for q in upsampling_charges.values()))+.25)
np.savez_compressed(UPSAMPLING_RESULTS/'charge_comparison.npz', config_indices=upsampling_ids,
                    flow_times=flow_times, **upsampling_charges)
print('Two-flow checkpoint:', upsampling_hash)
print('Maximum double-reunitarization field changes:', projection_changes)
print('Paired sources:', upsampling_ids.tolist())
'''),
nbf.v4.new_code_cell('''plot_charge_curves(upsampling_charges['flow1'], upsampling_ids, flow_times, upsampling_limits,
                   r'Flow 1 output L24: 60 held-out charge-evolution curves',
                   'topo_charge_flow1', output=UPSAMPLING_RESULTS, colorbar_label='Source configuration ID')
'''),
nbf.v4.new_code_cell('''plot_charge_curves(upsampling_charges['flow2'], upsampling_ids, flow_times, upsampling_limits,
                   r'Flow 2 output L24: 60 held-out charge-evolution curves',
                   'topo_charge_flow2', output=UPSAMPLING_RESULTS, colorbar_label='Source configuration ID')
'''),
nbf.v4.new_markdown_cell(r'''## Compare the reference and both generated stages

These four panels show the **same 60 source IDs**, with identical colors and
axis limits. The fixed-time distribution below uses **unrounded $Q(t/a^2=5)$**.
The paired differences are descriptive; the generative flows are not constrained
to reproduce each source's topological charge, and charge agreement is not
used for training or checkpoint selection.
'''),
nbf.v4.new_code_cell('''stage_labels = {'reference': 'Reference L24', 'smeared_reference': 'Smeared reference L24',
                'flow1': 'Flow 1 output L24', 'flow2': 'Flow 2 output L24'}
colors = plt.get_cmap('turbo', len(upsampling_ids))(np.arange(len(upsampling_ids)))
figure, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True, constrained_layout=False)
figure.subplots_adjust(left=.08, right=.98, bottom=.09, top=.95, hspace=.28, wspace=.12)
for axis, (stage, label) in zip(axes.flat, stage_labels.items()):
    for curve, color in zip(upsampling_charges[stage], colors):
        axis.plot(flow_times, curve, color=color, linewidth=.8, alpha=.7)
    guide_step = 1 if upsampling_limits[1]-upsampling_limits[0] <= 20 else 10
    for integer in range(int(np.ceil(upsampling_limits[0]/guide_step))*guide_step,
                         int(np.floor(upsampling_limits[1]))+1, guide_step):
        axis.axhline(integer, color='.75', linestyle=':', linewidth=.5, zorder=0)
    axis.set(title=label, xlabel=r'Gradient-flow time $t/a^2$', ylabel=r'$Q(t)$',
             xlim=(0, T_MAX), ylim=upsampling_limits)
figure.savefig(UPSAMPLING_RESULTS/'topo_charge_stage_comparison.png', dpi=180, bbox_inches='tight')
figure.savefig(UPSAMPLING_RESULTS/'topo_charge_stage_comparison.pdf', bbox_inches='tight')
plt.show()

comparison_time = float(flow_times[-1])
column = len(flow_times)-1
figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
bins = np.histogram_bin_edges(np.concatenate([q[:, column] for q in upsampling_charges.values()]), bins=18)
for color, (stage, label) in zip(['#333333', '#777777', '#d17618', '#1565c0'], stage_labels.items()):
    axis.hist(upsampling_charges[stage][:, column], bins=bins, density=True, histtype='step',
              linewidth=1.8, color=color, label=label)
axis.set(xlabel=rf'Unrounded topological charge $Q(t/a^2={comparison_time:g})$',
         ylabel='Probability density', title='Same 60 held-out sources: topological-charge distributions')
axis.legend(fontsize=11)
figure.savefig(UPSAMPLING_RESULTS/'topo_charge_stage_distribution.png', dpi=180, bbox_inches='tight')
figure.savefig(UPSAMPLING_RESULTS/'topo_charge_stage_distribution.pdf', bbox_inches='tight')
plt.show()

summary = {'checkpoint_sha256': upsampling_hash,
           'downsampling_checkpoint_sha256': upsampling_run['downsampling_checkpoint_sha256'],
           'config_indices': upsampling_ids.tolist(), 'flow_time': comparison_time,
           'epsilon': EPSILON, 't_max': T_MAX, 'measure_every': MEASURE_EVERY,
           'charge_operator': 'unrounded QUDA clover Q after Wilson flow',
           'maximum_reunitarization_field_changes': projection_changes, 'stages': {}}
rows = ['| Stage | Mean Q(0) | Mean Q(5) | SD Q(5) | ⟨Q(5)²⟩ | Mean absolute paired ΔQ(5) |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
for stage, label in stage_labels.items():
    q = upsampling_charges[stage]
    target = upsampling_charges['smeared_reference' if stage == 'flow1' else 'reference'][:, column]
    final = q[:, column]
    record = dict(mean_q0=float(q[:, 0].mean()), mean_q5=float(final.mean()),
                  std_q5=float(final.std(ddof=1)), mean_q5_squared=float(np.mean(final**2)),
                  mean_absolute_paired_difference=float(np.abs(final-target).mean()))
    summary['stages'][stage] = record
    rows.append(f"| {label} | {record['mean_q0']:+.4f} | {record['mean_q5']:+.4f} | {record['std_q5']:.4f} | {record['mean_q5_squared']:.4f} | {record['mean_absolute_paired_difference']:.4f} |")
(UPSAMPLING_RESULTS/'charge_comparison.json').write_text(json.dumps(summary, indent=2)+'\\n')
display(Markdown('\\n'.join(rows)))
upsampling_validation_path = UPSAMPLING_RESULTS/'validation.json'
if upsampling_validation_path.exists():
    validation = json.loads(upsampling_validation_path.read_text())
    assert validation['checkpoint_sha256'] == upsampling_hash
    print('Wilson-flow half-step checks (first held-out source per stage):', validation['stages'])
'''),
]
reproduction = notebook.cells.pop()
reproduction.source = '''## Reproduction

The reference and HMC result bundles are reused. To recompute these branches:

```bash
.venv/bin/python scripts/heatbath/topo_charge.py
.venv/bin/python scripts/heatbath/topo_charge.py --stride 1 --samples 300 --output presentation/results/topo_charge/full --reuse-from presentation/results/topo_charge
.venv/bin/python scripts/heatbath/hmc_continue.py
.venv/bin/python scripts/heatbath/topo_charge.py --input temp/topo_charge_hmc_L24 --stride 1 --samples 30 --output presentation/results/topo_charge/hmc/flow
```

After completing the current two-flow run, replace the stage measurements,
validate the flow step size, then execute this notebook:

```bash
.venv/bin/python scripts/upsample/measure_topology.py
.venv/bin/python scripts/upsample/validate_topology.py
.venv/bin/python scripts/upsample/update_topo_notebook.py
.venv/bin/python temp/execute_topo_charge_notebook.py
```

The two-flow charge curves and four-stage comparison are in `upsampling/`.
`upsampling/charge_comparison.npz` preserves all four unrounded charge arrays,
source IDs and flow times; its JSON companion records checkpoint hashes,
measurement settings and the charge summaries.
'''
for index, cell in enumerate(new_cells):
    cell.id = f'twoflowtopo_{index}'
notebook.cells.extend(new_cells)
notebook.cells.append(reproduction)
nbf.validate(notebook)
nbf.write(notebook, path)
print('Updated', path)
