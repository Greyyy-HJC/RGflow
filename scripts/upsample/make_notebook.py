#!/usr/bin/env python3
"""Replace the two-flow presentation with the current eight-operator experiment."""

import argparse
import os
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run', type=Path,
                    default=ROOT/'artifacts/4dsu3/upsample/L12_to_L24_2flow_v6_8ops_6train')
args = parser.parse_args()
relative_run = os.path.relpath(args.run.resolve(), ROOT)
required = ['observable_distributions.npz', 'metrics.json', 'history.json', 'run.json',
            'flows.pt', 'numerics.json', 'execution_notes.json', 'training_arguments.json',
            'training_invocations.json', 'calibration_stage1.json', 'calibration_stage2.json',
            'cache_metadata.json', 'source_hashes.json', 'promotion.json']
for name in required:
    if not (args.run/name).exists():
        raise FileNotFoundError(args.run/name)
notebook = nbf.v4.new_notebook()
notebook.metadata.kernelspec = {'display_name': 'Python (RGflow .venv)', 'language': 'python', 'name': 'python3'}
notebook.cells = [
nbf.v4.new_markdown_cell(r'''# SU(3) upsampling with two normalizing flows

**L12 → L24 · βfine = 6.20 · βcoarse = 5.775 · six training operators, eight evaluation operators**

Use the frozen **v4_8ops_6train** downsampling checkpoint fitted against the
lattice-spacing-matched L12 ensemble. The input here is the held-out
**downsampled fine ensemble**, $C=B(S(U))$. The native coarse ensemble fixes
the downsampling target; direct generation from native L12 inputs is a separate experiment.
No heatbath or HMC updates are applied to either flow output.

| Stage | Operation | Degrees of freedom per coarse cell |
| --- | --- | --- |
| Input | $C=B(S(U))$ | four SU(3) links |
| Flow 1 | $\widetilde S=F_1(C,z)$, conditional Haar details | 60 free links; four two-link products fixed |
| Flow 2 | $\widetilde U=F_2(\widetilde S)$ | transport of all 64 links |

Train on $W_{11},W_{12},W_{23},W_{33},\lvert P\rvert^2,\lvert P\rvert^4$.
**$W_{13}$ and $W_{22}$ are evaluation-only** and enter neither moment fitting
nor validation selection. The fourth Polyakov moment is
$\frac14\sum_\mu\lvert P_\mu\rvert^4$, after the transverse average in each direction.
Training, validation and test sources retain the downsampling 180/60/60 split.
'''),
nbf.v4.new_code_cell('''%matplotlib inline
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

ROOT = next(p for p in (Path.cwd(), Path.cwd().parent) if (p/'pyproject.toml').exists())
RUN = ROOT / __RUN_DIRECTORY__
data = np.load(RUN/'observable_distributions.npz')
metrics = json.loads((RUN/'metrics.json').read_text())
run = json.loads((RUN/'run.json').read_text())
history = json.loads((RUN/'history.json').read_text())
notes = json.loads((RUN/'execution_notes.json').read_text())
names = metrics['observable_names']
held_out = metrics['held_out_observables']
labels = [r'$W_{11}$', r'$W_{12}$', r'$W_{13}$', r'$W_{22}$', r'$W_{23}$', r'$W_{33}$',
          r'$|P|^2$', r'$|P|^4$']
np.testing.assert_array_equal(data['observable_names'], names)
plt.rcParams.update({'figure.dpi': 125, 'axes.grid': True, 'grid.alpha': .15})
print('Downsampling run:', Path(run['arguments']['downsampling_run']).name)
print('Source run:', RUN)
print('Downsampling checkpoint:', run['downsampling_checkpoint_sha256'])
print('Two-flow checkpoint:', run['checkpoint_sha256'])
print(f"{len(run['train_indices'])} training / {len(run['validation_indices'])} validation / {len(run['test_indices'])} test sources")
print('Training IDs:', run['train_indices'])
print('Validation IDs:', run['validation_indices'])
print('Test IDs:', data['test_indices'].tolist())
print('Training operators:', metrics['training_observables'])
print('Held-out operators:', held_out)
print('Flow sweeps:', run['arguments']['sweeps'], '+', run['arguments']['fine_sweeps'])
print('Initialization:', notes['initialization'])
print('Rethermalization sweeps:', metrics['rethermalization_sweeps'])
m = metrics['flow2_fine']
passed = ((np.abs(m['mean_shift_in_target_sigma']) <= 1)
          & (np.abs(np.asarray(m['std_ratio'])-1) <= .2))
display(Markdown(f"**Final Flow-2 test outcome: {'PASS' if m['acceptance']['distribution_match'] else 'FAIL'}**; "
                 f"{passed.sum()}/{len(names)} individual operators pass both mean and width criteria. "
                 "The two unseen operators are marked in every distribution figure."))
'''),
nbf.v4.new_markdown_cell(r'''## Fresh training and stage-wise calibration

Both covariant SU(3) maps start from seed 1234 with fresh neural weights.
The warm-up uses single precision: Flow 1 trains its exact conditional Haar
likelihood for four epochs on 12 training sources, and Flow 2 trains paired
link transport for three epochs. Both parameter sets are then promoted to
double precision without changing their values. Flow 1's six shared offsets
(scale, cubic shape and long-staple weight in each half) are fitted to six
means and six log variances. After freezing Flow 1, Flow 2 fits its own six
offsets on **generated Flow-1 inputs**.

Each calibration uses fixed, independent training and validation latent draws,
12 training sources and eight validation sources. Means use per-target standard
deviations; log variances use the relative floor $10^{-12}s_{\mathrm{target},j}^2$.
The log-variance squared residual has weight 10. Best stable validation loss
selects the checkpoint, with at most 30 objective evaluations, 65 minutes and
six evaluations of validation patience per stage. The actual stopping reasons
are printed below. No test targets enter training or selection.

Calibration and final generation retain complex128 fields and float64 conditioners. Alternating
checkerboards, directions and embedded SU(2) rotations use covariant inactive
staple frames. Radial Möbius and monotone cubic maps have analytic Haar
Jacobians and inverses. Flow 1 freezes both links of each retained two-link
product; Flow 2 transports every link. Small-lattice inverse, gauge covariance
and density checks gate updates; full L24 checks gate calibration.
The smoother includes a polar projection, so Flow 2 learns a distributional
transport rather than a global inverse of that smoother.
'''),
nbf.v4.new_code_cell('''fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
for ax, phase, title in zip(axes.flat,
        ['conditional_nll', 'smeared_moments', 'paired_transport', 'generated_moments'],
        ['Flow 1: conditional NLL / free link', 'Flow 1: six-operator calibration',
         'Flow 2: paired transport', 'Flow 2: six-operator calibration']):
    rows = [r for r in history if r['phase'] == phase]
    ax.plot([r['epoch'] for r in rows], [r['train_loss'] for r in rows], 'o-', label='Training')
    ax.plot([r['epoch'] for r in rows], [r['validation_loss'] for r in rows], 'o-', label='Validation')
    rejected = [r for r in rows if not r['stability']['passed']]
    if rejected:
        ax.scatter([r['epoch'] for r in rejected], [r['validation_loss'] for r in rejected],
                   marker='x', s=70, color='#b71c1c', zorder=4, label='Failed numerical check')
    selected = min((r for r in rows if r['stability']['passed']), key=lambda r: r['validation_loss'])
    ax.scatter([selected['epoch']], [selected['validation_loss']], marker='*', s=130,
               color='#2e7d32', zorder=5, label='Selected stable checkpoint')
    if phase in ['smeared_moments', 'generated_moments']:
        ax.set_yscale('log')
    ax.set(title=title, xlabel='Calibration evaluation' if 'moments' in phase else 'Epoch')
    ax.legend(fontsize=8)
fig.savefig(RUN/'upsampling_training.png', dpi=180, bbox_inches='tight')
fig.savefig(RUN/'upsampling_training.pdf', bbox_inches='tight')
plt.show()
for stage in (1, 2):
    fit = json.loads((RUN/f'calibration_stage{stage}.json').read_text())
    print(f"Flow {stage}: {fit['evaluations']} evaluations; stop: {fit['stopping_reason']}; optimizer converged: {fit['optimizer_converged']}; best validation loss: {fit['best_validation_loss']:.6g}")
'''),
nbf.v4.new_markdown_cell(r'''## Eight-operator distributions

Use one fresh Haar latent draw for each of the **60 held-out coarse sources**,
with the same draw passed through Flow 1 and Flow 2. Results are shown in stage
order: **Flow 1 versus the smeared reference $S(U)$**, then **Flow 2 versus the
original fine reference $U$**. Each stage has its own eight-operator raw
distributions, followed by its statistical tables.
Reference and generated fields each use 18 bins over their own data range,
overlaid on the same axes with probability-density normalization.

The distribution criterion requires each mean shift to be within **one target
standard deviation** and each width ratio in **[0.8, 1.2]**. The combined-SE shift
is descriptive for these paired sources. Results are grouped into the six
training operators, two held-out operators and all eight.
'''),
nbf.v4.new_code_cell('''def plot_operators(candidate, target, title, filename):
    figure, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    reference_label = 'Smeared reference S(U)' if target == 'smeared_reference' else 'Fine reference U'
    for k, ax in enumerate(axes.flat):
        reference, generated = data[target][:, k], data[candidate][:, k]
        ax.hist(reference, bins=18, density=True, alpha=.4, color='#333333', label=reference_label)
        ax.hist(generated, bins=18, density=True, histtype='step', linewidth=1.8,
                color='#1565c0', label=title)
        suffix = ' [HELD OUT]' if names[k] in held_out else ''
        ax.set_title(labels[k]+suffix, fontsize=12)
        ax.set(xlabel=labels[k], ylabel='Density')
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3), useOffset=False)
        m = metrics[candidate]
        ax.text(.03, .96, f"shift/σ={m['mean_shift_in_target_sigma'][k]:+.2f}\\nwidth={m['std_ratio'][k]:.2f}",
                transform=ax.transAxes, va='top', fontsize=9)
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc='outside upper center', ncol=2)
    figure.savefig(RUN/f'{filename}.png', dpi=180, bbox_inches='tight')
    figure.savefig(RUN/f'{filename}.pdf', bbox_inches='tight')
    plt.show()

def report_operators(candidate, title):
    m = metrics[candidate]
    rows = [f'**{title}**', '',
            '| Operator | Role | Target mean | Generated mean | Shift / target σ | Shift / combined SE | Width ratio | Pass |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: |']
    for k, name in enumerate(names):
        passed = abs(m['mean_shift_in_target_sigma'][k]) <= 1 and .8 <= m['std_ratio'][k] <= 1.2
        role = 'held out' if name in held_out else 'training'
        rows.append(f"| {name} | {role} | {m['target_mean'][k]:.7g} | {m['mean'][k]:.7g} | {m['mean_shift_in_target_sigma'][k]:+.3f} | {m['standardized_shift'][k]:+.3f} | {m['std_ratio'][k]:.3f} | {'YES' if passed else 'NO'} |")
    display(Markdown('\\n'.join(rows)))
    summary = ['| Operator group | Means within one target σ | Widths within 20% | Distribution match |',
               '| --- | :---: | :---: | :---: |']
    for group, result in m['group_acceptance'].items():
        summary.append(f"| {group} | {result['test_all_means_within_one_target_sigma']} | {result['test_all_std_ratios_within_20_percent']} | {result['distribution_match']} |")
    display(Markdown('\\n'.join(summary)))
'''),
nbf.v4.new_markdown_cell(r'''## Flow 1: generated smeared fields versus smeared reference

Compare $F_1(C,z)$ with the actual v4-smeared reference $S(U)$ on all 60 test
sources. The figure shows all eight operators; $W_{13}$ and $W_{22}$ are held
out of training. The horizontal axes show the original operator values.
'''),
nbf.v4.new_code_cell('''plot_operators('flow1_smeared', 'smeared_reference', 'Flow 1 output', 'upsampling_flow1_distributions')
report_operators('flow1_smeared', 'Flow 1 vs smeared reference')
'''),
nbf.v4.new_markdown_cell(r'''## Flow 2: final generated fields versus fine reference

Compare $F_2(F_1(C,z))$ with the original fine reference $U$ on the same 60 test
sources. These outputs use the generated Flow-1 fields as inputs. The figure
shows all eight operators on their original scales.
'''),
nbf.v4.new_code_cell('''plot_operators('flow2_fine', 'fine_reference', 'Flow 2 output', 'upsampling_fine_distributions')
report_operators('flow2_fine', 'Flow 2 vs fine reference')
'''),
nbf.v4.new_markdown_cell(r'''## Additional diagnostic: Flow 2 with teacher input

This comparison feeds the actual smeared reference $S(U)$ into Flow 2 and
compares $F_2(S(U))$ with $U$. It diagnoses transport separately from Flow-1
generation. The final upsampling output is the generated-input result in the
preceding section. A successful comparison of eight observables alone would
not establish equality of the full gauge-field probability measures.
'''),
nbf.v4.new_code_cell('''plot_operators('teacher_fine', 'fine_reference', 'Flow 2, teacher input', 'upsampling_teacher_distributions')
report_operators('teacher_fine', 'Flow 2 with teacher input vs fine reference')
'''),
nbf.v4.new_markdown_cell('''## Numerical checks and topological charge

Both generated stages are exported as FP64 NERSC fields with source IDs,
latent seeds and checkpoint hashes. SU(3) reunitarization in double precision
prepares the Wilson-flow inputs; its maximum field change is recorded per
configuration. No Monte Carlo rethermalization is performed.

The companion [topological-charge notebook](topo_charge.ipynb) applies the
same Wilson gradient flow to both outputs: $t/a^2=0$–5, RK3 step 0.02,
measurements every 0.1, and unrounded clover topological charge.
'''),
nbf.v4.new_code_cell('''print('Maximum raw Flow 1 / Flow 2 SU(3) unitarity and determinant errors:', metrics['max_su3_errors'])
print('Maximum |block(Flow 1) - coarse|:', metrics['max_blocking_error'])
print('Three PyQUDA checks, max difference per operator:', np.abs(metrics['pyquda_differences']).max(axis=0))
print('Three Flow-1 PyQUDA checks, max difference per operator:', np.abs(metrics['pyquda_stage1_differences']).max(axis=0))
numerical = json.loads((RUN/'numerics.json').read_text())
assert numerical['checkpoint_sha256'] == run['checkpoint_sha256']
print('Local Haar Jacobian checks:', numerical['haar_jacobians'])
print('Full L24 checks:', numerical['full_lattice_generation'])
'''),
nbf.v4.new_markdown_cell('''## Reproduction

Run all commands through the root `.venv`:

```bash
.venv/bin/python scripts/upsample/rerun.py --output artifacts/4dsu3/upsample/new_run
.venv/bin/python scripts/upsample/make_notebook.py --run artifacts/4dsu3/upsample/new_run
```

The runner resolves the selected downsampling experiment, records all source
hashes and settings, trains both flows, calibrates each stage, validates the
frozen checkpoint and evaluates all 60 test sources. Select a fresh training
directory for a new run. The notebook reads that run directly; exported figures
are saved alongside its measurements, without copying a presentation bundle.
'''),
]
notebook.cells[1].source = notebook.cells[1].source.replace('__RUN_DIRECTORY__', repr(relative_run))
path = ROOT/'presentation/upsampling_2flow.ipynb'
nbf.write(notebook, path)
manager = KernelManager(kernel_name='python3', transport='ipc', ip=str(ROOT/'temp/upsampling_notebook_kernel'))
manager.kernel_spec.argv = [str(ROOT/'.venv/bin/python'), '-m', 'ipykernel_launcher', '-f', '{connection_file}']
NotebookClient(notebook, timeout=300, km=manager, resources={'metadata': {'path': str(ROOT)}}).execute()
assert any('image/png' in output.get('data', {}) for cell in notebook.cells
           for output in cell.get('outputs', [])), 'Notebook contains no embedded figures'
nbf.validate(notebook)
nbf.write(notebook, path)
print(path)
