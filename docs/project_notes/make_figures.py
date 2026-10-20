#!/usr/bin/env python3
"""Rebuild the progress-note figures from a small, archived measurement bundle."""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--refresh-data', action='store_true', help='archive current measurements from artifacts')
selection = HERE/'results/selected_upsampling_run.json'
selected_run = (json.loads(selection.read_text())['run'] if selection.exists()
                else 'L12_to_L24_2flow_equivariant_v2')
parser.add_argument('--upsampling-run', type=Path,
                    default=ROOT/'artifacts/4dsu3/upsample'/selected_run)
down_selection = json.loads((HERE/'results/selected_downsampling_run.json').read_text())
parser.add_argument('--downsampling-run', type=Path,
                    default=ROOT/'artifacts/4dsu3/downsample'/down_selection['run'])
parser.add_argument('--coarse-ensemble', default=down_selection['coarse_ensemble'])
parser.add_argument('--downsampling-only', action='store_true', help='rebuild reference and downsampling figures without regenerating upsampling figures')
args = parser.parse_args()
DATA, FIGS = HERE/'results', HERE/'figs'
DATA.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)
down_name = args.downsampling_run.name
up_name = args.upsampling_run.name
if args.refresh_data:
    groups = {
        'L24_beta6p20': ['ensemble.json', 'static_potential.json', 'static_potential.npz', 'autocorrelation.json'],
        args.coarse_ensemble: ['ensemble.json', 'static_potential.json', 'static_potential.npz', 'autocorrelation.json'],
        'downsample/'+down_name: ['kernel.json', 'kernel.pt', 'metrics.json', 'fit_history.json', 'observable_distributions.npz',
                                 'source_hashes.json', 'environment.json', 'training_arguments.json', 'provenance.json',
                                 'execution_notes.json', 'gpu_attempt_history.json', 'gpu_attempt_source_hashes.json',
                                 'backend_crosscheck.json', 'evaluate_source.py', 'fit.log'],
        'upsample/'+up_name: ['run.json', 'metrics.json', 'history.json', 'observable_distributions.npz',
                             'calibration_stage1.json', 'calibration_stage2.json', 'training_arguments.json',
                             'training_invocations.json', 'execution_notes.json', 'numerics.json', 'validation.log', 'environment.json', 'flows.pt',
                             'initial_numerics.json', 'pipeline_status.txt',
                             'initial_flows.pt', 'source_hashes.json', 'calibration.log', 'targets.npz', 'tests.log',
                             'overlap_selection.json', 'selection.log',
                             'offload_validation.json', 'offload_validation.log'],
    }
    if up_name != 'L12_to_L24_2flow_v1':
        groups['upsample/L12_to_L24_2flow_v1'] = ['run.json', 'metrics.json', 'observable_distributions.npz']
    if args.downsampling_only:
        groups = {key: value for key, value in groups.items() if not key.startswith('upsample/')}
    for group, names in groups.items():
        destination = DATA/group.split('/')[-1]
        destination.mkdir(exist_ok=True)
        for name in names:
            source = (args.upsampling_run if group == 'upsample/'+up_name
                      else ROOT/'artifacts/4dsu3'/group)/name
            if source.exists():
                shutil.copy2(source, destination/name)

plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': .18,
                     'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
COLORS = ['#333333', '#2878b5', '#d17822']
NAMES = ['Plaquette', r'$W_{1\times2}$', r'$W_{2\times2}$', r'$|\overline{P}|^2$']


def save(fig, name):
    fig.savefig(FIGS/(name+'.pdf'), bbox_inches='tight')
    plt.close(fig)


fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout='constrained')
for tag, color in zip(['L24_beta6p20', args.coarse_ensemble], COLORS[1:]):
    d = json.loads((DATA/tag/'static_potential.json').read_text())
    x = np.array([p['r_over_a'] for p in d['potential']])
    y = np.array([p['aV'] for p in d['potential']])
    e = np.array([p['error'] for p in d['potential']])
    label = rf"$L={d['ensemble']['lattice_size'][0]},\ \beta={d['ensemble']['beta']}$"
    axes[0].errorbar(x, y, e, fmt='o', ms=4, color=color, label=label)
    fit = d['cornell_fit']; r = np.linspace(2, 6, 150)
    axes[0].plot(r, fit['offset']+fit['sigma_a2']*r-fit['alpha']/r, color=color)
    for i, (key, err) in enumerate([('a_from_r0_fm', 'a_from_r0_fm_error'),
                                     ('a_from_string_tension_fm', 'a_from_string_tension_fm_error')]):
        offset = -.07 if tag.startswith('L24') else .07
        axes[1].errorbar(i+offset, d['scale'][key], d['scale'][err], fmt='o', color=color)
axes[0].set(xlabel=r'$r/a$', ylabel=r'$aV(r)$', title='Static potential and Cornell fit')
axes[0].legend()
axes[1].set(xticks=[0, 1], xticklabels=[r'$r_0=0.5$ fm', r'$\sqrt{\sigma}=440$ MeV'],
            ylabel='Lattice spacing [fm]', title='Two scale-setting conventions')
save(fig, 'reference_scales')

fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout='constrained')
for row, tag in enumerate(['L24_beta6p20', args.coarse_ensemble]):
    d = json.loads((DATA/tag/'ensemble.json').read_text())
    p = np.array([x['plaquette'] for x in d['configurations']])
    axes[row, 0].plot(np.arange(len(p)), (p-p.mean())/p.std(ddof=1), lw=.7, color=COLORS[row+1])
    axes[row, 0].axvline(49.5, color='black', ls='--', lw=.8, label='New hot-start seed stream')
    axes[row, 0].set(title=tag.replace('_', ' '), xlabel='Saved configuration index', ylabel=r'$(P-\langle P\rangle)/s_P$')
    w = np.load(DATA/tag/'static_potential.npz')['wilson_loops'][:, 1, 1]
    for values, label, marker in [(p, 'Plaquette', 'o'), (w, '20-step APE W(2,2)', 's')]:
        z = values-values.mean()
        acf = np.array([z[:-k]@z[k:]/(z@z) for k in range(1, 13)])
        axes[row, 1].plot(np.arange(1, 13), acf, marker=marker, ms=3, label=label)
    axes[row, 1].axhspan(-.1777135, .1777135, color='grey', alpha=.12)
    axes[row, 1].axhspan(-1.96/np.sqrt(300), 1.96/np.sqrt(300), color='grey', alpha=.18)
    axes[row, 1].set(xlabel='Lag (100 compound steps)', ylabel='Sample ACF', ylim=(-.25, .25))
    axes[row, 1].legend(fontsize=10)
save(fig, 'reference_autocorrelation')

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.set(xlim=(0, 10), ylim=(0, 4.5)); ax.axis('off')
boxes = [(0.1, 3, 'Fine heatbath U\n24⁴, β = 6.20'), (3.4, 3, 'Projected polynomial S(U)\nFine smeared field V'),
         (6.8, 3, 'Two-link blocking B(V)\nCoarse C on 12⁴'),
         (6.8, .7, 'Haar lift L(C,z)\n60 free SU(3) links/cell'),
         (3.4, .7, 'Constrained Flow 1\nGenerated smeared field'),
         (.1, .7, 'Unconstrained Flow 2\nGenerated fine field')]
for x, y, text in boxes:
    ax.add_patch(FancyBboxPatch((x, y), 2.8, 1, boxstyle='round,pad=.08', fc='#edf3f7', ec=COLORS[1]))
    ax.text(x+1.4, y+.5, text, ha='center', va='center', fontsize=10)
for start, end in [((3, 3.5), (3.3, 3.5)), ((6.3, 3.5), (6.7, 3.5)), ((8.2, 2.9), (8.2, 1.8)),
                   ((6.7, 1.2), (6.3, 1.2)), ((3.3, 1.2), (3, 1.2))]:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=15, color=COLORS[0]))
ax.text(5, 2.3, 'Training targets: paired U and S(U); generation receives only C and fresh z.', ha='center', fontsize=9)
ax.text(5, .1, 'Flow 1: B(Vgenerated) = C. Flow 2: learned transport, not an exact inverse of S.', ha='center', fontsize=9)
save(fig, 'workflow')

down = DATA/down_name
down_kernel = json.loads((down/'kernel.json').read_text())
down_metrics = json.loads((down/'metrics.json').read_text())
down_names = down_metrics['observables']
training_names = down_kernel.get('training_observables', down_names)
loop_labels = {'plaquette': r'$W_{1\times1}$', 'polyakov_abs2': r'$|P|^2$', 'polyakov_abs4': r'$|P|^4$'}
for name in down_names:
    if name not in loop_labels:
        size = name.rsplit('_', 1)[1].replace('x', r'\times')
        loop_labels[name] = '$W_{'+size+'}$'
history = json.loads((down/'fit_history.json').read_text())
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout='constrained')
axes[0].semilogy([r['evaluation'] for r in history], [r['loss'] for r in history], color=COLORS[1])
axes[0].set(xlabel='Objective evaluation', ylabel='Sum of squared scaled residuals', title='60-configuration local-polynomial fit')
for i, name in enumerate(training_names):
    axes[1].plot([r['evaluation'] for r in history], [r['mean_residual'][i] for r in history], label=loop_labels[name])
axes[1].set(xlabel='Objective evaluation', ylabel='Scaled training mean residual'); axes[1].legend()
axes[1].set_yscale('symlog', linthresh=1.)
save(fig, 'downsampling_fit')

d = np.load(down/'observable_distributions.npz')
columns = (len(down_names) + 1) // 2
fig, axes = plt.subplots(2, columns, figsize=(10, 6), layout='constrained')
for i, ax in enumerate(axes.flat):
    if i >= len(down_names):
        ax.axis('off')
        continue
    ref = d['reference_test'][:, i]; mean, sigma = ref.mean(), ref.std(ddof=1)
    arrays = [(d[k][:, i]-mean)/sigma for k in ['reference_test', 'trained_test']]
    bins = np.histogram_bin_edges(np.concatenate(arrays), bins=22)
    for a, label, color in zip(arrays, ['Native L12', 'Fitted polynomial'], [COLORS[0], COLORS[2]]):
        ax.hist(a, bins=bins, density=True, histtype='step', lw=1.6, color=color, label=label)
    baseline_shift = (d['baseline_test'][:, i].mean()-mean)/sigma
    ax.text(.03, .94, f'Naive mean: {baseline_shift:+.1f} σ', transform=ax.transAxes,
            va='top', fontsize=9, color=COLORS[1],
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .85, 'pad': 2})
    role = 'training' if down_names[i] in training_names else 'HELD OUT'
    ax.set(title=loop_labels[down_names[i]]+' ['+role+']', xlabel=r'$(O-\langle O\rangle_{\rm ref})/\sigma_{\rm ref}$', ylabel='Density')
handles, legend_labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc='outside upper center', ncols=2, frameon=False)
save(fig, 'downsampling_distributions')

if args.downsampling_only:
    raise SystemExit(0)

up = DATA/up_name
if not (up/'metrics.json').exists():
    print('Reference and downsampling figures complete; upsampling evaluation is pending.')
    raise SystemExit(0)
h = json.loads((up/'history.json').read_text())
# Each panel shows the complete moment objective, rather than its components.
# Likelihood and paired-link warm-up have other units and remain in the archive.
fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout='constrained')
calibration = json.loads((up/'calibration_stage2.json').read_text())
for ax, phase, title in zip(axes, ['smeared_moments', 'generated_moments'],
                            ['Flow 1: total distribution loss', 'Flow 2: total distribution loss']):
    rows = [r for r in h if r['phase'] == phase and r.get('optimizer', 'Adam') == 'Adam']
    for key, style, color, label in [('train_loss', '.-', COLORS[1], 'Training'),
                                    ('validation_loss', '.--', COLORS[2], 'Validation')]:
        ax.plot(np.arange(1, len(rows)+1), [r[key] for r in rows], style,
                color=color, label=label)
    if phase == 'generated_moments':
        ax.text(.27, .97, 'Adam', transform=ax.transAxes, ha='center', va='top', fontsize=10)
        trials = [r for r in h if r['phase'] == phase
                  and r.get('optimizer') in ('global_least_squares', 'scalar_grid')]
        if 'baseline_checkpoint' in calibration:
            trials = [r for r in trials if r.get('baseline_checkpoint') == calibration['baseline_checkpoint']]
        x = len(rows)+np.arange(1, len(trials)+1)
        # Different weighting and warm start: do not join these segments.
        for key, style, color in [('train_loss', '.-', COLORS[1]),
                                 ('validation_loss', '.--', COLORS[2])]:
            ax.plot(x, [r[key] for r in trials], style, color=color)
        ax.axvline(len(rows)+.5, color=COLORS[0], ls=':', lw=1)
        ax.text(.8, .97, 'Calibration', transform=ax.transAxes, ha='center', va='top', fontsize=10)
        if 'selected_epoch' in calibration:
            selected = next(i for i, row in enumerate(trials, 1)
                            if row['epoch'] == calibration['selected_epoch'])
            ax.scatter(len(rows)+selected, calibration['best_validation_loss'],
                       marker='*', s=110, color='black', zorder=5, label='Selected checkpoint')
    ax.set(title=title, xlabel='Recorded fitting step', ylabel='Total mean + width loss', yscale='log')
    ax.legend(loc='lower left', fontsize=9)
save(fig, 'upsampling_training')

d = np.load(up/'observable_distributions.npz'); m = json.loads((up/'metrics.json').read_text())
fig, axes = plt.subplots(3, 4, figsize=(12, 8), layout='constrained')
for row, (candidate, reference, label) in enumerate([
    ('flow1_smeared', 'smeared_reference', 'Flow 1'), ('flow2_fine', 'fine_reference', 'Two-flow fine'),
    ('teacher_fine', 'fine_reference', 'Flow 2, teacher input')]):
    for i in range(4):
        ax = axes[row, i]; ref = d[reference][:, i]; sigma = ref.std(ddof=1)
        a, b = (ref-ref.mean())/sigma, (d[candidate][:, i]-ref.mean())/sigma
        bins = np.histogram_bin_edges(np.r_[a, b], bins=18)
        ax.hist(a, bins=bins, density=True, histtype='stepfilled', alpha=.25, color=COLORS[0], label='Target')
        ax.hist(b, bins=bins, density=True, histtype='step', lw=1.5, color=COLORS[1], label=label)
        ax.set(title=NAMES[i], xlabel=f"shift={m[candidate]['mean_shift_in_target_sigma'][i]:.2f} σ\nwidth={m[candidate]['std_ratio'][i]:.2f}")
        ax.legend(fontsize=11)
save(fig, 'upsampling_distributions')

columns = (len(down_names) + 1) // 2
fig, axes = plt.subplots(2, columns, figsize=(10, 6), layout='constrained')
for i, ax in enumerate(axes.flat):
    if i >= len(down_names):
        ax.axis('off')
        continue
    ref = d['fine_reference'][:, i]
    mean, sigma = ref.mean(), ref.std(ddof=1)
    target = (ref-mean)/sigma
    generated = (d['flow2_fine'][:, i]-mean)/sigma
    bins = np.histogram_bin_edges(np.r_[target, generated], bins=20)
    ax.hist(target, bins=bins, density=True, histtype='stepfilled', alpha=.2,
            color=COLORS[0], label='Fine reference')
    ax.hist(generated, bins=bins, density=True, histtype='step', lw=1.6,
            color=COLORS[1], label='Two-flow proposal')
    ax.set(title=NAMES[i], xlabel='(Operator − target mean) / target σ', ylabel='Density')
    ax.legend(fontsize=10)
save(fig, 'upsampling_fine_distributions')

# Compare each stage with its own target on the same test configuration indices.
fig = plt.figure(figsize=(12, 6.4), layout='constrained')
for subfig, (candidate, reference, heading, color) in zip(fig.subfigures(2, 1), [
    ('flow1_smeared', 'smeared_reference', r'Flow 1: $\widehat V$ versus $S_\theta(U)$', COLORS[2]),
    ('flow2_fine', 'fine_reference', r'Two flows: $\widehat U$ versus $U$', COLORS[1]),
]):
    subfig.suptitle(heading, fontsize=13)
    grid = subfig.add_gridspec(1, 4)
    for i in range(4):
        ref = d[reference][:, i]
        mean, sigma = ref.mean(), ref.std(ddof=1)
        target = (ref-mean)/sigma
        generated = (d[candidate][:, i]-mean)/sigma
        shift, width = generated.mean(), generated.std(ddof=1)
        separated = abs(shift) > 20
        if separated:
            split = grid[0, i].subgridspec(1, 2, wspace=.08)
            left = subfig.add_subplot(split[0, 0])
            right = subfig.add_subplot(split[0, 1], sharey=left)
            axes = [left, right]
            groups = sorted([target, generated], key=np.mean)
            span = 1.3*max(np.ptp(values) for values in groups)
            for ax, values in zip(axes, groups):
                center = (values.min()+values.max())/2
                ax.set_xlim(center-span/2, center+span/2)
                ax.xaxis.set_major_locator(plt.MaxNLocator(2))
            left.spines['right'].set_visible(False)
            right.spines['left'].set_visible(False)
            right.tick_params(axis='y', left=False, labelleft=False)
            # Diagonal marks explicitly identify the omitted horizontal gap.
            for ax, x in [(left, 1), (right, 0)]:
                for y in [0, 1]:
                    ax.plot([x-.04, x+.04], [y-.025, y+.025],
                            transform=ax.transAxes, color='black', lw=.8, clip_on=False)
            left.set_xlabel(r'$(O-\bar O_t)/s_t$ (broken axis)', x=1.08, fontsize=9)
        else:
            axes = [subfig.add_subplot(grid[0, i])]
            axes[0].set_xlabel(r'$(O-\bar O_t)/s_t$', fontsize=9)
        bins = (None if separated else
                np.histogram_bin_edges(np.r_[target, generated], bins=20))
        for ax in axes:
            limits = ax.get_xlim() if separated else None
            ax.hist(target, bins=20 if separated else bins, density=True,
                    histtype='stepfilled', alpha=.2, color=COLORS[0], label='Reference')
            ax.hist(generated, bins=20 if separated else bins, density=True,
                    histtype='step', lw=1.5, color=color, label='Generated')
            if separated:
                ax.set_xlim(limits)
            ax.tick_params(labelsize=8)
        axes[0].set_title(NAMES[i]+f'\n'+rf'$\Delta_\sigma={shift:.2f},\ R_s={width:.2f}$',
                          x=1.08 if separated else .5, fontsize=10)
        if i == 0:
            axes[0].set_ylabel('Density', fontsize=9)
        if i == 3:
            axes[0].legend(fontsize=8)
fig.savefig(FIGS/'upsampling_stage_distributions.png', dpi=150, bbox_inches='tight')
save(fig, 'upsampling_stage_distributions')

previous_path = DATA/'L12_to_L24_2flow_v1/observable_distributions.npz'
if previous_path.exists() and len(d['test_indices']) == 60:
    previous = np.load(previous_path)
    np.testing.assert_array_equal(previous['test_indices'], d['test_indices'])
    np.testing.assert_allclose(previous['fine_reference'], d['fine_reference'], rtol=1.e-5, atol=1.e-10)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout='constrained')
    for i, ax in enumerate(axes.flat):
        ref = d['fine_reference'][:, i]
        mean, sigma = ref.mean(), ref.std(ddof=1)
        values = [(ref-mean)/sigma, (previous['flow2_fine'][:, i]-mean)/sigma,
                  (d['flow2_fine'][:, i]-mean)/sigma]
        bins = np.histogram_bin_edges(np.concatenate(values), bins=20)
        ax.hist(values[0], bins=bins, density=True, histtype='stepfilled', alpha=.2,
                color=COLORS[0], label='Fine reference')
        ax.hist(values[1], bins=bins, density=True, histtype='step', lw=1.2,
                color=COLORS[2], label='Previous prototype')
        ax.hist(values[2], bins=bins, density=True, histtype='step', lw=1.8,
                color=COLORS[1], label='Equivariant (new)')
        ax.set(title=NAMES[i], xlabel='(Operator − target mean) / target σ', ylabel='Density')
        ax.legend(fontsize=9)
    fig.savefig(FIGS/'upsampling_fine_comparison.png', dpi=140, bbox_inches='tight')
    save(fig, 'upsampling_fine_comparison')

n = json.loads((up/'numerics.json').read_text())
run = json.loads((up/'run.json').read_text())
if n['checkpoint_sha256'] != run['checkpoint_sha256']:
    print('Observable figures rebuilt. Numerical report belongs to an older checkpoint; its figure is not regenerated.')
    raise SystemExit(0)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout='constrained')
for device, color in [('cpu', COLORS[1]), ('cuda', COLORS[2])]:
    rows = [r for r in n['records'] if r['device'] == device]
    for key, marker in [('gauge_error', 'o'), ('roundtrip_error', 's'), ('inverse_gauge_error', '^')]:
        axes[0].semilogy(np.arange(len(rows)), [max(r[key], 1.e-18) for r in rows], marker=marker,
                         ms=4, lw=.8, color=color, label=device+' '+key.replace('_error', ''))
axes[0].set(xlabel='Flow / random-gauge trial', ylabel='Maximum link error'); axes[0].legend(fontsize=9)
if 'full_lattice_generation' in n:
    errors = n['full_lattice_generation']['stage_gauge_errors']
    axes[0].text(.02, .03, 'L24 synchronized-noise chain:\n'+', '.join(f'{v:.2e}' for v in errors),
                 transform=axes[0].transAxes, fontsize=9)
axes[1].semilogy([1, 2], [max(r['error'], 1.e-18) for r in n['haar_jacobians']], 'o', color=COLORS[1])
axes[1].axhline(1.e-8, ls='--', color=COLORS[0], label='Acceptance tolerance')
axes[1].set(xticks=[1, 2], xlabel='Flow stage', ylabel='|8D numerical − analytic log J|'); axes[1].legend()
save(fig, 'numerical_integrity')
print('All progress-note figures rebuilt in', FIGS)
