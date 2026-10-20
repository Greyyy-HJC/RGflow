#!/usr/bin/env python3
"""Verify archived observables and insert compact result tables into the note."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
parser = argparse.ArgumentParser(description=__doc__)
selection = HERE/'results/selected_upsampling_run.json'
selected_run = (json.loads(selection.read_text())['run'] if selection.exists()
                else 'L12_to_L24_2flow_equivariant_v2')
parser.add_argument('--upsampling-run', type=Path,
                    default=ROOT/'artifacts/4dsu3/upsample'/selected_run)
down_selection = json.loads((HERE/'results/selected_downsampling_run.json').read_text())
parser.add_argument('--downsampling-run', type=Path,
                    default=ROOT/'artifacts/4dsu3/downsample'/down_selection['run'])
args = parser.parse_args()
UP = HERE/'results'/args.upsampling_run.name
DOWN = HERE/'results'/args.downsampling_run.name
run = json.loads((UP/'run.json').read_text())
metrics = json.loads((UP/'metrics.json').read_text())
numerics = json.loads((UP/'numerics.json').read_text())
execution = json.loads((UP/'execution_notes.json').read_text()) if (UP/'execution_notes.json').exists() else {}
quick = execution.get('mode') == 'quick_scalar'
fine_calibration = execution.get('mode') == 'six_offset_fine'
data = np.load(UP/'observable_distributions.npz')
for name, target in [('flow1_smeared', 'smeared_reference'), ('flow2_fine', 'fine_reference'),
                      ('teacher_fine', 'fine_reference')]:
    x, y = data[name].astype(np.float64), data[target].astype(np.float64)
    shift = (x.mean(0)-y.mean(0))/y.std(0, ddof=1)
    ratio = x.std(0, ddof=1)/y.std(0, ddof=1)
    np.testing.assert_allclose(metrics[name]['mean_shift_in_target_sigma'], shift, rtol=.002, atol=.002)
    np.testing.assert_allclose(metrics[name]['std_ratio'], ratio, rtol=.002, atol=.002)
    criterion = bool(np.all(np.abs(shift) <= 1) and np.all(np.abs(ratio-1) <= .2))
    assert criterion == metrics[name]['acceptance']['distribution_match']
assert hashlib.sha256((UP/'flows.pt').read_bytes()).hexdigest() == run['checkpoint_sha256']
assert run['gauge_equivariant']
assert data['test_indices'].tolist() == run['test_indices']
stable = run['coupling'] == 'covariant_frame_mobius_monotone_cubic_fp64_v2'
if stable:
    assert numerics['checkpoint_sha256'] == run['checkpoint_sha256']

names = ['Plaquette', '$W_{12}$', '$W_{22}$', 'Polyakov']


def number(value, digits=6):
    mantissa, exponent = f'{value:.{digits-1}e}'.split('e')
    power = int(exponent)
    if -3 <= power <= 3:
        return f'{value:.{digits}g}'
    return mantissa.rstrip('0').rstrip('.')+r'\times10^{'+str(power)+'}'


x = data['flow2_fine'].astype(np.float64)
y = data['fine_reference'].astype(np.float64)
shift = (x.mean(0)-y.mean(0))/y.std(0, ddof=1)
ratio = x.std(0, ddof=1)/y.std(0, ddof=1)
test_count = len(data['test_indices'])
lines = [r'\begin{table}[t]',
         rf'\caption{{Two-flow fine proposals on {test_count} test configurations.}}',
         r'\label{tab:up}\small\begin{tabular}{lrrrr}\toprule',
         r'Observable & Target mean & Generated mean & $\Delta_\sigma$ & $R_s$\\\midrule']
for j, name in enumerate(names):
    lines.append(name+' & $'+number(y.mean(0)[j])+'$ & $'+number(x.mean(0)[j])+
                 f'$ & ${shift[j]:.3f}$ & ${ratio[j]:.3f}$'+r'\\')
lines += [r'\bottomrule\end{tabular}\end{table}']
mean_fail = [name for name, value in zip(names, shift) if abs(value) > 1]
width_fail = [name for name, value in zip(names, ratio) if abs(value-1) > .2]
if not mean_fail and not width_fail:
    lines.append(r'The final fine output passes the stated mean and width criterion '
                 r'(Table~\ref{tab:up}, Fig.~\ref{fig:up}); this tests four observables, '
                 'rather than establishing equality of the full field distributions.')
else:
    lines.append(r'The final fine output fails the stated distribution criterion '
                 r'(Table~\ref{tab:up}, Fig.~\ref{fig:up}).')
    if mean_fail:
        lines.append('Mean shifts outside one target width: '+', '.join(mean_fail)+'.')
    if width_fail:
        lines.append(r'Widths outside the 20\% window: '+', '.join(width_fail)+'.')
for key, label in [('flow1_smeared', 'Intermediate smeared outputs'),
                   ('teacher_fine', 'Flow~2 with teacher inputs')]:
    verdict = 'pass' if metrics[key]['acceptance']['distribution_match'] else 'fail'
    lines.append(label+' '+verdict+' the same criterion.')
calibrations = [json.loads((UP/f'calibration_stage{s}.json').read_text()) for s in [1, 2]]
stop_names = {'`xtol` termination condition is satisfied.': 'step tolerance',
              '`ftol` termination condition is satisfied.': 'objective tolerance',
              '`gtol` termination condition is satisfied.': 'gradient tolerance',
              'Stopped because `callback` function raised `StopIteration` or returned `True`': 'time budget',
              'The maximum number of function evaluations is exceeded.': 'evaluation budget'}
stops = [stop_names.get(c.get('stopping_reason', c['message']),
                        c.get('stopping_reason', c['message']).replace('_', r'\_')) for c in calibrations]
if fine_calibration:
    lines.append(f"Flow~1 stays fixed. The fine-flow calibration used "
                 f"{calibrations[1]['evaluations']} objective evaluations, stopping on "
                 f"{stops[1]}. Validation selects the numerically stable checkpoint.")
elif quick:
    lines.append(f"The scalar scans tested {calibrations[0]['evaluations']} and "
                 f"{calibrations[1]['evaluations']} values. Validation selected "
                 f"$\\delta_1={calibrations[0]['best_offsets'][0]:g}$ and "
                 f"$\\delta_2={calibrations[1]['best_offsets'][0]:g}$. "
                 f"This {test_count}-configuration pilot does not replace the earlier 60-configuration evaluation.")
else:
    lines.append(f"Calibration used {calibrations[0]['evaluations']} and "
                 f"{calibrations[1]['evaluations']} objective evaluations, stopping on "
                 f"{stops[0]} and {stops[1]}, respectively. Checkpoint selection uses validation loss.")

raw = np.load(DOWN/'observable_distributions.npz')
down_metrics = json.loads((DOWN/'metrics.json').read_text())
down_kernel = json.loads((DOWN/'kernel.json').read_text())
down_names = down_metrics['observables']
training_names = down_kernel.get('training_observables', down_names)
x = raw['trained_test'].astype(np.float64)
y = raw['reference_test'].astype(np.float64)
shift = (x.mean(0)-y.mean(0))/y.std(0, ddof=1)
ratio = x.std(0, ddof=1)/y.std(0, ddof=1)
np.testing.assert_allclose(down_metrics['splits']['test']['trained']['mean_shift_in_target_sigma'], shift)
np.testing.assert_allclose(down_metrics['splits']['test']['trained']['std_ratio'], ratio)
z = (x.mean(0)-y.mean(0))/np.sqrt(x.var(0, ddof=1)/len(x)+y.var(0, ddof=1)/len(y))
down_summary = dict(observables=down_names, training_observables=training_names,
                    target_mean=y.mean(0).tolist(), mean=x.mean(0).tolist(),
                    target_std=y.std(0, ddof=1).tolist(), std=x.std(0, ddof=1).tolist(),
                    standardized_shift=z.tolist(),
                    ks=[float(ks_2samp(x[:, j], y[:, j]).statistic) for j in range(len(down_names))],
                    wasserstein_in_target_sigma=[float(wasserstein_distance(x[:, j], y[:, j])/
                                                        y[:, j].std(ddof=1)) for j in range(len(down_names))])
(DOWN/'test_statistics.json').write_text(json.dumps(down_summary, indent=2)+'\n')
down_lines = [r'\begin{table}[t]',
              r'\caption{'+str(len(down_names))+r' downsampling observables on 60 test fine and coarse configurations.}',
              r'\label{tab:down}\small\begin{tabular}{llrrr}\toprule',
              r'Observable & Role & $\Delta_\sigma$ & Combined SE & $R_s$\\\midrule']
for j, name in enumerate(down_names):
    label = {'plaquette': '$W_{11}$', 'polyakov_abs2': '$|P|^2$', 'polyakov_abs4': '$|P|^4$'}.get(name)
    if label is None:
        label = '$W_{'+name.rsplit('_', 1)[1].replace('x', '')+'}$'
    role = 'Train' if name in training_names else 'Held out'
    down_lines.append(f'{label} & {role} & ${shift[j]:.3f}$ & ${z[j]:.3f}$ & ${ratio[j]:.3f}$'+r'\\')
down_lines += [r'\bottomrule\end{tabular}\end{table}']
for group, columns in [(str(len(training_names))+' training', [i for i,n in enumerate(down_names) if n in training_names]),
                       (str(len(down_names)-len(training_names))+' held-out', [i for i,n in enumerate(down_names) if n not in training_names]),
                       ('all '+str(len(down_names)), list(range(len(down_names))))]:
    if not columns:
        continue
    means_pass = bool(np.all(np.abs(shift[columns]) <= 1))
    widths_pass = bool(np.all(np.abs(ratio[columns]-1) <= .2))
    down_lines.append('For '+group+' observables, the mean criterion '+
                      ('passes' if means_pass else 'fails')+' and the width criterion '+
                      ('passes' if widths_pass else 'fails')+r' (Eq.~\eqref{eq:criteria}).')

history_down = json.loads((DOWN/'fit_history.json').read_text())
optimizer_down = down_kernel['optimizer']
fit_lines = [f"The objective decreases from {history_down[0]['loss']:.6g} to "
             f"{min(row['loss'] for row in history_down):.6g}. "
             f"The solver used {optimizer_down['nfev']} of {optimizer_down['max_nfev']} "
             "objective evaluations and " +
             ("satisfied its convergence criterion." if optimizer_down['success'] else "stopped at its evaluation budget without declaring convergence."),
             r'Table~\ref{tab:coefficients} gives the frozen parameters.',
             r'\begin{table}[t]',
             r'\caption{Physical coefficients of the six-operator downsampling fit.}',
             r'\label{tab:coefficients}',
             r'\begin{tabular}{crrr}\toprule',
             r'Order & $a_n$ & $b_{n-1}$ & $c_n$\\\midrule']
architecture_down = down_kernel['architecture']
for i in range(4):
    values = [architecture_down['coefficients'][i],
              architecture_down['hook_coefficients'][i] if i < len(architecture_down.get('hook_coefficients', [])) else None,
              architecture_down['local_coefficients'][i] if i < len(architecture_down.get('local_coefficients', [])) else None]
    fit_lines.append(str(i+1)+' & '+' & '.join('$'+number(value, 8)+'$' if value is not None else '---' for value in values)+r'\\')
fit_lines += [r'\bottomrule\end{tabular}\end{table}']

# Observable results are usable even when the final numerical validation fails.
# An older successful validation report must not be attributed to this checkpoint.
revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
paths = ['src/rgflow/su3/upsampling.py', 'scripts/upsample/train.py',
         'scripts/upsample/calibrate.py', 'scripts/upsample/validate.py']
if quick:
    paths.append('scripts/upsample/quick.py')
if fine_calibration:
    paths += ['scripts/upsample/overlap.py', 'scripts/upsample/select_overlap.py']
hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
frozen_down = HERE/'results'/Path(run['arguments']['downsampling_run']).name
provenance_path = UP/'provenance.json'
previous_provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
base_revision = (previous_provenance['base_revision']
                 if previous_provenance.get('checkpoint_sha256') == run['checkpoint_sha256'] else revision)
provenance = dict(base_revision=base_revision, source_sha256=hashes,
                  checkpoint_sha256=run['checkpoint_sha256'],
                  numerical_report_matches_checkpoint=numerics['checkpoint_sha256'] == run['checkpoint_sha256'],
                  downsampling_checkpoint_sha256=hashlib.sha256((frozen_down/'kernel.pt').read_bytes()).hexdigest())
if (UP/'source_hashes.json').exists():
    provenance['executed_source_sha256'] = json.loads((UP/'source_hashes.json').read_text())
provenance_path.write_text(json.dumps(provenance, indent=2)+'\n')
source = HERE/'jinchen_progress.tex'
text = source.read_text()
evaluation_lines = [
    f'Evaluation uses one fresh draw for each of {test_count} test coarse configurations,',
    'obtained by downsampling fine reference configurations. No rethermalization,',
    'operator rescaling or test-target fitting is applied. Independent coarse',
    'heatbath inputs have not been evaluated.']
caption_lines = [r'\caption{Stage-by-stage distributions on the same '+str(test_count)+' test configurations.',
                 r'Top: Flow~1 output $\widehat V$ versus the reference fine configurations after',
                 r'the frozen smoother, $S_{\bm\theta}(U)$, both on $24^4$.',
                 r'Bottom: final output $\widehat U=F_2(\widehat V)$ versus original fine',
                 r'references $U$. Each panel is standardized by its own reference mean and',
                 r'width; annotations give $\Delta_\sigma$ and $R_s$ from Eq.~\eqref{eq:criteria}.',
                 'The top three loop panels use explicitly marked broken horizontal axes to',
                 'show the separated distributions; no samples are shifted to create overlap.}']
replacements = [('DOWNSAMPLING FIT', fit_lines), ('DOWNSAMPLING STATISTICS', down_lines), ('GENERATED RESULTS', lines),
                ('EVALUATION DESCRIPTION', evaluation_lines), ('UPSAMPLING CAPTION', caption_lines)]
if stable:
    full = numerics['full_lattice_generation']
    gauge = max(full['stage_gauge_errors'])
    inverse = max(full['stage_roundtrip_errors'])
    jacobian = max(row['error'] for row in numerics['haar_jacobians'])
    inverse_lines = [
        'The inverse reverses the coupling order, solves the cubic with eight Newton',
        r'steps, and applies the radial map with $-\eta$. The current version keeps',
        'fields and conditioners in double precision. Whole-flow checks gate selected',
        'checkpoints, since deep contractions can amplify roundoff even',
        'when each local inverse is accurate.',
        r'For the selected checkpoint, synchronized-noise $24^4$ generation gives',
        '$'+number(gauge, 3)+r'$ maximum gauge error and $'+number(inverse, 3)+
        r'$ maximum roundtrip error. Full eight-dimensional local Haar Jacobians',
        'agree with the analytic values to $'+number(jacobian, 3)+'$.',
        'The retained-product error is $'+number(metrics['max_blocking_error'], 3)+
        r'$; generated $\SU$ residuals are at most $'+number(max(metrics['max_su3_errors']), 3)+
        '$ with the cached single-precision constraints.']
    history = json.loads((UP/'history.json').read_text())
    counts = [sum(row['phase'] == phase and row.get('optimizer', 'Adam') == 'Adam'
                  for row in history) for phase in ['conditional_nll', 'smeared_moments',
                                                    'paired_transport', 'generated_moments']]
    training_lines = [
        'The smoother is frozen. The covariant-frame flows were initialized from',
        'scratch with 8/2 sweeps, seed 1234, 12 training configurations and eight validation',
        'configurations. The four objectives are Flow~1 conditional likelihood, generated',
        'smeared moments, Flow~2 paired transport, and generated fine moments.',
        r'The conditional density is $\log q_1(V\mid B(V))=\log J_{F_1^{-1}}(V)$;',
        r'paired transport minimizes the matrix difference between $F_2(S(U))$ and $U$.',
        f'The recorded Adam epoch counts are {"/".join(map(str, counts))}.',
        'The double-precision continuation starts from the uncalibrated covariant',
        'checkpoint. Its moment objective sums squared target-width-normalized mean',
        r'residuals and ten times squared log-variance residuals; only stable updates',
        'are accepted. Earlier moment training used covariance-weighted means.',
        'Moment gradients use streaming vector--Jacobian products; checkpoint',
        'inputs reside in CPU memory during double-precision GPU training.',
        '',
        'Six-offset calibration shifts scale, shape and long-staple weight in each',
        'half of a flow, using fixed Haar draws and a 30-evaluation budget per stage.',
        f'The offset bounds are $\\pm{calibrations[0]["bound"]:g}$ and '
        f'$\\pm{calibrations[1]["bound"]:g}$.',
        'Flow~1 is calibrated before continuing Flow~2 moment training, so the latter',
        'uses the selected Flow~1 output distribution. Validation selects checkpoints.']
    if quick:
        training_lines = [
            'The frozen smoother and a pre-existing covariant-frame checkpoint are reused.',
            'The checkpoint has 8/2 sweeps and was previously trained on 12 training and',
            'eight validation configurations, using conditional likelihood, smeared moments,',
            'paired transport and fine moments; its Adam epoch counts are',
            f'{"/".join(map(str, counts))}. The quick experiment performs no new gradient epochs.',
            '',
            r'Only one scalar $\delta_k$ per flow is changed: it is added to all three',
            'raw scale outputs in every sweep, leaving feature weights, cubic shapes',
            'and staple weights fixed. Up to four predefined values per flow are tested',
            'with fixed Haar details on four training and four validation inputs.',
            r'The score is $\sum_j\Delta_{\sigma,j}^2+10\sum_j[\log((v_j+10^{-12})/',
            r'(v_{t,j}+10^{-12}))]^2$, where $v=s^2$. Validation selects the best',
            'numerically stable value, then Flow~1 is frozen before scanning Flow~2.',
            'This short scan avoids backpropagation and finite-difference Jacobian fits.']
    if fine_calibration:
        training_lines = [
            'The smoother is frozen and the flows have 8/2 sweeps. Adam updates all',
            'conditioner weights, biases and long-staple weights by backpropagation.',
            'The original fit uses 12 training and 8 validation configurations, with',
            'conditional likelihood, smeared moments, paired transport and fine moments',
            f'trained for {"/".join(map(str, counts))} epochs. Flow~1 uses',
            r'$\log q_1(V\mid B(V))=\log J_{F_1^{-1}}(V)$ on its constrained Haar details;',
            r'paired transport compares $F_2(S(U))$ with $U$.',
            'The Adam moment loss sums covariance-weighted squared mean residuals and',
            'squared log-variance residuals with width weight 1.',
            '',
            'For the current calibration, Flow~1 is frozen at its stable uncalibrated',
            'checkpoint and Flow~2 starts from a previously calibrated checkpoint.',
            'This is not a direct continuation of the last plotted Adam checkpoint.',
            'Calibration trains the same Flow~2 in a restricted six-parameter subspace:',
            r'for each sweep $s=1,2$, $\delta_{\eta,s}$ shifts all three raw scale',
            r'output biases, $\delta_{h,s}$ shifts all three cubic-shape output biases,',
            r'and $\delta_{\lambda,s}$ changes the long-staple weight.',
            'Each offset is shared across sites, directions and colour pairs; other',
            'network parameters stay fixed. The original coupling bounds and spectral',
            'gates remain in place, preserving the equivariant construction.',
            '',
            'The inputs are cached actual Flow~1 proposals, not reference smeared fields.',
            f'Fixed Haar draws use seed $1234+60000+i$ for configuration $i$, on',
            f'{len(calibrations[1]["train_ids"])} training and',
            f'{len(calibrations[1]["validation_ids"])} validation configurations. Flow~2 changes the generated',
            'fine configurations themselves. Its calibration score is',
            r'$\sum_j\Delta_{\sigma,j}^2+10\sum_j[\log((v_j+10^{-12})/(v_{t,j}+10^{-12}))]^2$,',
            r'where $v=s^2$. This replaces the Adam mean metric by target-width',
            'normalization and increases the width weight to 10.',
            'Bounded least squares uses an eight-component residual and its',
            'finite-difference derivative with respect to six offsets (step 0.002),',
            f'with bounds $\\pm{calibrations[1]["bound"]:g}$ and at most 30 objective evaluations.',
            'These derivatives are not the Haar Jacobian; no lattice backpropagation',
            'or likelihood term is used. Finite-difference measurements are additional',
            'to the recorded objective count. Training residuals drive the search;',
            'selection minimizes validation score among whole-flow inverse passes with',
            f'all validation mean shifts within {calibrations[1].get("validation_mean_bound", 2.):g} target widths.',
            'Six evaluations without validation improvement stop the recorded fit.',
            r'The stricter test criterion in Eq.~\eqref{eq:criteria} is unchanged.',
            'Test configurations do not enter fitting or selection.',
            '',
            'Adam could optimize these same offsets. The two stages differ in parameter',
            'subspace, objective weighting and starting checkpoint, so their histories',
            'do not establish an advantage of least squares over Adam.',
            'The small fixed-noise fitting sample can overfit; independent test',
            'configurations with fresh noise assess only the measured observables.']
    replacements += [('INVERSE STATUS', inverse_lines), ('TRAINING DESCRIPTION', training_lines)]
for marker, content in replacements:
    text, count = re.subn(f'(?s)% BEGIN {marker}\\n.*?% END {marker}',
                         lambda _: f'% BEGIN {marker}\n'+'\n'.join(content)+f'\n% END {marker}', text)
    assert count == 1
source.write_text(text)
selection.write_text(json.dumps({'run': args.upsampling_run.name}, indent=2)+'\n')
print('Verified archived observables and updated', source)
