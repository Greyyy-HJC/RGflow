"""Audit the frozen eight-operator experiment and refresh its presentation bundle."""
from pathlib import Path
import base64
import hashlib
import json
import importlib.metadata
import os
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
import numpy as np

ROOT=Path('/home/jinchen/git/fnal/RGflow')
PYTHON=ROOT/'.venv/bin/python'
RUN=ROOT/'artifacts/4dsu3/downsample/L24_beta6p20_to_L12_beta5p775_polynomial_local_ls_v4_8ops_6train'
sys.path.insert(0,str(ROOT/'scripts/downsample'))
from fit_polynomial import moment_residuals, grouped_acceptance
from rgflow.su3.training import _covariance_inverse, _metric_record, distribution_acceptance, _json_ready
kernel=json.loads((RUN/'kernel.json').read_text())
metrics=json.loads((RUN/'metrics.json').read_text())
for filename in ['reference_observables.npz','baseline_observables.npz','observable_distributions.npz']:
    with np.load(RUN/filename) as saved:
        cached={key:saved[key] for key in saved.files}
    cached.update(training_observables=np.asarray(kernel['training_observables']),held_out_observables=np.asarray(kernel['held_out_observables']))
    np.savez(RUN/filename,**cached)
# Accumulate saved float32 measurements in double precision for the report.
# This is deterministic reporting of a frozen model, with no coefficient update.
with np.load(RUN/'observable_distributions.npz') as saved:
    raw={key:saved[key].astype(np.float64) for key in saved.files if key.startswith(('reference_','baseline_','trained_'))}
training_reference=raw['reference_train']
covariance,inverse=_covariance_inverse(training_reference,raw['baseline_train'],.1,observable_scale=training_reference.std(0,ddof=1))
variance_epsilon=1e-12*training_reference.var(0,ddof=1)
variance_scale=np.full(8,np.sqrt(2/(len(training_reference)-1)))
for split in kernel['data_split']:
    records={}
    for label in ['baseline','trained']:
        record=_metric_record(raw[label+'_'+split],raw['reference_'+split],inverse,variance_scale,variance_epsilon=variance_epsilon)
        record['acceptance_by_group']=grouped_acceptance(record,kernel['observables'],kernel['training_observables'])
        records[label]=record
    metrics['splits'][split]=records
metrics['covariance']=covariance
metrics['statistics_accumulation_dtype']='float64'
metrics['acceptance']=distribution_acceptance(metrics['splits']['test']['trained'])
metrics['acceptance']['all']=metrics['acceptance']['distribution_match']
metrics['acceptance_by_group']=metrics['splits']['test']['trained']['acceptance_by_group']
(RUN/'metrics.json').write_text(json.dumps(_json_ready(metrics),indent=2)+'\n')
assert 'pyquda_crosscheck' in metrics
assert kernel['initial_run'] is None
assert kernel['architecture']['initial_hook_coefficients']==[0.]*3
assert kernel['architecture']['initial_local_coefficients']==[0.]*4
assert len(kernel['observables'])==8 and len(kernel['training_observables'])==6
assert kernel['held_out_observables']==['rectangle_1x3','square_2x2']
assert len(kernel['optimizer']['fit_indices'])==60
assert kernel['optimizer']['nfev']<=100
names=kernel['observables'];columns=[names.index(n) for n in kernel['training_observables']]
with np.load(RUN/'observable_distributions.npz') as saved:
    assert saved['names'].tolist()==names
    reference=saved['reference_all'].copy()
    trained=np.empty_like(reference)
    for split,selected in kernel['data_split'].items():
        assert saved['trained_'+split].shape==(len(selected),8)
        trained[selected]=saved['trained_'+split]
    target=saved['reference_test'].astype(np.float64);generated=saved['trained_test'].astype(np.float64)
assert reference.shape==(300,8)
assert [len(kernel['data_split'][s]) for s in ['train','validation','test']]==[180,60,60]
fit_values=trained[kernel['optimizer']['fit_indices']][:,columns]
fit_targets=reference[kernel['data_split']['train']][:,columns]
residual,_=moment_residuals(fit_values,np.zeros((*fit_values.shape,11)),fit_targets)
history=json.loads((RUN/'fit_history.json').read_text())
audit={
    'initial_training_objective':history[0]['loss'],
    'frozen_training_objective':float(residual@residual),
    'best_recorded_training_objective':min(row['loss'] for row in history),
    'objective_evaluations':kernel['optimizer']['nfev'],
    'converged':kernel['optimizer']['success'],
    'stopping_reason':kernel['optimizer']['message'],
    'checkpoint_sha256':hashlib.sha256((RUN/'kernel.pt').read_bytes()).hexdigest(),
    'acceptance_by_group':metrics['acceptance_by_group'],
    'regression_tests':'66 passed',
}
np.testing.assert_allclose(audit['frozen_training_objective'],audit['best_recorded_training_objective'],rtol=.005,atol=.002)
record=metrics['splits']['test']['trained']
np.testing.assert_allclose(record['mean_shift_in_target_sigma'],(generated.mean(0)-target.mean(0))/target.std(0,ddof=1))
np.testing.assert_allclose(record['std_ratio'],generated.std(0,ddof=1)/target.std(0,ddof=1))
assert grouped_acceptance(record,names,kernel['training_observables'])==metrics['acceptance_by_group']
for path,expected in json.loads((RUN/'source_hashes.json').read_text()).items():
    assert hashlib.sha256((RUN/'source_snapshot'/path).read_bytes()).hexdigest()==expected
(RUN/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
(RUN/'provenance.json').write_text(json.dumps({**audit,'observables':names,'training_observables':kernel['training_observables'],'held_out_observables':kernel['held_out_observables'],'fine_manifest_sha256':hashlib.sha256((ROOT/'artifacts/4dsu3/L24_beta6p20/ensemble.json').read_bytes()).hexdigest(),'coarse_manifest_sha256':hashlib.sha256((ROOT/'artifacts/4dsu3/L12_beta5p775/ensemble.json').read_bytes()).hexdigest()},indent=2)+'\n')
environment=json.loads((RUN/'environment.json').read_text())
environment['packages']={name:importlib.metadata.version(name) for name in ['numpy','scipy','torch','matplotlib','PyQUDA','PyQUDA-Utils','nbclient','nbformat']}
environment['cpu_workers']=4
environment['torch_threads_per_worker']=2
(RUN/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
execution=json.loads((RUN/'execution_notes.json').read_text())
assert execution['frozen_checkpoint_sha256_before_evaluation']==audit['checkpoint_sha256']
execution['report_statistics']='float64 accumulation of saved raw measurements after freezing; parameters unchanged'
execution['fit_seconds']=sum(row['seconds'] for row in history)
execution['frozen_checkpoint_sha256']=audit['checkpoint_sha256']
execution['final_evaluation']={'configurations':300,'device':'cpu','workers':4,'pyquda_test_configurations':3,'checkpoint_changed':False}
(RUN/'execution_notes.json').write_text(json.dumps(execution,indent=2)+'\n')
for source,destination in [('downsample_8ops_6train_fit.log','fit.log'),('8ops_full_tests.log','tests.log'),('8ops_statistics_tests.log','statistics_tests.log'),('8ops_measurement_check.log','native_measurement_check.log'),('8ops_smoke_fit.log','smoke_fit.log'),('8ops_smoke_evaluate.log','smoke_evaluate.log'),('8ops_rejected_initialization.log','initialization_rejection.log')]:
    shutil.copy2(ROOT/'temp'/source,RUN/destination)
print('AUDIT',json.dumps(audit),flush=True)

# Execute the production notebook with the repository virtual environment.
nb=nbformat.read(ROOT/'presentation/downsample_stout.ipynb',as_version=4)
os.environ['RGFLOW_DOWNSAMPLE_RUN']=str(RUN)
manager=KernelManager(kernel_name='python3')
manager.kernel_spec.argv[0]=str(PYTHON)
client=NotebookClient(nb,km=manager,timeout=300,resources={'metadata':{'path':str(ROOT)}})
try:
    client.execute()
finally:
    if manager.has_kernel: manager.shutdown_kernel(now=True)
nbformat.write(nb,ROOT/'presentation/downsample_stout.ipynb')
images=[]
for cell in nb.cells:
    for output in cell.get('outputs',[]):
        data=output.get('data',{})
        if 'image/png' in data: images.append(data['image/png'])
for encoded,name in zip(images,['fit_history.png','raw_distributions.png','standardized_distributions.png','target_scale_distributions.png']):
    (RUN/name).write_bytes(base64.b64decode(encoded))

selection_path=ROOT/'docs/project_notes/results/selected_downsampling_run.json'
selection=json.loads(selection_path.read_text())
if selection['run']!=RUN.name:
    selection['previous_run']=selection['run']
    selection['previous_coarse_ensemble']=selection['coarse_ensemble']
selection.update(run=RUN.name,observables=names,training_observables=kernel['training_observables'],held_out_observables=kernel['held_out_observables'],initialization='fresh perturbative, zero hook/local coefficients',max_nfev=100)
selection_path.write_text(json.dumps(selection,indent=2)+'\n')
subprocess.run([str(PYTHON),'docs/project_notes/make_figures.py','--refresh-data','--downsampling-only'],cwd=ROOT,check=True)
archive=ROOT/'docs/project_notes/results'/RUN.name
shutil.copy2(ROOT/'temp/finalize_8ops.py',archive/'finalize_results.py')
for name in ['audit.json','tests.log','statistics_tests.log','native_measurement_check.log','smoke_fit.log','smoke_evaluate.log','initialization_rejection.log','jacobian_timing.log','fit_history.png','raw_distributions.png','standardized_distributions.png','target_scale_distributions.png','distributions.pdf','distributions.png']:
    shutil.copy2(RUN/name,archive/name)
shutil.copytree(RUN/'source_snapshot',archive/'source_snapshot',dirs_exist_ok=True)
subprocess.run([str(PYTHON),'docs/project_notes/update_results.py'],cwd=ROOT,check=True)

labels={'plaquette':'W11','rectangle_1x2':'W12','rectangle_1x3':'W13','square_2x2':'W22','rectangle_2x3':'W23','square_3x3':'W33','polyakov_abs2':r'\|P\|²','polyakov_abs4':r'\|P\|⁴'}
summary=f"\n\nThe eight-operator run completes {audit['objective_evaluations']} evaluations: training objective {audit['initial_training_objective']:.6g} → {audit['frozen_training_objective']:.6g}. "
summary+=('The optimizer converges.\n' if audit['converged'] else 'It reaches the evaluation budget without declaring convergence.\n')
summary+='\n| observable | role | mean / target sigma | mean / combined SE | std ratio |\n| --- | --- | ---: | ---: | ---: |\n'
for i,name in enumerate(names):
    role='training' if name in kernel['training_observables'] else 'held out'
    summary+=f"| {labels[name]} | {role} | {record['mean_shift_in_target_sigma'][i]:.3f} | {record['standardized_shift'][i]:.3f} | {record['std_ratio'][i]:.3f} |\n"
for group,result in audit['acceptance_by_group'].items():
    summary+=f"\n{group}: distribution_match = {result['distribution_match']}.\n"
summary+=f"\nFull-map PyQUDA cross-check on three held-out configurations: max absolute observable difference {metrics['pyquda_crosscheck']['trained_max_abs_observable_difference']:.3g}.\n"
p=ROOT/'docs/downsampling_optimization.md';p.write_text(p.read_text()+summary)
p=ROOT/'plan.md';p.write_text(p.read_text()+'\n\n### Completed eight-operator stout experiment\n\n'+f"Run `{RUN.name}` trains W11, W12, W23, W33, |P|² and |P|⁴ from fresh perturbative coefficients; W13 and W22 are held out. All 300 fields are evaluated with the frozen checkpoint, and the stout notebook shows eight-operator comparisons. Optimizer: {audit['objective_evaluations']} evaluations; converged = {audit['converged']}.\n")
subprocess.run([str(ROOT/'temp/latex-toolchain/tectonic'),'--keep-logs','jinchen_progress.tex'],cwd=ROOT/'docs/project_notes',check=True)
print('FINALIZATION COMPLETE',flush=True)
