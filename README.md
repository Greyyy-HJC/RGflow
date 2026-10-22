# RGflow: 4D SU(3)

This branch is the starting point for the 4D SU(3) lattice gauge-theory
study. The repository is intentionally focused on this theory rather than
providing a framework for several gauge groups.

## Environment

Create and use the virtual environment at the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/pytest
```

The root `.venv/` directory is ignored by Git. There is no committed
`uv.lock`; use the Python environment directly for this project.

## Layout

- `docs/` — papers, notes, and analysis documentation
- `presentation/` — notebooks and presentation material
- `src/` — 4D SU(3) implementation
- `temp/` — local scratch data and generated intermediates
- `tests/` — automated tests
- `plan.md` — the current 4D SU(3) research plan

The implementation will be added under `src/rgflow/su3/` as the study is
developed.

## Topological charge under gradient flow

`presentation/topo_charge.ipynb` plots unrounded clover topological charge
against Wilson gradient-flow time `t/a^2` on L24 at beta 6.20. Reference
plots retain the 30-source subset (IDs 0, 10, ..., 290) and all 300 fields.
The current two-flow comparison pairs all 60 held-out sources (IDs 4, 9,
..., 299): original reference, the exact v4-smeared reference, Flow 1 and
Flow 2 outputs. The v4 smoother was fitted against the lattice-spacing-matched
L12 reference at beta 5.775. Each field starts Wilson flow at time zero.
All branches use double-precision RK3, integration step 0.02, maximum time
5 and measurements every 0.1. The generated stages share colors and axes.

```bash
.venv/bin/python scripts/heatbath/topo_charge.py
.venv/bin/python scripts/heatbath/topo_charge.py --stride 1 --samples 300 --output presentation/results/topo_charge/full --reuse-from presentation/results/topo_charge
.venv/bin/python scripts/upsample/measure_topology.py
.venv/bin/python scripts/upsample/validate_topology.py
.venv/bin/python scripts/upsample/update_topo_notebook.py
.venv/bin/python temp/execute_topo_charge_notebook.py
```

Scalar measurements are saved after each field and matching incomplete runs
resume. Original gauge files are read only; the two generated stages and
paired smeared reference are exported as FP64 NERSC fields for their common
measurement. The stage bundle records source IDs, latent seeds, checkpoint
hashes, unrounded charge arrays, SU(3) preparation changes and half-step checks.

`presentation/gradient_flow_diagnostics.ipynb` contains two figures: plaquette
and clover energy versus gradient-flow time. Each figure has separate panels
for reference L24, Flow 1 and Flow 2, with shared axes. Plaquette uses the original
six paired test configurations; clover energy uses all 60. Each configuration has
its own curve, with consistent
source-ID colors across panels; no ensemble-mean curve is added. Numerical
checks, including the exact-identity control failure, remain in the result JSON files.

```bash
.venv/bin/python scripts/upsample/diagnose_gradient_flow.py
.venv/bin/python scripts/upsample/check_gradient_flow_observables.py
.venv/bin/python scripts/upsample/make_gradient_flow_notebook.py
```

`presentation/topo_charge.ipynb` also continues the last reference field (the 300th configuration,
file index 299) with 300 Wilson pure-gauge HMC trajectories at beta=6.20.
It adds a plaquette history and one total charge-evolution plot for the 30
chain states saved at trajectories 10, 20, ..., 300. Each trajectory includes
a Metropolis decision, has MD length 1, and uses 50 second-order Omelyan
steps. Rejections retain the previous field and count as trajectories.
Each proposal is projected to SU(3) with tolerance 2e-15 before the final
Hamiltonian evaluation and Metropolis decision, matching the native PyQUDA example.
The temporary fields use FP64 NERSC storage; a field/RNG checkpoint allows
resuming at each saved group of 10.

```bash
.venv/bin/python scripts/heatbath/hmc_continue.py
.venv/bin/python scripts/heatbath/topo_charge.py --input temp/topo_charge_hmc_L24 --stride 1 --samples 30 --output presentation/results/topo_charge/hmc/flow
```

The chain history and added figures are saved in
`presentation/results/topo_charge/hmc/`; the temporary gauge fields and
checkpoint are in `temp/topo_charge_hmc_L24/`.

## SU(3) downsampling

The current coarse reference is `L12_beta5p775` (300 configurations), replacing
`L12_beta5p80` in the downsampling defaults. The same static-potential analysis
gives `a_c = 0.144038(1749) fm` and `a_c/a_f = 1.9940(249)`, compared with
the previous ratio `1.9009(230)`. The target is twice the measured fine spacing,
`0.144470 fm`. The beta scan and selection are recorded in
`docs/project_notes/results/selected_downsampling_run.json`; the initial estimate
uses the relative [Necco–Sommer scale relation, Eq. (2.6)](https://arxiv.org/pdf/hep-lat/0108008),
anchored to the original measured coarse spacing. Direct measurements select
the new reference. Previous ensembles and completed runs remain available as
historical results.

The production stout experiment measures eight operators: W11, W12, W13,
W22, W23, W33, Polyakov second moment and Polyakov fourth moment. Only W11,
W12, W23, W33 and the two Polyakov moments enter training. W13 and W22 are
held out to test generalization across operators. The fourth moment is the
average of the four directional global Polyakov fourth powers, rather than
the square of the direction-averaged second moment.

Reproduce the fresh perturbative-initialized fit (zero hook/local initial
coefficients), using a new output directory for another reproduction:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/downsample/fit_polynomial.py --hook --local --workers 4 --device cuda --max-nfev 100 --pyquda-check
```

It fits six means and six log variances using 60 fixed fine training fields
and all 180 coarse training references, with the original 180/60/60 split.
Each variance floor is relative to that operator's training target variance;
this preserves the tiny Polyakov fourth-moment fluctuations. All eight
operators are evaluated after freezing the checkpoint. Raw measurements,
operator roles, source snapshots and actual stopping reason are saved in
`artifacts/4dsu3/downsample/L24_beta6p20_to_L12_beta5p775_polynomial_local_ls_v4_8ops_6train/`.
The stout notebook shows all eight and reports distribution checks separately
for the six fitted operators, the two unseen operators and the full set.
The shared measurement API retains its four-operator default for archived
upsampling experiments; pass `names` to select the extended set.

Install the optional Torch training dependency and run the L24 to L12
experiment with:

```bash
.venv/bin/python -m pip install -e '.[test,train,report]'
.venv/bin/python scripts/downsample/train.py --method stout
```

The legacy Adam `stout-polynomial` run first optimizes the perturbative
Brillouin-zone residual and then directly fits the four global coefficients in
the covariant polynomial smoother
`[1 + a1 L + a2 L^2 + a3 L^3 + a4 L^4] U`. The projected field is then
factor-two blocked. The earlier four-term path mixture remains available as
`--method stout`. Results are written under `artifacts/4dsu3/downsample/`,
including the checkpoint, chi-squared metrics, and diagnostic plot. Use
`--max-configs 5 --epochs 2 --configs-per-epoch 3 --validation-configs 1`
for a small smoke run.

The reversible field-transform method remains available with:

```bash
.venv/bin/python scripts/downsample/train.py --method field-transform --backend compile
```

Compare eager and compiled field-transform throughput with:

```bash
.venv/bin/python scripts/downsample/benchmark.py --backend both
```

For a completed checkpoint, `scripts/downsample/evaluate.py` performs the
eager final evaluation and PyQUDA cross-check. Use
`presentation/downsample_stout.ipynb` or `presentation/downsample_ft.ipynb`
to inspect the corresponding training history and distributions.

For a scaled least-squares fit of the polynomial plus longitudinal hooks:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/downsample/fit_polynomial.py --hook
```

This fits the means and log variances on a fixed training subset with an
analytic Jacobian. The cached basis stores only links used by the final
two-link products. The coefficients use the scales `12^n` during fitting,
then are saved in the same physical convention as `PolynomialStoutKernel`.
Use `--train-configs 0` to fit the complete training split, or
`--initial-run PATH` to refine an existing polynomial fit. Validation and
test configurations are used only for the final evaluation.

Add `--local` to fit four plaquette-conditioned polynomial coefficients.
Their local scalar is the mean adjacent plaquette trace minus one, so
these terms preserve the weak-field linear kernel. The archived four-operator refinement used the previous local-polynomial
checkpoint against the new coarse reference:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/downsample/fit_polynomial.py --hook --local --initial-run artifacts/4dsu3/downsample/L24_beta6p20_to_L12_beta5p80_polynomial_local_ls_v2 --pyquda-check --workers 4
```

`presentation/downsample_stout.ipynb` displays the selected production run's fit history,
complete held-out test distributions, and mean/width acceptance checks.
Set `RGFLOW_DOWNSAMPLE_RUN` to inspect another completed polynomial run.
The fitter measures the requested `--coarse-dir` with PyQUDA on every run;
naive blocking is also remeasured for all eight operators. `--reference-run`
is an optional provenance reference and supplies no measurement cache.
It saves both source manifests with the new checkpoint. The optional PyQUDA
check compares the complete projected map on three held-out configurations
with the retained-basis measurements, using CPU polar projection.
`--workers 4` evaluates independent training configurations concurrently on
the CPU. With multiple workers, basis construction and final eight-operator
evaluation also run across the CPU workers; `--device` selects the full-map
PyQUDA cross-check backend. Single-worker mode uses `--device` throughout.
The default single-worker mode performs all stages on `--device`.
For a completed polynomial checkpoint, evaluate all 300 configurations with
the same retained basis in parallel on the CPU, then cross-check three
held-out configurations with the full map and PyQUDA:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/downsample/evaluate.py --device cpu --workers 4 --pyquda-configs 3
```

The perturbative initialization uses a consistent left-endpoint Fourier
convention. For the covariant operator in Eq. (2.39) of the blocking note,
the linearization of `L_U^n U` is `d_mu^(n-1) L_mu,nu`, where
`d_mu = sum_(nu != mu) (2 cos(k_nu) - 2)`; it is not the matrix power
`L(k)^n`. The zero-background field after the first application explains
this distinction. The note's residuals for the unrestricted path basis
should not be assumed for this compact polynomial family.

Evaluation reports both mean shifts divided by the target distribution's
standard deviation and shifts divided by the combined standard errors.
Distribution acceptance requires every mean to lie within one target
standard deviation and every standard deviation ratio to lie in
`[0.8, 1.2]`. A separate field records the stricter one-standard-error
criterion. Improvement over naive blocking alone no longer counts as a
successful distribution match.

## Two-flow SU(3) upsampling

The current experiment uses the selected `v4_8ops_6train` downsampling
checkpoint, fitted against the lattice-spacing-matched L12 reference at
beta 5.775. Both flows start from fresh neural weights, seed 1234. Flow 1
trains conditional Haar likelihood for four epochs, then fits six shared
scale/shape/staple offsets. Flow 2 trains paired transport for three epochs,
then fits its six offsets on generated Flow-1 inputs. Each fit uses 12 training
and eight validation sources, at most 30 objective evaluations and 65 minutes,
with six evaluations of validation patience.

Moment fitting and checkpoint selection use W11, W12, W23, W33, Polyakov
|P|² and |P|⁴. W13 and W22 are held out. Means are normalized by target
standard deviations, and log variances use a per-operator relative floor.
Final evaluation measures all eight operators on all 60 test sources with
fresh Haar noise and three independent PyQUDA checks. Acceptance requires
mean shifts within one target standard deviation and width ratios in
[0.8, 1.2], reported separately for training, held-out and all operators.

The coarse input is `C = block(smear(U))` from the held-out fine sources.
The native L12 reference sets the downsampling target; this experiment does
not measure transfer from independently generated native coarse inputs.
No heatbath/HMC rethermalization is applied after either flow.

Warm-up training uses single precision; calibration and generation use double
precision after exact parameter promotion. Inactive-staple frames define
embedded SU(2) radial and monotone cubic couplings with analytic Haar
Jacobians. Flow 1 retains the four coarse two-link products. Flow 2 transports
all fine links. Inverse, covariance and density checks gate updates and
calibration; the final checkpoint also receives full L24 checks.

```bash
.venv/bin/python scripts/upsample/rerun.py
.venv/bin/python scripts/upsample/make_notebook.py
.venv/bin/python scripts/upsample/measure_topology.py
.venv/bin/python scripts/upsample/update_topo_notebook.py
.venv/bin/python temp/execute_topo_charge_notebook.py
```

Training artifacts are under
`artifacts/4dsu3/upsample/L12_to_L24_2flow_v6_8ops_6train/`.
`presentation/upsampling_2flow.ipynb` shows the fresh training history,
eight-operator raw and reference-standardized distributions, intermediate
and teacher-input diagnostics, grouped acceptance and numerical checks.
Its small result bundle is replaced when rebuilding the notebook.

Both generated stages are exported as FP64 NERSC fields with source IDs,
latent seeds and checkpoint hashes. Double-precision SU(3) reunitarization
prepares these fields for the same Wilson gradient flow used in
`presentation/topo_charge.ipynb`: t/a² = 0–5, RK3 step 0.02, measurements
every 0.1 and unrounded clover charge. That notebook shows both 60-curve
stage plots and paired reference/stage comparisons. The measurement script
replaces stale results when the smoother or flow checkpoint changes and
resumes matching incomplete measurements.
