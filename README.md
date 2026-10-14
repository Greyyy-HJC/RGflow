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

## SU(3) downsampling

Install the optional Torch training dependency and run the L24 to L12
experiment with:

```bash
.venv/bin/python -m pip install -e '.[test,train,report]'
.venv/bin/python scripts/downsample/train.py --method stout
```

The default `stout-polynomial` run first optimizes the perturbative
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
these terms preserve the weak-field linear kernel. For example, refine
the hook-polynomial checkpoint with:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/downsample/fit_polynomial.py --hook --local --initial-run artifacts/4dsu3/downsample/L24_beta6p20_to_L12_beta5p80_polynomial_ls_v1 --output artifacts/4dsu3/downsample/L24_beta6p20_to_L12_beta5p80_polynomial_local_ls_v2
```

`presentation/downsample_stout.ipynb` displays this run's fit history,
complete held-out test distributions, and mean/width acceptance checks.
Set `RGFLOW_DOWNSAMPLE_RUN` to inspect another completed polynomial run.

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

Train on the existing L24 heatbath ensemble and the frozen projected
polynomial smoothing checkpoint:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/upsample/train.py
.venv/bin/python scripts/upsample/make_notebook.py
```

Flow 1 draws 60 Haar detail links per coarse cell, fixes the four retained
two-link products to the input coarse links, and trains a conditional flow
with exact Haar likelihood, followed by generated-smeared operator fitting.
Flow 2 is an invertible transport trained first
on paired smeared/fine fields and then on the operator means and widths of
actual two-flow proposals. Its default moment batch has 12 independent
coarse sources, and four latent draws per training source plus fixed validation
draws are cached after Flow 1 is frozen. Couplings use embedded SU(2) stereographic radial
maps followed by bounded monotone cubic angle deformations, with analytic
Haar Jacobians. Nearest-neighbour and transverse-distance-three staples
contain only inactive links in each checkerboard substep. The revised
coupling acts in the covariant eigenframe of the inactive staple context
times its adjoint. Its conditioner is gauge invariant and eigenvector phases
cancel, making both flows gauge equivariant. Repeated context spectra use
the identity update; a relative-gap gate suppresses near-degenerate updates.
CUDA uses a differentiable analytic 3x3 spectral solver with CPU fallback
near repeated roots. Legacy checkpoints retain their original fixed-colour
coupling through the saved coupling version.

The `covariant_frame_mobius_monotone_cubic_fp64_v2` version retains double
precision throughout both flows. The completed `equivariant_v2` experiment
failed whole-flow inverse checks after calibration, despite accurate individual
coupling inverses. The separate `L12_to_L24_2flow_stable_v3` continuation starts
from its uncalibrated covariant checkpoint, uses stronger width fitting, and
rejects updates that fail whole-flow numerical checks, including a physical
L24 roundtrip. Its pipeline is `scripts/upsample/continue_stable_v3.sh`; current phase and
logs are saved in the new run directory. This long continuation was stopped
at the user's request before any new optimizer update. The old checkpoint
semantics are preserved.
Stable CUDA training offloads the exact checkpoint inputs to pinned CPU
memory: retaining all 64 double-precision L24 fields on the GPU otherwise
exhausts 12 GB. `scripts/upsample/check_training_memory.py` verifies a complete
L24 moment VJP before training; offloaded gradients are checked against
direct gradients in the tests.

For a short experiment, `scripts/upsample/quick.py` reuses that checkpoint,
scans one global scale offset per flow without backpropagation or a
finite-difference Jacobian fit, and uses 4/4/12 train/validation/test sources.
It preserves the covariant maps, double precision and final numerical checks.
The short run is `L12_to_L24_2flow_stable_quick_v4`; its small-sample results
are a pilot, rather than the earlier 60-source study. It completed in 14.4
minutes, passes the numerical checks, and fails the fine distribution criterion.
The first three generated fields agree with PyQUDA to machine precision.
To reproduce:

```bash
.venv/bin/python scripts/upsample/quick.py --output artifacts/4dsu3/upsample/quick_reproduction --minutes 16
```

`scripts/upsample/overlap.py` fixes the stable uncalibrated Flow 1 and refits
six offsets of the previously calibrated Flow 2: scale, cubic shape and
long-staple weight in each of its two sweeps. It uses 8 training and 8
validation sources, a 65-minute calibration budget, whole-flow stability
checks and a final 60-source test. Feature weights stay fixed; no new gradient
epochs or test-target fitting are performed. The independent run is
`L12_to_L24_2flow_overlap_v5`.
Checkpoint selection requires all validation mean shifts to stay within two
target widths; the original stricter test acceptance criterion remains unchanged.
Calibration also stops after six evaluations without validation improvement.
The completed 60-source run has mean shifts of approximately 1.068, 0.513,
0.425 and -0.130 target widths, with width ratios 1.727, 1.245, 0.914 and
0.724. It gives rough distribution overlap and passes the numerical checks;
the stricter complete mean/width criterion still fails. Calibration and
measurement took 54.2 minutes, with 15 calibration evaluations and selection
of evaluation 9.

```bash
.venv/bin/python scripts/upsample/overlap.py --output artifacts/4dsu3/upsample/overlap_reproduction
```

The default run uses held-out downsampled fine configurations as input,
not independent coarse heatbath configurations. The same downsampling
train/validation/test partition is retained. Final evaluation compares
the generated fine fields with the held-out L24 heatbath fields using
the four existing operators and the same distribution criteria. No
rethermalization or test-target fitting is performed.

Results and cached smoothed fields are stored in
`artifacts/4dsu3/upsample/L12_to_L24_2flow_equivariant_v2/`. The frozen smoother uses
CPU SVD for projection because large batches of tiny 3x3 CUDA SVDs are
slow on the RTX 3060; the default differentiable downsampling path is
unchanged. Use `--evaluate-only --pyquda-check` to cross-check generated
operators with PyQUDA, and `--resume --epochs 0 --transport-epochs 0`
`--smeared-moment-epochs 0` to extend only the generated-fine moment training.
For faster deterministic refinement of six shared shift/shape/staple offsets,
run `.venv/bin/python scripts/upsample/calibrate.py --stage 1`, then the same
command with `--stage 2`. This fits fixed training draws and selects the
checkpoint by validation loss; reevaluate afterwards to regenerate the
held-out metrics. No operator rescaling or test-set fitting is applied.
The refinement defaults to independent per-operator mean errors. Stage 2
also supports `--anneal-init` to expand details before recontracting them
through the last flow sweep; these are invertible flow couplings, not MCMC.
Run `.venv/bin/python scripts/upsample/validate.py` on the final checkpoint
to record CPU double-precision and CUDA single-precision gauge covariance,
inverse covariance, roundtrips and full eight-dimensional Haar Jacobians.
For the stable version, CUDA single-precision inputs are promoted and the
reported computation and gauge transformations use double precision.
Samplewise gauge tests transform the Haar details along with the coarse
field; using the same random seed after a gauge transformation does not
perform this detail transformation.

`presentation/upsampling_2flow.ipynb` displays the training history,
held-out final distributions, stage diagnostics and numerical checks.
Its small `presentation/results/upsampling_2flow/` bundle includes the
measurements, provenance and both trained flow state dictionaries, so
the executed report also works without the large ensembles. Set
`RGFLOW_UPSAMPLE_RUN` to display another completed result directory.

The old `L12_to_L24_2flow_v1` fixed-colour prototype passes the one-target-sigma test for all four
held-out operator means, but **does not pass the full distribution criterion**:
its width ratios are approximately 2.48, 2.38, 1.95 and 1.24. The first-stage
smeared distributions and teacher-input diagnostics also retain bias. These
samples should not yet be treated as a heatbath-equivalent fine ensemble.
