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
Haar Jacobians, including checkerboard-safe transverse
length-three staples. The current colour-subgroup implementation is not
exactly gauge equivariant.

The default run uses held-out downsampled fine configurations as input,
not independent coarse heatbath configurations. The same downsampling
train/validation/test partition is retained. Final evaluation compares
the generated fine fields with the held-out L24 heatbath fields using
the four existing operators and the same distribution criteria. No
rethermalization or test-target fitting is performed.

Results and cached smoothed fields are stored in
`artifacts/4dsu3/upsample/L12_to_L24_2flow_v1/`. The frozen smoother uses
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

`presentation/upsampling_2flow.ipynb` displays the training history,
held-out final distributions, stage diagnostics and numerical checks.
Its small `presentation/results/upsampling_2flow/` bundle includes the
measurements, provenance and both trained flow state dictionaries, so
the executed report also works without the large ensembles. Set
`RGFLOW_UPSAMPLE_RUN` to display another completed result directory.

The initial two-flow prototype passes the one-target-sigma test for all four
held-out operator means, but **does not pass the full distribution criterion**:
its width ratios are approximately 2.48, 2.38, 1.95 and 1.24. The first-stage
smeared distributions and teacher-input diagnostics also retain bias. These
samples should not yet be treated as a heatbath-equivalent fine ensemble.
