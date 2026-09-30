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
