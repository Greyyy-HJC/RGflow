# RGflow

RGflow studies renormalization-group accelerated lattice-field sampling. The
current implementation generates native two-dimensional scalar phi-four
ensembles and trains an approximate perfect-blocking kernel.

## 2D phi-four sampler

The target action is

```text
S = -2 kappa sum_(x,mu) phi_x phi_(x+mu)
    + sum_x [phi_x^2 + lambda (phi_x^2 - 1)^2].
```

The sampler uses periodic boundaries and alternates:

1. checkerboard Metropolis updates of the radial field magnitude; and
2. a Wolff cluster flip of the embedded Ising signs.

The proposal width adapts during warmup and is fixed during production. The
cluster step is important near the critical point because radial updates alone
do not efficiently decorrelate the field signs.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest
```

## Generate configurations

The defaults generate 1,000 stored configurations at each of `L=8,16,32`,
split over four chains, with `lambda=1` and `kappa=0.3401`:

```bash
.venv/bin/rgflow generate phi4
```

For a quick run:

```bash
.venv/bin/rgflow generate phi4 \
  --sizes 8 \
  --chains 2 \
  --samples-per-chain 10 \
  --warmup-cycles 100 \
  --sample-interval 2 \
  --output artifacts/smoke
```

Use `rgflow generate phi4 --help` for all sampling controls. Without an
explicit `--output`, generated files are written to `artifacts/phi4`.

## Output and diagnostics

Each lattice size produces:

- a compressed `.npz` file containing `configurations` with shape
  `(chains, samples_per_chain, L, L)`, saved cycle numbers, and theory
  parameters;
- a `.json` file containing action density, magnetization observables,
  susceptibility, Binder cumulant, acceptance and cluster statistics; and
- two `.svg` figures showing final configurations, production histories, and
  observable distributions; and
- integrated autocorrelation-time, effective-sample-size, and first-half vs.
  second-half drift estimates for key observables.

Stored configurations remain Markov-chain correlated. Check the reported
autocorrelation times, ESS, and drift before using an ensemble for physics or
machine-learning analysis. Production warmup and saving intervals should be
chosen from pilot runs rather than assumed from the defaults.

## Train the blocking kernel

Train a D4-symmetric five-parameter 5x5 kernel on the native L16 and L8
ensembles:

```bash
.venv/bin/rgflow train kernel phi4
```

The kernel applies periodic convolution followed by even-even downsampling.
Its sum is fixed to `2^(1/8)`, the critical field rescaling for the two-
dimensional Ising value `eta=1/4`; the five off-center orbit coefficients are
trained and the center coefficient enforces this normalization.

Chains 0-1 are used for training, chain 2 selects among deterministic
multi-start optimizations, and chain 3 is evaluated once as the test set.
Native L32 configurations are excluded from optimization and used only for an
L32-to-L16 transfer test.

The production run selected a kernel with:

- L16-to-L8 test RMS standardized shift `0.0530`, compared with `0.9150` for
  the scaled identity;
- L32-to-L16 transfer RMS standardized shift `0.1472`;
- dense-grid `min|K(p)| = 0.6335` and condition number `2.0905`.

The operational `kernel.json`, complete `metrics.json`, compact diagnostics
PDF, and the 27-panel `operator_distributions.pdf` are written under
`artifacts/phi4/kernel/L16_to_L8`. The operator figure compares native coarse
and blocked distributions on both the held-out test chain and the full
ensemble. Susceptibility, Binder cumulant, and second-moment correlation length
are shown as circular moving-block bootstrap distributions.

## Repository structure

```text
src/rgflow/cli.py              top-level command-line dispatch
src/rgflow/diagnostics.py      reusable Markov-chain diagnostics
src/rgflow/phi4/action.py      phi-four action and local action differences
src/rgflow/phi4/blocking.py    blocking kernels and Fourier inversion
src/rgflow/phi4/kernel_training.py kernel optimization and evaluation
src/rgflow/phi4/operator_diagnostics.py distribution-level kernel checks
src/rgflow/phi4/sampling.py    radial Metropolis and Wolff updates
src/rgflow/phi4/observables.py phi-four observables and ensemble summaries
src/rgflow/phi4/plotting.py    SVG production figures
src/rgflow/phi4/cli.py         phi-four generation command
artifacts/phi4/                generated configurations and figures
tests/                         correctness and end-to-end tests
docs/                          reference material
```
