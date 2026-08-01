# RGflow

RGflow studies renormalization-group accelerated lattice-field sampling. The
first implemented component generates native two-dimensional scalar phi-four
ensembles for later blocking and upscaling experiments.

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

## Repository structure

```text
src/rgflow/cli.py              top-level command-line dispatch
src/rgflow/diagnostics.py      reusable Markov-chain diagnostics
src/rgflow/phi4/action.py      phi-four action and local action differences
src/rgflow/phi4/sampling.py    radial Metropolis and Wolff updates
src/rgflow/phi4/observables.py phi-four observables and ensemble summaries
src/rgflow/phi4/plotting.py    SVG production figures
src/rgflow/phi4/cli.py         phi-four generation command
artifacts/phi4/                generated configurations and figures
tests/                         correctness and end-to-end tests
docs/                          reference material
```
