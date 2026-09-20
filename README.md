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
split over four chains, with `lambda=1` and the reference critical coupling
`kappa=0.340301`:

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

The production ensembles used for the two volume-scaling chains are generated
with one explicit command so that every volume has identical theory and MCMC
settings:

```bash
.venv/bin/rgflow generate phi4 \
  --sizes 8 12 16 24 32 48 64 \
  --kappa 0.340301 \
  --chains 4 \
  --samples-per-chain 250 \
  --warmup-cycles 1000 \
  --sample-interval 10 \
  --radial-sweeps 1 \
  --seed 1234 \
  --output artifacts/2dphi4
```

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

Train the default two-channel sum-of-squares (SOS) kernel with an equal-weight
multi-volume one-step objective:

```bash
.venv/bin/rgflow train kernel phi4
```

The command writes its artifacts to
`artifacts/phi4/kernel/multivolume_sos2_k0.340301` by default. It trains on
L16-to-L8, L24-to-L12, and L32-to-L16, with equal weight per volume pair.
L48-to-L24 and L64-to-L32 are excluded from optimization and retained as
unseen-volume transfer tests. `--sos-channels` controls the number of factors
and `--sos-floor-fraction` controls the strict spectral floor.

### By-construction invertibility

Each trainable factor \(A_r\) has 3x3 support. The first factor is constrained
to have unit sum and all remaining factors have zero sum:

\[
\sum_x A_1(x)=1,
\qquad
\sum_x A_r(x)=0\quad(r>1).
\]

The factors first form a positive-semidefinite autocorrelation kernel and then
are averaged over the D4 symmetry group:

\[
Q
=
\frac{1}{|D_4|}
\sum_{g\in D_4}
\sum_{r=1}^{R}
(gA_r)^\dagger(gA_r).
\]

A 3x3 factor autocorrelation has 5x5 support, so this construction preserves
the desired locality. The group average makes \(Q\) exactly D4 symmetric. The
factor-sum constraints give \(\widehat Q(0)=\sum_x Q(x)=1\).

The operational kernel is

\[
K
=
2^{1/8}
\left[(1-\rho)Q+\rho I\right],
\qquad 0<\rho<1.
\]

It therefore has the required critical-field normalization,

\[
\sum_x K(x)=\widehat K(0)=2^{1/8}.
\]

More importantly, its Fourier spectrum satisfies

\[
\widehat Q(p)
=
\frac{1}{|D_4|}
\sum_{g\in D_4}
\sum_r
\left|\widehat A_r(g^{-1}p)\right|^2
\ge 0,
\]

and hence

\[
\widehat K(p)
=
2^{1/8}
\left[(1-\rho)\widehat Q(p)+\rho\right]
\ge 2^{1/8}\rho>0
\]

for every momentum, independently of lattice size. Thus \(K\) is invertible by
construction rather than merely accepted after a numerical spectrum check.
The inverse is applied in Fourier space by dividing each mode by
\(\widehat K(p)\). The downsampling operator \(D\) is still noninvertible;
inverse blocking must restore the three missing detail sublattices before
applying \(K^{-1}\).

The default uses two channels and \(\rho=0.05\), giving a guaranteed spectral
floor of \(2^{1/8}\rho=0.0545\). Spectrum diagnostics remain part of training
because a much stronger empirical minimum and a small condition number are
desirable for numerical stability even though mathematical invertibility is
already guaranteed.

### Training objective and diagnostics

Each volume-specific objective combines covariance-weighted mean matching with
standardized quantile matching for ten distribution-support observables:
`phi2`, `phi4`, local kurtosis, `NN`, diagonal and distance-two correlators,
`m2`, `m4`, minimum-momentum power, and action density. The quantile term
includes the 1%, 5%, 10%, 25%, 50%, 75%, 90%, 95%, and 99% levels; its
relative weight is controlled by `--support-weight`. The three pair losses are
averaged with equal weights, so lattice size does not implicitly reweight the
fit.

Two additional terms improve stability and infrared fidelity: a mild loss on
the Binder cumulant and `chi/L^(2-eta)`, controlled by `--physical-weight`,
and a smooth condition-number penalty above a target of 2, controlled by
`--conditioning-weight`. The SOS spectral floor still guarantees
invertibility independently of this numerical conditioning term.

Chains 0-1 are used for training, chain 2 selects among deterministic
multi-start optimizations, and chain 3 is evaluated once as the held-out test
set. Optimization uses an equal deterministic cap of 128 configurations per
training chain and volume pair by default; all validation and reported test
metrics use the complete chain. The cap is configurable with
`--optimization-samples-per-chain`.

The current two-channel production candidate has:

- mean held-out 24-feature RMS standardized shift `0.1078` over the three
  training volume pairs;
- worst unseen-volume transfer RMS standardized shift `0.2170`;
- dense-grid `min K(p) = 0.6540` and condition number `2.1058`;
- mean ten-observable RMS shift `0.1241` across all five one-step links,
  compared with `0.1410` for the reference kernel on the same held-out data.

The improvement is not uniform: This work wins three of five links, while
L24-to-L12 is slightly worse and L64-to-L32 remains the main scaling
limitation.

The operational `kernel.json` records the trained 3x3 factors, their sums, the
guaranteed spectral floor, the derived 5x5 kernel, and numerical spectra.
`metrics.json` contains all distribution metrics. The output directory also
contains the dedicated ten-panel `kernel_observable_histograms.pdf` and
`.png`, a compact `diagnostics.pdf`, and the broader 27-panel
`operator_distributions.pdf`.

## Multi-volume scaling presentation

[`presentation/kernel_volume_scaling.ipynb`](presentation/kernel_volume_scaling.ipynb)
is a pre-executed comparison of the current SOS and `temp/Inverse_RG`
reference kernels on identical held-out ensembles. It covers the five one-step
links L64-to-L32, L32-to-L16, L16-to-L8, L48-to-L24, and L24-to-L12. The
notebook distinguishes trained links from unseen-volume transfers and includes
ten-observable histograms, aggregate one-step size scaling, bootstrap
finite-size scaling, and Fourier-spectrum conditioning. Cumulative multi-step
application is intentionally omitted from the main presentation.

## Conditional flow, exact corrections, and efficiency benchmarks

Install the optional PyTorch and notebook dependencies:

~~~bash
uv sync --extra test --extra flow --extra presentation
~~~

The flow funnel keeps the original affine model, adds the prescribed stronger
affine model, and adds an affine-plus-residual rational-quadratic spline model:

~~~bash
.venv/bin/rgflow train flow phi4 --model affine --device cuda
.venv/bin/rgflow train flow phi4 --model strong-affine --device cuda
.venv/bin/rgflow train flow phi4 \
  --model residual-spline \
  --initialize-from artifacts/phi4/flow/L8_to_L16_strong_affine/checkpoint.pt \
  --reverse-kl-weight 0.05 \
  --device cuda
~~~

The three detail sectors 'd01', 'd10', and 'd11' are transformed-field cosets.
They are rich in short-distance information but are not physical links.
Checkpoint selection uses chain 2 only: conditional importance-weight ESS ratio
is primary, observable error is secondary, and conditional NLL breaks ties.
Chains 0-1 train and chain 3 remains a final held-out test. New checkpoints have
format version 2 and record 'model_type'; version-1 affine checkpoints remain
loadable.

Exact post-flow corrections are selected with '--correction':

~~~bash
.venv/bin/rgflow upscale phi4 \
  --checkpoint artifacts/phi4/flow/L8_to_L16_affine/checkpoint.pt \
  --coarse artifacts/2dphi4/phi4_L8_k0.340301_lam1.npz \
  --native-fine artifacts/2dphi4/phi4_L16_k0.340301_lam1.npz \
  --output artifacts/phi4/upscale/L8_to_L16_detail_wolff \
  --correction detail-wolff \
  --detail-passes 2 \
  --sweeps 100 \
  --replicates 3 \
  --device cuda
~~~

Available corrections are 'coordinate', 'coordinate-wolff', 'detail-wolff',
'flow-detail-wolff', and the screening-only 'hmc-wolff'. Coordinate/detail MH
repairs continuous amplitudes and UV structure. One embedded-Ising Wolff
cluster per chain and sweep repairs sign clusters and long-distance
magnetization modes. A Wolff update acts on the fine field and then recomputes
'psi=K phi' and all inverse coordinates. Conditional flow refresh fixes the
current coarse field and uses the exact independence ratio

'-S_f(new) + S_f(old) + log q(old|c) - log q(new|c)'.

Every run writes per-chain, per-sweep observable arrays to
'observable_history.npz'. Autocorrelation and ESS are computed along the time
axis of each chain and aggregated across chains. A set containing only the last
configuration of each trajectory is reported as one-shot throughput and is
never assigned a time-series ESS.

Run the common native/inverse benchmark with one or both factor-two cases:

~~~bash
.venv/bin/rgflow benchmark phi4 \
  --case \
    artifacts/phi4/flow/L8_to_L16_affine/checkpoint.pt \
    artifacts/2dphi4/phi4_L8_k0.340301_lam1.npz \
    artifacts/2dphi4/phi4_L16_k0.340301_lam1.npz \
  --case \
    artifacts/phi4/flow/L16_to_L32_affine/checkpoint.pt \
    artifacts/2dphi4/phi4_L16_k0.340301_lam1.npz \
    artifacts/2dphi4/phi4_L32_k0.340301_lam1.npz \
  --chains 128 \
  --sweeps 100 \
  --device cuda
~~~

The benchmark creates a timestamped, non-overwriting artifact directory. It
reports time-to-tolerance, worst-observable ESS/s, paired-bootstrap efficiency
ratios, one-shot throughput, setup/online costs, and the end-to-end break-even
sample count. ESS/s times only sampler transitions, including GPU
synchronization and transfers; full wall time with diagnostics and checkpoint
I/O is stored separately. The native baseline is radial checkerboard Metropolis
plus one
Wolff cluster per chain and sweep. HMC advances beyond its single-seed screen
only if it beats the best non-HMC correction in both thermalization time and
worst-observable ESS/s.

The original 128-chain, 400-sweep coordinate-MH artifacts remain as a historical
baseline: their final ten-observable RMS shifts are '0.2158' for L8-to-L16 and
'0.2043' for L16-to-L32. Both narrowly miss the fixed RMS threshold of '0.20';
they are not relabeled as converged.

[presentation/flow_inverse_blocking.ipynb](presentation/flow_inverse_blocking.ipynb)
retains the original pre-executed analysis and now includes a benchmark-results
section that reads the newest versioned summary when one is available.

## Repository structure

```text
src/rgflow/cli.py              top-level command-line dispatch
src/rgflow/diagnostics.py      reusable Markov-chain diagnostics
src/rgflow/phi4/action.py      phi-four action and local action differences
src/rgflow/phi4/blocking.py    blocking kernels and Fourier inversion
src/rgflow/phi4/affine_flow.py conditional affine and residual-spline flows
src/rgflow/phi4/flow_training.py flow training and held-out evaluation
src/rgflow/phi4/inverse_mh.py  composable exact inverse-RG corrections
src/rgflow/phi4/benchmark.py   native/inverse efficiency benchmark
src/rgflow/phi4/kernel_training.py kernel optimization and evaluation
src/rgflow/phi4/operator_diagnostics.py distribution-level kernel checks
src/rgflow/phi4/sampling.py    radial Metropolis and Wolff updates
src/rgflow/phi4/observables.py phi-four observables and ensemble summaries
src/rgflow/phi4/plotting.py    SVG production figures
src/rgflow/phi4/cli.py         phi-four generation command
artifacts/phi4/                generated configurations and figures
tests/                         correctness and end-to-end tests
docs/                          reference material
presentation/                  pre-executed analysis notebooks
```
