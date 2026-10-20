# Polynomial blocking and ensemble matching

The implementation follows Eqs. (2.39)–(2.43) of
`Classically perfect blocking.pdf`: a covariant polynomial on the fine
lattice, optional symmetry-completed P5/P6 hooks, one SU(3) polar projection,
then the product of two links at all-even anchors.

## Correct linearization

Let `U_mu = I + i A_mu + O(A^2)` and use the left-endpoint Fourier
convention `A_mu(x) = exp(i k.x) A_mu(k)`. Write

```
g_mu(k) = 1 - exp(i k_mu)
d_mu(k) = sum_(nu != mu) [2 cos(k_nu) - 2]
L_mu,mu(k) = d_mu(k)
L_mu,nu(k) = [1-exp(i k_mu)] [1-exp(-i k_nu)]   (nu != mu)
```

The first application of the covariant operator gives
`L_U U = i L(k) A + O(A^2)`. Its constant term vanishes. When applying
`L_U` again, variations of the transporting links multiply a field with
zero constant term, and therefore only contribute at quadratic order.
At linear order, transport is the identity and each output direction is
acted on by its scalar transverse Laplacian. Consequently,

```
linearization(L_U^n U)_mu = i d_mu^(n-1) L_mu,nu A_nu
omega_mu,nu = delta_mu,nu + sum_(n=1..4) a_n d_mu^(n-1) L_mu,nu
```

The matrix polynomial `I + sum a_n L(k)^n` describes a different smoother.
A finite-difference SU(3) plane-wave test now checks the momentum kernel
against the actual nonlinear implementation through fourth order.

In the same convention the Wilson propagator is

```
G_mu,nu(k) = [delta_mu,nu - g_mu g_nu* / |g|^2] / |g|^2
D_mu,nu(k) = delta_mu,nu [1 + exp(i k_mu)]
```

The midpoint propagator with real `2 sin(k/2)` cannot be combined directly
with the endpoint kernels `L` and `D`. The corrected propagator annihilates
the endpoint pure-gauge mode, and the blocking objective projects onto the
same transverse subspace on the coarse lattice.

The independent, unrestricted path family in the note has many more
linear directions than this compact polynomial. Its quoted residuals
are not a validated accuracy claim for the four-parameter implementation.
With the corrected kernel, grid-8 optimization gives a residual of
0.0188374; an independent 4096-point scrambled Sobol sample gives
0.0188286. The fitted coefficients are approximately
`(0.0213491, -0.0190803, -0.00250032, -0.000157110)`.

## Nonperturbative fit

`scripts/downsample/fit_polynomial.py` fits means and log variances with
SciPy's trust-region least squares and an analytic ensemble Jacobian. It
uses a fixed training subset, avoiding changes in the objective from
rotating configurations. Dimensionless coefficients multiply
`L^n U / 12^n`; saved coefficients retain the original normalization.
The optional hooks add `H U/12`, `L H U/12^2`, and `L^2 H U/12^3`.
They are constructed for all four link directions before applying `L`.

The `--local` extension adds four coefficients multiplying
`f_mu(x) L^n U_mu`, with `f_mu` the mean real trace of the six adjacent
plaquettes minus one. This scalar is gauge invariant and starts at
quadratic order in the weak field, so these local terms leave the linear
momentum kernel unchanged. The cached basis includes their products at
the same retained fine sites. The local fit can warm-start from the
hook-polynomial checkpoint with `--initial-run`.

For independent fine and target ensembles of sizes `n_f` and `n_c`, the
mean residual scale is `sigma_target sqrt(1/n_f + 1/n_c)`. The log-variance
residual scale is `sqrt(2/(n_f-1) + 2/(n_c-1))`. Parameter derivatives of
the observables give derivatives of both moments. Only training targets
enter the fit; complete validation and test splits are measured afterwards.

The cached basis discards fine sites absent from the final two-link
products before the site-local projection. This is exactly equivalent to
the full map, rather than an approximation to it. Tests compare both the
blocked fields and coefficient gradients for nonzero hook coefficients.

The existing streaming Adam implementation also had a missing factor:
the gradient of `(delta_log_variance / scale)^2` includes division by
`scale^2`. The regression test now uses nonunit scales.

## Reading the metrics

`standardized_shift` divides the mean difference by the combined standard
errors of the two samples. `mean_shift_in_target_sigma` divides by the
target distribution's standard deviation. They answer different questions.
`std_ratio` checks the width rather than just the mean.

The explicit distribution criterion is all means within one target sigma
and all width ratios within 20%. The stricter criterion that all means
are within one combined standard error is reported separately. These
thresholds are recorded directly; beating the old baseline is insufficient.
This verifies the selected observables, not equality of the complete
gauge-field measures.

## Previous local polynomial refinement at beta 5.80

Run `L24_beta6p20_to_L12_beta5p80_polynomial_local_ls_v2` refines the
hook-polynomial checkpoint with four local coefficients, using 60 fixed
training configurations (seed 1234) and a budget of 40 objective
evaluations. The training objective falls from 304.015 to 5.68530. The
solver stops at the evaluation budget rather than declaring convergence.
The full 180/60/60 partition is then measured with the saved coefficients.
`presentation/downsample_stout.ipynb` shows the fit history and the
complete held-out test distributions, with acceptance computed directly
from the saved samples.

The held-out test measurements are:

| observable | mean shift / target sigma | std ratio |
| --- | ---: | ---: |
| plaquette | 0.208 | 1.259 |
| rectangle_1x2 | 0.288 | 1.033 |
| square_2x2 | 0.294 | 0.970 |
| polyakov_abs2 | -0.101 | 0.725 |

All test means are within one target sigma, but plaquette and Polyakov
widths fail the 20% criterion. Thus `distribution_match` is false.
The stricter combined-standard-error check also fails; the four mean
shifts in those units are approximately `(1.004, 1.554, 1.632, -0.632)`.


## Scale-matched coarse reference at beta 5.775

The active L12 reference now contains 300 fresh configurations at beta 5.775,
using the same 50/250 hot-start seed streams, 1000 warmup steps and 100-step
separation. The same static-potential protocol measures
`a_c = 0.144038(1749) fm` and `a_c/a_f = 1.9940(249)`, with target
`2a_f = 0.144470 fm`. The original beta 5.80 reference remains archived.
The three trial ensembles and the initial Necco–Sommer relative-scale estimate
are recorded in `docs/project_notes/results/selected_downsampling_run.json`.

Run `L24_beta6p20_to_L12_beta5p775_polynomial_local_ls_v3` starts from the
previous local-polynomial checkpoint and refits the same fixed 60 training
configurations against all 180 new coarse training targets. Forty objective
evaluations reduce the training objective from 1222.66 to 5.68758; the solver
stops at its evaluation limit. Independent CPU configurations run concurrently
with `--workers 4`. Their observables and analytic Jacobians agree with CUDA
to about `4.2e-7` on a physical L24 field. Serial/parallel equality and an
independent finite-difference Jacobian check are included in the tests.

The frozen checkpoint is evaluated on all 300 fine configurations, using the
exact retained polynomial basis and the original 180/60/60 partition. Targets
are remeasured with PyQUDA from the new coarse ensemble. The cached naive
baseline depends only on the unchanged fine ensemble; its source hashes and
measurements have been checked against the original baseline run.

| observable | mean shift / target sigma | std ratio |
| --- | ---: | ---: |
| plaquette | -0.199 | 1.223 |
| rectangle_1x2 | -0.083 | 1.127 |
| square_2x2 | -0.062 | 1.014 |
| polyakov_abs2 | -0.097 | 0.750 |

All four held-out means pass both the one-target-sigma criterion and the
stricter actual combined-standard-error criterion. Plaquette and Polyakov
widths fail the 20% window, so `distribution_match` remains false. This is
an operator-distribution check rather than a proof of equality of gauge-field
measures. Full-map PyQUDA measurements on test indices 4, 9 and 14 agree
with the retained-basis values to `1.04e-7` in the largest absolute observable
difference. Generated links have maximum unitarity/determinant errors of
about `2.4e-6` and `2.6e-6` on those three configurations.

`presentation/downsample_stout.ipynb` and the selected progress-note bundle
show this new result. Upsampling measurements retain their previous frozen
smoother and are identified as an archived experiment.


## Eight-operator experiment with two unseen operators

The `polynomial_local_ls_v4_8ops_6train` experiment restarts from the grid-8
perturbative polynomial, with all hook and local coefficients initially zero.
It fits W11, W12, W23, W33, Polyakov |P|² and |P|⁴. W13 and W22 are evaluated
only after freezing. The fourth moment is `(1/4) sum_mu |P_mu|^4`, with each
P_mu the spatially averaged normalized trace in that direction.

The unchanged 180/60/60 partition uses the same fixed 60 fine training fields
(seed 1234) and all 180 native coarse training targets. Trust-region least
squares uses twelve residuals and eleven parameters, with a 100-evaluation
limit and early convergence at the existing tolerances. The per-channel
variance floor is `1e-12 * target_training_variance`, including in the analytic
Jacobian. Covariance regularization operates in target-standard-deviation
units; raw measured values are saved without rescaling.

All 300 native targets and naive-blocking samples are measured for the
extended set, with names saved in the caches. No four-channel baseline or
old fitted parameters enter this experiment. The shared measurement API's
unnamed default remains the legacy four-channel set. Eight-channel test
figures mark W13 and W22 as held out, and summarize matching separately for
training, held-out, and all operators. Older four-channel aggregate losses
are not compared numerically to this experiment.


The eight-operator run completes 100 evaluations: training objective 400572 → 47.6273. It reaches the evaluation budget without declaring convergence.

| observable | role | mean / target sigma | mean / combined SE | std ratio |
| --- | --- | ---: | ---: | ---: |
| W11 | training | -0.235 | -1.220 | 1.110 |
| W12 | training | -0.270 | -1.510 | 0.959 |
| W13 | held out | 0.753 | 4.222 | 0.954 |
| W22 | held out | -1.894 | -11.311 | 0.826 |
| W23 | training | -0.467 | -2.707 | 0.889 |
| W33 | training | 0.778 | 4.409 | 0.932 |
| \|P\|² | training | 0.020 | 0.113 | 0.899 |
| \|P\|⁴ | training | -0.041 | -0.251 | 0.764 |

training: distribution_match = False.

held_out: distribution_match = False.

all: distribution_match = False.

Full-map PyQUDA cross-check on three held-out configurations: max absolute observable difference 4.83e-08.
