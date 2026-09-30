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

## Local polynomial refinement

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
