# Project Log

## 2026-08-24

- Added affine, strong-affine, and residual rational-quadratic spline flow
  configurations, optional reverse-KL fine-action tuning, conditional
  importance-weight ESS checkpoint selection, and backward-compatible
  checkpoint format version 2.
- Added exact 'coordinate-wolff', detail-heavy 'detail-wolff', conditional
  'flow-detail-wolff', and screening-only 'hmc-wolff' corrections. Fine-field
  Wolff updates rebuild all transformed inverse coordinates.
- Added per-chain trajectory histories, time-to-tolerance, observable-wise
  autocorrelation/ESS, worst-observable ESS/s, one-shot throughput, setup cost,
  end-to-end break-even, and paired-bootstrap efficiency comparisons.
- Added the timestamped 'rgflow benchmark phi4' experiment runner with native
  radial-MH+Wolff, Gaussian-initializer controls, and explicit HMC promotion
  criteria.
- Ran a non-production 16-chain, 30-sweep, single-seed screening artifact for
  both mappings. No method passed the final convergence gate and every inverse
  candidate remained slower in worst-observable ESS/s than native
  radial-MH+Wolff. HMC therefore remains a non-promoted exploratory control.

- Added volume-specific three-stage conditional affine flows for L8-to-L16 and
  L16-to-L32, trained with conditional NLL followed by observable and tail
  matching. The selected checkpoints are epochs 294 and 278; their held-out
  sweep-zero ten-observable RMS shifts are `0.3516` and `0.6046`.
- Added exact physical-coordinate inverse-blocking MH with a reversible coarse
  action kernel, sequential `d01`, `d10`, `d11` updates, divide-two residue
  classes, proposal-width calibration, full checkpointing, and reblocking
  validation below `1.8e-15`.
- Completed 128-chain, 400-sweep runs for both mappings. Final RMS shifts are
  `0.2158` and `0.2043`; both pass the maximum-shift and bootstrap physical
  checks but remain explicitly marked for extension because they narrowly miss
  the fixed RMS threshold of `0.20`.
- Added the pre-executed `presentation/flow_inverse_blocking.ipynb`, including
  Gaussian, sweep-zero flow, and sweep-400 comparisons plus contextual metrics
  from the checkpoint-free `temp/Inverse_RG` reference.
- Added PyTorch flow and presentation extras, flow/upscale CLI commands, and
  inverse-coordinate, logdet, exact-ratio, rejection, reproducibility, and
  end-to-end CLI tests.

- Replaced single-volume kernel fitting with an equal-weight one-step objective
  over L16-to-L8, L24-to-L12, and L32-to-L16; L48-to-L24 and L64-to-L32
  remain unseen-volume transfer tests.
- Added mild Binder and scaled-susceptibility matching, smooth conditioning
  regularization, and balanced 128-sample-per-chain optimization subsets while
  retaining complete held-out evaluation.
- Retrained the production two-channel SOS kernel. Its mean training-pair test
  RMS is `0.1078`, worst transfer RMS is `0.2170`, minimum Fourier spectrum is
  `0.6540`, and condition number is `2.1058`. Across the five held-out
  one-step links, its mean ten-observable RMS is `0.1241` versus `0.1410` for
  the reference kernel, with improvements on three of five links.
- Updated the presentation to use the multi-volume artifact and focus on
  one-step size scaling; cumulative multi-step plots were removed.
- Changed the production generator and kernel-training defaults to the
  reference critical coupling `kappa=0.340301`.
- Generated matched four-chain, 1,000-configuration ensembles for
  `L=8,12,16,24,32,48,64` with consistent MCMC settings.
- Retrained the two-channel SOS kernel at the unified coupling. It passes all
  acceptance checks, with test RMS shift `0.0479`, minimum spectrum `0.5733`,
  and condition number `2.2472`.
- Added the pre-executed `presentation/kernel_volume_scaling.ipynb` comparing
  the SOS and reference kernels on one-step and cumulative scaling chains,
  including distribution histograms and bootstrap physical observables.

- Added distribution-aware blocking-kernel training using standardized
  quantile matching at the 1%, 5%, 10%, 25%, 50%, 75%, 90%, 95%, and 99%
  levels for the ten observables used by the `temp/Inverse_RG` reference.
- Added JS divergence, total variation, target-tail coverage, KS, Wasserstein,
  standardized mean shift, and width-ratio diagnostics.
- Added held-out ten-panel density histograms comparing native coarse
  configurations with `DK` applied to fine configurations.
- The distribution-aware L16-to-L8 candidate passes the existing acceptance
  criteria. Its L32-to-L16 transfer RMS shift improves from 0.1472 to 0.1349,
  while its local-feature test RMS shift changes from 0.0530 to 0.0572.
- Improved the held-out phi2 and phi4 width ratios from 1.088 and 1.087 to
  1.038 and 1.031. Kept the new candidate in a separate artifact directory so
  the mean-only baseline remains available.
- Added a multichannel sum-of-squares kernel parameterization. Two 3x3 factors
  produce a D4-averaged 5x5 autocorrelation kernel with a strictly positive
  identity spectral floor, making invertibility true by construction.
- Trained a two-channel SOS candidate. It differs from the free
  distribution-aware kernel by at most 0.00126 per matrix entry and has nearly
  identical held-out observable distributions. Its test and transfer RMS
  shifts are 0.0572 and 0.1356, with min K(p)=0.5332 and condition 2.9860.
- Evaluated the free, SOS, and `temp/Inverse_RG` kernels during parameterization
  selection. These superseded comparison artifacts were removed after adopting
  the unified coupling.
- Promoted multichannel SOS to the only trainable kernel parameterization,
  removed the free/SOS CLI branch, and documented the analytic positive
  spectral bound that guarantees invertibility on every lattice size.

## 2026-08-01

- Added a D4-symmetric five-parameter 5x5 blocking kernel with fixed
  `sum(K)=2^(1/8)`, periodic convolution, even-even downsampling, Fourier
  diagnostics, and inversion.
- Added `rgflow train kernel phi4` with covariance-weighted observable
  matching, deterministic multi-start L-BFGS-B optimization, chain-separated
  training/validation/test data, and an isolated L32-to-L16 transfer test.
- Trained on the existing native L16 and L8 ensembles. The selected kernel
  reduced the held-out RMS standardized shift from 0.9150 for the scaled
  identity to 0.0530. Its L32-to-L16 transfer shift is 0.1472, dense-grid
  minimum Fourier magnitude is 0.6335, and condition number is 2.0905.
- Wrote the operational kernel, full metrics, and PDF diagnostics under
  `artifacts/phi4/kernel/L16_to_L8`.
- Added a single-page 27-panel comparison of native coarse and blocked
  operator distributions. It covers local moments, action components,
  magnetization, D4-averaged real- and momentum-space two-point functions, and
  moving-block bootstrap distributions for susceptibility, Binder cumulant,
  and second-moment correlation length.
- Confirmed that the expanded diagnostics do not change the selected kernel or
  the original acceptance metrics. They also expose larger held-out
  distribution shifts in susceptibility and xi/L than in the local training
  observables.

## 2026-07-31

- Reorganized the Python package into a standard `src` layout with the
  theory-specific implementation under `rgflow/phi4`.
- Moved reusable Markov-chain diagnostics to `rgflow/diagnostics.py` without
  imposing a shared field representation or sampler interface on future
  lattice theories.
- Replaced the theory-specific console entry point with the extensible
  `rgflow generate phi4` command and changed its default output directory to
  `artifacts/phi4`. Existing production data under `artifacts/2dphi4` remains
  unchanged.
- Verified the reorganized package with 11 passing tests and a minimal
  end-to-end generation run through the new command.

## 2026-07-30

- Started the native 2D phi-four configuration generator.
- Selected the kappa-lambda action with radial Metropolis and embedded-Ising
  Wolff updates.
- Added a NumPy implementation with periodic boundaries, adaptive warmup,
  multiple chains, compressed ensemble output, observables, autocorrelation
  times, ESS, and equilibration drift diagnostics.
- Added the `rgflow-generate-phi4` command. Its defaults target 1,000 stored
  configurations for each of `L=8,16,32` at `lambda=1`, `kappa=0.340301`.
- Verified the implementation with 9 passing tests, including local action
  differences, Z2 symmetry, reproducibility, the decoupled-site moment, and a
  CLI smoke run.
- Before a full production run, use pilot ensembles to select warmup and
  saving intervals from the slowest measured autocorrelation time.
- Moved the conventional generator to `2dphi4/generation`, changed the default
  output to `artifacts/2dphi4`, and added SVG configuration and diagnostic
  figures for each lattice size.
- Completed the default production run in 29 seconds. Generated 1,000 stored
  configurations each for `L=8,16,32`; production acceptance was
  0.497, 0.500, and 0.502, respectively. Binder cumulants were 0.5970,
  0.6084, and 0.6122. All arrays, JSON diagnostics, and SVG figures are under
  `artifacts/2dphi4`.
