# Project Log

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
  configurations for each of `L=8,16,32` at `lambda=1`, `kappa=0.3401`.
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
