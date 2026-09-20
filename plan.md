# 4D SU(3) research plan

The sole target of this repository is four-dimensional SU(3) lattice gauge
theory. The immediate objective is to build and validate an RG-assisted
configuration sampler for the Wilson gauge action; portability to other
theories is out of scope.

## Initial scope

1. Define 4D periodic SU(3) link fields and the Wilson plaquette action.
2. Implement numerically stable staple construction, local link updates, and
   reproducible Markov-chain sampling.
3. Measure plaquette observables, action density, Polyakov loops, and short
   Wilson loops, with autocorrelation and thermalization diagnostics.
4. Establish reference ensembles at selected lattice sizes and couplings.
5. Implement scale-two blocking and compare blocked ensembles against native
   coarse ensembles.
6. Develop the inverse-blocking proposal and exact Metropolis correction only
   after the reference sampler and blocking diagnostics are trusted.

## Validation requirements

- Check SU(3) unitarity and determinant-one constraints after every update.
- Verify detailed balance for local updates on small lattices.
- Compare observables across independent chains and report effective sample
  sizes rather than only saved-configuration counts.
- Test every blocking or inverse-blocking step at more than one lattice size.
- Keep generated data and scratch calculations under `artifacts/` and
  `temp/`; neither is part of the source tree.

## Milestones

- [ ] Project skeleton and environment setup
- [ ] SU(3) algebra and link-field representation
- [ ] Wilson action and local reference sampler
- [ ] Ensemble diagnostics and baseline data
- [ ] Scale-two RG blocking
- [ ] Inverse proposal and exact correction
- [ ] End-to-end performance comparison
