# 4D SU(3) research plan

generate reference SU(3) ensembles using heatbath + overrelaxation: 
- L24, beta = 6.20, ref lattice spacing a = 0.06775(24) fm, measured lattice spacing a = 0.0722350(2175) fm
- L12, beta = 5.775, measured lattice spacing a = 0.144038(1749) fm, spacing ratio a_c/a_f = 1.9940(249); replaces beta = 5.80 with a = 0.137312(1608) fm

- L16, beta = 5.95, ref lattice spacing a = 0.10208(25) fm, actual lattice spacing a = 0.1047 fm



## Downsampling

### Try the naive blocking in scripts/downsample: L24 to L12

- firstly try the stout smearing + blocking, the stout smearing is parameterized by a coefficient kernel K, optimize K to match the coarse reference lattice, the procedure is similar to the "perfect blocking" in the phi4 branch.
   - the stout smearing is defined as U_smeared = k1 * U + k2 * staple link + ... (firstly try only the staple terms in 4 directions)
   - the kernel K should be parameterized by a CNN for training.
   - note that after stout smearing, we need to project the smeared link back to the SU(3) group;
   - in this 4d case, blocking only choose 1 link in the 2^4 sublattice.
   - "match the coarse reference lattice" is defined as the minimize the covariance-weighted mismatch of some observables, i.e. loss function in docs/2608.28581.pdf (firstly try only the plaquette, 1*2 rectangle, 2*2 square and one long-distance observable)

   - use pyquda for measurements, use pytorch for CNN and training.

### Completed eight-operator stout experiment

Run `L24_beta6p20_to_L12_beta5p775_polynomial_local_ls_v4_8ops_6train` trains W11, W12, W23, W33, |P|² and |P|⁴ from fresh perturbative coefficients; W13 and W22 are held out. All 300 fields are evaluated with the frozen checkpoint, and the stout notebook shows eight-operator comparisons. Optimizer: 100 evaluations; converged = False.
