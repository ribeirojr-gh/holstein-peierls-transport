# D6c local validation — 2026-09-05

## Scope

Local Ubuntu/WSL2 validation of the D6c finite-temperature pair BAOAB lattice bath for the correlated singlet bipolaron and distinguishable electron-hole exciton sectors.  The electronic propagation remains coherent Ehrenfest CF4-Lanczos dynamics.  No electronic decoherence, electric field, mobility, or transport fit is included in this gate.

The uploaded validation archive was generated from a ZIP checkout, so `git_commit`, `git_branch`, and `git_status` are reported as `unknown`.  The numerical artifact is accepted with that provenance limitation and corresponds to the D6c implementation distributed from branch `d6-pair-dynamics` after commit `f8c0c4c34b7aaa6ed5022bb50873ac7bc479e125`.

## Environment

- Python 3.12.3
- NumPy 2.5.2
- SciPy 1.18.1
- WSL2 x86_64
- `OPENBLAS_NUM_THREADS=1`
- `OMP_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`

## Test gates

- D6 focused tests: PASS (D6a + D6b + D6c; 24 tests)
- full pytest: PASS
- D6c pair finite-temperature benchmark: PASS
- failed gates: none

## Numerical control

- lattice: 4 x 4
- pair electronic ordered-basis dimension: 256
- temperature: 300 K
- `gamma_u = gamma_v = 0.01 fs^-1`
- `dt = 0.2 fs`
- trajectory length: 2 ps
- burn-in: 0.5 ps
- sampling interval: 5 fs
- four seeds per sector: 20260905–20260908
- CF4-Lanczos Krylov dimension: 8
- projected intermolecular zero modes
- kinetic degrees of freedom: 46 = 3N - 2

## Ensemble results

### Bipolaron

- mean lattice temperature: 296.546 K
- standard deviation across trajectory means: 20.778 K
- maximum `|Delta E_matter - Q_bath|`: 4.93544e-6 eV
- maximum electronic norm error: 3.109e-15
- maximum singlet-sector/RDM constraint error: 1.777e-15
- maximum projected zero-mode mean: 2.082e-17
- maximum lattice excursion: 0.5363 angstrom

### Exciton

- mean lattice temperature: 297.834 K
- standard deviation across trajectory means: 23.261 K
- maximum `|Delta E_matter - Q_bath|`: 4.51743e-6 eV
- maximum electronic norm error: 3.331e-15
- maximum e/h RDM constraint error: 8.882e-16
- maximum projected zero-mode mean: 2.082e-17
- maximum lattice excursion: 0.6622 angstrom

The individual trajectory-mean temperatures fluctuate substantially, as expected for only 46 kinetic degrees of freedom and 1.5 ps of post-burn sampling; the four-trajectory ensemble means are nevertheless close to the 300 K bath target.

## Pre-registered closure

All 12 checks passed:

- bipolaron temperature
- bipolaron energy balance
- bipolaron norm
- bipolaron sector constraints
- bipolaron zero modes
- bipolaron bounded lattice
- exciton temperature
- exciton energy balance
- exciton norm
- exciton sector constraints
- exciton zero modes
- exciton bounded lattice

## D6c conclusion

D6c is closed.  The D4 exact-OU BAOAB lattice thermostat transfers cleanly to both D6 pair sectors while preserving the pair-specific electronic constraints and the generalized zero-field energy balance

`Delta E_matter ~= Q_bath`.

This result is deliberately not an electronic-equilibrium claim.  The next checkpoint must diagnose electronic thermalization in the many-body adiabatic basis before any IDC/decoherence model is introduced.  For the bipolaron, the thermal reference must be restricted to the symmetric spatial singlet sector; antisymmetric spatial states must not enter the canonical comparison.  For the distinguishable e-h exciton, the full ordered `N^2` pair space is the appropriate reference.