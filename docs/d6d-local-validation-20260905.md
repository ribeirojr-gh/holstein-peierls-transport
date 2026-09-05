# D6d local validation and pair electronic-overheating diagnosis

Local artifact: `d6d-local-validation/20260905T115702Z/` (ZIP workflow; git metadata inside the artifact is therefore `unknown`).

## Numerical validation

- Python 3.12.3, NumPy 2.5.2, SciPy 1.18.1, WSL2 x86_64.
- `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1`.
- D6 focused tests: 30/30 passed.
- Full test suite: 311 passed.
- D6d benchmark completed in 334.56 s.
- All eight numerical gates passed for both sectors: lattice temperature, D6c energy/bath balance, electronic norm, and sector constraints.

## Control

`4x4`, 300 K, `gamma_u=gamma_v=0.01 fs^-1`, `dt=0.2 fs`, 4 ps, 1 ps burn-in, diagnostics every 50 fs, four seeds per sector, CF4-Lanczos `m=8`, projected intermolecular zero modes.

The bipolaron canonical reference is restricted to the symmetric spatial singlet sector, dimension `N(N+1)/2=136`; the distinguishable e-h exciton uses the full ordered pair space, dimension `N^2=256`.

## Ensemble diagnosis

| metric | bipolaron | exciton |
|---|---:|---:|
| lattice T [K] | 302.767 +/- 17.515 | 303.651 +/- 18.576 |
| mean heating coordinate | 0.174568 +/- 0.017738 | 0.124737 +/- 0.027367 |
| early heating | 0.138521 | 0.120336 |
| late heating | 0.213572 | 0.129766 |
| heating slope [ps^-1] | +0.037657 +/- 0.008348 | +0.004799 +/- 0.003634 |
| beta_eff / beta_bath | 0.153532 | 0.142940 |
| TV to canonical | 0.496286 | 0.205847 |
| TV to uniform | 0.864260 | 0.874552 |
| ground-manifold population | 0.497624 | 0.794151 |
| canonical ground-manifold population | 0.993357 | 0.999998 |
| max sampled |dE-Q_bath| [eV] | 1.238e-5 | 5.003e-6 |
| max norm error | 4.219e-15 | 3.997e-15 |

All four seeds have positive heating slopes in both sectors. The bipolaron effect is statistically much stronger; the exciton drift is slower but the propagated ground-manifold population is still substantially below its instantaneous 300 K canonical reference.

## Decision

D6d demonstrates model-specific electronic overheating under coherent pair Ehrenfest dynamics while the classical lattice bath remains correctly thermalized. An explicit electronic-relaxation/decoherence correction is therefore justified for **both** pair sectors.

This does **not** justify copying the one-polaron D5 production choice. In particular:

- IDC-BM is not preselected for pair dynamics;
- the one-polaron `t_d=180 fs` control is not transferred;
- bipolaron collapse must remain inside the symmetric spatial singlet sector;
- exciton collapse uses the full distinguishable e-h sector;
- all collapse energy jumps must enter the generalized balance explicitly, with no hidden lattice-velocity rescaling.

D6e will compare DP, BM and MA at a common numerical-control interval before any interval-sensitivity sweep or production selection.