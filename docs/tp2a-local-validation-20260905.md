# TP2a local validation — paired-field linear-response screening

Date: 2026-09-05

## Status

**Numerical screening: PASS. Mobility convergence: NOT YET ESTABLISHED.**

The local artifact `tp2a-local-validation/20260905T205533Z` was generated from a ZIP checkout, so Git commit/branch/status are reported as `unknown`. The archive was manually associated with the requested `transport-production` branch state used for TP2a. This provenance limitation does not affect the numerical results.

## Environment

- Python 3.12.3
- NumPy 2.5.2
- SciPy 1.18.1
- WSL2 Linux x86_64
- `OPENBLAS_NUM_THREADS=1`
- `OMP_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`

## Numerical closure

- focused TP0/TP1/TP2 regression tests: PASS (39 tests)
- full pytest suite: PASS (339 tests)
- paired-field benchmark: PASS
- trajectories: 24 = 4 seeds × 3 field magnitudes × 2 signs
- mean lattice temperature: 298.5306 K
- maximum generalized energy-balance residual: 1.5010e-5 eV
- maximum electronic norm error: 2.3848e-13
- maximum projected zero-mode mean: 1.2629e-17
- maximum difference between direct field work and `-E Δx`: 1.0481e-14 eV
- maximum instantaneous power/velocity identity error: 0

All nine pre-registered numerical closure checks passed.

## Statistical transport screening

Control:

- lattice: 20×20
- temperature: 300 K
- `gamma_u = gamma_v = 0.01 fs^-1`
- `dt = 0.2 fs`
- total time: 6 ps
- burn-in: 2 ps
- IDC-BM, `t_d = 180 fs`
- CF4-Lanczos, Krylov dimension 6
- paired fields: ±0.5, ±1.0, ±2.0 mV/Å
- four independent seed groups with common random numbers inside each ±E pair

Field-wise electron-mobility diagnostics (cm²/Vs):

| | 0.5 mV/Å | 1.0 mV/Å | 2.0 mV/Å |
|---|---:|---:|---:|
| ensemble mean | 0.12596 | 0.08684 | 0.02838 |
| 95% CI lower | -0.08965 | -0.09988 | -0.01437 |
| 95% CI upper | 0.34157 | 0.27357 | 0.07113 |

Seed-level through-origin mobility estimates using all three fields were:

- 20260905: 0.01654 cm²/Vs (`R²_origin = 0.432`)
- 20260906: 0.02535 cm²/Vs (`R²_origin = 0.802`)
- 20260907: 0.01761 cm²/Vs (`R²_origin = 0.999`)
- 20260908: 0.11715 cm²/Vs (`R²_origin = 0.635`)

The seed-level ensemble estimate was

`mu_e = 0.04417 cm²/Vs`,

with 95% Student-t CI

`[-0.03351, 0.12184] cm²/Vs`.

Additional diagnostics:

- ensemble through-origin `R² = 0.6952`
- maximum fractional linearity residual = 0.4832
- maximum even/odd mean ratio = 1.2336
- the 95% mobility CI includes zero

## Interpretation

TP2a establishes that the PBC-safe current/displacement machinery remains numerically consistent on a 20×20 transport cell and that the paired ±E protocol is operational. It does **not** establish a converged mobility.

All four seed-level all-field mobility estimates are positive, which is encouraging, but the small four-seed ensemble is dominated by stochastic variation (especially seed 20260908), the field dependence is not convincingly linear over 0.5–2.0 mV/Å, and the even component remains comparable to the odd signal. Therefore the value 0.04417 cm²/Vs must not be promoted as a material or production mobility.

The next checkpoint (TP2b) must increase both independent-seed count and trajectory duration, retain paired common-random-number ±E sampling, and evaluate temporal convergence from checkpoints within the same long trajectories. The 2.0 mV/Å control will be retained as a nonlinearity diagnostic rather than assumed to belong to the linear regime.
