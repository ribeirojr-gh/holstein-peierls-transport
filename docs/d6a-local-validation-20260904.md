# D6a local validation — frozen pair propagation

## Status

D6a is numerically accepted from the local validation artifact produced on 2026-09-04 UTC.

The submitted archive reported `overall_status = pass`, with no failed gates.  The run used Python 3.12.3, NumPy 2.5.2, SciPy 1.18.1 and explicit single-thread controls

- `OPENBLAS_NUM_THREADS=1`
- `OMP_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`

The artifact was generated from a source tree without Git metadata, so `git_commit`, `git_branch`, and `git_status` were all recorded as `unknown`.  D6a is therefore accepted numerically with an explicit provenance caveat: the exact local source SHA cannot be established from the validation archive itself.

## Regression gates

- focused D6a tests: 8/8 passed;
- full test suite: 289 passed in 10.24 s;
- frozen-pair benchmark: passed all pre-registered closure checks.

## Frozen-H benchmark

Control: 3x3 lattice, pair Hilbert-space dimension 81, `dt = 0.2 fs`, 40 steps (8 fs total).

### Bipolaron

| Krylov m | phase-aligned error | max norm error | energy error [eV] | H applications |
|---:|---:|---:|---:|---:|
| 6 | 3.247e-9 | 1.554e-15 | 5.690e-16 | 240 |
| 8 | 2.073e-13 | 6.217e-15 | 5.274e-16 | 320 |
| 12 | 2.098e-14 | 1.710e-14 | 2.776e-15 | 480 |
| 16 | 1.669e-14 | 1.998e-15 | 3.192e-16 | 640 |

Exchange symmetry stayed at roundoff and the spin-summed one-body RDM retained trace two.

### Distinguishable electron-hole exciton

| Krylov m | phase-aligned error | max norm error | energy error [eV] | H applications |
|---:|---:|---:|---:|---:|
| 6 | 1.408e-9 | 1.332e-15 | 1.249e-15 | 240 |
| 8 | 8.340e-14 | 7.327e-15 | 3.053e-16 | 320 |
| 12 | 2.196e-14 | 1.910e-14 | 1.804e-15 | 480 |
| 16 | 1.254e-14 | 1.021e-14 | 1.082e-15 | 640 |

Electron and hole reduced density matrices each retained trace one.

## Decision

The complex-safe matrix-free pair action is validated for frozen Hamiltonians.  `m=8` is the economical frozen-H reference for the 3x3 control, but it is **not yet promoted as the coupled moving-lattice production dimension**.  D6b must revalidate Krylov dimension, time-step convergence, pair-state forces, total-energy conservation, bipolaron exchange symmetry, and electron/hole RDM traces with a moving lattice.

No finite-temperature, decoherence, field-driven, or transport claim follows from D6a.
