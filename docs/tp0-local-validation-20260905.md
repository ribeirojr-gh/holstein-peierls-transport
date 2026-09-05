# TP0 local validation and closure — 2026-09-05

## Scope

TP0 combines the validated one-polaron ingredients needed before defining transport observables:

- D3 uniform electric field through Peierls phases;
- D4 finite-temperature BAOAB lattice bath with projected intermolecular zero modes; and
- D5 IDC-BM electronic decoherence in the instantaneous **field-dependent** adiabatic basis.

The bookkeeping identity under test is

`Delta E_matter ~= Q_lattice + Q_electronic + W_field`.

This checkpoint validates the combined energy accounting and numerical stability only. It does **not** establish a steady state or a mobility.

## Local validation artifact

The user supplied `tp0-local-validation.zip`, containing the run directory
`tp0-local-validation/20260905T202048Z/`.

Environment recorded by the runner:

- Python 3.12.3;
- NumPy 2.5.2;
- SciPy 1.18.1;
- WSL2 Linux x86_64;
- `OPENBLAS_NUM_THREADS=1`;
- `OMP_NUM_THREADS=1`;
- `MKL_NUM_THREADS=1`.

Because the source was downloaded as a GitHub ZIP, the local metadata reports `git_commit`, `git_branch`, and `git_status` as `unknown`. The benchmark is therefore numerically reproducible but its exact local Git SHA cannot be inferred from the artifact itself. The intended source branch was `transport-production`, whose TP0 runner head before this closure record was `20f0b8016e40bd7edf0b0d90de7e1519311546be`.

## Regression gates

- focused field/BAOAB/IDC tests: PASS;
- complete pytest suite: **324 passed** in 9.09 s;
- TP0 driven thermal IDC benchmark: PASS;
- failed gates: none.

## TP0 control

- lattice: 4 x 4;
- temperature: 300 K;
- `gamma_u = gamma_v = 0.01 fs^-1`;
- field: `E_x = 2 mV/angstrom`;
- timestep: 0.2 fs;
- total time: 4 ps;
- burn-in: 1 ps;
- electronic propagator: CF4-Lanczos, Krylov dimension 6;
- IDC: BM, fixed interval 180 fs;
- projected intermolecular zero modes;
- four stochastic seeds.

The field and the 180 fs IDC interval remain numerical controls, not material-calibrated parameters.

## Aggregate results

- mean lattice temperature: **303.350338 K**;
- maximum sampled generalized energy-balance residual: **1.49782e-6 eV**;
- maximum electronic norm error: **1.98952e-13**;
- maximum projected zero-mode mean: **4.16334e-17**;
- maximum lattice excursion: **0.620777 angstrom**.

All pre-registered TP0 checks passed: temperature, full energy balance, electronic norm, projected zero modes, bounded lattice motion, event count, finite field work, and finite electronic-environment exchange.

## Per-trajectory energy exchanges

| seed | <T> [K] | Q_lattice [eV] | Q_electronic [eV] | W_field [eV] | Delta E_matter [eV] | max |residual| [eV] |
|---:|---:|---:|---:|---:|---:|---:|
| 20260905 | 299.0200 | 0.7211533 | -0.00587638 | 0.00242314 | 0.7177012 | 1.20674e-6 |
| 20260906 | 317.0606 | 0.9231057 | -0.03188840 | 0.01950286 | 0.9107211 | 1.03797e-6 |
| 20260907 | 311.4828 | 1.3148517 | -0.01701901 | -0.00021748 | 1.2976165 | 1.49782e-6 |
| 20260908 | 285.8379 | 0.9713621 | -0.00664472 | -0.00582203 | 0.9588965 | 1.15220e-6 |

Each trajectory produced the expected 22 IDC events.

The sign of the finite-time field work varies among four short stochastic trajectories. That is not interpreted as a transport result; TP0 was deliberately an energy-closure gate rather than a drift or mobility benchmark.

## Decision

**TP0 is closed.**

The validated combined propagation can now be used to define transport observables in TP1. The next checkpoint must avoid naive Cartesian centers under periodic boundaries. The preferred transport coordinate will be obtained from bond probability flux/current and its time integral, with periodic circular-center diagnostics used only as a localization aid.
