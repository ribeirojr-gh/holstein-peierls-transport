# IP1e size-aware recheck — attempt 1

Date: 2026-09-07

## Status

**Execution environment failure. No physics or numerical recheck result was produced.**

The uploaded `ip1e-recheck-local-validation.zip` contains a runner summary with overall status FAIL, but all three failed gates are explained by using the system Python instead of the project virtual environment:

- Python executable: `/usr/bin/python`;
- `pytest` was not installed (`No module named pytest`);
- the editable project package was not installed (`No module named holstein_peierls`);
- the size-aware audit therefore never executed;
- pycompile itself passed.

This attempt must not be interpreted as a failure of the size-aware energy criterion or of the previously completed IP1e trajectories. The expensive dynamics do not need to be repeated.

## Independent artifact check

The original completed IP1e JSON artifacts remain internally consistent. Applying the proposed size-aware extensive energy gate directly to their recorded values gives:

| protocol | cell | max generalized residual [eV] | size-aware tolerance [eV] | expected size-aware result |
|---|---:|---:|---:|---:|
| baseline20 | 20x20 | 1.51860856e-5 | 5.0e-5 | PASS |
| large40 | 40x40 | 5.33097094e-5 | 2.0e-4 | PASS |
| weak40 | 40x40 | 5.26707978e-5 | 2.0e-4 | PASS |

All non-energy numerical checks in the original artifacts are already PASS. This table is an independent consistency check only; formal closure still requires the recheck runner to execute successfully in the project environment.

## Correct rerun procedure

A root-level `run.sh` is now provided. It creates/uses `.venv`, installs `.[dev]`, fixes BLAS/OpenMP thread counts to one, locates or extracts the original IP1e artifact directory, and runs the no-dynamics recheck.

From a fresh download of branch `isotropic-polaron-barrier`, place `ip1e-local-validation.zip` in the repository root and run:

```bash
bash run.sh
```

Alternatively pass the artifact directory explicitly:

```bash
bash run.sh ip1e-local-validation/20260907T183017Z
```

Expected full regression count is 405 tests if no regressions are present.

## Interpretation rule

Do not rerun IP1e dynamics because of this failed attempt. Do not promote the independent table above to formal closure until the runner itself passes. After a clean recheck, IP1e can be closed and the next stage can focus on explicit carrier-centered phonon-wake observables, including the sign of group velocity relative to carrier velocity.
