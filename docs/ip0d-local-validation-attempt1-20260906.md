# IP0d local validation — attempt 1 (2026-09-06)

## Status

**Overall runner status: FAIL. IP0d remains OPEN pending rerun.**

The production-scale 20x20 IP0d benchmark itself completed successfully and all eight IP0d numerical physics checks passed. The overall runner failed because three newly added unit tests used a 4x4 fully isotropic static fixture whose relaxed charge density is effectively translation invariant. For that finite-size fixture, a full-charge-cloud order parameter between one-site translated endpoints is mathematically undefined, and the implementation correctly raises `ValueError("translated endpoint charge densities are not distinguishable")`.

This is a test-fixture error, not a failure of the 20x20 IP0d method. The tests have been corrected to use a localized 4x4 anisotropic fixture for positive constraint tests and to retain the 4x4 isotropic case as an explicit negative/guard test.

## Environment

- Python 3.12.3
- NumPy 2.5.2
- SciPy 1.18.1
- Linux / WSL2 x86_64
- `OPENBLAS_NUM_THREADS=1`
- `OMP_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
- ZIP provenance fields are `unknown`, as expected for a downloaded branch archive without `.git` metadata.

## Runner results

- `py_compile`: PASS
- focused pytest: FAIL — 3 IP0d fixture failures
- full pytest: FAIL — **360 passed, 3 failed**
- 20x20 IP0d benchmark: PASS

The only failures were:

1. `test_translation_charge_order_parameter_has_unit_endpoint_coordinates`
2. `test_midpoint_constraint_hits_full_charge_order_target`
3. `test_biased_eigenvalue_identity_removes_constraint_bias`

All failed at order-parameter construction on the same 4x4 isotropic fixture because the translated endpoint densities were indistinguishable.

## 20x20 benchmark numerical checks

All passed:

- all static relaxations converged;
- endpoint charge-order span normalized to exactly 2;
- charge constraint targets achieved;
- electronic norm preserved;
- biased-eigenvalue identity satisfied after removing the auxiliary constraint energy;
- translated endpoint energies invariant;
- constrained physical state obeys the variational bound;
- all charge-constrained path barriers finite and non-negative.

Aggregate diagnostics:

- maximum endpoint energy mismatch: `2.7756e-16 eV`;
- maximum charge-coordinate residual: `1.2335e-11`;
- maximum norm error: `2.2204e-16`;
- maximum biased-energy identity error: `2.2204e-16 eV`;
- minimum constraint energy penalty: `-4.4409e-16 eV` (roundoff);
- total constraint diagonalizations: `302`.

## Production-scale physical diagnostic

| J0y/J0x | direction | adiabatic frozen barrier (meV) | charge-constrained barrier (meV) | maximum off-path constraint penalty (meV) |
|---:|:---:|---:|---:|---:|
| 0.15 | +x | 13.046228 | 13.046228 | 1.1851 |
| 0.15 | +y | 101.522258 | 101.522258 | 35.8954 |
| 0.50 | +x | 11.311174 | 11.311174 | 0.9053 |
| 0.50 | +y | 51.241725 | 51.241725 | 11.2692 |
| 1.00 | +x | 12.013953 | 12.013953 | 0.9335 |
| 1.00 | +y | 12.013959 | 12.013959 | 0.9335 |

For every case, the maximum frozen-path energy occurs at the symmetric midpoint. At that midpoint the natural adiabatic state already satisfies the centered full-charge-cloud coordinate, so the required bias is essentially zero and the charge-constrained barrier equals the IP0a adiabatic frozen barrier. Away from the midpoint, especially along the hard `+y` direction of the anisotropic model, enforcing a prescribed continuous charge transfer can require a substantial positive electronic penalty.

This result confirms that the IP0a frozen midpoint is a genuine electronic charge-sharing configuration. It does **not** validate the failed IP0c relaxed lattice path, nor does it provide a relaxed MEP/free-energy barrier.

## Corrective action

Commit `7943cc213ca35517461f20784f2b861e4c8772a1` changes only the IP0d unit-test fixture:

- positive order-parameter tests now use a localized `J0y/J0x=0.5` 4x4 fixture;
- the fully isotropic 4x4 fixture is retained as an explicit test that indistinguishable endpoints are rejected.

IP0d will be closed only after a fresh local runner artifact passes focused tests, the complete pytest suite, and the 20x20 benchmark.
