# IP0d final local validation — 2026-09-06

## Status

**IP0d is formally CLOSED.**

The rerun was performed locally from a ZIP checkout of `isotropic-polaron-barrier`; therefore Git provenance fields are `unknown`, as expected for the established ZIP-based local-validation workflow. Numerical environment: Python 3.12.3 on WSL2, NumPy 2.5.2, SciPy 1.18.1, with OpenBLAS/OMP/MKL each restricted to one thread.

## Regression and numerical closure

- focused tests: **32/32 passed**;
- full suite: **364/364 passed**;
- 20x20 IP0d benchmark: **PASS**;
- all eight IP0d numerical checks: **PASS**;
- maximum endpoint-energy mismatch: `2.776e-16 eV`;
- maximum charge-constraint residual: `1.234e-11`;
- maximum norm error: `2.220e-16`;
- maximum biased-energy identity error: `2.220e-16 eV`;
- minimum constrained-state variational penalty: `-4.441e-16 eV` (roundoff);
- maximum normalized endpoint-span error: `2.220e-16`;
- total constrained diagonalizations: 302.

The corrected 4x4 tests now explicitly reject a translation charge order parameter when the small isotropic cell produces endpoint charge densities that are not distinguishable. This is a valid domain guard, not a relaxation of the physics checks.

## 20x20 physical diagnostics

| J0y/J0x | direction | adiabatic frozen barrier [meV] | charge-constrained barrier [meV] | maximum off-midpoint constraint penalty [meV] |
|---:|:---:|---:|---:|---:|
| 0.15 | +x | 13.046228 | 13.046228 | 1.185074 |
| 0.15 | +y | 101.522258 | 101.522258 | 35.895391 |
| 0.50 | +x | 11.311174 | 11.311174 | 0.905289 |
| 0.50 | +y | 51.241725 | 51.241725 | 11.269220 |
| 1.00 | +x | 12.013953 | 12.013953 | 0.933465 |
| 1.00 | +y | 12.013959 | 12.013959 | 0.933466 |

The constrained and adiabatic barriers coincide because their maxima occur at the symmetric frozen midpoint. At that midpoint the unconstrained adiabatic state already satisfies the full-charge-cloud translation coordinate, so the Lagrange multiplier is numerically zero. Away from the midpoint, forcing continuous charge transfer can cost substantial additional energy, especially for the hard transverse direction of the strongly anisotropic control.

For the isotropic 20x20 case the midpoint is symmetric in both directions, with source and target populations about 0.3124 each and participation number about 4.6847. The charge constraint therefore confirms that the IP0a frozen midpoint is a genuine shared-charge configuration rather than the lattice-only bypass identified in IP0c.

## Scientific interpretation

IP0a–IP0d now consistently show that the absence of isotropic transport in the historical dynamics is **not explained by a large static nearest-neighbour translation barrier**. The isotropic frozen barrier remains about 12 meV in both directions, while IP0b showed a substantially larger and slower collective intermolecular reorganization coordinate as isotropy is approached.

This does not establish a finite-temperature activation barrier, hopping rate, diffusion coefficient, or mobility. The next question is dynamical: whether thermal lattice fluctuations generate persistent nearest-neighbour hops, how the hop rate depends on temperature, and whether hops are preceded by transient local anisotropy of the instantaneous transfer integrals.

## Decision

Proceed to **IP1a: zero-field thermal hopping and transient-anisotropy screening**. No further static MEP claim is made from IP0d.
