# TP2b — paired-field convergence and mobility-readiness gate

## Motivation

TP2a passed all numerical checks but did not establish a converged linear-response mobility. Four independent seeds over a 4 ps post-burn window gave a positive mean mobility diagnostic, but its 95% confidence interval included zero, the ensemble through-origin R² was only 0.695, the maximum fractional linearity residual was 0.483, and the finite-time even component was comparable to or larger than the odd response.

TP2b therefore changes **sampling**, not physics.

## Frozen physics/numerics

- one-polaron Holstein-Peierls dynamics
- 20×20 periodic lattice
- 300 K
- BAOAB lattice bath
- `gamma_u = gamma_v = 0.01 fs^-1`
- field-aware IDC-BM
- `t_d = 180 fs` (numerical control, not material-calibrated)
- `dt = 0.2 fs`
- CF4-Lanczos, Krylov dimension 6
- projected intermolecular zero modes
- electron-like field convention already validated by TP0/TP1
- paired common-random-number `+E/-E` trajectories

## Sampling change

TP2b uses eight independent seed groups instead of four and extends each trajectory from 6 ps to 12 ps. The burn-in remains 2 ps.

Fields remain

- ±0.5 mV/Å
- ±1.0 mV/Å
- ±2.0 mV/Å

for 48 total trajectories.

The 2.0 mV/Å field is retained specifically as a nonlinearity diagnostic; it is not assumed to be inside linear response.

## Temporal convergence without reruns

Each 12 ps trajectory records cumulative post-burn mean velocities at final times

- 6 ps (4 ps post-burn window)
- 9 ps (7 ps post-burn window)
- 12 ps (10 ps post-burn window)

so temporal convergence can be assessed from nested windows of the same stochastic trajectory.

## Numerical closure gates

TP2b must preserve:

1. converged static initial state;
2. exact requested trajectory count;
3. ensemble lattice temperature between 240 and 360 K;
4. generalized energy-balance residual below `1e-4 eV`;
5. electronic norm error below `1e-10`;
6. projected zero-mode mean below `1e-12`;
7. exact expected IDC event counts;
8. direct field work vs `-E Δx` disagreement below `1e-10 eV`;
9. instantaneous field-power/velocity identity error below `1e-12 eV/fs`.

A failure of any numerical gate is a software/algorithmic failure.

## Pre-registered mobility-readiness diagnostics

These diagnostics are deliberately reported separately from numerical PASS. The TP2b control is considered ready for a production mobility estimate only if all hold at the final 12 ps checkpoint:

1. the seed-level 95% Student-t mobility CI is strictly positive;
2. ensemble through-origin `R² >= 0.90`;
3. maximum fractional linearity residual `<= 0.25`;
4. maximum mean even/odd response ratio `<= 0.50`;
5. mobility changes by at most 25% between the 9 ps and 12 ps cumulative checkpoints;
6. the low-field (0.5 and 1.0 mV/Å) mobility differs from the all-field mobility by at most 25%;
7. at least 75% of independent seed-level mobility estimates are positive.

These are benchmark readiness criteria, not universal physical constants.

## Interpretation rule

- **Numerical PASS + mobility readiness PASS:** proceed to the final TP2 production/statistical mobility stage.
- **Numerical PASS + mobility readiness NOT CONVERGED:** do not report a mobility; inspect which criterion failed and adapt sampling or field range in TP2c.
- **Numerical FAIL:** repair the implementation before any further transport inference.

No pair transport, material calibration, CPU threading, or GPU claim is part of TP2b.
