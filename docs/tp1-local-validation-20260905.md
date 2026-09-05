# TP1 local validation and closure — 2026-09-05

## Scope

TP1 validates periodic-boundary-safe one-polaron transport observables on top of the already validated TP0 field + BAOAB + IDC-BM dynamics. The primary transport observable is the nearest-neighbour probability current and its time-integrated unwrapped displacement. TP1 does **not** claim a steady state or a mobility.

## Local validation artifact

The user supplied `tp1-local-validation.zip`, containing the run directory `tp1-local-validation/20260905T204309Z/`.

The run used Python 3.12.3 under WSL2 Linux x86_64 with NumPy 2.5.2, SciPy 1.18.1, and

- `OPENBLAS_NUM_THREADS=1`;
- `OMP_NUM_THREADS=1`;
- `MKL_NUM_THREADS=1`.

Because the validation tree was downloaded as a GitHub ZIP, its local metadata reports `git_commit`, `git_branch`, and `git_status` as `unknown`. The numerical artifact corresponds to the TP1 code distributed from branch `transport-production` at the stage whose branch head was `14c761180daec797725f9713cc57e3a0a120d130` before this closure note was added.

## Regression gates

- TP1 focused tests: **PASS** (`37 passed`).
- Full test suite: **PASS** (`332 passed in 9.08 s`).
- TP1 transport-observables benchmark: **PASS**.
- Failed gates: none.

## Control

The numerical benchmark used

- `4 x 4` periodic lattice;
- `T = 300 K`;
- `gamma_u = gamma_v = 0.01 fs^-1`;
- electron-like uniform field `E_x = +2 mV/angstrom`;
- `dt = 0.2 fs`;
- total time `4 ps`;
- burn-in `1 ps`;
- IDC-BM with `t_d = 180 fs`;
- CF4-Lanczos with Krylov dimension `m = 6`;
- projected intermolecular zero modes; and
- four stochastic seeds: 20260905--20260908.

The field and IDC interval remain numerical controls and are not material-calibrated parameters.

## Closure results

All eight pre-registered TP1 checks passed.

| quantity | result |
|---|---:|
| ensemble mean lattice temperature | 303.350338 K |
| max `|W_field(current) - W_field(D3)|` | `3.702e-15 eV` |
| max instantaneous `|P_field + E dot v|` | `8.098e-19 eV/fs` |
| max generalized energy-balance residual | `1.498e-6 eV` |
| max electronic norm error | `1.990e-13` |
| max projected zero-mode mean | `4.163e-17` |
| ensemble mean post-burn `v_x` | `-5.072e-4 angstrom/fs` |
| ensemble std post-burn `v_x` | `1.603e-3 angstrom/fs` |

The exact numerical agreement between field work evaluated from the D3 power expression and from the TP1 particle velocity is the strongest TP1 identity:

`P_field = - E dot v_particle`.

This establishes that the PBC-safe probability-current observable has the same field-coupling convention as the validated energy bookkeeping.

## Per-trajectory stochastic spread

The four post-burn x velocities were approximately

- seed 20260905: `-6.413e-4 angstrom/fs`;
- seed 20260906: `-3.077e-3 angstrom/fs`;
- seed 20260907: `+8.222e-4 angstrom/fs`;
- seed 20260908: `+8.675e-4 angstrom/fs`.

The spread is much larger than the four-trajectory mean. This is expected for short stochastic trajectories and is direct evidence that a single finite-field trajectory, or even this four-trajectory TP1 control, is insufficient for a mobility claim.

## Interpretation

TP1 is closed for its intended purpose:

1. the nearest-neighbour probability current satisfies the validated periodic transport convention;
2. its integrated displacement is unwrapped across periodic boundaries;
3. the current/velocity is exactly consistent with external field power to numerical precision; and
4. TP0 energy, thermostat, norm, and zero-mode controls remain intact.

The next phase must therefore address statistics and linear response, not redefine the current. Field reversal with paired stochastic seeds is the preferred next diagnostic because the odd-in-field component isolates drift while the even component measures finite-sample stochastic bias.

No mobility, steady-state, material prediction, or CPU/GPU performance claim follows from TP1 alone.
