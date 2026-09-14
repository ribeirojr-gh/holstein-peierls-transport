# IP1i field-driven moving-polaron screen: local validation closure

Date: 2026-09-14
Branch: `isotropic-polaron-barrier`
Validated commit: `6af2da2fc308e30ba711ddc659aa9018a8f57ca0`

## Numerical closure

The uploaded local artifact `ip1i-local-validation/20260914T155718Z` completed all requested gates. The run used Python 3.12.3 from the repository `.venv`, NumPy 2.5.3, SciPy 1.18.1, and single-thread BLAS/OpenMP settings. `git_status` was clean.

- focused tests: PASS;
- full suite: 420/420 PASS;
- 40x40 zero-damping PBC recurrence preflight for 5 ps: PASS;
- six deterministic D3 field-driven trajectories: PASS;
- all static convergence, energy-work balance, electronic norm, projected-zero-mode and finite-diagnostic gates: PASS.

The benchmark required about 878 s locally.

## Physical screening result

The field points along +x. For the electron-like one-polaron convention used by TP1, the observed carrier displacement is toward -x, consistent with the established sign convention for field power.

| J0y/J0x | E [mV/A] | TP1 dx [A] | persistent NN events | net event dx [sites] | transport-qualified |
|---:|---:|---:|---:|---:|---:|
| 1.00 | 2 | -0.207 | 0 | 0 | no |
| 1.00 | 5 | -0.782 | 0 | 0 | no |
| 1.00 | 10 | -3.864 | 1 | -1 | no |
| 0.15 | 2 | -0.278 | 0 | 0 | no |
| 0.15 | 5 | -3.365 | 1 | -1 | no |
| 0.15 | 10 | -7.158 | 2 | -2 | yes |

For the anisotropic 10 mV/A control, the two persistent -x events start at 2496 fs and 4108 fs, giving a 1612 fs inter-event interval. The isotropic 10 mV/A control has one persistent -x event starting at 2826 fs and no second persistent event before 5 ps.

## Interpretation

IP1i closes as a protocol-selection stage, not as a mobility or field-threshold calculation. The 40x40, deterministic T=0, 10 mV/A anisotropic control is the first screened condition that provides self-consistent field-driven motion with multiple persistent nearest-neighbor relocations and TP1/event-sign agreement. The isotropic 10 mV/A case is retained as a near-pinned comparison.

The next stage must measure lattice radiation in the laboratory frame around the first natural persistent relocation. Because the first anisotropic event is followed by another event after 1612 fs, a post-event window of at most about 1.5 ps is appropriate for an event-isolated diagnostic. At the IP1h packet speed scale (~1.6-1.7 sites/ps), this window can robustly probe the d=2 boundary but is too short for a clean long-range d=4-6 event-isolated analysis.
