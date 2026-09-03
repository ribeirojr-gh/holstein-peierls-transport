# D4a local validation record

## Context

Hosted GitHub Actions minutes were exhausted during D4 development, so the D4a finite-temperature lattice-bath validation gates were executed locally on the user's Ubuntu/WSL2 workstation using `scripts/run_local_validation.py`.

The source tree was downloaded as a GitHub ZIP rather than cloned with `.git` metadata. Consequently the runner correctly recorded `git_commit`, `git_branch`, and `git_status` as `unknown`. The archive did contain the D4a local runner and D4a test files. This provenance limitation is recorded explicitly and does not alter the numerical results below.

## Environment

- validation timestamp: 2026-09-03T15:54:01Z;
- Python: 3.12.3, GCC 13.3.0;
- platform: Linux 6.18.33.2 Microsoft WSL2, x86_64, glibc 2.39;
- NumPy: 2.5.2;
- SciPy: 1.18.1;
- `OPENBLAS_NUM_THREADS=1`;
- `OMP_NUM_THREADS=1`;
- `MKL_NUM_THREADS=1`.

## D4a focused gates

`tests/test_d4_langevin.py` passed all eight focused tests. These cover:

1. exact Ornstein-Uhlenbeck damping coefficient and fluctuation-dissipation variance;
2. frictionless O-step identity without RNG consumption;
3. deterministic zero-temperature exponential damping;
4. fixed-seed reproducibility and independent coordinate channels;
5. the corrected kinetic temperature `T = 2K/(N_dof k_B)`;
6. exact frictionless BAOAB reduction to velocity Verlet;
7. stationary OU kinetic variance `m <v^2> = k_B T`;
8. harmonic-oscillator equipartition with the legacy mass/spring unit conventions.

## Full regression suite

The complete test suite passed:

`259 passed in 9.99 s`

The deterministic D0a 4x4 propagator gate also passed. Representative phase-aligned errors were:

- RK4: `5.18164e-9`;
- RKF7(8): `1.34011e-15`;
- Lanczos m=8: `2.63029e-15`;
- frozen CFM4 m=8: `9.27687e-15`.

All six strict 4x4 S0 relaxation gates converged:

- singlet / onsite;
- singlet / bond_x;
- singlet / bond_y;
- triplet / onsite;
- triplet / bond_x;
- triplet / bond_y.

The largest reported final structural gradient among these six branches remained below `3.0e-7 eV/angstrom`, and all neutral/excited orbital convergence flags were true.

## D4a decision

D4a is accepted as numerically validated. The BAOAB + exact Ornstein-Uhlenbeck thermostat primitives, persistent caller-owned RNG convention, corrected kinetic-temperature diagnostic, and frictionless reduction are cleared for use in D4b.

D4b must still validate the coupled electronic-lattice thermostat ordering, zero-friction reductions to D2/D3, electronic norm preservation, finite-temperature statistics, treatment of the two intermolecular collective zero modes, timestep convergence, and a multi-picosecond 300 K stability gate before transport observables are interpreted.
