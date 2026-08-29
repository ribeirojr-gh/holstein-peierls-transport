# Static solver validation record

This record applies to version `0.1.0a1` of the static polaron solver. The time-dependent code is outside this milestone.

## Automated checks

The local test suite verifies parameter parsing, periodic Hamiltonian matrix elements, analytical gradients against finite differences, agreement among dense and sparse electronic solvers, deterministic RPROP behaviour, and a converged cross-language reference case.

## Cross-language reference case

The archived `rprop.f90` source was compiled with gfortran and system LAPACK using typed equivalents of the generated include constants. For a 4 x 4 lattice with the archived physical/RPROP parameters, polaron position 6, maximum 2000 iterations, and convergence criterion 1e-8, the archived algorithm gives:

- polaron formation energy: `0.5864295596422433 eV`;
- maximum molecular charge: `0.49634800043158167`.

The Python `dense_full` solver reproduces the formation energy to machine precision and the maximum charge within approximately 2e-8 in the tested environment.

## Floating-point note

RPROP depends only on derivative signs. At symmetric sites, derivatives that are mathematically zero can appear numerically at approximately 1e-18. Different LAPACK/compiler stacks may therefore take symmetry-equivalent intermediate paths. Regression should compare invariant physical observables rather than require bitwise equality of every intermediate coordinate across platforms.

## 20 x 20 reference calculation

Using the `parameters1.inc` reconstructed from the archived shell script and `dense_lowest`:

- legacy u-only stopping rule: 408 iterations, total energy `-0.4054590577418 eV`, formation energy `0.6354590577418 eV`; `vx` and `vy` are still flagged as not converged;
- modern all-coordinate stopping rule: 547 iterations, total energy `-0.4054590577561 eV`, formation energy `0.6354590577561 eV`, conventional IPR `0.4477810858560`, with all three convergence flags true.

The energy difference is only about 1.4e-11 eV for this case, but requiring all three lattice fields to converge is scientifically clearer and is therefore the modern default. The historical rule remains available through `--legacy-convergence` for regression.
