# S1 — spin-adapted RPROP bridge preregistration

Date: 2026-09-26  
Status: prospective numerical validation for Paper 1 static workflow.

## Purpose

Paper 1 is intended to present a static extended Holstein–Peierls model in which the structural relaxation methodology is as uniform as possible across one-polaron, bipolaron, direct electron-hole and spin-adapted singlet/triplet sectors.

The promoted spin-adapted S0 benchmark currently uses a harmonic-preconditioned Newton/Armijo structural optimizer. A generic component-wise RPROP path exists but has not been validated on the same checkerboard-gapped, recovery-aware canonical control.

S1 therefore asks a narrow numerical question:

> Can a gapped, recovery-aware non-backtracking RPROP structural optimizer reproduce the promoted stationary singlet/triplet minima of the accepted S0 control without relaxing any electronic or structural convergence gate?

S1 does not change the spin-adapted Hamiltonian, interaction kernel, spin definitions, orbital optimizer, neutral-reference force convention, or checkerboard validation control.

## Frozen physical control

Use exactly the canonical S0 benchmark:

- 4x4 periodic square lattice;
- `Jx = Jy = 0.100 eV`;
- `alpha_intra = alpha_interx = alpha_intery = 3.0 eV/A`;
- `K1 = 16.51 eV/A^2`;
- `K2 = 0.51 eV/A^2`;
- density-density validation interactions `U = 0.525 eV`, nearest-neighbour `V = 0.08 eV`;
- checkerboard one-particle gap `Delta = 2.0 eV`;
- half-filled neutral reference;
- open-shell singlet and high-spin triplet;
- seeds `onsite`, `bond_x`, `bond_y`;
- seed amplitude `1e-3 A`.

The checkerboard gap remains a numerical conditioning control, not a material band gap.

## Electronic solver

At every accepted lattice geometry:

- neutral and excited state-specific orbital optimizations must satisfy the existing orbital-gradient tolerance `1e-8`;
- the same deterministic warm-start/cold-recovery logic as the promoted preconditioned S0 benchmark must be used;
- an electronically unconverged geometry must not be accepted as a converged structural state.

No change to spin coefficients, exchange conventions or orbital degrees of freedom is permitted in S1.

## RPROP structural rule

Use a non-backtracking component-wise RPROP rule consistent with the validated two-particle/exciton relaxation convention:

- initial step `update_start = 1e-3 A`;
- maximum step `update_max = 1e-2 A`;
- minimum step `update_min = 2^-52 A`;
- acceleration factor `1.2`;
- deceleration factor `0.5`;
- when the gradient product is positive, increase the local step and move opposite the current gradient;
- when the gradient product is negative, reduce the local step and make **no coordinate move** on that component for that iteration;
- otherwise use the current local step.

After each RPROP coordinate update, fix the Peierls translational gauge by subtracting:
- each row mean from `vx`;
- each column mean from `vy`.

This removes only exact harmonic zero modes and makes comparison with the promoted preconditioned benchmark unambiguous.

## Structural convergence

Retain the promoted S0 gates unchanged:

- true last maximum coordinate update < `1e-8 A`;
- final maximum structural gradient < `1e-6 eV/A`;
- neutral electronic state converged;
- excited electronic state converged.

Maximum lattice iterations: **2000**.

S1 must not declare convergence merely because RPROP step sizes have collapsed if the structural-gradient gate fails.

## Reference comparison

Recompute the accepted harmonic-preconditioned branches in the same GitHub Actions run. This avoids comparing different software environments or commits.

For each optimizer, run all six combinations:

`singlet/triplet x onsite/bond_x/bond_y`.

The primary cross-optimizer comparison is performed on the **lowest strictly converged branch within each multiplicity**, not by assuming that the same seed must select the same local open-shell root.

Define:
- `E_S^R`, `E_T^R`: lowest converged RPROP energies;
- `E_S^P`, `E_T^P`: lowest converged preconditioned energies;
- `Delta_ST = E_S - E_T`.

## Locked S1 acceptance criteria

S1 passes only if all are true:

1. all six RPROP branches satisfy the strict electronic and structural convergence gates;
2. all six preconditioned reference branches satisfy the same gates;
3. `|E_S^R - E_S^P| <= 1e-5 eV`;
4. `|E_T^R - E_T^P| <= 1e-5 eV`;
5. `|Delta_ST^R - Delta_ST^P| <= 1e-5 eV`;
6. `|Tr(Delta gamma)| < 1e-10` for every promoted RPROP branch;
7. all final lattice arrays and energies are finite;
8. complete repository pytest suite passes.

The `1e-5 eV` cross-optimizer tolerance is a numerical stationary-state equivalence gate for this 4x4 regression control. It is not an uncertainty estimate for a physical singlet-triplet gap.

## Secondary diagnostics

Report without using them to rescue a failed primary gate:

- branch-by-branch energy differences between optimizers;
- iteration counts;
- final update and gradient;
- maximum/rms lattice difference for the promoted lowest states after the fixed zero-mode gauge;
- excitation-density norms;
- seed/root ordering changes.

A difference in which seed reaches the lowest branch is acceptable if the promoted stationary energies agree under the primary gates. It must nevertheless be reported.

## Stop rule

If S1 passes, RPROP becomes an explicitly validated structural option for the canonical spin-adapted static control and may be used for Paper-1 stationary S/T results.

If S1 fails, do not loosen tolerances after seeing results. Keep the preconditioned solver as the validated spin-adapted method and either:
- describe the optimizer difference transparently in Paper 1, or
- preregister a distinct numerical investigation of the RPROP failure mechanism.

No Paper-1 production spin scan begins until S1 is closed.
