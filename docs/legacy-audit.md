# Legacy Code Audit

This document records observed behavior in the archived Fortran workflow. It is intentionally descriptive: legacy behavior must first be reproduced and tested before any scientific or numerical correction is introduced.

## Workflow confirmed from `scritp.sh`

For each electric-field value the script:

1. creates a dedicated `ef<field>` directory;
2. copies the Fortran sources;
3. generates `parameters1.inc` for the static RPROP stage;
4. generates `parameters.inc` for the dynamics/post-processing stage;
5. sets `MKL_NUM_THREADS=4` and `OMP_NUM_THREADS=4`;
6. compiles `rprop.f90` and `hp2D.f90` with Intel Fortran and parallel MKL;
7. runs `rprop -> hp2D -> graphics -> velocities`.

The field sweep starts at 0.2, increments by 0.4, and contains ten calculations, ending at 3.8.

## `rprop.f90`

Role: static optimization of the lattice in the presence of one excess charge.

Important points already identified for regression testing:

- The electronic Hamiltonian is diagonalized densely.
- The full electronic density matrix is constructed even though the gradient uses only diagonal and nearest-neighbour elements.
- Only the lowest electronic eigenvector is required by the static calculation.
- The main RPROP loop termination condition explicitly checks the `u` convergence flag; convergence flags for the intermolecular coordinates are computed separately and should be audited against actual legacy runs.
- RPROP sign-change branches and their relationship to the stored previous displacement should be reproduced exactly in a `legacy` update mode before considering a corrected implementation.

## `hp2D.f90`

Role: coupled electronic/lattice dynamics initialized from the RPROP geometry.

Observed algorithmic structure:

- Reads `outU.dat`, `outVX.dat`, and `outVY.dat` from the static calculation.
- Initializes the electronic wavefunction from the lowest eigenstate of the instantaneous Hamiltonian.
- Uses a complex Hermitian Hamiltonian when an electric field is present.
- Introduces the electric field through phase factors multiplying intermolecular transfer integrals.
- Propagates the electronic wavefunction in the instantaneous eigenbasis.
- Propagates classical lattice coordinates using alternating half-kick/drift operations with optional damping and stochastic forces.
- Uses Intel MKL VSL Gaussian random numbers for the stochastic terms.
- The archived field-sweep input has `Temp = 0`, damping constants set to zero, and `rfOFF = 1`, so stochastic forces are disabled in that workflow.
- Writes charge density, lattice coordinates, and lattice velocities through Fortran logical units 10–16, plus energy components through logical units 17–21.

### Dormant/reporting issues to preserve in the audit

The program body does not use `implicit none`. In the simulation summary it prints identifiers named `a2` and `E0x`, whereas the generated input defines `a2x`, `a2y`, and `E0`. These summary-only identifiers therefore require verification; they are not the variables used in the Hamiltonian construction.

The damping terms in some intermolecular derivative expressions index `vxdt(i)` or `vydt(i)` while the force is being evaluated for flattened site index `k`. Because the archived field-sweep inputs set the corresponding damping coefficient to zero, this does not affect those zero-damping runs, but it must be treated explicitly before implementing finite damping/temperature dynamics.

## `velocities.f90`

Role: derive transport observables from time-dependent charge density.

Observed behavior:

- Reads charge-density blocks from Fortran logical unit 13 output.
- Reconstructs a periodic center coordinate using sine/cosine moments.
- Applies a manual trajectory unwrapping rule.
- Converts the mean site velocity to a physical velocity using a hard-coded 3.5 Å spacing.
- Computes a time-dependent inverse participation ratio and a late-time average.

The position estimator operates on the flattened `nxy` site index rather than explicitly on `(x, y)` coordinates. Its exact interpretation for the two-dimensional periodic lattice must therefore be validated against known legacy trajectories before it is ported. The modern implementation should ultimately expose explicit `x` and `y` center-of-charge coordinates and a direction-resolved drift velocity.

## `graphics.f90`

Role: legacy plotting/post-processing.

This functionality should not be coupled to the numerical solver in the Python implementation. Simulation output will be structured independently and plotting will be handled by a separate Matplotlib-based module.

## `mkl_vsl.f90`

Role: Intel MKL VSL interface used by the stochastic dynamics.

The new implementation should depend on maintained Python numerical libraries rather than redistribute this interface file. The original file remains only in the private immutable legacy archive for provenance.

## Porting rule

Any behavior marked above as questionable is a **legacy observation**, not an automatic bug fix. The modernization sequence is:

1. reproduce relevant legacy behavior;
2. establish numerical regression tests;
3. isolate questionable behavior behind explicit tests/options;
4. introduce a documented correction in a separate commit;
5. quantify its impact on published observables.
