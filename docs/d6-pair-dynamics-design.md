# D6 pair-dynamics design

## Scope

D6 ports the validated one-particle matrix-free propagation architecture to the existing correlated two-particle sectors:

1. the symmetric spatial (spin-singlet) bipolaron in `holstein_peierls.two_particle`; and
2. the distinguishable electron-hole reference exciton in `holstein_peierls.exciton`.

The existing distinguishable exciton backend is intentionally spin-blind and contains no electron-hole exchange. D6 must therefore never relabel it as a physical singlet/triplet exciton. Spin-adapted excited-state dynamics remain a separate backend problem.

## Why D6 starts with a frozen lattice

The pair Hilbert space has dimension `N^2`. Complete diagonalization is therefore not a production option beyond small validation lattices. Before pair forces, thermostats, fields, or decoherence are coupled in, the electronic propagator must be validated against exact small-system references.

D6 is split into the following gates.

### D6a — frozen-H complex propagation

Use fixed lattice coordinates and compare matrix-free propagation against an exact dense spectral exponential on small lattices.

Mandatory checks:

- complex amplitudes are preserved by the matrix-free Hamiltonian action;
- the action agrees with the existing static real-valued operators on real vectors;
- the small dense matrix assembled from the action is Hermitian;
- short-Lanczos propagation agrees with the exact spectral reference;
- electronic norm and propagated-state energy remain controlled;
- bipolaron exchange symmetry is preserved;
- bipolaron one-body RDM trace remains two;
- distinguishable exciton electron and hole RDM traces remain one each;
- no accidental exchange symmetrization is applied to the distinguishable e-h state.

The D6a reference includes the full Holstein-Peierls hopping modulation already implemented in the static bipolaron and exciton sectors. Interaction matrices remain lattice-geometry independent exactly as in the validated static models.

### D6b — zero-temperature coupled pair-lattice dynamics

Only after D6a closes, introduce moving classical coordinates and propagated-state Hellmann-Feynman forces.

Required reductions and gates:

- zero pair interaction must reduce to the corresponding independent-carrier limit;
- bipolaron forces must use the spin-summed one-body RDM with trace two;
- exciton forces must use separate electron and hole RDMs, each with trace one;
- total energy at zero field and zero thermostat must remain controlled;
- electronic norm and bipolaron exchange symmetry must remain controlled;
- timestep convergence must be demonstrated before long trajectories.

### D6c — finite-temperature pair dynamics

After D6b, port the validated D4 BAOAB lattice bath. The same explicit treatment of the two intermolecular collective zero modes is required. Electronic thermalization/decoherence for the pair sectors must not be assumed to be identical to the one-particle D5 model without a new diagnostic gate.

### D6d — field-driven pair dynamics and observables

Only after D6b/D6c are stable should Peierls phases and external work accounting be introduced. Relevant observables include pair separation, onsite/nearest-neighbour/CT channel probabilities, one-body occupations, center-of-charge observables, and the already validated projector-based channel yields.

No steady-state mobility, dissociation yield, or experimental quantum yield is claimed merely from reaching D6d.

## Complex-valued dynamics versus real static solvers

The current static pair solvers use real symmetric Hamiltonians and `LinearOperator(dtype=float)`, which is appropriate for adiabatic ground-state eigensolvers. Time propagation, however, necessarily generates complex amplitudes. D6 therefore introduces a dedicated complex-safe matrix-free Hamiltonian action rather than silently passing complex states through the real static `LinearOperator` implementation.

For real vectors, the new action must agree with the existing static operator to roundoff. This cross-check fixes sign, basis ordering, interaction, and periodic-boundary conventions.

## Production propagator

D6 initially reuses the D0/D1/D2 decision: short Hermitian Lanczos exponential action is the production candidate for linear frozen-H pair propagation, with dense spectral propagation retained only as a small-system reference. The Krylov dimension is revalidated in the pair Hilbert space rather than inherited blindly from the one-particle benchmark.

GPU/thread optimization remains downstream. The first goal is to expose and validate the actual `N^2` matrix-free Hamiltonian-action kernel.
