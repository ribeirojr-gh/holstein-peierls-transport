# v0.7.0a1 — validated static exciton milestone

## Scope

Version `0.7.0a1` closes the generic static-physics phase of the modernization
project. The repository now contains validated stationary solvers for:

1. one Holstein-Peierls polaron;
2. one correlated singlet bipolaron with short- and long-range repulsion; and
3. one distinguishable electron-hole exciton with attractive short- and
   long-range interactions.

This release remains deliberately generic. It does not assign the exciton,
bipolaron, dielectric response, HOMO/LUMO transfer integrals, or electron-phonon
couplings to a specific molecular crystal.

## Static exciton model

The exciton wavefunction is an ordered distinguishable-particle amplitude

`Psi[i_e, i_h]`,

normalized to one. For a fixed lattice,

`H_X = H_e(q) tensor I + I tensor H_h(q) + V_eh`.

Electron and hole hopping, Holstein, and Peierls parameters are independent in
the API. The two carriers share the classical lattice fields `u`, `vx`, and
`vy`. The pair Hamiltonian is applied matrix-free rather than assembling an
`N^2 x N^2` dense matrix.

The interaction layer supports positive user-facing attraction magnitudes for:

- onsite attraction;
- cardinal nearest-neighbour attraction;
- optional minimum-image screened `1/r` attraction; and
- explicit shell-specific replacements of selected continuum values.

As in the validated bipolaron long-range model, distances are frozen to the
equilibrium molecular lattice in this milestone, so there is no explicit
Coulomb contribution to the lattice-coordinate derivatives.

## Relaxation and observables

The shared lattice is relaxed with RPROP using Hellmann-Feynman forces from both
carrier reduced density matrices. Strict promotion requires both:

- maximum coordinate update below `1e-8 angstrom`; and
- maximum structural gradient below `1e-6 eV/angstrom`.

The solver exposes electron and hole densities, separate IPRs, onsite pair
probability, mean/RMS electron-hole separation, total energy, and the binding
convention

`E_bind^X = E_e-pol + E_h-pol - E_X`,

where positive binding denotes a bound relaxed exciton relative to separately
relaxed electron and hole polarons.

Five initialization topologies are retained: `frenkel`, `ct_x`, `ct_y`,
`diagonal`, and `separated`. Near-degenerate relaxed minima are grouped with an
energy tolerance of `1e-10 eV` so floating-point noise does not relabel the
physical basin.

## Validation gates

The v0.7 exciton implementation is covered by tests for:

- the noninteracting Kronecker-sum energy;
- unit-trace electron and hole reduced density matrices;
- atomic Frenkel and nearest-neighbour charge-transfer limits;
- equal-density symmetry for the equal-carrier control;
- short-/long-range interaction replacement semantics;
- seed topology;
- finite-difference validation of `u`, `vx`, and `vy` gradients;
- strict stationary relaxation;
- positive-binding sign convention; and
- safe branch promotion requiring convergence.

The final pre-merge suite passed with `154` tests.

## Generic reference benchmark

The reference benchmark uses the same generic Holstein-Peierls scale already
used to validate the one-polaron framework:

- `Jx = 0.100 eV`;
- `Jy = 0.015 eV`;
- `alpha_1 = 3.0 eV/angstrom`;
- `alpha_2x = alpha_2y = 0.4 eV/angstrom`;
- `K1 = 16.51 eV/angstrom^2`;
- `K2 = 0.51 eV/angstrom^2`; and
- onsite electron-hole attraction `0.525 eV` as a numerical control scale.

The `0.525 eV` attraction is not a fitted exciton parameter.

Strict benchmarks were performed on `6x6`, `10x10`, and `20x20` periodic cells.
For the final `20x20` control, the Frenkel, CT-x, CT-y, and separated seeds
relax to the same minimum within approximately `1e-14 eV`. The diagonal seed
converges to a distinct higher-energy metastable exciton.

For the common lowest-energy `20x20` minimum:

- `E_X = -1.695788941148 eV`;
- `E_bind = 0.884870826 eV`;
- onsite probability `= 0.86043`;
- electron IPR `= 0.85456`;
- hole IPR `= 0.85456`; and
- mean electron-hole separation `= 0.14360` lattice sites.

The diagonal branch lies approximately `0.41327 eV` above that minimum and is
substantially less localized.

These values characterize the generic validation Hamiltonian only and are not
predictions for pentacene or another molecular semiconductor.

## Reproducible benchmark

The permanent benchmark driver is

`experiments/exciton_reference_branch_benchmark.py`.

The model definition and detailed validation record are in

- `docs/static-exciton-reference-model.md`; and
- `docs/static-exciton-validation-results.md`.

## What is intentionally not in v0.7

This release does not introduce:

- time propagation;
- electric-field transport;
- a new finite-temperature model;
- a choice among RK4, RK8, Krylov, exponential, split-operator, or other
  dynamical propagators;
- a production strategy for avoiding or reducing repeated Hamiltonian
  diagonalization;
- material-specific electron/hole parameterization;
- triplet/exchange excitons;
- explicit Coulomb-lattice force terms; or
- GPU acceleration.

These are design questions for the next phase and should be benchmarked against
the archived dynamics before a production implementation is selected.

## Next milestone

The next milestone begins with a structured audit of the archived dynamic code,
with special attention to the dominant cost of repeated eigenvalue/eigenvector
solutions, the electronic time propagator, the classical lattice integrator,
and the finite-temperature thermostat/noise model. Candidate propagation and
thermalization strategies will be compared for accuracy, stability, physical
consistency, and wall-clock cost before one becomes the new reference dynamics.
