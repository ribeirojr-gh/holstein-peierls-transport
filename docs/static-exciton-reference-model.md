# Static exciton reference model

## Scope

This milestone extends the generic Holstein-Peierls framework from one carrier
(polaron) and two equal carriers (singlet bipolaron) to one distinguishable
electron-hole pair.  It deliberately does **not** assign the model to a specific
material.  Material-specific FHI-aims/FO-DFT infrastructure remains available
under the material layer but is not required by this solver.

The initial control model reuses the validated reference parameter scale already
used throughout the project.  Electron and hole parameters remain independent
in the API so later work is not constrained to equal conduction- and
valence-sector hopping or electron-phonon couplings.

## Electronic Hilbert space

The real ordered-pair wavefunction is

`Psi[i_e, i_h]`,

with normalization

`sum_ij |Psi[i,j]|^2 = 1`.

No particle-exchange projection is applied: electron and hole are
distinguishable.  Their reduced density matrices are

`gamma_e = Psi Psi^T`,

`gamma_h = Psi^T Psi`,

and both have trace one.

For a fixed lattice the Hamiltonian is

`H_X = H_e(q) tensor I + I tensor H_h(q) + V_eh`,

applied matrix-free without building an `N^2 x N^2` sparse matrix.

## Holstein-Peierls lattice coupling

Electron and hole share the classical lattice fields `u`, `vx`, and `vy`, but
have separate hopping and coupling parameters.  For carrier `c=e,h`,

`t_x^c = -J0x^c + alpha_x^c [vx(i+x)-vx(i)]`,

`t_y^c = -J0y^c + alpha_y^c [vy(i+y)-vy(i)]`,

and the onsite Holstein term is `alpha_1^c u_i`.

The shared-lattice Hellmann-Feynman force uses both reduced density matrices.
For example,

`dE/du_i = K1 u_i + alpha_1^e n_i^e + alpha_1^h n_i^h`.

The Peierls gradients are the corresponding sum of the electron and hole bond
orders.  The attractive interaction is frozen with respect to molecular
coordinates in this reference milestone, matching the frozen-distance
convention already used for the static correlated Coulomb sector.

## Electron-hole attraction

User-facing interaction parameters are positive attraction magnitudes; the
Hamiltonian applies them with a negative sign.

Supported controls are:

- onsite attraction (`onsite_attraction`);
- cardinal nearest-neighbour attraction;
- optional minimum-image `-1/r` attraction with explicit dielectric constant
  and lattice spacings;
- shell-specific short-range replacements of that continuum tail.

Short-range replacements follow the same no-double-counting rule used in the
bipolaron interaction infrastructure: selected shell values replace the
continuum value rather than being added to it.

The first numerical benchmark uses `0.525 eV` as an onsite attraction magnitude
only because the same energy scale was already exercised in the correlated
bipolaron validation.  It is a control value, not a fitted exciton parameter.

## Competing relaxed branches

Five physically distinct seeds are retained:

1. `frenkel`: electron and hole initially on the same site;
2. `ct_x`: nearest-neighbour charge-transfer seed along x;
3. `ct_y`: nearest-neighbour charge-transfer seed along y;
4. `diagonal`: first diagonal charge-transfer seed;
5. `separated`: maximally separated finite-cell seed.

All requested branches are relaxed independently.  Only converged branches are
eligible for promotion; the lowest total energy among them is selected.

## Observables

The reference layer exposes:

- electron density and hole density;
- electron and hole IPR;
- onsite electron-hole probability;
- mean and RMS electron-hole separation in lattice-site units;
- physical separation in angstrom when a physical lattice geometry is supplied;
- exciton total energy;
- binding convention

  `E_bind^X = E_e-pol + E_h-pol - E_X`,

  so positive binding means a bound relaxed exciton relative to two separately
  relaxed polarons.

## Validation gates

The milestone is accepted only if:

- the noninteracting pair energy equals the sum of the two one-particle band
  minima;
- `Tr gamma_e = Tr gamma_h = 1`;
- strong atomic onsite attraction yields unit onsite probability;
- strong atomic nearest-neighbour attraction yields a charge-transfer pair at
  one-site separation;
- the equal-carrier control gives equal electron and hole densities;
- analytic Hellmann-Feynman gradients for `u`, `vx`, and `vy` agree with central
  finite differences;
- a full reference Holstein-Peierls relaxation satisfies both the coordinate
  update and structural-gradient convergence criteria;
- competing branch selection never promotes a non-converged state.

## Deferred work

This static milestone does not yet introduce time propagation, electric fields,
carrier-specific masses, material-specific HOMO/LUMO parameters, triplet
excitons, exchange integrals, spin-orbit coupling, explicit Coulomb lattice
forces, GPU acceleration, or parameter-space phase diagrams.

After the static exciton is validated, the next major stage is a unified dynamics
layer for polaron, bipolaron, and exciton based on the legacy Fortran dynamics
and modernized for parallel CPU/GPU execution.
