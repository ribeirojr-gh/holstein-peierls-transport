# Static bipolaron validation plan

## Objective

The first two-particle implementation should be validated in limits with analytic or well-established behaviour before any combined Holstein-Peierls phase diagram is interpreted physically.

## 1. Noninteracting rigid-lattice limit

Set A = alpha_x = alpha_y = 0 and U = V = 0.

The two-particle Hamiltonian is H_1 tensor I + I tensor H_1. The singlet ground-state energy must equal twice the one-particle band minimum (with the same finite periodic lattice and sign convention). The two-particle density factorizes into the product of one-particle ground states.

This tests indexing, periodic boundaries, normalization, and the matrix-free Hamiltonian action.

## 2. Atomic Holstein limit

Take J_x,J_y -> 0 and alpha_x=alpha_y=0. For a site with occupation n,

E(u,n) = (K_1/2) u^2 + A u n.

Minimization gives

u*(n) = -A n/K_1,

E_min(n) = -A^2 n^2/(2K_1).

For two separated single polarons,

E_sep = 2 E_min(1) = -A^2/K_1.

For an onsite singlet bipolaron,

E_on = E_min(2) + U = -2A^2/K_1 + U.

Therefore the onsite bipolaron binding energy in this limit is

Delta_BP = E_sep - E_on = A^2/K_1 - U.

The exact atomic-limit threshold is

U_c = A^2/K_1.

For the current example input A=3.0 eV/A and K_1=16.51 eV/A^2,

U_c approximately 0.545 eV.

The numerical solver must reproduce this threshold as J approaches zero.

This is the most important first regression test because it verifies the factor-of-two structure of the reduced density matrix and the lattice gradient for double occupation.

## 3. Finite-hopping Holstein-Hubbard limit

Set alpha_x=alpha_y=0 and vary A, K_1, U, and hopping anisotropy.

Expected qualitative behaviour from established 2D adiabatic Holstein-Hubbard studies includes:

- unbound/extended carriers at weak coupling;
- onsite singlet bipolarons at sufficiently strong Holstein coupling and weak U;
- intersite/nearest-neighbour pairs at larger U in appropriate regions;
- competing multi-site structures near phase boundaries.

The purpose is not to reproduce another publication point-by-point, because conventions differ, but to recover the same topology of physical regimes before adding Peierls coupling.

## 4. Translational degeneracy

As in the single-polaron problem, a localized bipolaron may converge at any periodically equivalent location. Regression tests must align pair densities by periodic translation before comparing configurations.

For an onsite pair, alignment can use the maximum of the total one-particle density. For intersite states, alignment should use the pair center of mass and preserve relative orientation.

## 5. Exchange symmetry

For the singlet implementation,

Psi_ij = Psi_ji.

Numerical tests must verify symmetry to tolerance and normalization

sum_ij |Psi_ij|^2 = 1.

When the triplet sector is added,

Psi_ij = -Psi_ji,
Psi_ii = 0.

These identities are exact structural tests.

## 6. Reduced density matrix

For the singlet,

gamma_mn = 2 sum_j Psi_mj Psi*_nj.

Tests must verify

Tr(gamma) = 2,

gamma = gamma^dagger,

and agreement between analytic Holstein/Peierls lattice gradients constructed from gamma and finite-difference derivatives of the total energy.

This is the two-particle analogue of the single-polaron gradient regression.

## 7. Coulomb interaction tests

For onsite-only U, the interaction expectation is

E_U = U sum_i |Psi_ii|^2.

For a general pair potential V_ij,

E_C = sum_ij V_ij |Psi_ij|^2.

Both should be tested independently on synthetic wavefunctions with known support.

## 8. Binding criterion and system-size test

A negative energy relative to the rigid lattice is not sufficient to identify a bipolaron. The bound state must be lower than the best separated two-polaron solution:

Delta_BP = E_2P^sep - E_BP^bound > 0.

The separation distribution P(r) must remain localized as the lattice size increases. At minimum, compare 20x20 and 40x40 for candidate bound states and verify that the binding energy is insensitive to the cell once the pair is well separated from its periodic images.

## 9. Peierls-only and combined limits

After the Holstein-Hubbard implementation is validated:

1. set A=0 and explore Peierls-only pairing;
2. compare with the known fact that Peierls coupling can support bound pairs in quantum lattice models, while recognizing that the mechanism in the present semiclassical model may differ;
3. restore A and map whether simultaneous Holstein and Peierls relaxation is additive, competitive, or synergistic for the pair binding energy.

The last point is the main new physical question.

## 10. Exciton atomic-limit benchmark for the later electron-hole solver

For one electron and one hole on separate sites, the independent local Holstein relaxation energy is

E_sep^latt = -A_e^2/(2K_1) - A_h^2/(2K_1).

If electron and hole occupy the same molecular site, the shared coordinate couples to A_e + A_h:

E_same^latt = -(A_e+A_h)^2/(2K_1).

The additional lattice contribution to same-site binding is therefore

Delta_X^latt = E_sep^latt - E_same^latt = A_e A_h/K_1.

Consequences:

- A_e A_h > 0: local lattice relaxation reinforces electron-hole co-localization;
- A_e A_h < 0: local lattice relaxation opposes co-localization;
- A_e = -A_h: the same-site Holstein distortion cancels exactly in this minimal model, even though the separated electron and hole each form polarons.

This analytic result should be retained as a central physical benchmark when the exciton solver is developed.
