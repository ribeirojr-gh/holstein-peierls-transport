# G5a pentacene structural data layer

## Purpose

G1-G4 established a material-agnostic periodic molecular graph, one-particle Hamiltonian, generalized Peierls algebra, and correlated-pair interaction/observable layer. G5 begins assigning a real material, but only where the evidence is sufficiently unambiguous.

G5a therefore fixes the **293 K bulk pentacene crystal geometry** and records published electronic-coupling evidence without prematurely converting unsigned transport couplings into a signed tight-binding model.

## Selected crystal

The initial target is the 293 K single-crystal structure of Mattheus et al., Acta Crystallographica C 57, 939-941 (2001), DOI `10.1107/S010827010100703X`, CCDC 170186.

The reported triclinic cell is

- `a = 6.266 angstrom`;
- `b = 7.775 angstrom`;
- `c = 14.530 angstrom`;
- `alpha = 76.475 deg`;
- `beta = 87.682 deg`;
- `gamma = 84.684 deg`;
- `Z = 2`.

For the two-dimensional conducting-plane model, G5a projects the crystallographic `a-b` plane into Cartesian coordinates as

`a1 = (a, 0)`

and

`a2 = (b cos(gamma), b sin(gamma))`.

The crystallographic report places inversion centers on the two molecules at fractional coordinates `(0,0,0)` and `(1/2,1/2,0)`. G5a therefore uses projected molecular basis positions

- `A = (0,0)`;
- `B = (1/2,1/2)`.

This is a molecular-center lattice representation, not an atomistic reconstruction of each pentacene molecule.

## Minimal in-plane transport graph

The structural layer exposes a candidate four-family herringbone graph:

- `a_AA`: A to A translated by `(1,0)`;
- `a_BB`: B to B translated by `(1,0)`;
- `diag_plus_AB`: A to B in the same primitive cell, displacement `(a+b)/2`;
- `diag_minus_AB`: A to B in the neighboring `-a` cell, displacement `(-a+b)/2`.

These are geometric labels only. They represent the two inequivalent same-orientation paths along the crystallographic a direction and the two inequivalent A-B diagonal directions in the a-b plane.

They are deliberately **not** yet labeled A, A-prime, B, or C from any particular electronic-structure paper.

## Published hole-coupling evidence

Stehr et al., Phys. Rev. B 83, 155208 (2011), DOI `10.1103/PhysRevB.83.155208`, report four dominant pentacene hole-coupling magnitudes

- `90.69 meV`;
- `55.05 meV`;
- `39.68 meV`;
- `36.62 meV`;

with a hole reorganization energy of `92 meV`.

Comparative literature identifies the two lower-magnitude paths with the two inequivalent a-axis contacts and the two higher-magnitude paths with the two diagonal herringbone contacts. However, the Marcus transport calculation uses coupling magnitudes; the evidence currently stored in G5a does not establish a signed one-to-one tight-binding assignment to `a_AA`, `a_BB`, `diag_plus_AB`, and `diag_minus_AB` at the level required for band-structure validation.

Consequently G5a stores these values as `TransferIntegralEvidence` records with `signed_value_known=False` and only a coarse geometry group (`a_axis_same_basis` or `diagonal_AB`).

## Deliberately unresolved fields

The material module keeps the following quantities explicitly unresolved:

- signed hopping assignment;
- bond-resolved Peierls derivatives;
- effective bond stiffness matrices;
- screened onsite Hubbard U;
- screened short-range contact interactions V_ij;
- long-range dielectric convention;
- mapping from molecular reorganization energy to the effective Holstein coordinate.

This prevents an incomplete record from being silently promoted into a runnable "pentacene" bipolaron parameter set.

## Validation

G5a-specific local tests verify:

1. the projected a and b lengths and gamma angle reproduce the Mattheus 293 K cell;
2. the projected A/B basis reproduces the reported inversion-center positions;
3. the four candidate geometric bond families have the intended crystallographic displacement vectors;
4. the Stehr values remain unsigned evidence rather than model hoppings;
5. all scientifically blocking parameters remain explicitly unresolved.

Test record:

- G5a-specific tests: `5/5` passed;
- complete project suite after G5a: `76/76` passed.

## Next research task

G5b should resolve the **signed hopping assignment** before any pentacene band structure is claimed. Preferred evidence, in order, is:

1. signed transfer integrals computed on the same Mattheus 293 K geometry with explicit pair definitions;
2. an original electronic-structure source whose figures/tables map signs and molecular pairs unambiguously onto the four G5a geometric families;
3. if neither is available, a small reproducible electronic-structure calculation rather than an inferred sign assignment from Marcus magnitudes.

Only after G5b passes a noninteracting band/symmetry check should the project begin mapping Peierls derivatives and elastic modes for material-specific relaxation.
