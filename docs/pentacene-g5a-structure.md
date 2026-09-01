# G5a pentacene structural data layer

## Purpose

G1-G4 established a material-agnostic periodic molecular graph, one-particle Hamiltonian, generalized Peierls algebra, and correlated-pair interaction/observable layer. G5 begins assigning a real material, but only where the evidence is sufficiently unambiguous.

G5a therefore fixed the **293 K bulk pentacene crystal geometry** and recorded published electronic-coupling evidence without prematurely converting unsigned transport couplings into a signed tight-binding model.

> **G5b correction.** The original G5a implementation described four independent geometric hopping groups and represented each A-B diagonal by only one translational `BondFamily`. That representation was sufficient to identify the four parameter groups, but it was not a complete finite herringbone transport graph: unlike same-basis A-A or B-B bonds, an A-B family does not automatically generate the opposite displacement when the Hermitian matrix element is added. G5b therefore retains four independent hopping groups but represents the two diagonal groups by explicit forward/backward families, giving six translational bond families in total. This correction is documented in `docs/pentacene-g5b-signed-band.md`.

## Selected crystal

The initial target is the 293 K single-crystal structure of Mattheus et al., *Acta Crystallographica C* **57**, 939-941 (2001), DOI `10.1107/S010827010100703X`, CCDC 170186.

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

## Minimal in-plane parameter groups

The structural layer identified four independent nearest-neighbour hopping groups in the `a-b` herringbone plane:

- one same-basis A-A group along `a`;
- one same-basis B-B group along `a`;
- one A-B group along `+(a+b)/2` and its opposite neighbour;
- one A-B group along `+(a-b)/2` and its opposite neighbour.

After the G5b connectivity correction these four independent parameter groups are represented by six explicit translational bond families:

- `a_AA`;
- `a_BB`;
- `diag_plus_AB_forward` and `diag_plus_AB_backward`;
- `diag_minus_AB_forward` and `diag_minus_AB_backward`.

The duplication of each diagonal is topological, not a new material parameter: the forward and backward members share the same transfer integral in the undistorted inversion-symmetric reference crystal.

## Published hole-coupling evidence

Stehr et al., *Phys. Rev. B* **83**, 155208 (2011), DOI `10.1103/PhysRevB.83.155208`, report four dominant pentacene hole-coupling magnitudes

- `V1 = 90.69 meV`;
- `V2 = 55.05 meV`;
- `V3 = 39.68 meV`;
- `V4 = 36.62 meV`;

with a hole reorganization energy of `92 meV`.

The full-text discussion identifies the largest hole coupling with the `[1 -1 0]` direction and the second largest with `[1 1 0]`. G5b therefore maps the two large magnitudes to the two diagonal herringbone groups and the two lower magnitudes to the two same-basis `a`-axis contacts. The Marcus-rate calculation still does not provide the orbital-phase signs required by a coherent tight-binding Hamiltonian, so these records remain `signed_value_known=False`.

## Deliberately unresolved fields after G5a

G5a deliberately left unresolved:

- a signed hopping assignment;
- bond-resolved Peierls derivatives;
- effective bond stiffness matrices;
- screened onsite Hubbard `U`;
- screened short-range contact interactions `V_ij`;
- long-range dielectric convention;
- mapping from molecular reorganization energy to the effective Holstein coordinate.

G5b partially resolves the first item by adding a signed 90 K reference and a provenance-labelled cross-source 293 K candidate. A **single-source signed 293 K parameterization** remains unresolved, as do all interaction and electron-phonon quantities.

## Original G5a validation record

The G5a-specific local tests verified:

1. the projected `a` and `b` lengths and `gamma` angle reproduce the Mattheus 293 K cell;
2. the projected A/B basis reproduces the reported inversion-center positions;
3. the four independent geometric hopping groups have the intended crystallographic displacement vectors;
4. the Stehr values remain unsigned evidence rather than promoted model hoppings;
5. all scientifically blocking parameters remain explicitly unresolved.

Original test record at merge:

- G5a-specific tests: `5/5` passed;
- complete project suite after G5a: `76/76` passed.

The G5b test suite supersedes the transport-graph portion of item 3 by requiring the complete six-family finite graph and degree-six nearest-neighbour connectivity.

## Next research task

After G5b validates the signed noninteracting band layer, the project should proceed in this order:

1. seek or compute a single-source signed 293 K hopping set and compare it with the cross-source candidate;
2. map the `92 meV` molecular reorganization energy to a documented effective Holstein convention;
3. obtain bond-resolved Peierls derivatives together with compatible stiffness/mode information;
4. obtain internally consistent screened `U`, short-range `V_ij`, and long-range dielectric screening;
5. only then enable graph-based material-specific polaron/bipolaron relaxation.
