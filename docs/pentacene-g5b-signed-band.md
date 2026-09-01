# G5b signed pentacene band layer

## Purpose

G5a fixed the bulk pentacene molecular-center geometry but intentionally stopped before creating a signed coherent tight-binding model. G5b resolves enough of the one-particle electronic sector to validate the herringbone band topology while preserving a strict provenance distinction between a **single-source reference** and a **cross-source room-temperature candidate**.

No Holstein, Peierls, Hubbard, or screened intermolecular interaction parameter is promoted in G5b. Consequently G5b does **not** yet justify material-specific polaron or bipolaron relaxation.

## Topological correction to the G5a transport graph

The two-molecule basis is

- `A = (0,0)`;
- `B = (1/2,1/2)`

in primitive `a-b` fractional coordinates.

For same-basis contacts, one family such as `A -> A + a` is sufficient: Hermitian completion of each undirected edge gives the `+a` and `-a` neighbours of every A site. The same is true for B-B.

For an A-B contact the situation is different. A single family `A -> B + d` gives one B neighbour per A and its Hermitian reverse gives B -> A, not the second physical A -> B neighbour at `-d`. Therefore each diagonal herringbone direction requires explicit forward and backward translational representatives.

The complete nearest-neighbour graph uses six `BondFamily` objects but only four independent hopping groups:

| independent group | explicit families | displacement |
|---|---|---|
| A-A along `a` | `a_AA` | `+a` |
| B-B along `a` | `a_BB` | `+a` |
| plus diagonal | `diag_plus_AB_forward`, `diag_plus_AB_backward` | `+(a+b)/2`, `-(a+b)/2` |
| minus diagonal | `diag_minus_AB_forward`, `diag_minus_AB_backward` | `+(a-b)/2`, `-(a-b)/2` |

On a normal finite supercell every molecular site therefore has six nearest-neighbour graph edges: two same-basis neighbours along `±a` plus four opposite-basis diagonal neighbours.

This is a correction to the executable G5a graph representation, not a change in the crystallographic data or in the number of independent material hopping parameters.

## Signed 90 K reference: de Wijs et al.

G. A. de Wijs, C. C. Mattheus, R. A. de Groot, and T. T. M. Palstra, *Synthetic Metals* **139**, 109-114 (2003), DOI `10.1016/S0379-6779(03)00020-1`, calculated the pentacene band structure with DFT and fitted a molecular-orbital tight-binding model.

The calculation used the experimental 90 K Mattheus crystal. For the HOMO complex the authors identify the three dominant transfer integrals as

- `t_a = +31 meV`;
- `t_(a+b)/2 = -56 meV`;
- `t_(a-b)/2 = +91 meV`;

with an inequivalent-molecule energy parameter `e = 0.042 eV`. The remaining fitted hopping terms are much smaller.

G5b encodes this as `pentacene_90k_dewijs_homo_model()`. The same-basis value is attached to both A-A and B-B families. Each diagonal value is attached to both the forward and backward representative of that physical direction.

### Orbital-phase gauge

The individual signs of inter-sublattice transfer integrals are not absolute observables. Multiplying every B-basis HOMO by `-1` changes the sign of every A-B hopping while leaving all eigenvalues unchanged. G5b therefore records the de Wijs sign convention as one explicit gauge and tests that simultaneously reversing both diagonal A-B signs produces exactly the same band eigenvalues.

The physically relevant information is the relative sign/frustration pattern after a gauge has been fixed, together with the same-basis hopping sign.

## Physical graph regression: HOMO bandwidth

The signed de Wijs reference provides a useful test that goes beyond source-code equivalence. With the complete six-family herringbone connectivity, the reduced dominant-hopping model gives a full two-band HOMO-complex width

`W_HOMO = 0.5894980916 eV`

on an even commensurate reciprocal mesh containing the `M=(1/2,1/2)` point. Both extrema occur at M for this reduced fit.

De Wijs et al. report a DFT HOMO bandwidth of approximately `0.6 eV`, and their dominant three hopping terms are stated to reproduce the band structure closely. The `0.5895 eV` result is therefore adopted as a **physical regression of the graph connectivity and sign implementation**.

The earlier incomplete G5a A-B graph produced only roughly `0.324 eV` with the same dominant parameters. The bandwidth discrepancy exposed the missing opposite A-B neighbours and motivated the G5b topology correction.

## 293 K coupling magnitudes: Stehr et al.

V. Stehr, J. Pfister, R. F. Fink, B. Engels, and C. Deibel, *Physical Review B* **83**, 155208 (2011), DOI `10.1103/PhysRevB.83.155208`, report the four dominant hole couplings

- `V1 = 90.69 meV`;
- `V2 = 55.05 meV`;
- `V3 = 39.68 meV`;
- `V4 = 36.62 meV`;

and a hole reorganization energy `lambda = 92 meV`.

Their discussion identifies the highest hole coupling with the `[1 -1 0]` direction and the second highest with `[1 1 0]`. In the G5 molecular-center graph this maps

- `90.69 meV -> (a-b)/2` diagonal group;
- `55.05 meV -> (a+b)/2` diagonal group;
- `39.68` and `36.62 meV ->` the two inequivalent same-basis `a`-axis groups.

The Marcus hopping-rate treatment depends on coupling magnitudes, so this source is retained as unsigned evidence rather than being interpreted as an independent sign determination.

## Cross-source 293 K signed candidate

For band validation only, G5b defines `pentacene_293k_cross_source_hole_model()` by combining

1. the 293 K coupling **magnitudes and direction identities** from Stehr et al.; and
2. the gauge-fixed **relative HOMO sign pattern** from the de Wijs band fit.

The resulting candidate is

| family group | Hamiltonian hopping |
|---|---:|
| A-A along `a` | `+39.68 meV` |
| B-B along `a` | `+36.62 meV` |
| `(a+b)/2` A-B, both directions | `-55.05 meV` |
| `(a-b)/2` A-B, both directions | `+90.69 meV` |

The assignment of `39.68` versus `36.62 meV` to the names A-A and B-B is a basis-label convention when the two onsite energies are equal. G5b tests that exchanging those two labels leaves the two-band spectrum invariant.

A dense reciprocal-space inspection of this candidate gives a HOMO-complex width of about `0.585 eV`; a reproducible `6x6` commensurate mesh gives `0.5829921234 eV`. This closeness to the low-temperature signed reference is encouraging, but it is **not** used to promote the mixed-source candidate as a final material parameter set.

## Why the 293 K candidate is not yet the final electronic model

The cross-source candidate mixes two electronic-structure protocols and two temperatures:

- de Wijs: signed DFT/TB fit on the 90 K structure;
- Stehr: first-principles Marcus-coupling magnitudes used for room-temperature mobility calculations on the bulk structure.

The relative sign topology is physically plausible and independently consistent with later tight-binding descriptions, but a final 293 K electronic parameterization should preferably come from one signed calculation on the selected 293 K geometry, or from a reproducible calculation performed within this project.

Therefore `single_source_signed_hopping_parameterization` remains explicitly unresolved.

## G5b validation gates

The G5b tests require:

1. the 293 K transport graph contains all six translational families and every site has degree six;
2. all four diagonal displacement signs equal the analytic `±(a+b)/2` and `±(a-b)/2` vectors;
3. the Stehr V1/V2 directions map to the two correct diagonal groups while all Stehr values remain flagged unsigned;
4. the de Wijs signed 90 K model reproduces `W_HOMO = 0.5894980916 eV` and lies within 20 meV of the reported approximately 0.6 eV width;
5. simultaneous reversal of both A-B hopping signs is spectrally gauge-equivalent;
6. the 293 K cross-source candidate contains the correct Stehr magnitudes and de Wijs relative signs;
7. swapping the two same-basis 293 K magnitudes is spectrally invariant for equal A/B onsite energies;
8. interaction and electron-phonon material parameters remain unresolved.

## Next stage

G5b completes the first defensible noninteracting band layer. The next material work should not yet be a bipolaron phase diagram.

Recommended sequence:

1. obtain or compute a single-source signed 293 K hopping set and compare its bands with the present candidate;
2. map the `92 meV` intramolecular reorganization energy onto an explicit Holstein `alpha_intra/K1` convention;
3. obtain bond-resolved nonlocal electron-phonon derivatives together with compatible effective stiffness information;
4. establish screened onsite `U`, contact-specific `V_ij`, and a long-range dielectric convention without double counting;
5. only after those gates, connect G1-G4 to a material-specific graph-based static polaron/bipolaron relaxer.
