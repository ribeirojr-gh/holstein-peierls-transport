# Pentacene material-parameterization audit

## Purpose

This document defines the evidence threshold required before the generic
Holstein-Peierls-Hubbard model is described as a **material-specific pentacene
model**. No pentacene phase boundary or transport result is promoted at this
stage.

The central conclusion of the audit is structural: the present `v0.6.0a1`
rectangular one-molecule-per-site lattice is a useful generic two-dimensional
molecular-crystal model, but it is not a literal representation of bulk
pentacene. A crystal-specific parameterization therefore requires a generalized
periodic molecular lattice before numerical pentacene phase diagrams are
scientifically defensible.

## Legacy-model status

The current code descends from the semiclassical two-dimensional
Holstein-Peierls framework used in the Ribeiro Junior/Stafstrom studies of
polaron stability, electron-phonon coupling symmetry, and anisotropic polaron
dynamics. Those papers deliberately study generic molecular-crystal parameter
spaces and anisotropy rather than a crystallographic reconstruction of one
specific pentacene polymorph.

Relevant lineage:

- L. A. Ribeiro Junior and S. Stafstrom, *Phys. Chem. Chem. Phys.* **17**,
  8973-8982 (2015), DOI `10.1039/C4CP06028H`.
- L. A. Ribeiro Junior and S. Stafstrom, *Phys. Chem. Chem. Phys.* **18**,
  1386-1391 (2016), DOI `10.1039/C5CP06577A`.
- L. A. Ribeiro Junior and S. Stafstrom, *Phys. Chem. Chem. Phys.* **19**,
  4078-4084 (2017), DOI `10.1039/C6CP07478B`.

Accordingly, the existing default parameters must not be relabeled as a
pentacene parameter set without a separate mapping argument.

## Crystal topology: the blocking issue

Bulk pentacene is triclinic and herringbone packed, with two molecules per unit
cell. Mattheus *et al.* report `P-1`, `Z = 2`, with the following structures:

| temperature | a (angstrom) | b (angstrom) | c (angstrom) | alpha | beta | gamma |
|---:|---:|---:|---:|---:|---:|---:|
| 293 K | 6.266 | 7.775 | 14.530 | 76.475 deg | 87.682 deg | 84.684 deg |
| 90 K | 6.239 | 7.636 | 14.330 | 76.978 deg | 88.136 deg | 84.415 deg |

Reference: C. C. Mattheus *et al.*, *Acta Crystallogr. C* **57**, 939-941
(2001), DOI `10.1107/S010827010100703X`.

The conducting layers are therefore not an orthogonal square/rectangular lattice
with one equivalent molecular site per primitive cell. The two molecular
orientations and herringbone bond network generate several inequivalent transfer
paths. This is not a small parameter correction: it changes the graph on which
the electronic Hamiltonian and Peierls forces are defined.

### Initial target phase

The recommended first material-specific target is the **293 K bulk crystal**
reported by Mattheus *et al.*. This choice is pragmatic rather than fundamental:
a widely used first-principles pentacene coupling data set was evaluated on this
same 293 K structure, allowing geometry and electronic couplings to be taken
from an internally consistent source. The 90 K bulk structure should be retained
as the first polymorph/temperature transferability test. Thin-film polymorphs
should not be mixed into the initial bulk parameter set.

## Parameter evidence table

The values below are candidate evidence, not yet a single self-consistent input
file. Values obtained under different screening conventions, temperatures, or
transport models are deliberately kept separate.

| quantity | candidate evidence | source / method | confidence for initial mapping | mapping issue |
|---|---|---|---|---|
| bulk cell | 293 K: `a=6.266`, `b=7.775`, `c=14.530 A`; `alpha=76.475`, `beta=87.682`, `gamma=84.684 deg`; `Z=2` | Mattheus et al. 2001, X-ray | high | requires non-orthogonal Bravais vectors and two-site basis |
| low-T cell | 90 K: `a=6.239`, `b=7.636`, `c=14.330 A`; `alpha=76.978`, `beta=88.136`, `gamma=84.415 deg`; `Z=2` | Mattheus et al. 2001, X-ray | high | use as transferability test, not mixed with 293 K couplings |
| dominant hole couplings | `90.69, 55.05, 39.68, 36.62 meV` for four important paths | Stehr et al. 2011, first-principles + Marcus network on 293 K crystal | medium-high | path identities must be mapped to explicit A/B bond families |
| alternative dominant couplings | `t1=75 meV`, `t2=32 meV`; two principal herringbone paths | Bakulin et al. 2015 / associated electronic-structure calculation | medium | different electronic-structure protocol; useful cross-check, not to be averaged blindly |
| hole intramolecular reorganization energy | about `92 meV`; earlier values cited around `95-98 meV` | Stehr et al. 2011; Bredas-group literature | high for scale | reorganization energy is not identical to the current single effective `alpha_intra` and `K1`; a mapping convention is required |
| Holstein/Peierls separation | high-frequency intramolecular modes mainly modulate site energy; low-frequency modes mainly modulate transfer integrals | Girlando et al. 2011 | high qualitative | material mapping should be mode-aware; a single Peierls derivative may be an effective reduction |
| nonlocal e-ph coupling | Gamma-only phonons can underestimate transfer-integral variance by about 40% for herringbone dimers and >80% for cofacial dimers | JCP 2012, DOI `10.1063/1.4759040` | high qualitative | do not calibrate a material-specific Peierls sector from Gamma-point derivatives alone |
| high-frequency/electronic dielectric scale | `epsilon_infinity ~ 3.36` | self-consistent dielectric-dependent hybrid/RPA literature | medium-high | not interchangeable with a static dielectric constant in an adiabatic lattice model |
| scalar solid-state dielectric estimates | RPA scalar `epsilon ~ 3.6`; experimental scalar value reported as about `4.0` in a PCM parameter survey | JCTC 2016, DOI `10.1021/acs.jctc.6b00225` | medium | screening convention must match which lattice polarization is already explicit |
| p-doped-film effective dielectric | fit example `epsilon_s = 2.8` | Ha, Qi, Kahn 2010 STM of p-doped films | low for bulk mapping | surface/doped-film observable; context-only |
| onsite hole-hole repulsion | fitted cutoff has magnitude about `2.3 eV` and is interpreted as Hubbard `U` | Ha, Qi, Kahn 2010 | low for bulk mapping | surface/doped-film fit and sign convention; must not be promoted directly to bulk low-energy `U` |
| short-range `V_ij` | unresolved | must come from a consistent screened Coulomb calculation or a directly applicable literature data set | unresolved | scalar `epsilon` is not sufficient to define molecular-contact `V_ij` after the v0.6 range-sensitivity result |
| Peierls derivative per bond family | unresolved as a single clean static data set | Girlando et al.; full-BZ nonlocal e-ph literature | unresolved | need derivatives or variance-equivalent effective couplings for the same crystal geometry and bond definitions |
| effective elastic constants | unresolved | phonon/e-ph mapping required | unresolved | current `K2` is an effective lattice stiffness and cannot be assigned independently of the selected Peierls coordinates |

Key references for the numerical evidence above:

- V. Stehr, J. Pfister, R. F. Fink, B. Engels, and C. Deibel,
  *Phys. Rev. B* **83**, 155208 (2011), DOI `10.1103/PhysRevB.83.155208`.
- A. A. Bakulin *et al.*, *Nature Communications* **6**, 7880 (2015),
  DOI `10.1038/ncomms8880`.
- A. Girlando *et al.*, *J. Chem. Phys.* **135**, 084701 (2011),
  DOI `10.1063/1.3625293`.
- S. D. Ha, Y. Qi, and A. Kahn, *Chem. Phys. Lett.* **495**, 212-217
  (2010), DOI `10.1016/j.cplett.2010.06.085`.
- J. L. Bredas *et al.*, *J. Am. Chem. Soc.* **124**, 7918-7919 (2002),
  DOI `10.1021/ja0175892`.

## Why a scalar dielectric constant is not enough

Version `0.6.0a1` established that replacing only the highest-weight short-range
shells can shift the bipolaron dissociation boundary substantially even when the
continuum `1/r` tail is retained. Therefore a pentacene calculation in which
`epsilon_r` is fitted while all molecular contacts are forced to obey the same
continuum expression would discard a validated sensitivity of the model.

A material-specific calculation should instead separate:

1. onsite `U`;
2. explicit screened short-range pair matrix elements for the dominant molecular
   contact families;
3. a consistently defined longer-range dielectric tail.

The dielectric convention must also be matched to the degrees of freedom already
included explicitly. In particular, `epsilon_infinity`, static dielectric
constants, and surface/doped-film effective values cannot be treated as
interchangeable numbers.

## Model decision

### Option A: retain the current one-site rectangular lattice

This is acceptable only for a **pentacene-inspired effective model**. One could
choose transfer integrals and coupling scales within pentacene-like ranges and
study trends, but the result should not be called a crystal-specific pentacene
phase diagram because the primitive-cell topology and molecular orientations are
wrong.

### Option B: generalize the periodic molecular lattice

**Recommended.** Introduce a generic periodic lattice with:

- arbitrary two-dimensional Bravais vectors;
- an arbitrary molecular basis per primitive cell;
- explicit periodic bond families between basis sites in neighboring cells;
- bond-family transfer integrals and Peierls couplings;
- physical basis coordinates for minimum-image pair distances;
- shell/contact-specific screened interactions.

Pentacene then becomes a two-basis-site herringbone parameter file rather than a
special case in the solver. The existing rectangular lattice becomes a
one-basis-site parameterization of the same infrastructure.

This route is more work initially, but it prevents a structural model error and
also makes the code reusable for other molecular crystals.

## Acceptance criteria before the first pentacene phase diagram

A result may be labeled material-specific only after all of the following are
satisfied:

1. a documented pentacene polymorph and temperature are selected;
2. Bravais vectors and molecular basis positions come from a traceable crystal
   structure/CIF;
3. every retained hopping path is mapped to an explicit periodic bond family;
4. transfer integrals and Peierls derivatives use a mutually compatible geometry
   and sign convention;
5. the mapping from molecular reorganization energy to the effective Holstein
   sector is documented and tested;
6. onsite `U`, short-range `V_ij`, and the long-range dielectric convention are
   internally consistent and do not double count screening;
7. the generalized-lattice implementation reproduces `v0.6.0a1` exactly for the
   legacy rectangular one-site topology;
8. the noninteracting herringbone band structure and symmetry are validated
   before correlated lattice relaxation is enabled;
9. finite-size convergence and competing bipolaron branches are repeated on the
   material-specific graph.

## Immediate next implementation

The next code milestone should therefore be a **general periodic molecular
lattice**, not a pentacene parameter file. Its detailed compatibility contract is
specified in `docs/general-periodic-molecular-lattice-design.md`.
