# Pentacene G5g: atomistic 293 K dimer reconstruction

## Purpose

G5g supplies the atomistic geometry that G5e/G5f deliberately left unresolved.
It reconstructs the two complete pentacene molecules in the 293 K bulk unit cell
and maps them onto the six explicitly oriented herringbone bond families used by
the material-specific graph.

No electronic-structure calculation is performed in this stage.

## Primary crystallographic source

The source is Mattheus et al., *Polymorphism in pentacene*, Acta Cryst. C 57,
939-941 (2001), DOI `10.1107/S010827010100703X`, CCDC 170186.  The IUCr page
links the crystallographic information file `sk1477sup1.cif`, containing both
293 K and 90 K datablocks, and exposes the 293 K fractional-coordinate table.

The relevant 293 K cell is

- a = 6.266 A;
- b = 7.775 A;
- c = 14.530 A;
- alpha = 76.475 deg;
- beta = 87.682 deg;
- gamma = 84.684 deg;
- V = 685.15 A^3;
- Z = 2;
- triclinic P-1 (Hall symbol `-P 1`).

The article states that both molecules have crystallographically imposed
inversion centers:

- C1--C11 partner molecule: `(1/2, 1/2, 0)`, operation
  `(1-x, 1-y, -z)`;
- C12--C22 partner molecule: `(0, 0, 0)`, operation `(-x, -y, -z)`.

Seven independent H sites are given for each half molecule.  At 293 K the H
positions were inferred from neighboring sites and constrained during
refinement.

## Precision boundary

The current implementation uses the **rounded fractional coordinates printed in
the IUCr article/supporting-data web view**.  It records the linked CIF filename
and DOI, but does not claim that the raw CIF bytes have been parsed in this
environment.

This distinction matters.  The rounded coordinates are adequate for validating
molecular orientation and producing a traceable first atomistic geometry, but a
future production FHI-aims campaign should preferentially ingest and archive the
raw CIF itself if it becomes directly accessible.  That replacement must be
regression-tested against the current geometry rather than silently changing
coordinates.

## A/B convention

The existing material graph defines basis A at `(0,0)` and basis B at
`(1/2,1/2)` in the projected a-b plane.  G5g preserves that convention:

- code basis A = crystallographic C12--C22 molecule at `(0,0,0)`;
- code basis B = crystallographic C1--C11 molecule at `(1/2,1/2,0)`.

This mapping is documented because the atom-number sequence in the crystal
paper would otherwise make it easy to reverse A/B in later dimer calculations.

## Triclinic Cartesian embedding

The three crystallographic vectors are embedded as

```text
a = (a, 0, 0)
b = (b cos(gamma), b sin(gamma), 0)
c = (c cos(beta),
     c [cos(alpha)-cos(beta)cos(gamma)]/sin(gamma),
     c_z)
```

with positive `c_z` chosen to reproduce the cell volume.  Fractional positions
are converted with this complete 3D cell, not the earlier 2D projection.

The resulting volume from the rounded cell parameters is approximately
685.155 A^3, consistent with the reported 685.15(15) A^3.

## Molecular reconstruction

Each independent half molecule contains 11 carbon sites and 7 hydrogen sites.
The corresponding crystallographic inversion operation generates the other 18
atoms.  Each reconstructed molecule therefore contains exactly

```text
22 C + 14 H = 36 atoms.
```

Because every atom has an inversion partner of identical mass, the imposed
crystallographic inversion center is also the molecular center of mass up to
floating-point roundoff.  G5g uses that exact crystallographic center as the
rigid-body pivot.

## Principal molecular axes

The G5e coordinate convention requires long, short, and normal molecular axes.
G5g obtains them from the mass-weighted rigid-body inertia tensor using C and H
atomic masses.  Eigenvectors are ordered by increasing principal moment:

1. smallest moment -> long molecular axis;
2. intermediate moment -> short in-plane axis;
3. largest moment -> molecular normal.

Eigenvector signs are mathematically arbitrary, so G5g fixes them
deterministically: the largest Cartesian component of the long and short axes
is required to be positive, and the normal is defined by their right-handed
cross product.

This sign convention is a coordinate gauge, not a physical observable.  Any
calculated derivative and structural covariance must use the same convention.

## Independent orientation regression

In the conventional embedding, a and b lie in the Cartesian xy plane, so c* is
parallel to z.  From the reconstructed rounded coordinates, the two mass-weighted
long axes form angles of approximately

```text
25.12 deg and 24.40 deg
```

with c*.  Mattheus et al. report molecular heart-line angles of 25.17(2) deg
and 24.39(2) deg.  Agreement at this level provides an independent physical
regression of the inversion reconstruction and triclinic coordinate transform.

The first value differs by several hundredths of a degree because the code uses
rounded coordinates and a mass-weighted principal axis rather than the paper's
graphical heart-line definition.

## Six atomistic oriented dimers

G5g maps the complete molecules onto the exact G5b graph:

```text
a_AA                    A(0,0,0) -> A(1,0,0)
a_BB                    B(0,0,0) -> B(1,0,0)
diag_plus_AB_forward    A(0,0,0) -> B(0,0,0)
diag_plus_AB_backward   A(0,0,0) -> B(-1,-1,0)
diag_minus_AB_forward   A(0,0,0) -> B(0,-1,0)
diag_minus_AB_backward  A(0,0,0) -> B(-1,0,0)
```

The centroid displacements reproduce the corresponding 2D G5b bond vectors in
x-y and have zero z displacement because both molecular inversion centers lie
at fractional z=0 and the selected contacts use only a/b translations.

## Local coordinate convention for G5e

Neef et al. define the six dimer degrees of freedom by projecting the centroid
vector on the principal axes of **one molecule** and obtaining relative angles
by molecular alignment.  G5g makes a deterministic implementation choice:

- molecule A in `RigidDimerReference` is the fixed/source partner;
- molecule B is the target partner displaced by G5e;
- the local long/short/normal frame is the principal-axis frame of the fixed
  source molecule.

For same-basis contacts this choice is trivial.  For A-B contacts it is a
coordinate convention and must remain consistent when derivatives and thermal
covariances are compared.  It should not be interpreted as a claim that the
literature defines a unique source-versus-target gauge.

## Relation to G5f

Every `AtomisticPentaceneDimer.reference` is directly consumable by the G5f
FHI-aims renderer.  However, G5g still does **not** create executable production
jobs because the final FO-DFT step requires validated isolated-fragment state
indices (`fo_orbitals`).

Those indices must come from actual neutral pentacene fragment calculations in
the target FHI-aims installation.  Guessing them from the molecular formula is
not an acceptable substitute.

## Promotion gates before numerical Peierls derivatives

The next numerical stage requires:

1. archive the raw 293 K CIF if direct access becomes available;
2. materialize at least the equilibrium fragment/dimer FHI-aims inputs;
3. run neutral fragment calculations using the validated H2n@DA/PBE/tier2/tight
   protocol;
4. determine and record the actual fragment HOMO state indices;
5. validate one equilibrium transfer integral against the expected pentacene
   hopping scale/sign convention;
6. obtain a validated output fixture so the G5f parser can be implemented;
7. only then launch the multi-delta G5e scan.

No `g_b`, elastic constant, Hubbard/Coulomb parameter, or relaxed polaron result
is promoted by G5g.
