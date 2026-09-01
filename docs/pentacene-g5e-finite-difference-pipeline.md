# Pentacene G5e: bond-resolved finite-difference Peierls pipeline

## Purpose

G5e builds the reproducible bridge from material-specific dimer electronic
couplings to the generalized G3 Peierls representation.  It does **not** yet
claim numerical pentacene Peierls parameters.  The stage provides geometry
generation, multi-step central differences, local-to-global coordinate
projection, linearity gates, and a traceable electronic-coupling protocol.

The final target for each oriented bond family is

\[
\mathbf g_b = \frac{\partial t_b}{\partial \Delta\mathbf q},
\]

where the six global rigid-body coordinates are

\[
\Delta\mathbf q =
(\Delta x,\Delta y,\Delta z,
 \Delta\theta_x,\Delta\theta_y,\Delta\theta_z),
\]

with translations in angstrom and infinitesimal rotations in radians.

## Why the dimer-local and G3 coordinate systems are separated

The dynamical-disorder literature describes intermolecular motion using a
local right-handed molecular-dimer frame:

1. translation along the long molecular axis `l`;
2. translation along the short molecular axis `s`;
3. translation along the molecular normal `n`;
4. rotation around `l`;
5. rotation around `s`;
6. rotation around `n`.

For pentacene A/B molecules in the herringbone crystal, these local frames
cannot be subtracted blindly between sites.  G3 therefore receives derivatives
only after an explicit proper-rotation transform from the dimer-local frame to
one global Cartesian translation/rotation convention.

This also means that G5e does **not** copy one derivative vector onto the
forward/backward or A/B bond families by assumption.  The six oriented graph
families remain explicit:

- `a_AA`;
- `a_BB`;
- `diag_plus_AB_forward`;
- `diag_plus_AB_backward`;
- `diag_minus_AB_forward`;
- `diag_minus_AB_backward`.

Any future symmetry reduction must be expressed as a tested coordinate
transformation, including the sign of every derivative component.

## Electronic-coupling provenance

The reference protocol follows Neef et al., *Frontier orbitals control
dynamical disorder in molecular semiconductors*, arXiv:2412.06030:

- H2n@DA variant of fragment-orbital DFT;
- FHI-aims implementation;
- PBE exchange-correlation functional;
- `tier2` numeric atom-centered basis;
- `tight` integration grids;
- electronic-level convergence below `1e-6 eV`;
- no vdW correction for the static isolated-dimer coupling calculations;
- hole coupling obtained from the Kohn-Sham matrix element between donor and
  acceptor fragment HOMOs.

The source explicitly generated dimers by translating and rotating molecules
along their principal axes and evaluated partial derivatives by varying one
coordinate around its equilibrium value while leaving the remaining degrees of
freedom fixed.  This is the numerical definition adopted by G5e.

No FHI-aims input writer is included yet.  The H2n@DA-specific keywords must be
implemented only from an authoritative input specification or validated legacy
example; they will not be guessed from the paper prose.

## Scan grid

The initial translation grid is

\[
\delta r = 0.0025,\ 0.005,\ 0.010\ \mathrm{\AA}.
\]

The smallest value also matches the finite-displacement convergence scale used
in the independent peer-reviewed full-Brillouin-zone EPC study of Gnoli et al.
(DOI `10.1021/acs.jpcc.5c04906`).  The larger values are convergence probes.

The initial rotational grid is

\[
\delta\theta = 0.25^\circ,\ 0.5^\circ,\ 1.0^\circ.
\]

These angular values are a numerical proposal, not published pentacene
parameters.  They must pass the same step-size convergence tests as the
translations and can be tightened automatically if needed.

For one oriented bond the initial grid requires

\[
6\ \mathrm{DOF}\times 3\ \mathrm{steps}\times 2\ \mathrm{signs}=36
\]

non-equilibrium coupling calculations.  Evaluating all six oriented graph
families conservatively requires 216 displaced calculations, plus optional
reference calculations at the equilibrium geometry for gauge/sign diagnostics.

## Numerical derivative

For each positive step `h`, G5e computes

\[
d(h)=\frac{t(+h)-t(-h)}{2h}.
\]

For a smooth transfer-integral surface,

\[
d(h)=g+c h^2+\mathcal O(h^4).
\]

The code therefore performs a least-squares fit of `d(h)` against `h^2`; the
zero-step intercept is the promoted derivative candidate.  The raw derivatives
at all step sizes are retained, together with their maximum relative spread
and fit RMS.  A derivative that fails the requested linearity tolerance cannot
be exported to G3 by the default projection gate.

This extrapolation is intentionally more informative than selecting a single
finite-difference step.

## Geometry generation

`RigidMoleculeGeometry` stores atom labels, Cartesian coordinates, and an
explicit rotation pivot.  `RigidDimerReference` additionally stores a
right-handed orthonormal local frame as a 3x3 matrix whose columns are the
long, short, and normal axes expressed in global Cartesian coordinates.

`prepare_scan_geometries()` then generates every +h/-h geometry while keeping
the first molecule fixed and translating or rotating the second molecule
rigidly.  Rotations use Rodrigues' formula around the explicit target pivot,
so intramolecular distances are preserved exactly up to floating-point
roundoff.

The geometry layer is backend-independent: FHI-aims, another FO-DFT
implementation, or a validated surrogate can consume the same generated
geometries.

## Projection into G3

After all six local derivatives of one oriented bond pass the numerical gate,
G5e transforms the local vector with the bond's explicit proper-rotation matrix.
Translations and infinitesimal rotation pseudovectors are transformed in
separate 3-component blocks.

`assemble_g3_bond_mode_couplings()` then requires an explicit result for every
expected bond label and returns the exact tuple structure consumed by
`MolecularPeierlsModel.bond_mode_couplings`.

Missing families, duplicate labels, or failed linearity gates are errors.

## Later room-temperature validation

A physically acceptable set of local derivatives must also predict the
observed thermal hopping variance when combined with structural correlations:

\[
\operatorname{Var}(t_b)
\simeq
\mathbf g_b^T\,\mathbf C_{\Delta q,b}\,\mathbf g_b.
\]

G5e already implements this quadratic projection.  G5f will supply a
coordinate covariance matrix in the **same global six-coordinate convention**
and compare the resulting standard deviations with the 295 K Neef values:

- `a`: 12.0 meV;
- `+`: 18.0 meV;
- `-`: 18.4 meV.

This covariance test is mandatory because Neef explicitly shows that several
in-plane degrees of freedom are strongly correlated.  Replacing the full
covariance by a scalar average displacement would not be a valid validation.

## Promotion gates

A numerical pentacene Peierls derivative is not material-ready until all of the
following hold:

1. the transfer integral is signed and follows a continuous orbital gauge;
2. the reference dimer and rigid-body pivots are archived;
3. each oriented bond has an explicit right-handed local frame;
4. at least three central-difference scales are present;
5. the zero-step derivative is stable under step-size reduction;
6. all six rigid-body coordinates are evaluated;
7. no untested symmetry copy is used between bond families;
8. the local vector is transformed into the global G3 convention;
9. every scan point uses the same electronic-structure protocol;
10. the resulting model is checked against the 295 K hopping covariance data.

Only after these gates should the numerical vector enter a material-specific
`MolecularPeierlsModel` and be used for a pentacene polaron/bipolaron claim.
