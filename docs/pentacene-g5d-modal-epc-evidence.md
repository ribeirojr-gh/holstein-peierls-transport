# G5d pentacene modal EPC evidence

## Purpose

G5c established a signed room-temperature electronic reference and independent
295 K hopping-disorder statistics for pentacene. G5d begins the material-specific
Peierls sector, but deliberately stops before constructing local G3 coupling
vectors.

The reason is representational: the best peer-reviewed full-Brillouin-zone EPC
data currently available for pentacene are **normal-mode deformation potentials**,
whereas G3 requires local molecular-coordinate bond derivatives

`g_b = partial t_b / partial q_local`.

These are related, but they are not the same quantity.

## Peer-reviewed benchmark

Gnoli et al., *J. Phys. Chem. C* **129**, 21738-21750 (2025), DOI
`10.1021/acs.jpcc.5c04906`, computed phonons and nonlocal electron-phonon
coupling in the three pentacene polymorphs at several Brillouin-zone q points.
The underlying data are reported as openly available at Zenodo DOI
`10.5281/zenodo.17368135`.

The calculation used periodic first-principles electronic structure and phonons.
The EPC finite-displacement convergence study selected `0.0025 angstrom` for the
EPC calculation.

For the low-temperature bulk polymorph, the two strongest tabulated EPC points
retained in G5d are:

| polymorph | q point | frequency (cm^-1) | EPC (eV/A) | dominant molecular component |
|---|---|---:|---:|---|
| LT | U | 26.7 | 0.27 | long-axis translation, 67% |
| LT | X | 39.2 | 0.24 | long-axis translation, 46% |

Both dominant points are away from Gamma. This is physically important: the
paper concludes that the strongest nonlocal EPC is not generally a Gamma-point
phenomenon, and for high-mobility LT pentacene the strongest modes prominently
involve translations along the molecular long inertia axis.

## Why these numbers are not G3 coupling vectors

G3 uses a local real-space expression

`t_b = t_b0 + g_b . (q_j - q_i)`.

The Gnoli EPC constant is defined for a periodic phonon eigenmode at a specific
wave vector. Schematically, a modal derivative is a projection of many local
responses,

`D_(nu,q) ~ sum_(b,mu) (partial observable / partial q_(b,mu)) e_(b,mu;nu,q)`,

with a normalization inherited from the phonon eigenvector and normal coordinate.
Therefore the numerical values `0.27` and `0.24 eV/A` cannot be copied into a
single local bond `g_b` without reconstructing this projection.

The code encodes this distinction explicitly:

- representation: `normal_mode_bandwidth_deformation_potential`;
- `direct_g3_compatible = False`;
- attempting to construct a modal record with direct compatibility enabled is
  rejected.

## Required projection information

Before a peer-reviewed modal EPC can become a local G3 parameterization, we need:

1. the phonon eigenvector expressed in molecular local coordinates;
2. the exact normal-mode coordinate normalization;
3. signed, bond-resolved transfer-integral responses;
4. the mapping from the periodic mode to local bond-coordinate differences;
5. a consistent elastic/Hessian representation.

The first priority is to recover these objects from the openly cited Zenodo data.
If the raw archive does not expose a sufficient local bond decomposition, the
modal data will remain a validation benchmark and local derivatives will be
computed independently from controlled dimer scans.

## Relation to the Neef room-temperature disorder data

Neef et al. provide at 295 K the observed/computed hopping standard deviations
for the three main pentacene contact directions. Those statistics are a powerful
validation target, but not a direct derivative measurement.

Once local G3 vectors and the structural covariance matrix are available, the
material-specific Peierls model should satisfy, to linear order,

`Var(t_b) = g_b^T Cov(Delta q) g_b`,

including covariance terms for correlated molecular motions.

This is stricter than estimating `g ~ sigma_t / sigma_q` from a single average
translational fluctuation and avoids attributing rotational and correlated
in-plane disorder to one arbitrary coordinate.

## G5d acceptance boundary

This checkpoint adds **evidence and projection requirements only**. It does not
add:

- local Peierls coupling vectors;
- stiffness matrices;
- a pentacene lattice relaxation;
- Hubbard U or short-range Coulomb matrix elements;
- a material-specific polaron/bipolaron result.

A subsequent G5e/G6 checkpoint may promote local derivatives only after the
projection or independent electronic-structure scan is numerically reproducible
and passes the hopping-variance validation above.
