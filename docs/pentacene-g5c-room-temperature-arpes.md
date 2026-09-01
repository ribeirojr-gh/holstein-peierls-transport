# G5c room-temperature pentacene ARPES and dynamical-disorder evidence

## Purpose

G5b established a correct six-neighbour herringbone graph and two signed electronic references: a low-temperature de Wijs DFT/TB fit and a cross-source 293 K candidate. G5c adds a stronger room-temperature electronic benchmark from a **single experimental/computational study** while keeping its publication status and modeling limits explicit.

The source is A. Neef et al., *Frontier orbitals control dynamical disorder in molecular semiconductors*, arXiv:2412.06030, DOI `10.48550/arXiv.2412.06030`.

As of September 2026 the work is still listed by the authors' Max Planck/Fritz Haber publication pages as a **preprint**. G5c therefore treats it as the best room-temperature signed experimental reference currently available to the project, but does not relabel it as a final peer-reviewed material parameterization.

## Experimental signed tight-binding fit

Neef et al. measured the valence-band structure of pentacene single crystals with angle-resolved photoemission spectroscopy. The Methods state explicitly that the ARPES data were acquired at **room temperature**. Fitting the standard two-dimensional herringbone tight-binding model gives

- `t_a = +35 +/- 10 meV` for neighbours at `+/- a`;
- `t_plus = +55 +/- 5 meV` for neighbours at `+/-(a+b)/2`;
- `t_minus = -70 +/- 5 meV` for neighbours at `+/-(a-b)/2`.

The sign product is negative,

`t_a * t_plus * t_minus < 0`,

so the experimental fit has the frustrated hopping topology expected for pentacene.

The paper's model is exactly the graph implemented in G5b: each molecule has six nearest neighbours at

`+/-a`, `+/-(a+b)/2`, and `+/-(a-b)/2`.

For pentacene the Bloch Hamiltonian is

`H(k) = [[h0(k), h1(k)], [h1(k), h0(k)]]`

with

`h0(k) = 2 t_a cos(k.a)`

and

`h1(k) = 2 t_plus cos[k.(a+b)/2] + 2 t_minus cos[k.(a-b)/2]`.

G5c tests the numerical `bloch_hamiltonian()` directly against this analytic expression at a generic k point, rather than merely checking stored constants.

## Executable room-temperature reference

`pentacene_room_temperature_neef_arpes_homo_model()` attaches the ARPES values to the G5b six-family graph:

| explicit graph family | hopping |
|---|---:|
| `a_AA` | `+35 meV` |
| `a_BB` | `+35 meV` |
| `diag_plus_AB_forward/backward` | `+55 meV` |
| `diag_minus_AB_forward/backward` | `-70 meV` |

The finite graph uses the Mattheus **293 K** molecular-center geometry as the nearest explicit crystallographic host for a room-temperature experiment. The roughly 2 K difference from a nominal 295 K room temperature is not fitted away and should not be interpreted as a structural correction.

The Neef tight-binding expression uses equivalent diagonal onsite terms, so the executable G5c reference uses equal A/B onsite energies.

## Gauge relation to de Wijs

The G5b de Wijs reference was stored in the gauge

- `t_a > 0`;
- `t_plus < 0`;
- `t_minus > 0`.

Multiplying every B-sublattice HOMO by `-1` reverses both A-B transfer integrals and gives

- `t_a > 0`;
- `t_plus > 0`;
- `t_minus < 0`,

which is the Neef gauge. Thus the two sources have the **same frustrated sign topology up to the allowed molecular-orbital phase gauge**. G5c tests this explicitly.

## 295 K MD + FO-DFT evidence

The same Neef study performs ab-initio-quality molecular dynamics at room temperature; Extended Data specifies **295 K**. Nearest-neighbour dimers extracted from the MD trajectory are evaluated by fragment-orbital DFT.

For pentacene the reported hopping distributions have means

- `t_a: 32.0 meV`;
- `t_plus: 39.5 meV`;
- `t_minus: -78.8 meV`,

and standard deviations

- `sigma_t,a = 12.0 meV`;
- `sigma_t,+ = 18.0 meV`;
- `sigma_t,- = 18.4 meV`.

The average translational structural fluctuation quoted for pentacene is about `0.21 angstrom`.

These values are stored as `DynamicalDisorderEvidence` records with the simulation temperature and method attached.

## Why sigma_t is not a Peierls derivative

G5c deliberately does **not** map the thermal hopping widths into G3 `coupling_vectors` or `stiffness_matrices`.

For one structural coordinate `r`, a linearized contribution may be written

`sigma_t,r = (partial t / partial r) sigma_r`,

but the total hopping distribution results from several translational and rotational degrees of freedom, including correlations. Therefore a measured/calculated `sigma_t` alone does not determine either

- the derivative `partial t / partial r`, or
- the elastic stiffness that produces `sigma_r`.

Promoting `sigma_t` directly to a static Peierls coupling would mix thermal statistics with the Hamiltonian derivative and could double count lattice fluctuations when dynamics are introduced later.

The correct next e-ph task is to extract the **degree-of-freedom-resolved derivatives and structural variances** from the source data/supplement or to calculate them reproducibly.

## Evidence status after G5c

G5c resolves the previous broad question “does a signed single-source room-temperature pentacene band fit exist?”: **yes**.

It does not close the stricter publication/parameterization gate because the Neef result is still a preprint. The unresolved electronic field is therefore renamed to

`peer_reviewed_temperature_matched_signed_hopping_parameterization`.

The following remain unresolved regardless of the ARPES advance:

- bond-resolved Peierls derivatives;
- effective stiffness matrices / mode mapping;
- screened onsite Hubbard `U`;
- screened short-range contact interactions `V_ij`;
- the compatible long-range dielectric convention;
- the Holstein reorganization-energy mapping.

## G5c validation gates

The G5c tests require that

1. the ARPES evidence retains the signs, uncertainties, room-temperature status, source DOI, and `peer_reviewed=False` flag;
2. the executable G5c model reproduces the published analytic herringbone Bloch Hamiltonian;
3. Neef and de Wijs have the same frustrated sign topology after the allowed B-sublattice gauge flip;
4. the 295 K MD/FO-DFT means and standard deviations are stored exactly as statistical evidence;
5. the mean pentacene translational fluctuation remains separately recorded as `0.21 angstrom`;
6. no disorder statistic is promoted to a Peierls derivative or stiffness;
7. Coulomb and Holstein material parameters remain unresolved.

## Next step

G5d should focus on the **electron-phonon mapping**, not on running a material-specific bipolaron yet.

The highest-value target is to extract, for the `a`, `+`, and `-` dimers, the individual derivatives of `t` with respect to the local relative translations/rotations and the corresponding 295 K structural variances. Those data can be projected onto a controlled low-dimensional G3 mode basis. Only after the resulting effective Peierls Hamiltonian passes finite-difference and thermal-variance checks should material-specific static relaxation begin.

In parallel, the project still needs a consistent screened `U/V_ij/dielectric` set before a pentacene bipolaron phase diagram is scientifically complete.
