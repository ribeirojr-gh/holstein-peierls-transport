# General periodic molecular-lattice design

## Goal

Generalize the current rectangular one-site Holstein-Peierls lattice to an
arbitrary two-dimensional periodic molecular crystal **without changing the
validated physics of the legacy topology**.

The implementation must support pentacene's two-molecule herringbone unit cell,
but the core API must remain material agnostic. Pentacene should be a data/model
configuration, not a hard-coded branch in the Hamiltonian.

## Design principles

1. **Backward compatibility is a physics requirement.** The existing one-site
   rectangular topology must reproduce `v0.6.0a1` energies, gradients,
   interaction matrices, pair observables, and converged structures to numerical
   precision.
2. **Topology and material parameters are separate.** Geometry defines sites,
   periodic images, distances, and bonds. Electronic/elastic parameter sets
   assign numbers to those objects.
3. **Bond families are explicit.** Inequivalent herringbone contacts cannot be
   inferred from a single `x/y` rule.
4. **The Peierls coordinate API must not assume only two modes.** The current
   `vx/vy` fields are a two-component special case of a more general local-mode
   coordinate.
5. **Pair distances use the physical non-orthogonal lattice.** Independent
   wrapping of Cartesian x/y indices is not valid for a general triclinic
   projection.
6. **No material fit enters the infrastructure PR.** Geometry/topology code is
   validated first; pentacene numbers are introduced only in a later data PR.

## Periodic geometry

For primitive-cell index `n = (n1, n2)` and basis index `s`, define the physical
molecular reference position

`R(n,s) = n1 a1 + n2 a2 + tau_s`,

where `a1` and `a2` are arbitrary two-dimensional Bravais vectors in angstrom and
`tau_s` is the basis position in the conducting layer.

A finite simulation cell contains `N1 x N2` primitive cells and `n_basis`
molecules per cell, so

`N_sites = N1 N2 n_basis`.

The implementation should expose a stable bijection

`(cell_y, cell_x, basis) <-> site_index`.

### Proposed immutable geometry objects

```python
@dataclass(frozen=True, slots=True)
class MolecularBasisSite:
    label: str
    position_fractional: tuple[float, float]


@dataclass(frozen=True, slots=True)
class BondFamily:
    label: str
    source_basis: int
    target_basis: int
    cell_offset: tuple[int, int]


@dataclass(frozen=True, slots=True)
class PeriodicMolecularLattice2D:
    n1: int
    n2: int
    a1_angstrom: tuple[float, float]
    a2_angstrom: tuple[float, float]
    basis: tuple[MolecularBasisSite, ...]
    bonds: tuple[BondFamily, ...]
```

The exact public names may change during implementation, but the information
content should not.

## Periodic bond graph

Each `BondFamily` defines one undirected physical contact from a source basis
site in one primitive cell to a target basis site in a translated primitive
cell. Translation generates all equivalent bonds in the finite supercell.

The generated graph must guarantee:

- no accidental duplicate undirected bonds;
- deterministic ordering;
- periodic wrapping at supercell boundaries;
- a unique bond-family label for each generated edge;
- Hermitian one-particle hopping without storing both directions manually.

For the legacy rectangular model the compatibility topology is:

- one basis site `A` at `(0,0)`;
- `a1=(a_x,0)`, `a2=(0,a_y)`;
- bond family `x: A -> A + (1,0)`;
- bond family `y: A -> A + (0,1)`.

The electronic graph generated from that configuration must be exactly the
current nearest-neighbour graph.

## Generalized Holstein-Peierls coordinates

The current solver stores one Holstein coordinate `u_i` and two intermolecular
fields `vx_i`, `vy_i`. The electronic transfer on the legacy bonds is

`t_x(i,j) = -J_x + alpha_x [v_x(j) - v_x(i)]`,

`t_y(i,j) = -J_y + alpha_y [v_y(j) - v_y(i)]`.

A general molecular crystal should use a local-mode vector `q_i` with an
arbitrary number `M` of effective intermolecular coordinates per molecule:

`q_i = (q_i1, ..., q_iM)`.

For bond family `b`, define a coupling vector `g_b` and write

`t_b(i,j) = -J_b + g_b . [q_j - q_i]`.

This reproduces the present code exactly when `M=2`,

- `q=(vx,vy)`;
- `g_x=(alpha_x,0)`;
- `g_y=(0,alpha_y)`.

The generalized representation can later accommodate effective translations,
rotations, or mode-projected coordinates without changing the electronic graph
API.

### Elastic energy

Use a positive-semidefinite stiffness matrix `K_b` for each bond family:

`E_P = 1/2 sum_(i,j,b) (q_j-q_i)^T K_b (q_j-q_i)`.

The strict legacy reduction is obtained with two orthogonal mode channels and
projectors that reproduce

`K2/2 sum_x [vx(j)-vx(i)]^2 + K2/2 sum_y [vy(j)-vy(i)]^2`.

The first infrastructure implementation may keep diagonal/projector stiffnesses,
but the data model should not make a full symmetric `K_b` impossible later.

The Holstein sector remains

`E_H = K1/2 sum_i u_i^2`,

with onsite energy modulation `alpha_intra u_i` unless a future material model
requires basis-dependent local parameters.

## Hellmann-Feynman gradient

For the spin-summed one-body density matrix `gamma`, each undirected bond
`(i,j,b)` contributes to the electronic Peierls force through the derivative of
its hopping. With real hoppings and the existing Hamiltonian convention, the
bond contribution is proportional to

`2 gamma_ij g_b`.

The generalized gradient should be assembled by scattering equal-and-opposite
bond contributions onto `q_i` and `q_j`, plus the elastic bond force. This
bond-loop/vectorized-scatter formulation replaces the current hard-coded
left/right/up/down expressions and naturally supports arbitrary periodic graphs.

A finite-difference gradient test is mandatory for:

1. the legacy one-site rectangular graph;
2. a two-basis skew lattice;
3. at least one bond family crossing each periodic boundary.

## Pair geometry and Coulomb interaction

For sites `(n,s)` and `(m,t)`, the physical displacement before periodic
minimization is

`Delta R = A[(m-n)] + tau_t - tau_s`,

where `A=[a1 a2]`.

For a finite `N1 x N2` supercell, the minimum-image vector must minimize

`|Delta R + p N1 a1 + q N2 a2|`

over nearby integer translations `(p,q)`.

For non-orthogonal cells this must **not** be implemented by independently
minimizing Cartesian components. A robust first implementation can transform to
supercell fractional coordinates, round to the nearest image, and explicitly
check the neighboring translation candidates around that rounded solution.

The resulting physical distance matrix replaces the current orthogonal
`sqrt[(a_x dx)^2+(a_y dy)^2]` helper.

## Short-range interaction families

The `v0.6.0a1` API uses integer-offset shells `(dx,dy)`. In a multi-basis lattice,
short-range contacts should instead be addressable by an explicit pair/contact
family, because two geometrically similar separations may involve different
basis orientations and screened matrix elements.

Recommended representation:

```python
@dataclass(frozen=True, slots=True)
class PairInteractionFamily:
    label: str
    source_basis: int
    target_basis: int
    cell_offset: tuple[int, int]
    value_ev: float
```

The continuum tail remains the baseline for all distinct pairs. Explicit
material-specific contact values replace that baseline only for generated pairs
belonging to the selected family, preserving the no-double-counting rule already
validated in `v0.6.0a1`.

The legacy shell API should remain supported through an adapter for one-basis
rectangular lattices during the migration period.

## Observables

Pair observables must be separated into topology-independent and topology-aware
quantities.

Topology-independent quantities remain meaningful:

- onsite probability;
- one-body density and IPR;
- physical mean/rms carrier separation in angstrom;
- binding relative to `2 E_polaron`.

Topology-aware observables should use named contact families rather than hard
coded `nearest_neighbour_x`, `nearest_neighbour_y`, and `diagonal` fields. A
compatibility adapter can continue exposing the old names for the rectangular
legacy topology.

For pentacene, output should eventually report probability on each important
herringbone contact family (`V1`, `V2`, ... or physically named equivalents).

## Band-structure validation before interactions

Before Holstein or Coulomb interactions are enabled on a new topology, validate
the one-particle Hamiltonian independently.

For a periodic basis with `n_basis` molecules, construct the Bloch Hamiltonian

`H_st(k) = sum_b t_b exp[i k . Delta R_b]`

with the appropriate Hermitian counterpart. Required checks:

- numerical supercell spectrum against sampled Bloch bands;
- Hermiticity at arbitrary k;
- invariance under bond-list ordering;
- expected degeneracies/symmetries for synthetic test lattices;
- exact recovery of the legacy rectangular dispersion
  `E(k) = -2 Jx cos(k.a1) - 2 Jy cos(k.a2)`.

Only after these tests pass should correlated bipolaron relaxation be promoted to
the generalized graph.

## Migration plan

### Phase G1: geometry only

- implement immutable periodic geometry and indexing;
- generate physical coordinates, periodic images, bond graph, and distance
  matrix;
- no solver behavior changes.

### Phase G2: one-particle Hamiltonian

- add graph-based hopping builder;
- prove exact rectangular equivalence;
- add Bloch-band validation utilities and tests.

### Phase G3: generalized Peierls sector

- replace hard-coded `vx/vy` force assembly by mode-vector/bond-family assembly;
- reproduce all existing one- and two-particle rectangular gradients and
  energies;
- add skew/two-basis finite-difference tests.

### Phase G4: generalized pair interaction and observables

- physical non-orthogonal minimum-image distances;
- contact-family short-range overrides;
- generic contact probabilities and physical separation observables;
- reproduce all `v0.6.0a1` Coulomb and shell tests through the compatibility
  adapter.

### Phase G5: pentacene data layer

- import one selected bulk crystal structure/CIF;
- define A/B molecular basis and retained contact families;
- map transfer integrals and e-ph parameters with provenance;
- validate noninteracting bands/contact graph;
- only then run polarons/bipolarons.

## Regression gate

The generalized solver must not replace the current implementation until a
single compatibility test suite demonstrates, for the same rectangular
parameters and seeds:

- identical one-particle Hamiltonian elements;
- identical two-particle interaction matrix;
- identical total energy to tight numerical tolerance;
- analytical gradients equal to current gradients;
- finite-difference gradients pass;
- converged one-polaron and bipolaron energies agree;
- pair observables agree;
- pure-tail and R1/R2/R3 validation points remain unchanged within their recorded
  numerical tolerances.

During G1-G4 the existing solver can remain as the reference implementation.
Deletion or replacement should occur only after this gate passes.

## Scientific consequence

This generalization is required for a literal pentacene model because the
herringbone crystal has two molecular orientations per primitive cell and
multiple inequivalent transfer paths. It also cleanly separates the historically
useful generic Holstein-Peierls model from future material-specific calculations,
without invalidating any result established through `v0.6.0a1`.
