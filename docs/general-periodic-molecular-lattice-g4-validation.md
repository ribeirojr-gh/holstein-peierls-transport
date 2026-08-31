# G4 generalized molecular pair interaction and observables

## Scope

Phase G4 generalizes the static correlated-pair geometry layer introduced in `v0.5.0a1`/`v0.6.0a1` from a one-basis rectangular lattice to the arbitrary periodic molecular lattice established in G1.

The implementation remains parallel to the validated production bipolaron solver. No existing two-particle eigensolver or relaxation routine is rerouted in this phase.

G4 adds:

- continuum Coulomb interactions evaluated from G1 physical minimum-image distances;
- explicit named `PairContactFamily` objects for multi-basis short-range contacts;
- contact-specific effective interaction values that replace, rather than add to, the continuum baseline;
- topology-independent onsite probability and physical mean/rms separation in angstrom;
- named contact probabilities;
- one-body IPR derived from the normalized ordered-pair state;
- a strict rectangular compatibility adapter for all `v0.6.0a1` interaction semantics.

## Contact-family representation

A contact family contains

- a label;
- source and target basis indices;
- one or more primitive-cell offsets.

Multiple offsets can intentionally share a label/value when they represent symmetry-equivalent contacts. For example, the absolute rectangular diagonal shell `(1,1)` is represented by offsets `(1,1)` and `(1,-1)` so both undirected diagonal orientations are included.

The generated contact mask is symmetric in ordered-pair space. The onsite pair is never a contact family because onsite interaction remains the independent Hubbard `U`.

If two explicitly overridden contact families overlap, G4 rejects the model as ambiguous instead of silently applying the last value.

## Interaction definition

For distinct sites, the optional continuum baseline is

`V_cont(i,j) = 14.3996454784255 / (epsilon_r r_ij)` eV,

where `r_ij` is the physical minimum-image distance returned by G1. This therefore works for non-orthogonal lattices and multi-molecule bases.

A named short-range value replaces the baseline only on the corresponding contact mask:

`V(i,j in contact c) = V_c`.

No addition is performed, preserving the no-double-counting convention validated in `v0.6.0a1`.

The diagonal remains

`V(i,i) = U`.

The continuum can also be disabled, allowing the same general layer to represent an override-only extended-Hubbard interaction.

## Exact rectangular regression

The compatibility adapter translates the established `BipolaronParameters` into the G4 representation. Absolute `(dx,dy)` shell semantics are converted to appropriate named contact offsets, and scalar `V1` becomes a cardinal contact family containing x and y nearest neighbors.

Four independent rectangular parameter sets were tested:

1. continuum Coulomb + onsite `U` only;
2. continuum + scalar `V1` + diagonal shell override;
3. continuum + independent x, y, diagonal, and second-axial shell overrides;
4. extended-Hubbard `U + V1` with the continuum disabled.

For all four cases, the complete G4 interaction matrix is **bitwise identical** (`numpy.array_equal`) to the established `v0.6.0a1` `pair_interaction_matrix` result.

For a normalized random symmetric pair state, the interaction expectation also agrees with the established implementation to machine precision.

## Observable compatibility

The G4 observable layer uses named contacts rather than hard-coded x/y fields. The compatibility observable families

- `nearest_x`;
- `nearest_y`;
- `diagonal`

map directly onto the old rectangular masks.

For a normalized symmetric random pair state, G4 reproduces the established onsite probability, x-nearest probability, y-nearest probability, and one-body IPR.

Physical mean and rms pair separations are now reported in angstrom using the G1 distance matrix rather than lattice-index units. A synthetic skew `2 x 2` cell verifies the analytic minimum-image distance `sqrt(5) angstrom` for a localized separated pair.

## Multi-basis validation

A synthetic skew two-basis lattice with A/B molecular sites was used to validate the genuinely new path. An A-B intracell contact is overridden by an explicit short-range value while an A-A pair outside that contact retains the non-orthogonal continuum value calculated from the physical distance matrix. The onsite A-A configuration retains the independent Hubbard `U`.

This confirms that contact-specific replacement and continuum geometry remain separated on a multi-basis crystal.

## Test record

Local validation of the completed G4 layer:

- G4-specific tests: `10/10` passed;
- complete project suite: `71/71` passed.

The G1-G3 regression stack and all pre-generalization static physics tests therefore remain green.

## Promotion boundary

G4 completes the infrastructure required to represent the static electronic, Peierls, and interaction sectors of a material-specific herringbone molecular crystal without hard-coded rectangular topology.

The next phase, G5, should be a **data/model layer rather than another generic algebra layer**. It should:

1. select one crystallographic pentacene structure and document its provenance;
2. define the projected conducting-plane Bravais vectors and A/B basis positions;
3. define retained hopping/contact families with literature or calculation provenance;
4. assign transfer integrals first and validate the noninteracting bands/contact graph;
5. keep unresolved Peierls derivatives, elastic constants, screened `U/V_ij`, and dielectric response explicitly unset rather than mixing incompatible literature values;
6. only after a self-consistent parameter set exists, connect the generalized G1-G4 layers into a graph-based static polaron/bipolaron relaxation path.
