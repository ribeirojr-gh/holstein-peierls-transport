# G1 periodic molecular-lattice validation

## Scope

Phase G1 introduces geometry and topology infrastructure only. It does not route any existing one- or two-particle Hamiltonian, lattice gradient, Coulomb interaction, or relaxation code through the new representation.

The implementation is material agnostic and provides:

- arbitrary two-dimensional Bravais vectors in angstrom;
- one or more molecular basis sites per primitive cell using fractional coordinates;
- deterministic `(cell_y, cell_x, basis) <-> site_index` mapping;
- explicit translational bond families and a unique undirected finite-cell bond graph;
- Cartesian molecular reference positions;
- minimum-image displacement and distance calculations for non-orthogonal finite supercells;
- a `rectangular_legacy_lattice` compatibility adapter.

## Legacy compatibility rule

The rectangular compatibility adapter uses the same site ordering and the same floating-point operation order as the `v0.6.0a1` long-range Coulomb distance helper:

`dx = min(|x_i-x_j|, nx-|x_i-x_j|)`

`dy = min(|y_i-y_j|, ny-|y_i-y_j|)`

`r = sqrt[(dx a_x)^2 + (dy a_y)^2]`.

This is deliberate. A generic metric calculation is mathematically equivalent for an orthogonal one-basis lattice but need not be bitwise identical because floating-point operations occur in a different order. The compatibility path therefore preserves the established numerical representation exactly rather than weakening the regression test.

For a `7 x 6` rectangular lattice with `a_x = 6.266 angstrom` and `a_y = 7.775 angstrom`, the complete new distance matrix is bitwise identical (`numpy.array_equal`) to the `v0.6.0a1` `minimum_image_distances_angstrom` result.

## Non-orthogonal minimum image

For general cells, a raw displacement is transformed to finite-supercell fractional coordinates. The nearest integer image is identified and the surrounding `3 x 3` translations are explicitly tested in Cartesian space. The shortest Cartesian vector is retained. This avoids independent Cartesian-component wrapping, which is invalid for skew cells.

A synthetic `2 x 2` lattice with

- `a1 = (2, 0) angstrom`;
- `a2 = (1, 2) angstrom`

has a periodic diagonal displacement with squared minimum-image length `5 angstrom^2`; the G1 implementation reproduces this analytic result.

## Bond graph

The legacy adapter generates two families:

- `x: A -> A + (1, 0)`;
- `y: A -> A + (0, 1)`.

For a `3 x 3` finite cell, G1 generates 9 x-bonds and 9 y-bonds, including the periodic boundary edges. Same-family duplicate undirected edges that arise in very small periodic cells are collapsed. If two differently named families map onto the same undirected edge, the geometry is rejected as ambiguous.

A separate two-basis skew-cell test verifies deterministic A/B positions and two independent periodic A-B bond families.

## Test record

The local validation was performed against the `v0.6.0a1` source snapshot plus the documentation-only pentacene audit checkpoint. Results:

- new G1 geometry tests: `7/7` passed;
- complete project suite after G1: `49/49` passed.

The pre-existing 42 tests therefore remain unchanged and passing. Because G1 is not connected to the production solvers, all previously validated energies, gradients, phase boundaries, and convergence behavior continue to execute through the original code paths.

## Performance note

The general distance-matrix path computes physical positions and the supercell matrix once per matrix construction. It then performs the neighboring-image search per unique pair. This avoids the redundant position reconstruction present in the first prototype. Large-cell vectorization/chunking may be added when G4 connects this geometry to correlated pair interactions, but it is intentionally not mixed into the geometry-only milestone.

## Promotion criterion

G1 is ready to merge when:

1. the complete test suite remains green;
2. the PR contains no modifications to existing solver modules;
3. the rectangular distance matrix remains bitwise identical to `v0.6.0a1`;
4. non-orthogonal and two-basis synthetic tests pass.

After G1, Phase G2 may build a graph-based one-particle Hamiltonian and validate its rectangular dispersion and finite-supercell spectrum before any generalized Peierls or correlated-pair physics is enabled.
