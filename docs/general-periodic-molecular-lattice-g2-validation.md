# G2 graph-based one-particle Hamiltonian validation

## Scope

Phase G2 adds a fixed, undistorted one-particle tight-binding Hamiltonian on the periodic molecular graph introduced in G1. It remains isolated from the validated production Holstein-Peierls solvers. No Peierls coordinate, lattice gradient, Coulomb interaction, correlated two-particle operator, or relaxation routine is routed through the new graph in this phase.

The G2 layer provides:

- one signed transfer integral per `BondFamily`;
- optional basis-resolved onsite energies;
- a finite-supercell sparse Hamiltonian assembled from the unique undirected G1 bond graph;
- a primitive-cell Bloch Hamiltonian for arbitrary Cartesian wavevector;
- reciprocal primitive vectors;
- the finite-supercell commensurate k-point mesh;
- sampled Bloch spectra for direct supercell/band validation.

The transfer-integral API stores the actual Hamiltonian element including sign. Thus the legacy convention with positive `Jx` and `Jy` maps to graph transfers `t_x = -Jx` and `t_y = -Jy`.

## Rectangular compatibility

For the one-basis rectangular compatibility lattice with zero Holstein and Peierls distortion, the graph Hamiltonian is required to reproduce the existing `build_dense_hamiltonian` matrix exactly.

The validation uses `nx=5`, `ny=4`, `Jx=0.100 eV`, and `Jy=0.015 eV`. The complete matrix produced by G2 is bitwise identical (`numpy.array_equal`) to the established rectangular Hamiltonian.

At arbitrary Cartesian wavevector `k`, the one-band Bloch Hamiltonian reproduces

`E(k) = -2 Jx cos(k_x a_x) - 2 Jy cos(k_y a_y)`

to floating-point precision.

The complete finite-supercell eigenvalue spectrum also agrees with the union of this dispersion sampled on the `nx x ny` commensurate k mesh.

## Skew two-basis validation

A synthetic two-basis lattice uses

- `a1 = (5.0, 0.0) angstrom`;
- `a2 = (1.2, 4.3) angstrom`;
- basis positions `A=(0,0)` and `B=(0.48,0.52)` in primitive fractional coordinates;
- three inequivalent A-B bond families and one A-A family;
- nonzero basis onsite offsets.

For this model:

1. the Bloch Hamiltonian is Hermitian at an arbitrary non-symmetry k point;
2. the finite-supercell Hamiltonian and Bloch Hamiltonian are invariant to the ordering of the declared bond families;
3. the finite-supercell eigenvalue spectrum agrees with the sampled two-band Bloch spectrum;
4. the reciprocal vectors satisfy `A^T B = 2 pi I`.

These checks validate both inter-basis phases and same-basis translated bonds in a non-orthogonal cell.

## Test record

Local validation after G2:

- G2-specific tests: `7/7` passed;
- complete project suite: `56/56` passed.

The previous 49-test G1 checkpoint therefore remains green, including all 42 pre-G1 tests from the `v0.6.0a1` physics paths.

## Promotion boundary

G2 does not replace the existing Hamiltonian builders. It is a parallel validation implementation for undistorted graph-based electronic structure.

The next phase, G3, may generalize the Peierls sector only after G2 is integrated. G3 must introduce local mode vectors and bond-family coupling vectors, then prove that the rectangular choice

- `q=(vx,vy)`;
- `g_x=(alpha_x,0)`;
- `g_y=(0,alpha_y)`

reproduces the existing transfer integrals, elastic energy, Hellmann-Feynman gradients, and converged rectangular physics before any generalized solver path is promoted.
