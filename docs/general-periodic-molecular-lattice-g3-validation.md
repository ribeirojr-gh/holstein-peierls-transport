# G3 generalized molecular Peierls validation

## Scope

Phase G3 generalizes the legacy pair of intermolecular Peierls fields into an arbitrary local mode vector on the periodic molecular graph, while keeping the new implementation parallel to the validated production solvers.

The generalized state is

`q_i = (q_i1, ..., q_iM)`

and a bond family `b` uses

`t_b(i,j) = t_b^0 + g_b . (q_j - q_i)`.

The generalized elastic energy is

`E_P = 1/2 sum_(i,j,b) (q_j-q_i)^T K_b (q_j-q_i)`,

where every `K_b` is symmetric positive semidefinite.

The Holstein sector remains

`E_H = K1/2 sum_i u_i^2`

with onsite modulation `alpha_intra u_i`.

## Bond orientation requirement

G1 originally stored generated undirected bond endpoints only in canonical index order. That representation is sufficient for fixed symmetric hopping, but it is not sufficient for a directed difference such as `q_j-q_i`.

G3 therefore keeps both concepts on each generated bond:

- `source` and `target`: canonical undirected endpoints used for graph uniqueness and symmetric matrix assembly;
- `family_source` and `family_target`: the physical orientation declared by the `BondFamily`.

The undirected G1/G2 graph is unchanged. A rectangular periodic-boundary test explicitly verifies that the canonical edge `(0,2)` for the x-wrap retains the family orientation `2 -> 0`.

## Strict legacy reduction

The validated rectangular Holstein-Peierls model is recovered with

- `M = 2`;
- `q = (vx, vy)`;
- `g_x = (alpha_x, 0)`;
- `g_y = (0, alpha_y)`;
- `K_x = diag(K2, 0)`;
- `K_y = diag(0, K2)`;
- fixed graph transfers `t_x^0 = -Jx`, `t_y^0 = -Jy`.

For random distorted `5 x 4` rectangular states, the complete generalized Hamiltonian is bitwise identical (`numpy.array_equal`) to the established dense Hamiltonian.

The generalized Holstein and Peierls elastic energies agree with the established lattice-energy implementation to approximately machine precision.

## Hellmann-Feynman force assembly

For a spin-summed one-body density matrix `gamma`, every oriented bond contributes

`2 Re(gamma_ij) g_b`

to the electronic part of the mode force. The complete bond contribution is assembled by equal-and-opposite scatter onto the source and target local-mode vectors, together with the elastic term `K_b(q_j-q_i)`.

This graph-scatter formulation replaces the conceptual dependence on hard-coded left/right/up/down neighbors and supports arbitrary bond families without changing the physical derivative.

### One-particle rectangular regression

For a random distorted rectangular state, the generalized gradient agrees with the optimized legacy one-polaron gradient for `u`, `vx`, and `vy` with absolute differences of order `10^-15`.

### Correlated bipolaron regression

A separate test solves the existing correlated singlet bipolaron Hamiltonian with

- a `4 x 3` rectangular lattice;
- `U = 0.525 eV`;
- `V1 = 0.008 eV`;
- nonzero x/y Peierls couplings.

The spin-summed correlated one-body density matrix from that two-particle ground state is passed directly to the generalized G3 gradient. The resulting `u`, `vx`, and `vy` derivatives agree with the established two-particle Peierls gradient to absolute tolerance `3e-15`.

This is an important regression gate: the generalized bond-scatter force is compatible not only with a single-particle orbital density matrix but also with the correlated RDM used by the bipolaron solver.

## Independent skew-cell finite differences

A synthetic non-orthogonal two-basis lattice with four inequivalent bond families uses two local intermolecular modes, nontrivial coupling vectors, and full symmetric `2 x 2` bond stiffness matrices. Analytical derivatives of selected `u` and `q` coordinates agree with central finite differences of the total adiabatic one-particle energy using a `2e-6 angstrom` displacement and the established `5e-5` relative / `5e-7 eV/angstrom` absolute tolerances.

This test exercises the generalized implementation away from the rectangular compatibility limit.

## Test record

Local validation of the completed G3 implementation:

- all G1 geometry tests passed;
- all G2 Hamiltonian/Bloch tests passed;
- all 5 G3 Peierls/gradient tests passed;
- complete project suite: `61/61` passed.

The complete pre-G3 physics suite therefore remains green.

## Promotion boundary

G3 still does not replace the production relaxation path. It establishes the generalized hopping, elastic-energy, and Hellmann-Feynman force algebra as a validated parallel implementation.

Phase G4 should next generalize the correlated-pair geometry and interaction layer:

1. use the G1 physical minimum-image distances for the continuum Coulomb tail;
2. replace integer `(dx,dy)` shell overrides by named contact/pair families for multi-basis lattices;
3. retain the no-double-counting replacement rule;
4. add topology-independent physical separation observables and named contact probabilities;
5. prove exact `v0.6.0a1` rectangular interaction/observable compatibility before routing correlated relaxation through the general graph.
