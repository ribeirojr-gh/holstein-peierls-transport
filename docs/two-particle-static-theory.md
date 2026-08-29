# Two-particle static Holstein-Peierls theory

## Purpose

This note defines the theoretical extension of the validated single-polaron solver to two-particle bound states before any implementation is attempted. Two physical sectors are considered:

1. two identical charge carriers (bipolaron problem);
2. one electron and one hole (exciton problem).

The goal is to preserve the structural content of the current semiclassical two-dimensional Holstein-Peierls model while replacing the one-particle electronic problem by a correlated two-particle problem.

## 1. Existing one-particle model

For a fixed lattice configuration q = {u_i, v_i^x, v_i^y}, the one-particle Hamiltonian used by the current code can be written schematically as

H_1[q] = sum_i (epsilon_i + A u_i) |i><i| + sum_<ij> t_ij[q] (|i><j| + |j><i|),

with hopping amplitudes modulated by intermolecular Peierls coordinates. For an x bond,

t_i^x = J_0^x - alpha_x (v_{i+x}^x - v_i^x),

and analogously in y. The lattice energy is

E_latt[q] = (K_1/2) sum_i u_i^2
          + (K_2/2) sum_i [(v_{i+x}^x-v_i^x)^2 + (v_{i+y}^y-v_i^y)^2].

The optimized single-polaron state minimizes

E_1[q,psi] = <psi|H_1[q]|psi> + E_latt[q]

with respect to q while psi is the instantaneous electronic ground state.

The original work showed that local Holstein and nonlocal Peierls relaxation are not simply additive: their simultaneous action can stabilize a polaron much more strongly than the sum of the isolated contributions in parts of parameter space. This nonlinearity is one of the main motivations for examining two-particle states.

## 2. Bipolaron model

### 2.1 Electronic Hilbert space

For two identical carriers in the same molecular band, use the real-space basis |i,j>. In the spin-singlet sector the spatial wavefunction is symmetric,

Psi_ij = Psi_ji,

whereas the triplet spatial wavefunction is antisymmetric and Psi_ii = 0.

The first implementation should target the singlet sector because it contains both onsite and intersite bound states and is the natural candidate for the lowest-energy bipolaron.

### 2.2 Hamiltonian

The minimal static bipolaron Hamiltonian is

H_BP[q] = H_1[q] tensor I + I tensor H_1[q] + V_ee,

with an extended-Hubbard interaction

V_ee(i,j) = U delta_ij + (1-delta_ij) V(r_ij).

A minimal first study can use onsite U plus nearest-neighbour V_1. A subsequent material-oriented version can use a screened long-range interaction,

V(r) = e^2 / (4 pi epsilon_0 epsilon_r r),

with an effective dielectric constant or a more refined anisotropic screening model.

In coordinate form,

(H_BP Psi)_ij = [epsilon_i(q)+epsilon_j(q)+V_ee(i,j)] Psi_ij
              + sum_{k in nn(i)} t_ik(q) Psi_kj
              + sum_{l in nn(j)} t_jl(q) Psi_il.

This operator can be applied without constructing the N^2 by N^2 matrix.

### 2.3 Energy functional and lattice relaxation

The total energy is

E_BP[q,Psi] = <Psi|H_BP[q]|Psi> + E_latt[q].

For a normalized singlet wavefunction, define the one-particle reduced density matrix

gamma_mn = 2 sum_j Psi_mj Psi*_nj,

which has trace 2. All derivatives of the one-body Holstein-Peierls terms have the same mathematical structure as in the single-polaron solver, with the one-particle density matrix rho replaced by gamma.

For the local coordinate,

dE/du_i = K_1 u_i + A gamma_ii.

For an x Peierls coordinate, if t_i^x = J_0^x - alpha_x(v_{i+x}^x-v_i^x),

dE/dv_i^x = K_2(2v_i^x-v_{i+x}^x-v_{i-x}^x)
           + 2 alpha_x [Re gamma_{i,i+x} - Re gamma_{i-x,i}],

up to the exact indexing/sign convention already validated in the single-particle code. The y expression is analogous.

The electron-electron term initially has no explicit lattice derivative. If V(r) is later allowed to depend on instantaneous intermolecular distances, the corresponding electrostatic forces must be added explicitly.

### 2.4 Binding energy

A bound bipolaron must be distinguished from two independently localized polarons. Define a positive binding energy as

Delta_BP = E_2P^sep - E_BP^bound,

where E_2P^sep is the lowest relaxed solution constrained or initialized so that the two polarons remain well separated in the same periodic cell. This is preferable to using 2E_P alone because it controls finite-size and shared-lattice effects.

An infinite-separation proxy can also be reported,

Delta_BP^inf = 2 E_1 - E_BP - E_0,

where E_0 is the neutral-lattice reference energy and all energies use the same convention.

A state is classified as bound only if Delta_BP > 0 and the pair-correlation function remains localized with increasing system size.

### 2.5 Bipolaron observables

Required observables include:

- onsite probability P_0 = sum_i |Psi_ii|^2;
- nearest-neighbour probability P_1;
- mean carrier separation <r_12>;
- separation distribution P(r);
- one-particle density n_i = gamma_ii;
- pair participation ratio / inverse participation ratio;
- Holstein displacement u_i;
- Peierls bond distortions Delta v^x and Delta v^y;
- binding energy Delta_BP;
- singlet/triplet energy difference when the triplet sector is implemented.

These quantities distinguish unbound polarons, onsite bipolarons, nearest-neighbour/intersite bipolarons, and more extended correlated states.

## 3. Exciton model

### 3.1 Electron-hole basis

Use a product basis |i_e,j_h> with an electron in a conduction-like molecular orbital (LUMO sector) and a hole in a valence-like molecular orbital (HOMO sector). The wavefunction Psi^X_ij is not required to be symmetric because electron and hole are distinguishable quasiparticles.

A same-molecule configuration i=j is Frenkel-like; configurations i!=j are charge-transfer-like. A single electron-hole real-space basis can therefore interpolate continuously between local and charge-transfer character.

### 3.2 Separate electron and hole one-particle Hamiltonians

The electron and hole must not share parameters by assumption. Define

H_e[q] = H(J_e^x,J_e^y,A_e,alpha_e^x,alpha_e^y;q),
H_h[q] = H(J_h^x,J_h^y,A_h,alpha_h^x,alpha_h^y;q).

The hole Hamiltonian is a quasihole Hamiltonian referenced to the valence band; it should not be implemented merely as the negative of the electron Hamiltonian. Its transfer integrals and electron-phonon derivatives must ultimately be parameterized from HOMO-based electronic structure data.

### 3.3 Electron-hole interaction

The minimal exciton Hamiltonian is

H_X[q] = H_e[q] tensor I + I tensor H_h[q] - W_eh + H_ex,

where W_eh(i,j) is the screened electron-hole attraction. A practical hierarchy is:

1. effective onsite attraction W_0 plus nearest-neighbour W_1;
2. screened 1/r interaction for charge-transfer states;
3. anisotropic or environment-dependent screening when material-specific data are available.

H_ex represents short-range exchange and is needed to distinguish singlet and triplet excitons quantitatively. It may initially be omitted in a charge-transfer-focused proof of concept, but that limitation must be explicit.

### 3.4 Lattice forces

Define reduced one-particle density matrices

gamma^e_mn = sum_j Psi^X_mj Psi^{X*}_nj,

gamma^h_mn = sum_i Psi^X_im Psi^{X*}_in.

Then the local Holstein derivative is

dE/du_i = K_1 u_i + A_e gamma^e_ii + A_h gamma^h_ii.

Peierls derivatives are the sum of independent electron and hole bond-density contributions, each weighted by its own coupling constant.

A central physical point is that charge neutrality does not imply zero structural relaxation. A_e and A_h are derivatives of quasiparticle/orbital energies with respect to molecular coordinates, not simply electrostatic charges with opposite signs. They may reinforce or cancel depending on the molecular electronic structure. The same applies to the Peierls derivatives of HOMO-HOMO and LUMO-LUMO transfer integrals.

### 3.5 Exciton binding and relaxation energies

The model should report at least two different quantities.

First, the relaxed binding relative to a separated polaron pair,

Delta_X = E_e+h^sep - E_X^relaxed.

Second, the structural relaxation (self-trapping) energy,

E_relax^X = E_X(q_neutral) - E_X(q_X^opt).

These answer different questions: electron-hole binding and lattice stabilization.

The present Holstein-Peierls code does not contain an absolute HOMO-LUMO quasiparticle gap, so the first implementation can predict relative binding and relaxation energies but not an absolute optical excitation energy unless additional material-specific electronic energies are introduced.

### 3.6 Exciton observables

Required observables include:

- electron density n_i^e;
- hole density n_i^h;
- electron-hole separation distribution P_eh(r);
- mean separation <r_eh>;
- onsite/Frenkel weight sum_i |Psi^X_ii|^2;
- charge-transfer weight 1 - onsite weight;
- exciton participation ratio;
- lattice relaxation fields u, Delta v^x, Delta v^y;
- binding energy Delta_X;
- lattice relaxation energy E_relax^X;
- singlet-triplet splitting when exchange is included.

## 4. Computational feasibility

For a 20x20 lattice, N=400 and the full two-particle coordinate space has N^2=160,000 amplitudes. The wavefunction itself is small (about 1.3 MB in float64 or 2.6 MB in complex128), but an explicit dense N^2 by N^2 Hamiltonian is impossible: it would contain approximately 2.56e10 elements.

The correct implementation is matrix-free. Reshape the wavefunction as an N by N array and apply the Hamiltonian as nearest-neighbour operations on each coordinate plus a diagonal interaction term. The matrix-vector product scales as O(z N^2), with z the coordination number, and memory scales as O(N^2).

For the bipolaron singlet, symmetry could later reduce the state space to N(N+1)/2, but the initial implementation should prefer the full N by N representation because it is simpler to validate.

A scipy.sparse.linalg.LinearOperator combined with eigsh or LOBPCG is a natural CPU reference implementation. The existing optimized single-particle sparse solver provides the required building blocks.

## 5. Initial parameter strategy

The current example parameter set is strongly coupled:

A = 3.0 eV/A,
K_1 = 16.51 eV/A^2,
J_x = 0.100 eV,
J_y = 0.015 eV,
alpha_x = alpha_y = 0.4 eV/A,
K_2 = 0.51 eV/A^2.

The atomic-limit Holstein relaxation scale for one full local occupation is

E_H = A^2/(2K_1) approximately 0.273 eV,

and the additional local lattice stabilization available when two occupations share a site relative to two separate sites is of order

A^2/K_1 approximately 0.545 eV.

This indicates that the present input is a useful strong-coupling numerical laboratory for a bipolaron proof of concept. It should not, however, be treated as a quantitatively realistic pentacene parameterization without independent validation. The original literature identifies 50-100 meV polaron formation energies as the more relevant range for molecular crystals, while the current example produces a much larger single-polaron stabilization.

For the first bipolaron phase diagram, scan dimensionless ratios rather than claiming material specificity:

- U / E_H;
- V_1 / E_H;
- J_y / J_x (anisotropy);
- A^2/(K_1 J_x) as a Holstein-coupling measure;
- alpha_x^2/(K_2 J_x) and alpha_y^2/(K_2 J_y) as Peierls-coupling measures.

## 6. Recommended implementation order

1. Bipolaron, singlet, onsite U only, Holstein coupling only.
2. Validate against limiting cases and known 2D Holstein-Hubbard behaviour.
3. Add nearest-neighbour / long-range repulsion.
4. Add Peierls coupling and map the combined Holstein-Peierls phase diagram.
5. Add triplet sector.
6. Only after the bipolaron solver is validated, implement electron-hole excitons with separate electron/hole parameters.
7. Add exchange and material-specific screening for quantitative exciton studies.

This order maximizes reuse of the existing static-polaron implementation while separating numerical validation from new material-specific assumptions.
