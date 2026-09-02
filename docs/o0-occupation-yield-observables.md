# O0 occupation-number and channel-yield observables

## Scope

O0 defines electronic observables before any new time propagator is selected.
The same functions must therefore work with static states and with future RK,
Krylov, Magnus, Ehrenfest, and MCTDHF trajectories.

This checkpoint does **not** introduce dynamics. It fixes what will be measured
once dynamics is implemented.

## Instantaneous occupation number

The nonadiabatic polaron/bipolaron literature supplied for validation defines
the occupation of an instantaneous electronic level `l` as

`n_l(t) = sum_k f_k |<phi_l(t)|psi_k(t)>|^2`,

where `psi_k(t)` are the actually evolved one-electron orbitals, `f_k` are their
initial occupation factors, and `phi_l(t)` are instantaneous eigenstates of the
electronic Hamiltonian. See Sun et al., *Organic Electronics* 11, 279-284
(2010), DOI `10.1016/j.orgel.2009.11.006`, and Sun and Stafstrom,
*J. Chem. Phys.* 135, 074902 (2011), DOI `10.1063/1.3624730`.

O0 uses the equivalent one-particle reduced-density-matrix form

`n_l(t) = <phi_l(t)|gamma^(1)(t)|phi_l(t)>`,

with

`gamma^(1) = sum_k f_k |psi_k><psi_k|`

for an independent-orbital representation. This form generalizes directly to
correlated and multiconfigurational states because only the one-particle RDM is
required.

The implementation is `instantaneous_occupation_numbers()` and the direct
orbital form is `occupation_numbers_from_propagated_orbitals()`.

### Conservation gate

If the instantaneous orbitals form a complete orthonormal basis,

`sum_l n_l = Tr(gamma^(1)) = N_e`.

This identity is a permanent O0 regression test.

### Distinguishable exciton baseline

The existing spin-blind distinguishable electron-hole solver already exposes
separate electron and hole RDMs, each with trace one. Both can therefore be fed
directly to the O0 occupation API. This keeps the already validated baseline
compatible with the later spin-adapted/MCTDHF layer.

### What is not called an occupation number here

The eigenvalues of `gamma^(1)` are **natural-orbital occupations**. They are
useful observables but are conceptually different from the instantaneous
Hamiltonian-level occupations used in the supplied dynamics literature. O0 does
not silently interchange these definitions.

## Occupation number and diagonalization cost

The O0 definition does not require full diagonalization at every integration
step. The propagated state/RDM exists independently of the instantaneous energy
basis. During production dynamics we may therefore evaluate instantaneous
energy-level occupations only at an output/analysis stride, and later benchmark
whether a partial eigensolver around the gap is sufficient for the requested
levels.

This is important because full dense diagonalization at every electronic step
is the dominant legacy dynamics bottleneck.

## Yield of a single electronic configuration

The supplied reaction-dynamics papers define the relative yield of an electronic
configuration `K` through projection,

`I_K(t) = |<Phi_K(t)|Psi(t)>|^2`.

Here `Psi(t)` is the evolved many-electron state and `Phi_K(t)` is a state of
interest constructed from instantaneous electronic orbitals. This definition is
used explicitly in the supplied polaron-bipolaron and bipolaron-recombination
references.

For two Slater determinants, O0 uses the standard relation

`<Phi_L|Phi_R> = det(M)`,

where `M_ij = <phi_i|psi_j>`. This is also the projection relation used by
Miranda et al., *J. Chem. Phys.* 134, 244102 (2011), DOI
`10.1063/1.3600404`.

The corresponding implementation is `slater_determinant_overlap()`.

## Channel yield

A physical reaction product is often represented by more than one electronic
configuration. O0 therefore distinguishes a single-configuration yield from a
**channel yield**.

For a normalized state and an orthogonal projector `P_C` onto channel `C`,

`Y_C(t) = <Psi(t)|P_C|Psi(t)>`.

For a density operator,

`Y_C(t) = Tr[rho(t) P_C]`.

If channel states are orthonormal this reduces to the familiar sum of squared
projection amplitudes. If they are not orthogonal, simply summing those squares
would double count their common subspace. For channel-state columns collected
in `S`, O0 constructs

`P_C = S (S^dagger S)^+ S^dagger`,

using the Moore-Penrose pseudoinverse. The resulting projector is Hermitian and
idempotent even if the supplied channel list is redundant.

The relevant functions are `channel_projector()`,
`channel_yield_from_state()`, and `channel_yield_from_density()`.

## Multiconfigurational interference

For a multiconfigurational state, determinant/configuration amplitudes must be
combined coherently before taking an absolute square. O0 therefore does not
calculate a MCTDHF yield by independently squaring every determinant overlap.

For two determinant expansions

`|L> = sum_A c_A^L |Phi_A^L>`

and

`|R> = sum_B c_B^R |Phi_B^R>`,

the determinant overlap matrix is

`S_AB = <Phi_A^L|Phi_B^R>`

and the coherent numerator is

`(c^L)^dagger S c^R`.

O0 evaluates the self-overlap metrics as well, so that the returned overlap is
normalized even when the two expansions use different instantaneous orbital
sets. `multiconfigurational_yield()` then returns the modulus squared of this
normalized coherent overlap.

The functions are `determinant_overlap_matrix()`,
`multiconfigurational_overlap()`, and `multiconfigurational_yield()`.

## Interference regression

A permanent test uses two orthogonal configurations `|A>` and `|B>`:

`|Psi_+> = (|A> + |B>)/sqrt(2)`

and

`|Psi_-> = (|A> - |B>)/sqrt(2)`.

Their correct yield against each other is zero. A naive sum of the individual
configuration probabilities would instead lose the destructive interference.
This test guards the future MCTDHF implementation against that error.

## Terminology: yield is not automatically radiative quantum yield

Throughout the code, `yield` means a probability associated with a quantum
projector or a configuration overlap. It is not automatically an experimental
radiative quantum yield. Predicting the latter requires additional physical
information such as radiative/nonradiative rates and photon-emission channels.

Likewise, real-space quantities such as exciton onsite probability, CT
probability, or carrier separation are useful structural observables but are not
renamed `yield` unless a corresponding many-electron channel projector has been
defined.

## Validation gates

O0 is accepted only if all of the following hold:

- RDM and direct-overlap occupation formulas agree;
- complete-basis occupations sum to `Tr(gamma^(1))`;
- occupations are invariant under independent orbital phases;
- the existing distinguishable exciton electron/hole RDMs use the same API;
- non-orthogonal/redundant channel states generate Hermitian idempotent
  projectors;
- pure-state and mixed-state channel yields reproduce analytic 0, 1, and
  fractional controls;
- Slater determinant overlaps reproduce `det(L^dagger R)`;
- multiconfigurational amplitudes exhibit the correct constructive/destructive
  interference before squaring;
- all observables support complex-valued states, as required for dynamics.

No time propagator, thermostat, electric field, or GPU implementation is part of
O0. After O0 is consolidated, D0a may compare the legacy spectral propagator,
RK4, RK8, Krylov exponential, and commutator-free Magnus/Krylov using one common
set of validated observables.
