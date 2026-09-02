# O0 occupation numbers and projection yields

## Scope

O0 introduces electronic observables before any new time propagator is chosen.
The implementation is therefore independent of RK, Krylov, Magnus, MCTDHF,
thermostats, fields, or GPU backends.

The definitions follow the attached charge-transfer/recombination literature.
The historical occupation number of instantaneous level `l` is

`n_l(t) = sum_k f_k |<phi_l(t)|psi_k(t)>|^2`,

where `psi_k` are propagated occupied orbitals and `phi_l` are instantaneous
one-particle eigenstates. The canonical implementation is the equivalent RDM
form

`n_l(t) = <phi_l(t)| gamma^(1)(t) |phi_l(t)>`.

This form is essential for the spin-adapted framework because it remains valid
when a single set of independently propagated occupied orbitals no longer
exists.

The implementation accepts either a complete instantaneous basis or a selected
spectral window. Consequently, future dynamics does not need a full dense
diagonalization solely to report occupations near the gap.

## Relative configuration yield

For a normalized evolved many-electron state `|Psi(t)>` and an electronic
configuration `|Phi_K>`, the projection-method yield is

`I_K(t) = |<Phi_K|Psi(t)>|^2`.

For two Slater determinants built from normalized occupied orbitals, the
many-electron overlap is evaluated without constructing the determinants
explicitly:

`<Phi_A|Phi_B> = det(S)`,

where `S_ij = <a_i|b_j>`.

For a multiconfigurational state

`|Psi> = sum_beta c_beta |D_beta>`,

configuration amplitudes are summed coherently before taking the modulus. This
preserves interference and is required by the future MCTDHF layer.

## Physical channels rather than single configurations

Several physical products can correspond to many electronic configurations.
The recombination references explicitly sum contributions from all
configurations belonging to the same state/channel. O0 generalizes this with a
projector onto the span of the reference configurations.

For reference configurations `|Phi_a>`, define

`G_ab = <Phi_a|Phi_b>`

and

`v_a = <Phi_a|Psi>`.

The normalized channel yield is

`Y_C = v^dagger G^+ v / <Psi|Psi>`,

where `G^+` is the Moore-Penrose pseudoinverse. For orthonormal reference
configurations this reduces to the ordinary sum of individual probabilities.
For non-orthogonal or redundant references it avoids double counting.

This quantity is a state/channel projection probability. It must not be called
a radiative quantum yield unless radiative and non-radiative kinetic processes
are included separately.

## Validation gates

The O0 kernel tests require:

- exact equivalence of the RDM occupation formula and the historical overlap
  formula;
- preservation of total particle number when a complete instantaneous basis is
  supplied;
- support for a partial instantaneous spectral window;
- unit yield under occupied-orbital unitary rotations of a determinant;
- zero yield for orthogonal determinants;
- coherent interference for multiconfigurational expansions;
- correct Gram-projector behavior for non-orthogonal and redundant channel
  references.

The next O0 checkpoint will add adapters that construct the relevant RDMs and
configuration/channel definitions directly from the validated polaron,
bipolaron, distinguishable electron-hole exciton, and spin-adapted S0 states.
