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

## Static-state adapters

O0 now exposes the validated stationary sectors through one common observable
boundary without modifying their Hamiltonians or relaxation solvers.

### One polaron

A one-polaron electronic ground state is mapped to

`gamma_pol = |psi><psi|`,

with trace one, and to the equivalent one-occupied-orbital Slater
configuration. The high-level `PolaronResult` is deliberately not converted
from charge density alone because `|psi_i|^2` does not contain the off-diagonal
coherences required by an RDM. Users must supply the underlying validated
`GroundState` rather than silently accepting a diagonal approximation.

### Correlated singlet bipolaron

The adapter reproduces the established spin-summed reduced density matrix

`gamma_BP = 2 Psi Psi^dagger`,

with trace two. The normalized ordered-pair wavefunction can also be flattened
in the same C-order convention used by the matrix-free solver.

### Distinguishable electron-hole exciton

The existing reference exciton remains spin-blind and distinguishable. O0 keeps
its two one-particle RDMs separate:

`gamma_e = Psi Psi^dagger`,

`gamma_h = Psi^dagger Psi`,

with unit trace for each carrier. No electron/hole RDM is relabelled as a
singlet/triplet object.

### Spin-adapted S0 state

The neutral and excited many-electron RDMs use the already validated S0
spin-summed convention

`gamma = sum_mu n_mu P_mu`.

The adapter returns `gamma_neutral`, `gamma_excited`, and

`Delta gamma = gamma_excited - gamma_neutral`.

For the neutral HOMO-to-LUMO excitation used by S0, the neutral and excited RDMs
have equal particle number and `Tr(Delta gamma)=0`.

For configuration-space projections, O0 also constructs an explicit
spin-orbital determinant expansion. The neutral state is one closed-shell
Slater determinant. Excited singlet and triplet states use the established
minimal two-determinant `M_S=0` coefficients from `spin_adapted.spin`. The
frontier determinant phases are fixed so the existing convention is preserved:
equal coefficient signs for the singlet and opposite signs for the triplet.
The static triplet orbital optimization may still use the equivalent high-spin
`M_S=1` open-shell functional; only the projection representation is converted
to the two-component `M_S=0` form.

## Ordered-product spatial channels

The bipolaron and distinguishable exciton live naturally in an ordered product
basis `|i,j>`, not in the single-determinant representation used by the Slater
yield kernel. O0 therefore defines a separate exact projector representation
for spatial pair channels.

A `ProductBasisChannel` stores only the canonical basis indices belonging to a
channel. The channel probability is

`Y_C = <Psi|P_C|Psi>/<Psi|Psi>`

and is evaluated by summing the probability on those selected orthogonal basis
states. The full projector is never materialized, avoiding O(N^4) memory.

For periodic rectangular controls, O0 provides minimum-image channel builders
for:

- onsite states, corresponding to the bipolaron onsite sector and the exciton
  Frenkel sector;
- nearest-neighbour x states;
- nearest-neighbour y states;
- first diagonal states; and
- an explicit user-selected minimum-separation channel.

Calling such a quantity a channel yield is justified here because an explicit
orthogonal projector has been defined. This remains distinct from an
experimental radiative quantum yield.

## Validation gates

The O0 kernel and adapter tests require:

- exact equivalence of the RDM occupation formula and the historical overlap
  formula;
- preservation of total particle number when a complete instantaneous basis is
  supplied;
- support for a partial instantaneous spectral window;
- unit yield under occupied-orbital unitary rotations of a determinant;
- zero yield for orthogonal determinants;
- coherent interference for multiconfigurational expansions;
- correct Gram-projector behavior for non-orthogonal and redundant channel
  references;
- preservation of the one-polaron rank-one RDM, including off-diagonal
  coherence;
- exact agreement of the bipolaron adapter with the existing spin-summed RDM
  and trace two;
- exact agreement of the exciton electron/hole adapters with their existing
  separate unit-trace RDMs;
- equal neutral/excited S0 particle numbers and zero-trace excitation RDM;
- normalized S0 determinant expansions with the established singlet/triplet
  coefficient convention; and
- ordered-product channel probabilities consistent with existing onsite and
  nearest-neighbour pair probabilities.

With these adapters in place, O0 no longer depends on a future propagation
choice. The next dynamics checkpoint is D0a: deterministic electronic
propagator benchmarking against the archived full-diagonalization spectral
reference before any production integrator is selected.
