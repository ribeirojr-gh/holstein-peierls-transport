# Spin-adapted exciton and projection-observable design

## Purpose

This document revises the exciton/dynamics roadmap after auditing the legacy
`hp2D.f90` occupation diagnostic and reviewing the projection/yield and
multiconfigurational literature supplied with the project.

The central conclusions are:

1. the legacy occupation-number calculation is not a reliable general
   definition and must be replaced by a reduced-density-matrix formulation;
2. reaction/exciton yields should be implemented as projections onto explicit
   many-body or pair-state subspaces, not inferred from single-level
   occupations;
3. the current static electron-hole solver is a useful spin-blind reference,
   but it cannot distinguish singlet and triplet excitons because it contains
   direct electron-hole attraction without the exchange/spin structure that
   generates singlet-triplet splitting; and
4. production singlet/triplet exciton calculations therefore require a
   spin-adapted electronic backend.  The minimal fixed-coefficient
   multiconfigurational time-dependent Hartree-Fock formulation of Miranda,
   Fisher, Stella, and Horsfield is the first implementation target because it
   was designed for large semiempirical conjugated-electron systems at a cost
   comparable to TDHF.  The software architecture must not assume that this is
   the only possible backend.

No existing validated spin-blind exciton result is invalidated by this design.
Those calculations remain reference results for the direct electron-hole
Holstein-Peierls model.

## 1. Occupation number: reference definition

Let `gamma1(t)` be the one-body reduced density matrix of the propagated
many-electron state and let `phi_l(t)` be an instantaneous orthonormal
single-particle adiabatic orbital.  The occupation of that orbital is

`n_l(t) = <phi_l(t) | gamma1(t) | phi_l(t)>`.

This is the definition that should be used throughout the modern code.  It
specializes correctly to all sectors of interest.

### One propagated carrier

For `gamma1 = |psi><psi|`,

`n_l(t) = |<phi_l(t)|psi(t)>|^2`,

with `sum_l n_l = 1`.

### Independent occupied orbitals / Slater determinant

For propagated occupied orbitals `psi_k` with occupation weights `f_k`,

`gamma1 = sum_k f_k |psi_k><psi_k|`,

and therefore

`n_l(t) = sum_k f_k |<phi_l(t)|psi_k(t)>|^2`.

This is the formula used in the supplied Sun/Stafstrom-type dynamics papers.

### Spin-adapted multiconfigurational state

For a multiconfigurational wavefunction, `gamma1` must be obtained from the
configuration/state definition rather than reconstructed as if the state were a
single determinant.  In the fixed-coefficient open-shell formalism of Miranda
et al., the equivalent instantaneous-orbital occupation is

`p_i = sum_mu sum_(j in shell mu) n_mu |<varphi_i|phi_(j,mu)>|^2`.

### Distinguishable electron-hole reference model

For the current pair wavefunction `Psi[i_e,i_h]`,

`gamma_e = Psi Psi^dagger`,

`gamma_h = Psi^dagger Psi`.

The corresponding instantaneous electron and hole occupations are

`n_a^e = <phi_a^e|gamma_e|phi_a^e>`,

`n_b^h = <phi_b^h|gamma_h|phi_b^h>`,

and each set sums to one.

Natural-orbital occupations (eigenvalues of a one-body RDM) are a different,
complementary diagnostic and must not be called the instantaneous adiabatic
level occupations.

## 2. Why the legacy `COEF` occupation code must not be retained

The archived dynamics stores the old instantaneous eigenvectors, diagonalizes
the new Hamiltonian, and computes

`COEF(i,j) = sum_k conj(evec_new(k,j)) evec_old(k,i)`.

It then combines `|COEF|^2` with the initial Fermi-like occupations.  This
measures overlap between two consecutive instantaneous eigenbases.  It does
not, in general, evaluate the projection of the **propagated occupied
orbitals/state** onto the current adiabatic basis.

The legacy code also propagates only the selected carrier wavefunction `psi`,
whereas the literature definition for a many-electron occupation sums the
projections of all currently propagated occupied orbitals (or, equivalently,
uses the one-body RDM).  The old `COEF` diagnostic is therefore retained only as
a historical observable for regression studies; it must not define modern
occupation numbers.

A key performance consequence is that the production propagator must not be
forced to diagonalize the full Hamiltonian at every integration step merely to
obtain occupations.  Instantaneous occupations are diagnostics and can be
sampled at output times.  For large systems, only the required spectral window
should be computed with a sparse eigensolver when a complete adiabatic spectrum
is unnecessary.

## 3. Yield: projection definition

For a normalized propagated many-body state `|Psi(t)>` and a normalized target
state `|Phi_K(t)>`, the state yield is

`Y_K(t) = |<Phi_K(t)|Psi(t)>|^2`.

This is the projection used in the supplied polaron-bipolaron and bipolaron-
recombination papers.

The more useful production definition is a **channel yield**.  For an
orthonormal set of target states spanning channel `C`,

`P_C = sum_(K in C) |Phi_K><Phi_K|`,

`Y_C(t) = <Psi(t)|P_C(t)|Psi(t)>`

`       = sum_(K in C) |<Phi_K(t)|Psi(t)>|^2`.

Examples include Frenkel-exciton, charge-transfer-exciton, separated/free-
carrier, biexciton, excited-polaron-plus-carrier, singlet-exciton, and
triplet-exciton channels.

If the target states are not mutually orthogonal, a naive sum of squared
overlaps double-counts probability.  The implementation must either construct
an orthonormal channel basis or use the corresponding Gram-matrix projector.

For two Slater determinants the overlap is

`<Phi|Psi> = det(S)`,

where `S_ij = <phi_i|psi_j>`.  For multiconfigurational states the complex
configuration amplitudes must be combined **before squaring**.  This is
essential to retain interference and spin symmetry.

The code must distinguish this projection yield from experimental radiative
quantum yield.  Radiative/nonradiative rates and photon emission are separate
physics and are not implied by `Y_C(t)` alone.

## 4. Singlet/triplet physics: what is actually required

A multiconfigurational representation is a natural way to construct spin-pure
open-shell excited states, but the statement "MCTDHF is the only correct method"
is too strong.

The indispensable ingredients are:

- a spin-adapted (or otherwise spin-resolved) electronic state;
- electron-electron/electron-hole exchange and the associated correlation
  structure needed to distinguish singlet and triplet sectors; and
- a propagation/static variational scheme consistent with that state
  definition.

Alternative frameworks can satisfy these requirements, including spin-adapted
configuration interaction, Bethe-Salpeter electron-hole Hamiltonians with a
proper exchange kernel, TD-CASSCF/TD-MCSCF variants, and other multireference or
spin-flip approaches.

For the present semiempirical Holstein-Peierls/PPP-style project, however, the
minimal open-shell formalism of Miranda et al. is an especially direct first
choice.  It keeps only the smallest spin-adapted determinant expansion, uses
fixed symmetry-determined configuration coefficients, and optimizes the
single-particle orbitals through the Dirac-Frenkel time-dependent variational
principle.

This should be described precisely as a **minimal fixed-coefficient
spin-adapted MCTDHF/open-shell TDHF variant**, not confused with a full
complete-active-space MCTDHF calculation with a large set of time-dependent CI
coefficients.

The distinction matters because the Miranda formalism itself notes that some
true two-electron processes require time-dependent configuration coefficients.
The minimal method therefore becomes the baseline spin-adapted backend, while
the architecture must permit a richer active-space/configuration-amplitude
backend later if validation shows that it is needed.

## 5. Consequence for the existing static exciton solver

The current `exciton` package uses a distinguishable ordered-pair wavefunction
`Psi[i_e,i_h]` with direct attractive interactions.  It explicitly omits
exchange integrals and triplet physics.  Consequently, its present Hamiltonian
is spin blind: attaching a singlet or triplet label to the same spatial pair
state would not generate a physical singlet-triplet splitting.

This solver remains valuable as:

- a direct electron-hole binding benchmark;
- a matrix-free two-particle numerical benchmark;
- a Frenkel/CT/separated topology reference; and
- a future comparison target for the spin-adapted model when exchange is taken
  to zero.

It must not be silently reinterpreted as a physical singlet exciton.

Before production exciton dynamics, the stationary sector should therefore gain
a spin-adapted excited-state milestone.  Since MCTDHF is intrinsically
*time-dependent*, the corresponding static calculation is a state-specific
spin-adapted multiconfigurational self-consistent-field/open-shell Hartree-Fock
problem using the same state parameters and energy functional.

## 6. First spin-adapted states

The first implementation should reproduce the shell constructions in Miranda et
al.

### Closed-shell singlet reference

One doubly occupied shell with occupation `n=2`.

### Open-shell singlet exciton

The minimal spin-pure open-shell singlet is a two-determinant state, with the
two singly occupied spatial orbitals in separate shells and the
symmetry-determined coefficients `1/sqrt(2)` (up to the selected determinant
phase convention).

The corresponding shell state parameters are

`a = [[1,1,1], [1,1,1], [1,1,1]]`

and

`b = [[ 1,  1,  1],`
`     [ 1,  2, -2],`
`     [ 1, -2,  2]]`.

### High-spin triplet exciton

A spin-pure `M_S=1` triplet can be represented with a high-spin open-shell
shell structure: one doubly occupied shell and one singly occupied shell.  The
state parameters are

`a = [[1,1], [1,1]]`

and

`b = [[1,1], [1,2]]`.

These are the first two excited-state sectors to validate.  Spin-orbit coupling
and intersystem crossing are deliberately excluded from this milestone; under
a spin-independent Hamiltonian singlet and triplet sectors are propagated
separately and do not interconvert.

## 7. Static validation gates before dynamics

The spin-adapted stationary extension should be promoted only after all of the
following tests pass:

1. closed-shell limit reproduces the ordinary restricted Hartree-Fock/static
   reference for the same semiempirical Hamiltonian;
2. zero electron-electron/exchange interaction makes the corresponding singlet
   and triplet spatial energetics degenerate within numerical tolerance;
3. finite exchange produces a nonzero singlet-triplet gap with the expected
   sign for the selected model/parameter convention;
4. spin-pure test states have the expected `<S^2>` values (0 for singlet, 2 for
   triplet, in units of hbar^2);
5. the analytic electronic energy agrees with direct many-body evaluation on
   tiny systems where exact determinant/configuration construction is feasible;
6. analytic structural forces agree with central finite differences;
7. a tiny active-space exact diagonalization/configuration-interaction control
   reproduces the qualitative state ordering and provides a quantitative error
   measure for the minimal fixed-coefficient approximation;
8. singlet and triplet relaxed lattice structures/energies are reproducible
   from multiple initial seeds; and
9. the existing spin-blind pair solver is recovered as an appropriate
   direct-interaction control when exchange/spin-dependent terms are disabled.

## 8. Revised dynamics protocol

The previous `D0-D6` sequence is retained but gains two prerequisites.

### S0 — spin-adapted static foundation

Implement and validate the stationary closed-shell, open-shell singlet, and
high-spin triplet state machinery described above.

### O0 — projection observables

Implement occupation-number and yield/projector APIs independently of the
dynamics integrator.  Validate probability sum rules, determinant overlaps,
multiconfigurational interference, spin-channel projectors, and reduced-density-
matrix occupations.

### D0a — linear frozen-H propagator benchmark

Retain the existing benchmark between exact diagonalization/exponential, RK4,
explicit RK8, Krylov/Lanczos exponential action, and commutator-free Magnus.
This benchmark remains essential for the one-carrier and direct pair-state
backends.

### D0b — spin-adapted multiconfigurational frozen-geometry benchmark

The MCTDHF orbital equations are nonlinear/state-dependent through their shell
Fock operators and projectors.  They therefore require a separate benchmark.
The supplied Miranda implementation explicitly recommends an eighth-order
Dormand-Prince integrator with adaptive step-size control; this should be the
high-accuracy baseline together with RK4 and progressively tightened adaptive
references.

Exponential/Krylov methods may still be useful for the instantaneous effective
orbital generator, but only after a self-consistent predictor/corrector or
commutator-free treatment of its time dependence is validated.  Results from
the linear TDSE benchmark must not be assumed automatically to transfer to the
nonlinear multiconfigurational equations.

D1-D5 then proceed as previously defined, with occupation/yield diagnostics
sampled at controlled output intervals.  D6 becomes the comparison and transfer
stage for polaron, bipolaron, spin-blind pair exciton, and spin-adapted
singlet/triplet exciton dynamics.

## 9. Performance rule for occupation/yield diagnostics

The old code tied adiabatic occupations to a full eigendecomposition at every
step.  The new architecture must avoid recreating that bottleneck.

- Propagation and diagnostics are separate interfaces.
- Site/local observables are evaluated every step when cheap.
- Adiabatic-level occupations and channel yields are evaluated at configurable
  sampling intervals.
- Full spectra are computed only when scientifically required.
- Near-gap or target states should use sparse/iterative eigensolvers when
  possible.
- GPU acceleration should focus first on batched matrix-free Hamiltonian/Fock
  actions and reduced-density/projector contractions rather than dense full
  diagonalization.

This separation is required both for scientific clarity and for preserving the
performance gains sought by the dynamics modernization.

## Literature anchors

The implementation and validation plan is anchored to the following references:

- R. P. Miranda, A. J. Fisher, L. Stella, and A. P. Horsfield, J. Chem. Phys.
  134, 244101 (2011), DOI 10.1063/1.3600397: fixed-coefficient
  multiconfigurational formalism and general open-shell states.
- R. P. Miranda, A. J. Fisher, L. Stella, and A. P. Horsfield, J. Chem. Phys.
  134, 244102 (2011), DOI 10.1063/1.3600404: coupling to Ehrenfest nuclei,
  shell Fock operators, occupations, singlet/triplet dynamics, and eighth-order
  Dormand-Prince integration.
- Z. Sun et al., Organic Electronics 11, 279-284 (2010), DOI
  10.1016/j.orgel.2009.11.006: instantaneous-level occupations and projected
  reaction yields.
- Z. Sun and S. Stafstrom, J. Chem. Phys. 135, 074902 (2011), DOI
  10.1063/1.3624730: determinant-projection yields for grouped recombination
  channels.
- Z. Sun and S. Stafstrom, J. Chem. Phys. 136, 244901 (2012), DOI
  10.1063/1.4729483: spin-dependent polaron recombination using a
  multiconfigurational spin-adapted dynamics.
- J.-W. van der Horst, P. A. Bobbert, M. A. J. Michels, and H. Bassler, J.
  Chem. Phys. 114, 6950-6957 (2001), DOI 10.1063/1.1356015: a Bethe-Salpeter
  electron-hole treatment reproducing singlet-triplet splittings, demonstrating
  that MCTDHF is not the unique spin-resolved exciton formalism.
- Z. Sun, S. Li, S. Xie, and Z. An, Organic Electronics 57, 277-284 (2018), DOI
  10.1016/j.orgel.2018.03.025: PPP-Peierls + CIS treatment of singlet and
  triplet excited states, providing another useful static comparison model.
