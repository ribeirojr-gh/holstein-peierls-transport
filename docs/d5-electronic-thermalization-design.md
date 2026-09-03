# D5 electronic thermalization and decoherence: diagnostic-first design

## Why D5 is separate from D4

D4 validated the classical Langevin lattice bath and its coupling to coherent one-carrier Ehrenfest dynamics. That does not guarantee that the electronic subsystem samples the correct finite-temperature equilibrium distribution.

Published organic-semiconductor studies report two important facts that motivate a diagnostic-first D5:

- Si and Wu, J. Chem. Phys. 143, 024103 (2015), DOI 10.1063/1.4926534, found that coherent Ehrenfest dynamics tends toward an electronic infinite-temperature limit in their nonlocal electron-phonon model, while instantaneous-decoherence schemes with detailed-balance reweighting restore near-equilibrium occupations.
- Runeson, Drayton and Manolopoulos, J. Chem. Phys. 161, 144102 (2024), DOI 10.1063/5.0226001, showed that a mapping surface-hopping approach preserves the equilibrium distribution by construction. They also found mobilities close to Ehrenfest in some regimes even though Ehrenfest overheats the electronic subsystem.
- Xie et al., J. Chem. Theory Comput. (2020), DOI 10.1021/acs.jctc.9b01271, showed that the performance of mixed quantum-classical methods depends on the transport regime and that mean-field Ehrenfest can still be quantitatively useful in sufficiently band-like cases.

Therefore D5 must not assume in advance that a correction is either mandatory or harmless.

## D5a: diagnostics only

D5a will reuse the validated D4 thermal trajectory without modifying any force, propagator, stochastic impulse, or electronic state.

At selected trajectory snapshots, diagonalize the instantaneous one-carrier Hamiltonian and evaluate

`p_l = |<phi_l(q(t))|psi(t)>|^2`.

Compare these propagated adiabatic occupations with the instantaneous canonical reference

`p_l^B = exp[-beta epsilon_l] / Z`,

at the same lattice-bath temperature.

Required scalar diagnostics are:

1. propagated electronic energy `sum_l p_l epsilon_l`;
2. canonical electronic energy at the bath temperature;
3. uniform/infinite-temperature electronic energy `mean(epsilon_l)`;
4. normalized heating coordinate

   `(E_propagated-E_canonical)/(E_infinite-E_canonical)`;

5. total-variation and Jensen-Shannon distances to the canonical occupation vector;
6. total-variation distance to the uniform occupation vector;
7. ground-state occupation versus canonical ground-state weight;
8. adiabatic occupation entropy and participation number;
9. an energy-matched effective inverse temperature `beta_eff`, reported primarily through `beta_eff/beta_bath`.

The adiabatic occupation entropy is a basis-dependent occupation diagnostic; it must not be called the von Neumann entropy of the pure propagated state.

## D5a decision gate

Run the same 20x20, 300 K, gamma=0.01 fs^-1, projected-zero-mode control validated in D4 for 10 ps. Sample the electronic diagnostics only after thermal burn-in and at a coarse interval so full eigendecomposition is a diagnostic cost rather than a propagation cost.

Interpretation:

- `beta_eff/beta_bath ~ 1`, small canonical distances and stable canonical ground-state weight: coherent Ehrenfest is thermally acceptable for this control; no correction is introduced solely for formal reasons.
- `beta_eff/beta_bath -> 0`, heating coordinate -> 1, and occupations become closer to uniform than canonical: direct evidence of electronic overheating in this model; proceed to D5b corrections.
- intermediate or regime-dependent behavior: retain Ehrenfest as a documented baseline and benchmark corrections before any transport claim.

## D5b only if required by D5a

Candidate extensions, in increasing architectural cost, are:

1. instantaneous decoherence with explicit detailed-balance reweighting, including a Boltzmann control;
2. corrected surface-hopping variants with explicit treatment of decoherence and trivial crossings;
3. mapping-based surface hopping as an equilibrium-consistent reference where feasible.

No correction will be selected from literature reputation alone. The comparison must use equilibrium occupations, energy relaxation, diffusion/drift observables, Einstein consistency where applicable, numerical stability, and computational cost on the actual Holstein-Peierls controls.

D6 pair dynamics remains downstream of this choice.
