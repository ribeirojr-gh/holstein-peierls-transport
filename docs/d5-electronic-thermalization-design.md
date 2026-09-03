# D5 electronic thermalization and decoherence: diagnostic-first design

## Why D5 is separate from D4

D4 validated the classical Langevin lattice bath and its coupling to coherent one-carrier Ehrenfest dynamics. That does not guarantee that the electronic subsystem samples the correct finite-temperature equilibrium distribution.

Published organic-semiconductor studies report two important facts that motivate a diagnostic-first D5:

- Si and Wu, J. Chem. Phys. 143, 024103 (2015), DOI 10.1063/1.4926534, found that coherent Ehrenfest dynamics tends toward an electronic infinite-temperature limit in their nonlocal electron-phonon model, while instantaneous-decoherence schemes with detailed-balance reweighting restore near-equilibrium occupations.
- Runeson, Drayton and Manolopoulos, J. Chem. Phys. 161, 144102 (2024), DOI 10.1063/5.0226001, showed that a mapping surface-hopping approach preserves the equilibrium distribution by construction. They also found mobilities close to Ehrenfest in some regimes even though Ehrenfest overheats the electronic subsystem.
- Xie et al., J. Chem. Theory Comput. (2020), DOI 10.1021/acs.jctc.9b01271, showed that the performance of mixed quantum-classical methods depends on the transport regime and that mean-field Ehrenfest can still be quantitatively useful in sufficiently band-like cases.

Therefore D5 must not assume in advance that a correction is either mandatory or harmless.

## D5a: diagnostics only

D5a reuses the validated D4 thermal trajectory without modifying any force, propagator, stochastic impulse, or electronic state.

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
7. ground-manifold occupation versus canonical ground-manifold weight;
8. adiabatic occupation entropy and participation number;
9. an energy-matched signed effective inverse temperature `beta_eff`, reported primarily through `beta_eff/beta_bath`.

The signed effective inverse temperature is well defined because the finite electronic spectrum is bounded. `beta_eff=0` is the uniform/infinite-temperature reference; negative values indicate population energies above the uniform mean. It is a diagnostic parameter, not an assertion that the pure electronic state is itself a canonical ensemble.

The adiabatic occupation entropy is basis-dependent and must not be called the von Neumann entropy of the pure propagated state.

## Implemented D5a API

`src/holstein_peierls/dynamics/electronic_thermalization.py` contains reusable definitions for:

- positive-temperature and signed-beta Boltzmann occupation vectors;
- distribution energies;
- total-variation and Jensen-Shannon distances;
- occupation Shannon entropy and participation number;
- energy-matched signed effective inverse temperature;
- the complete instantaneous `ElectronicThermalizationSnapshot`.

These routines are deliberately independent of the trajectory integrator so the same observables can later compare coherent Ehrenfest, instantaneous-decoherence corrections, surface hopping, or mapping-based schemes without changing definitions.

## D5a numerical definition tests

`tests/test_d5_electronic_thermalization.py` fixes the following controls:

1. Boltzmann weights are normalized and invariant under a uniform energy shift;
2. the effective-beta inversion recovers prescribed positive and negative canonical controls;
3. uniform populations map exactly to `beta_eff=0` and participation number `N`;
4. probability distances have the correct identical/disjoint limits;
5. a pure coherent superposition constructed with Boltzmann amplitudes reproduces the canonical adiabatic population vector, zero heating coordinate, and `beta_eff/beta_bath=1`;
6. a uniform adiabatic superposition gives heating coordinate 1 and infinite effective temperature;
7. propagated electronic energy agrees with the direct Hamiltonian expectation and all diagnostics are invariant under a global electronic phase.

## D5a production diagnostic gate

The implemented benchmark `experiments/d5a_electronic_thermalization.py` uses the same finite-temperature control already validated in D4:

- lattice: 20x20;
- lattice bath: 300 K;
- `gamma_u = gamma_v = 0.01 fs^-1`;
- timestep: 0.2 fs;
- projected uniform `vx`/`vy` zero modes;
- CF4-Lanczos, Krylov dimension 6;
- total trajectory time: 10 ps;
- thermal burn-in: 2 ps;
- electronic diagnostic interval: 100 fs;
- four independent RNG seeds: 20260903, 20260904, 20260905, 20260906.

Full instantaneous eigendecomposition is performed only at the coarse diagnostic snapshots, not during every propagation step.

For each trajectory the benchmark reports time-averaged and early/late values of the heating coordinate, its slope, `beta_eff/beta_bath`, distances to canonical and uniform populations, occupation entropy, participation number, ground-manifold population, lattice temperature, electronic norm error, and the D4 energy/bath residual. An ensemble mean and trajectory-to-trajectory standard deviation are then reported.

## D5a decision gate

Interpretation is based on the numerical evidence rather than an automatic threshold:

- `beta_eff/beta_bath ~ 1`, small canonical distances and stable canonical ground-state weight: coherent Ehrenfest is thermally acceptable for this control; no correction is introduced solely for formal reasons.
- `beta_eff/beta_bath -> 0`, heating coordinate -> 1, and occupations become closer to uniform than canonical: direct evidence of electronic overheating in this model; proceed to D5b corrections.
- negative `beta_eff`, heating coordinate greater than 1, or persistent drift beyond the uniform-energy reference: stronger evidence that coherent mean-field dynamics is not sampling the desired electronic equilibrium.
- intermediate or regime-dependent behavior: retain Ehrenfest as a documented baseline and benchmark corrections before any transport claim.

The local runner `scripts/run_d5a_local_validation.py` also repeats the complete pytest suite, D0a benchmark, and all six S0 relaxation gates. Its `PASS` means only that the diagnostic executed successfully and prior numerical regressions remain intact. It intentionally does **not** label the Ehrenfest physics as passed or failed; that decision is made from the generated D5a JSON.

## D5b only if required by D5a

Candidate extensions, in increasing architectural cost, are:

1. instantaneous decoherence with explicit detailed-balance reweighting, including a Boltzmann control;
2. corrected surface-hopping variants with explicit treatment of decoherence and trivial crossings;
3. mapping-based surface hopping as an equilibrium-consistent reference where feasible.

No correction will be selected from literature reputation alone. The comparison must use equilibrium occupations, energy relaxation, diffusion/drift observables, Einstein consistency where applicable, numerical stability, and computational cost on the actual Holstein-Peierls controls.

D6 pair dynamics remains downstream of this choice.
