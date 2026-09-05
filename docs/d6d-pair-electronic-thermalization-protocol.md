# D6d pair electronic thermalization diagnostic protocol

## Purpose

D6d asks whether the coherent D6c pair Ehrenfest dynamics approaches the appropriate finite-temperature electronic distribution.  It is diagnostic only: no collapse, decoherence correction, reweighting, surface hop, velocity rescaling, or additional force is applied.

The D6c lattice bath is already validated separately.  Therefore D6d must not confuse correct lattice thermalization with correct electronic thermalization.

## Physical reference spaces

### Correlated bipolaron

The propagated bipolaron belongs to the symmetric spatial sector corresponding to the spin singlet.  Its instantaneous canonical reference is therefore constructed only in the orthonormal basis

- `|ii>`;
- `(|ij> + |ji>)/sqrt(2)` for `i < j`.

For `N` sites the physical diagnostic dimension is `N(N+1)/2`, not `N^2`.  Antisymmetric spatial states are excluded because they belong to a different exchange/spin sector.

### Distinguishable e-h exciton

The electron and hole are distinguishable in the current direct-interaction reference model.  The physical diagnostic space is the complete ordered pair basis of dimension `N^2`.  No exchange symmetrization is imposed and this model must not be labeled singlet or triplet.

## Instantaneous observables

For the propagated normalized state and instantaneous many-body eigenstates `|Phi_mu(q)>`, D6d records

`p_mu = |<Phi_mu|Psi>|^2`.

These are many-body adiabatic state populations.  They are not natural occupations, one-body orbital occupations, pair-distance probabilities, channel yields, or experimental quantum yields.

The canonical reference is

`p_mu^can ~ exp[-E_mu/(k_B T)]`,

restricted to the appropriate physical sector.  The uniform distribution is the bounded finite-spectrum `beta=0` reference.

Diagnostics include:

- energy-based heating coordinate, where 0 is canonical and 1 is uniform/infinite-temperature;
- early and late heating coordinates;
- linear heating trend per ps;
- total-variation and Jensen-Shannon distance to canonical;
- total-variation distance to uniform;
- fraction of samples closer to uniform than canonical;
- ground-manifold population versus canonical ground-manifold population;
- signed energy-matched `beta_eff/beta_bath` as a secondary nonlinear diagnostic;
- adiabatic-population Shannon entropy and participation number.

The adiabatic-population Shannon entropy is not the von Neumann entropy of the pure propagated state.

## Numerical control

The first D6d benchmark uses:

- 4 x 4 lattice;
- 300 K;
- `gamma_u = gamma_v = 0.01 fs^-1` numerical bath controls;
- `dt = 0.2 fs`;
- 4 ps total time;
- 1 ps burn-in;
- diagnostics every 50 fs;
- four independent seeds per sector;
- projected intermolecular zero modes;
- CF4-Lanczos with `m=8`.

This gives physical dimensions 136 for the bipolaron singlet sector and 256 for the distinguishable exciton.

## Gate semantics

`PASS` from `scripts/run_d6d_local_validation.py` means only:

1. D6 focused regression tests pass;
2. full pytest passes;
3. the D6d diagnostic benchmark executes;
4. the validated D6c lattice temperature, energy-bath balance, norm, and pair-sector constraints remain controlled.

There is deliberately no automatic pass/fail criterion for electronic overheating and no weighted score combining thermalization metrics.

## Manual decision logic

After the local ensemble is available:

- If heating is near zero, late-time heating is stable, canonical distances are small, and the ground-manifold population is consistent with canonical values across both sectors, coherent pair Ehrenfest dynamics may be adequate for this numerical control and no pair IDC should be added solely by analogy with D5.
- If heating increases systematically toward the uniform limit while the lattice remains correctly thermalized, a pair-specific decoherence/thermalization correction is justified.
- If only one sector overheats, corrections must be sector-specific rather than imposed globally.

Even if a correction is required, the one-particle IDC-BM choice and `t_d=180 fs` are not transferred automatically.  Any pair decoherence rule must respect the bipolaron symmetric singlet manifold and the distinguishable e-h structure, and its phenomenological interval must be revalidated in the pair model.