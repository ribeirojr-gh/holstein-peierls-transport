# D6c finite-temperature pair dynamics design

## Purpose

D6c transfers the already validated D4 BAOAB Langevin lattice bath to the D6 pair sectors after D6b established deterministic zero-temperature moving-lattice dynamics.

The electronic propagation remains coherent Ehrenfest dynamics. D6c does **not** assume that the D5 one-carrier decoherence correction can be copied unchanged into pair Hilbert spaces.

## Physics held fixed

- bipolaron: symmetric spatial singlet pair state in the ordered site-product basis;
- exciton: distinguishable electron-hole pair, with no exchange symmetrization;
- matrix-free complex-safe pair Hamiltonian action;
- propagated-state electronic energy;
- current lattice-coordinate-independent pair interactions;
- explicit lattice masses;
- CF4-Lanczos moving-H propagation with `m=8` as the current D6 control;
- no field, no electronic collapse, no mobility fit.

## Thermostat

The classical bath uses the D4 exact Ornstein-Uhlenbeck step,

`v' = exp(-gamma dt) v + sigma R`,

with

`sigma^2 = [1-exp(-2 gamma dt)] k_B T / m`.

The pair implementation uses explicit D6 lattice masses rather than the one-carrier parameter object.

The full coupled ordering is BAOAB:

`B(dt/2) A(dt/2) O(dt) A(dt/2) B(dt/2)`.

The electronic state samples the actual two-segment lattice path over the BAOAB step. The O step acts only on classical velocities, so

`Q_bath = K_after_O - K_before_O`.

At zero field the finite-step balance diagnostic is

`Delta E_matter ~= Q_bath`.

## Zero modes

The two uniform intermolecular coordinate modes remain exact zero modes because the elastic and Peierls energies depend only on coordinate differences.

D6c preserves the explicit D4 policies:

- `retain`: 3N kinetic degrees of freedom;
- `project`: uniform vx/vy coordinate and velocity means removed, 3N-2 kinetic degrees of freedom.

The benchmark uses `project`. An arbitrary initial state is never projected silently; callers must explicitly invoke `project_pair_zero_modes`.

## Exact reduction gate

For `gamma_u = gamma_v = 0` and zero-mode policy `retain`, one D6c step delegates directly to the validated D6b velocity-Verlet/CF4-Lanczos step. No random number is consumed. This is an exact implementation-level reduction gate.

## Pre-registered benchmark

Control:

- lattice: 4x4;
- pair Hilbert dimension: 256;
- bath: 300 K;
- gamma_u = gamma_v = 0.01 fs^-1 numerical controls;
- dt = 0.2 fs;
- duration = 2 ps;
- burn-in = 0.5 ps;
- sample interval = 5 fs;
- four independent seeds: 20260905--20260908;
- zero-mode policy: project;
- CF4-Lanczos m=8;
- both bipolaron and distinguishable-exciton sectors.

The small lattice is deliberate: D6c validates transfer of the bath/numerics, not a thermodynamic-limit observable.

## Closure metrics

For each sector the ensemble must satisfy:

1. post-burn-in mean kinetic lattice temperature between 240 and 360 K;
2. maximum absolute generalized energy residual below 1e-4 eV;
3. maximum electronic norm error below 1e-10;
4. maximum sector/RDM constraint error below 1e-10;
5. maximum projected zero-mode mean below 1e-12;
6. maximum lattice excursion below 2 A.

These are deliberately broad transfer gates, not material predictions.

## Interpretation boundary

Passing D6c means that the classical 300 K BAOAB bath can be coupled consistently to the two pair sectors for this control. It does not show that coherent pair Ehrenfest dynamics yields the correct electronic equilibrium distribution.

The next checkpoint must therefore be diagnostic-first, analogous in spirit to D5a but adapted to pair observables. A one-carrier IDC-BM collapse rule must not be copied into a correlated bipolaron or distinguishable electron-hole pair without first defining what electronic ensemble, adiabatic basis, and conserved/relaxed quantities the pair correction is intended to reproduce.
