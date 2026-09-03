# D4b coupled finite-temperature lattice bath

## Scope

D4b couples the D4a-validated classical BAOAB Langevin bath to the validated one-carrier Ehrenfest dynamics. The electron remains coherently propagated; electronic decoherence and detailed-balance corrections remain D5.

## Operator ordering

The classical lattice step is

`B(dt/2) A(dt/2) O(dt) A(dt/2) B(dt/2)`.

The first B kick uses the propagated electronic state at the beginning of the interval. The two A drifts define a continuous piecewise-linear lattice path with a possible velocity change at the O step. The electronic CF4-Lanczos propagator samples this actual path at its internal stage times. The propagated state at the end of the interval supplies the final B force.

For the `retain` zero-mode policy and zero friction, the implementation delegates directly to the validated D2/D3 step. Therefore the deterministic limit is exact and consumes no random numbers.

## Bath heat and field work

The exact O substep changes only lattice velocity. D4b therefore records

`Q_bath = K_after_O - K_before_O`

for each step. This sign convention is positive when the bath delivers energy to the matter subsystem.

At zero electric field, the finite-step balance diagnostic is

`Delta E_matter - Q_bath`.

With a prescribed electric field, the D3 work term is retained and the diagnostic becomes

`Delta E_matter - Q_bath - W_field`.

These residuals are numerical splitting diagnostics, not conserved physical energies of the thermostatted subsystem.

## Intermolecular collective zero modes

The elastic and electronic Hamiltonians depend on spatial differences of `vx` and `vy`, so each field has one exact uniform collective mode. D4b exposes two explicit policies:

- `retain`: retain both collective coordinates and use `3N` kinetic degrees of freedom;
- `project`: constrain uniform `vx`/`vy` coordinates and velocities to zero and use `3N-2` kinetic degrees of freedom.

The projected Ornstein-Uhlenbeck velocity is obtained by removing its uniform component after the stochastic draw. This is the Gaussian OU process projected onto the constrained subspace. The same caller-owned RNG stream is used in both policies.

The production default is not accepted until the local benchmark confirms that projection removes only redundant collective diffusion and leaves internal lattice differences and electronic propagation unchanged within numerical precision.

## Validation gates

D4b must pass all of the following before closure:

1. exact zero-friction `retain` reduction to D2 at zero field;
2. exact zero-friction `retain` reduction to D3 with field, including field work;
3. zero-friction path consumes no RNG variates;
4. fixed-seed finite-temperature reproducibility;
5. CF4-Lanczos electronic norm preservation without renormalization;
6. projected `vx`/`vy` coordinate and velocity means remain at roundoff and use `3N-2` degrees of freedom;
7. `retain` and `project` give the same gauge-invariant intermolecular differences and electronic state for the same raw stochastic stream;
8. complete pytest regression remains green;
9. a 20x20, 300 K timestep/statistics comparison at `dt=0.2` and `0.1 fs`;
10. a projected-mode 20x20, 300 K, 10 ps stability gate at `dt=0.2 fs`.

The numerical bath controls used for this validation are `gamma_u = gamma_v = 0.01 fs^-1`, target temperature 300 K, and fixed seed `20260903`. These are validation controls, not material-specific damping predictions.

No mobility or steady-state transport coefficient is interpreted in D4b. Such observables require the D4 bath to be closed first and the D5 electronic-thermalization/decoherence question to be addressed explicitly.
