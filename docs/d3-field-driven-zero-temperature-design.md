# D3 field-driven zero-temperature coupled dynamics

## Scope

D3 extends the validated D2 deterministic Ehrenfest dynamics by adding only the D1 Peierls electric-field phase. The classical lattice remains deterministic and unthermostatted, and no decoherence or stochastic electronic process is introduced.

The central physical change relative to D2 is that the matter energy is no longer conserved. A prescribed external field performs work on the electronic-lattice subsystem.

## Energy-work balance

For the explicitly time-dependent Hamiltonian

\[
H(q,t),
\]

the matter energy is

\[
E_{\mathrm{matter}}(t)=V_{\mathrm{lattice}}(q)+T_{\mathrm{lattice}}+\frac{\langle\psi|H(q,t)|\psi\rangle}{\langle\psi|\psi\rangle}.
\]

The exact deterministic balance is

\[
E_{\mathrm{matter}}(t)-E_{\mathrm{matter}}(0)
=\int_0^t P_{\mathrm{field}}(t')dt',
\]

with

\[
P_{\mathrm{field}}=\frac{\langle\psi|\partial H/\partial t|\psi\rangle}{\langle\psi|\psi\rangle}.
\]

This residual, not raw energy drift, is the principal D3 validation observable.

## Field-aware Ehrenfest force

The D1 forward-bond hopping is

\[
t_{ij}(q)e^{i\phi_\alpha(t)}.
\]

Therefore the Peierls force must use the phase-rotated coherence

\[
\mathrm{Re}[\psi_i^* e^{i\phi_\alpha}\psi_j],
\]

rather than the zero-field real coherence used by D2. The Holstein population term is unchanged.

At zero field the D3 force must reduce exactly to the D2 complex-state Ehrenfest force.

## Instantaneous field power

For each directed +x/+y bond,

\[
\partial_t(t e^{i\phi})=i\dot\phi\,t e^{i\phi}.
\]

The corresponding pair contribution to the expectation value is

\[
-2\dot\phi\,t\,\mathrm{Im}[\psi_i^*e^{i\phi}\psi_j].
\]

The implementation validates this expression independently against a central finite difference of the electronic energy with respect to time at fixed lattice and fixed propagated state.

## Split integrator

The first D3 production candidate reuses the D2 velocity-Verlet + CF4-Lanczos structure:

1. evaluate the field-aware Ehrenfest force at `(q_n, psi_n, t_n)`;
2. perform the first half kick;
3. drift the lattice to `q_{n+1}`;
4. propagate the electronic state across the linearly interpolated lattice path, evaluating both lattice and Peierls field at the actual absolute CF4 stage times;
5. evaluate the new field-aware force at `(q_{n+1}, psi_{n+1}, t_{n+1})`;
6. perform the second half kick.

External work is initially accumulated with the trapezoidal rule using the endpoint field powers. This is deliberately second order, matching the classical split. A full-system adaptive DOP853 reference will be added as an independent D3 gate.

## Initial validation gates

The first D3 implementation is required to pass:

1. exact zero-field reduction of electronic energy, Ehrenfest gradient, force, power, and one coupled step to D2;
2. central finite differences of the field-aware `u`, `vx`, and `vy` gradients for a complex propagated state;
3. central finite-difference validation of `P_field=<partial H/partial t>`;
4. electronic norm preservation for CF4-Lanczos without routine renormalization;
5. systematic reduction of the matter-energy-minus-field-work residual under timestep halving.

## Next D3 gates

Before D3 is closed, the implementation must also add:

- a complete coupled DOP853 reference including the field and an accumulated work variable;
- 20x20 relaxed-polaron benchmarks at the legacy-scale 2 mV/angstrom field;
- comparison of CF4-Lanczos and RK4 under the same driven trajectory;
- verification that the work-balance residual shows the expected second-order convergence;
- a multi-ps field-driven stability run before finite-temperature D4 work begins.

The 2 mV/angstrom field remains a legacy-scale numerical control rather than a material-specific transport prediction.