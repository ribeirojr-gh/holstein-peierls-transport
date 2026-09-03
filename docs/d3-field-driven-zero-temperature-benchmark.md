# D3 field-driven zero-temperature coupled dynamics: validation results

## Scope

D3 adds the already validated D1 Peierls electric-field coupling to the D2 moving-lattice Ehrenfest equations. The lattice remains deterministic and unthermostatted. No stochastic force, decoherence model, electronic bath, or GPU approximation is present in this checkpoint.

The field used below, 2 mV/angstrom along +x, is a historical-scale numerical control. These results validate equations and integrators; they are not a material-specific mobility or transport prediction.

## Correct driven energy diagnostic

Because the prescribed vector potential makes the Hamiltonian explicitly time dependent, the matter energy is not conserved. The appropriate balance is

\[
E_{\mathrm{matter}}(t)-E_{\mathrm{matter}}(0)
= W_{\mathrm{field}}(t)
= \int_0^t \frac{\langle\psi|\partial H/\partial t|\psi\rangle}{\langle\psi|\psi\rangle}\,dt'.
\]

Therefore the principal D3 integration diagnostic is

\[
R_W(t)=\Delta E_{\mathrm{matter}}(t)-W_{\mathrm{field}}(t),
\]

not raw matter-energy drift.

The analytic field power and the phase-aware Ehrenfest force were independently validated by central finite differences before the production benchmark.

## Numerical methods

The D3 validation hierarchy is:

- **full coupled DOP853 + work**: independent adaptive reference integrating the electronic state, all lattice coordinates, all lattice velocities, and accumulated external work in a single `7N+1` ODE state;
- **velocity Verlet + CF4-Lanczos m=6**: structure-preserving production candidate inherited from D1/D2, now evaluated with the absolute field time at internal electronic stages;
- **velocity Verlet + RK4**: transparent nonunitary regression baseline.

For the split methods, the external work is integrated by the trapezoidal rule from endpoint field powers, consistent with the second-order classical splitting.

## 20x20 short benchmark

A 20x20 polaron was first relaxed with the validated sparse static solver. The lattice velocities were then set exactly to zero so that all subsequent motion is field induced. The field was switched on through the Peierls vector-potential gauge at `t=0` and the system was propagated for 10 fs.

The benchmark ran on a GitHub-hosted Ubuntu 24.04 runner with Python 3.12, NumPy 2.5.2, SciPy 1.18.1, and BLAS/OpenMP restricted to one thread. Timings are hardware dependent and are provided only for relative context.

| method | dt [fs] | elapsed [s] | H evals | H applications | matter dE [eV] | field work [eV] | max abs(dE-W) [eV] | final dE-W [eV] | norm error | electronic error vs DOP853 | lattice error vs DOP853 | velocity error vs DOP853 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CF4-Lanczos m=6 | 0.20 | 0.149398 | 100 | 600 | 4.786e-05 | 4.774e-05 | 2.037e-07 | 1.230e-07 | 3.331e-15 | 4.171e-09 | 6.650e-10 | 1.908e-04 |
| RK4 | 0.20 | 0.143010 | 200 | 200 | 4.786e-05 | 4.773e-05 | 2.031e-07 | 1.236e-07 | 2.424e-05 | 2.426e-05 | 6.371e-10 | 1.878e-04 |
| CF4-Lanczos m=6 | 0.10 | 0.302261 | 200 | 1200 | 4.786e-05 | 4.783e-05 | 5.090e-08 | 3.074e-08 | 3.775e-15 | 2.639e-10 | 1.672e-10 | 4.755e-05 |
| RK4 | 0.10 | 0.437950 | 400 | 400 | 4.786e-05 | 4.783e-05 | 5.085e-08 | 3.074e-08 | 7.605e-07 | 7.625e-07 | 1.656e-10 | 4.740e-05 |
| CF4-Lanczos m=6 | 0.05 | 0.779365 | 400 | 2400 | 4.786e-05 | 4.785e-05 | 1.272e-08 | 7.683e-09 | 3.020e-14 | 1.739e-11 | 4.187e-11 | 1.188e-05 |
| RK4 | 0.05 | 0.575168 | 800 | 800 | 4.786e-05 | 4.785e-05 | 1.272e-08 | 7.682e-09 | 2.379e-08 | 2.403e-08 | 4.178e-11 | 1.187e-05 |

### Tight full-system reference

For DOP853 with `rtol=1e-9`, `atol=1e-11`, and `max_step=0.02 fs`:

- elapsed: 3.8389 s;
- RHS/Hamiltonian evaluations: 6014;
- matter-energy change: `4.786e-05 eV`;
- accumulated field work: `4.786e-05 eV`;
- final energy-work residual: `1.0e-17 eV`;
- electronic norm error: `1.11e-16`.

The reference therefore closes the driven work balance essentially at floating-point accuracy for this control.

## Convergence interpretation

For CF4-Lanczos, halving the timestep gives

- maximum `|dE-W|`: `2.037e-07 -> 5.090e-08 -> 1.272e-08 eV`;
- final `dE-W`: `1.230e-07 -> 3.074e-08 -> 7.683e-09 eV`;
- velocity error vs DOP853: `1.908e-04 -> 4.755e-05 -> 1.188e-05`.

All three sequences decrease by approximately a factor of four, confirming that the global driven split is second order, as expected from velocity Verlet and trapezoidal work accumulation.

The CF4 electronic state remains much more accurate than RK4 at equal timestep. At `dt=0.2 fs`, the electronic errors are `4.17e-09` and `2.43e-05`, respectively, while CF4 preserves the norm at roundoff and RK4 shows a `2.42e-05` norm error. The lattice/velocity errors are nearly identical for the two electronic propagators because the second-order classical split dominates those observables at this resolution.

## 10 ps stability gate

The production candidate was then propagated for 10 ps (`50,000` steps at `dt=0.2 fs`) on the same relaxed 20x20 control, with zero initial lattice velocity, 2 mV/angstrom field, no thermostat, and one CPU thread.

Measured results:

- elapsed propagation time: `74.7481 s`;
- matter-energy change: `5.274807e-04 eV`;
- accumulated field work: `5.273615e-04 eV`;
- maximum `|dE-W|`: `2.037102e-07 eV`;
- final `dE-W`: `1.191885e-07 eV`;
- maximum electronic norm error: `1.089329e-11`;
- IPR: `0.44778108 -> 0.45067920`;
- population L2 change: `8.080075e-02`;
- maximum lattice-coordinate excursion: `5.696505e-02 angstrom`;
- Hamiltonian evaluations/applications: `100,000 / 600,000`.

The particularly useful observation is that the maximum energy-work residual over 10 ps remains essentially at the same `~2.04e-07 eV` scale already seen in the 10 fs `dt=0.2 fs` convergence run. It does not accumulate secularly over the 50,000-step trajectory. Meanwhile the population and lattice coordinates change substantially enough to demonstrate a genuinely driven, nonstationary trajectory.

## D3 decision

The validated deterministic field-driven one-carrier hierarchy carried forward is:

1. **full coupled DOP853 + accumulated work** as the high-accuracy reference;
2. **velocity Verlet + CF4-Lanczos m=6** as the production candidate for zero-temperature field-driven coupled dynamics;
3. **RK4** retained only as an independent regression baseline.

For the historical-scale 20x20 control, `dt=0.2 fs` passes both the short convergence study and the 10 ps stability gate. This timestep is a validated numerical control for this parameter set, not a universal timestep guarantee.

D3 does not establish steady-state transport because there is no thermal bath or electronic decoherence yet. In a periodic, nondissipative finite system the field-driven response must not be interpreted as a macroscopic DC mobility. Those physical ingredients belong to D4 and D5.

The next checkpoint, D4, introduces a finite-temperature lattice bath and must validate fluctuation-dissipation, random-number reproducibility, equilibrium temperature statistics, and the zero-friction/zero-temperature reduction back to D2/D3 deterministic dynamics.