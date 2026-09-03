# D2 coupled zero-temperature electron-lattice benchmark

## Scope

D2 validates deterministic one-carrier Ehrenfest dynamics with a moving classical Holstein-Peierls lattice at zero electric field and zero thermostat.  It is the first checkpoint in which the propagated complex electronic state and the classical lattice evolve together.

The implementation deliberately separates:

- propagated-state Ehrenfest forces and energy diagnostics;
- classical velocity-Verlet stepping;
- electronic propagation along a supplied moving-lattice path; and
- a full coupled DOP853 reference integrator.

No stochastic force, friction, external field, decoherence correction, GPU-specific approximation, or material reparameterization is introduced by D2.

## Corrected propagated-state physics

The electronic energy is evaluated as

`E_el = <psi|H(q)|psi>/<psi|psi>`

rather than with the instantaneous lowest eigenvector used by the archived dynamics diagnostic.

The Ehrenfest lattice force is

`F_q = -d/dq [V_lattice(q) + <psi|H(q)|psi>]`,

with the propagated complex `psi` held fixed in the coordinate derivative.  The implementation uses local populations and nearest-neighbour real coherences and therefore remains O(N) apart from the Hamiltonian action itself.

For a normalized real instantaneous ground state, this gradient reduces to the already validated optimized static-polaron gradient.

## Classical units and integration

The archived masses are retained in the effective `eV as^2 / angstrom^2` convention.  The femtosecond dynamics layer converts them through

`M_fs = M_as / 10^6`.

Thus the default controls become

- `M1 = 7.5e4 eV fs^2 / angstrom^2`;
- `M2 = 1.5e5 eV fs^2 / angstrom^2`.

A clean zero-temperature velocity-Verlet implementation is validated independently on an exactly solvable harmonic control.  It exhibits the expected second-order global convergence and bounded symplectic energy error.

## Coupled splitting

One full D2 production-candidate step is:

1. evaluate the Ehrenfest force at `(q_n, psi_n)`;
2. apply a classical half kick;
3. drift the lattice to `q_(n+1)`;
4. linearly interpolate the lattice path between the endpoints;
5. propagate `psi` through that moving Hamiltonian at the electronic method's actual internal stage times;
6. evaluate the Ehrenfest force at `(q_(n+1), psi_(n+1))`; and
7. apply the second classical half kick.

The primary electronic method is the D1-selected fourth-order two-exponential commutator-free Magnus propagator with a short Lanczos exponential action (`CF4-Lanczos`, Krylov dimension 6).  RK4 remains an independent baseline.

A separate adaptive DOP853 solver integrates the complete coupled ODE for small and moderate benchmark trajectories and serves as the tightened accuracy reference.

## Validation gates

All of the following gates pass:

- complex-wavefunction Ehrenfest gradients agree with central finite differences of `V_lattice + <psi|H|psi>`;
- the complex gradient reduces to the validated static optimized gradient for a real ground state;
- the corrected electronic energy agrees with direct matrix expectation values;
- the attosecond-to-femtosecond mass conversion is regression tested;
- lattice kinetic and total-energy decompositions are independently checked;
- velocity Verlet recovers second-order global convergence on a harmonic oscillator;
- the tightened complete-system DOP853 trajectory conserves total energy to numerical tolerance;
- Verlet + CF4-Lanczos converges to the complete DOP853 reference with the expected overall second-order behavior;
- total-energy error decreases by approximately a factor of four when the coupled time step is halved; and
- CF4-Lanczos preserves electronic norm to floating-point accuracy, unlike the explicit RK4 baseline at the same coarse step.

## 20x20 relaxed-polaron benchmark

The benchmark starts from the validated relaxed 20x20 polaron and its instantaneous ground-state wavefunction.  To produce a non-stationary but controlled trajectory, a local intramolecular velocity kick of `2.0e-4 angstrom/fs` is applied at the polaron center.  This adds a small deterministic lattice kinetic excitation without an electric field or thermostat.

The short comparison is run for 10 fs using one CPU thread.

| method | dt [fs] | elapsed [s] | H evals | H apps | max |dE| [eV] | relative max drift | norm error | electronic error vs DOP853 | lattice error vs DOP853 | velocity error vs DOP853 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CF4-Lanczos m=6 | 0.20 | 0.1103 | 100 | 600 | 1.445e-09 | 3.578e-09 | 7.327e-15 | 2.798e-09 | 1.890e-09 | 3.087e-07 |
| RK4 | 0.20 | 0.0976 | 200 | 200 | 1.440e-09 | 3.565e-09 | 2.394e-05 | 2.394e-05 | 1.889e-09 | 3.146e-07 |
| CF4-Lanczos m=6 | 0.10 | 0.2350 | 200 | 1200 | 3.610e-10 | 8.937e-10 | 1.332e-14 | 7.172e-10 | 4.724e-10 | 7.713e-08 |
| RK4 | 0.10 | 0.2061 | 400 | 400 | 3.607e-10 | 8.928e-10 | 7.510e-07 | 7.510e-07 | 4.723e-10 | 7.751e-08 |
| CF4-Lanczos m=6 | 0.05 | 0.4559 | 400 | 2400 | 9.026e-11 | 2.234e-10 | 2.420e-14 | 1.807e-10 | 1.181e-10 | 1.928e-08 |
| RK4 | 0.05 | 0.3888 | 800 | 800 | 9.023e-11 | 2.234e-10 | 2.349e-08 | 2.349e-08 | 1.181e-10 | 1.931e-08 |

The tightened complete-system DOP853 reference used `rtol=1e-9`, `atol=1e-11`, and `max_step=0.02 fs`.  It required 6014 RHS/Hamiltonian evaluations, 2.39 s on the same one-thread runner, and finished with energy drift `3.33e-16 eV` and norm error `2.22e-16`.

### Interpretation

The approximately fourfold reduction of the energy, lattice, and velocity errors when `dt` is halved confirms that the overall coupled scheme is second order, as expected from the classical velocity-Verlet splitting.  The fourth-order electronic CF4 kernel is therefore not the limiting error at these steps.

At `dt=0.2 fs`, CF4-Lanczos costs only about 13% more than RK4 in this benchmark while reducing the electronic-state/norm error by roughly four orders of magnitude.  The essentially identical lattice-energy drift of CF4 and RK4 demonstrates that the remaining error is controlled by the classical coupled splitting rather than by the CF4 electronic propagator.

## 10 ps stability gate

The primary D2 candidate was then propagated for the full 10 ps duration of the archived example:

- lattice: 20x20;
- `dt = 0.2 fs`;
- 50,000 coupled steps;
- CF4-Lanczos with Krylov dimension 6;
- zero field and no thermostat;
- same `2.0e-4 angstrom/fs` local velocity kick;
- one CPU thread.

Results:

- wall time: `111.853 s`;
- maximum absolute total-energy drift: `2.652067e-09 eV`;
- maximum relative total-energy drift: `6.565188e-09`;
- final energy drift: `1.872092e-09 eV`;
- maximum electronic norm error: `6.994627e-12`;
- IPR: `0.44778108 -> 0.45346387`;
- population L2 change: `5.531110e-02`;
- maximum lattice-coordinate excursion: `1.909402e-02 angstrom`;
- Hamiltonian evaluations/applications: `100000 / 600000`.

The nonzero IPR, population, and lattice changes confirm that this is not a numerically stationary trajectory.  Despite 50,000 coupled steps, the energy error remains at the few-nanoelectronvolt level rather than showing a significant secular growth, and the electronic norm remains extremely well controlled.

## D2 numerical decision

D2 promotes the following deterministic one-carrier zero-temperature scheme:

- **classical lattice:** velocity Verlet;
- **electronic propagation:** CF4-Lanczos with Krylov dimension 6;
- **moving-H treatment:** lattice interpolation sampled at the true CF4 internal Gauss nodes;
- **validated starting time step:** `dt = 0.2 fs` for the present legacy-parameter 20x20 control;
- **accuracy reference:** complete-system adaptive DOP853;
- **independent baseline:** RK4, retained for regression but not preferred for long production trajectories because it does not preserve electronic norm.

The `0.2 fs` value is a validated numerical control for this parameter set, not a universal material time step.  Later material-specific simulations must recheck the time step whenever masses, force constants, electronic bandwidths, electron-phonon couplings, or field protocols change substantially.

## D2 conclusion and next checkpoint

D2 closes the deterministic zero-field, zero-temperature one-polaron coupling checkpoint.  The modern code now has a corrected energy diagnostic, validated complex-state Ehrenfest force, a classical symplectic baseline, a stage-consistent moving-H electronic propagation, an independent complete-system reference, and a stable 10 ps trajectory.

The next roadmap checkpoint is **D3 — field-driven zero-temperature coupled dynamics**.  D3 should reuse the validated D2 coupling unchanged and add only the already validated D1 Peierls field phase.  Its new gates should focus on field work/energy balance, charge-center/current observables, direction reversal, and transport-response convergence rather than revalidating the zero-field force machinery from scratch.
