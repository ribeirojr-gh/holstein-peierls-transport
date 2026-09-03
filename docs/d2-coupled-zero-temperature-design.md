# D2 coupled zero-temperature electron-lattice dynamics design

## Purpose

D2 is the first fully coupled dynamical checkpoint.  It couples one propagated electronic carrier to the classical Holstein-Peierls lattice at **zero electric field and zero thermostat**.  The purpose is numerical and physical validation of the deterministic Ehrenfest equations before finite-temperature, stochastic, spin-relaxation, GPU, or material-specific production work.

The primary gate is **controlled conservation of the total energy of the propagated state**.

## State variables

The D2 one-carrier state is

- a normalized complex electronic wavefunction `psi`;
- lattice coordinates `u`, `vx`, `vy` in angstrom;
- lattice velocities `du/dt`, `dvx/dt`, `dvy/dt` in angstrom/fs.

The lattice Hamiltonian and electron-phonon couplings are the same validated Holstein-Peierls functions used by the static solver.  D2 does not re-fit or reinterpret any parameter.

## Electronic equation

At zero external field,

`i hbar d psi / dt = H(q(t)) psi`.

The electronic propagator must use the instantaneous moving-lattice Hamiltonian rather than the static ground-state eigensystem.  D1 selected CF4-Lanczos with Krylov dimension 6 as the primary linear candidate, RK4 as an independent baseline, and DOP853 as the tightened reference.

For D2, a time-dependent electronic step must sample/interpolate the moving lattice consistently at its internal stage times.  A frozen-H electronic step inside a moving-lattice full step is therefore only a legacy/control splitting, not the final accuracy reference.

## Ehrenfest force

For a normalized propagated state, the force follows the Hellmann-Feynman/Ehrenfest expectation

`F_q = - <psi | dH/dq | psi> - dV_lattice/dq`.

Equivalently, the lattice gradient is the derivative of

`E(q,psi) = V_lattice(q) + <psi|H(q)|psi>`

with `psi` held fixed.

For the rectangular model this gives

`dE/du_i = K1 u_i + alpha1 |psi_i|^2`,

`dE/dvx_i = K2(2 vx_i - vx_left - vx_right)`
`            + 2 alpha2x [Re(psi_left* psi_i) - Re(psi_i* psi_right)]`,

and the analogous expression along y.

These formulas reduce exactly to the validated optimized static gradients when `psi` is the real instantaneous ground state.  D2 must test this reduction explicitly.

## Energy diagnostic

The archived dynamics used the instantaneous lowest eigenvector in the electronic-energy diagnostic even while propagating a different state.  D2 must instead use

`E_el = <psi|H(q)|psi>/<psi|psi>`.

The deterministic total energy is

`E_total = E_el + V_lattice + T_lattice`,

with

`T_lattice = 1/2 M1 sum_i |u_dot_i|^2`
`          + 1/2 M2 sum_i (|vx_dot_i|^2 + |vy_dot_i|^2)`.

No thermostat or field-work term exists in D2.

## Mass/time units

The archived dynamics advances time in attoseconds while the Python dynamics layer uses femtoseconds.  The retained legacy masses `m1` and `m2` are therefore interpreted in the archived `eV as^2 / angstrom^2` convention.  For a femtosecond integrator the equivalent masses are

`M_fs = M_as / 10^6`,

because `1 fs = 1000 as`.

For the legacy default controls this yields

- `M1 = 7.5e4 eV fs^2 / angstrom^2`;
- `M2 = 1.5e5 eV fs^2 / angstrom^2`.

This conversion must be regression-tested against a direct attosecond-form update so that no silent unit change enters the dynamics.

## Classical integrator baseline

The archived deterministic zero-damping limit is velocity-Verlet/BBK-like.  D2 should first implement a clean velocity-Verlet reference:

1. half kick from the Ehrenfest force at `(q_n, psi_n)`;
2. coordinate drift;
3. propagate the electronic state consistently across the interval while the lattice moves;
4. evaluate the new Ehrenfest force at `(q_{n+1}, psi_{n+1})`;
5. second half kick.

A strict legacy splitting control may freeze the Hamiltonian during the electronic substep, but the promoted D2 scheme must benchmark a stage-consistent moving-H treatment.

## Initial conditions

The first conservation benchmark should start from the validated relaxed 20x20 polaron and its electronic ground state.  Exactly zero velocities would leave the system stationary up to numerical noise, so conservation tests also require controlled deterministic perturbations that preserve interpretability, for example:

- a small displacement kick in one lattice mode with zero initial electronic excitation; and
- a small electronic superposition at the relaxed geometry with zero initial lattice velocity.

Perturbation amplitudes are numerical controls, not material predictions.

## Validation gates

D2 is not complete until all of the following pass:

1. complex-wavefunction Ehrenfest gradients agree with central finite differences of `V_lattice + <psi|H|psi>`;
2. the complex gradient reduces to the validated static optimized gradient for a real ground state;
3. the corrected electronic-energy diagnostic agrees with direct matrix expectation values;
4. lattice kinetic-energy and mass conversion agree between as- and fs-based forms;
5. electronic norm remains controlled without routine renormalization;
6. velocity-Verlet reproduces the expected order on an isolated harmonic lattice control;
7. total-energy drift decreases systematically with the full coupled time step;
8. CF4-Lanczos is compared with RK4 and tightened DOP853 under the same coupled trajectory;
9. the selected scheme remains stable for a physically relevant multi-ps zero-field trajectory before D3 is opened; and
10. no thermostat, stochastic force, field, decoherence correction, or GPU-specific approximation is introduced into the conservation benchmark.

## Expected architecture

D2 should add separable interfaces for

- propagated-state Ehrenfest force evaluation;
- lattice kinetic/total-energy diagnostics;
- deterministic classical stepping;
- electronic propagation under a supplied time-dependent lattice path; and
- coupled trajectory orchestration.

This separation is required so D4 can later replace only the classical thermostat/integrator and so CPU/GPU backends can replace matrix actions without changing the validated physics.
