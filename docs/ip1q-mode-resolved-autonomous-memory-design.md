# IP1q — mode-resolved decomposition of autonomous isotropic post-hop memory

## Motivation

IP1o established that the long-lived post-hop trailing-current pattern survives for at least 2 ps on a conservative frozen-electronic surface. The complete complex electronic state is bitwise fixed, no later electronic hop is possible, no external power is applied, and yet the late four-boundary current vector remains almost collinear with the fully coupled zero-power continuation.

The next question is therefore no longer whether autonomous memory exists, but **what lattice modes carry it**.

The repository already contains the historical helper `ip1p_phonon_recurrence_audit.py`; this scientific stage is named IP1q to avoid collision with that established `ip1p` filename.

IP1q asks:

> Which Holstein/Peierls polarizations and wave-vector sectors store the autonomous post-hop memory, and does their time-frequency content follow the exact harmonic dispersion of the frozen electronic surface?

## Key simplification of the frozen surface

With the electronic wavefunction and held Peierls phase fixed, the electron-lattice coupling contributes only terms linear in `u`, `vx`, and `vy`. The lattice Hessian is therefore unchanged from the bare harmonic model.

The fixed electronic state shifts the equilibrium coordinates but does not renormalize the normal-mode frequencies.

Consequently, after subtracting the exact frozen-surface equilibrium:

- `u` is a dispersionless intramolecular (Holstein) oscillator field with `omega_u = sqrt(k1/m1)`;
- `vx` is an intermolecular Peierls chain field dispersing only along x,
  `omega_vx(qx) = 2 sqrt(k2/m2) |sin(qx/2)|`;
- `vy` is the analogous y-chain field,
  `omega_vy(qy) = 2 sqrt(k2/m2) |sin(qy/2)|`.

All masses are converted to eV fs^2/A^2 as in the validated dynamics.

## Production protocol

Repeat the validated IP1o branch point:

- 40x40 PBC;
- isotropic `J0y/J0x = 1.0`;
- initial field +10 mV/A along x;
- T=0;
- no thermostat;
- no IDC;
- dt=0.2 fs;
- detect the first persistent natural x hop rather than hard-code its time;
- expected screened event: 820 -> 819 (-x), accepted near 2874 fs;
- at the exact accepted sample, hold the Peierls phase and freeze the complete complex electronic state;
- propagate only the frozen-electronic classical lattice branch.

Extend the frozen continuation to **10 ps** after the switch. This remains well inside the 40x40 stationary-carrier harmonic wrap time (~21.7 ps) while improving frequency resolution to approximately `2*pi/10 ps = 6.3e-4 rad/fs`.

Store complete lattice displacements and velocities every **10 fs**:

- `u`, `vx`, `vy`;
- `du/dt`, `dvx/dt`, `dvy/dt`.

The 10 fs sampling is far faster than the highest lattice frequency and keeps the stored artifact manageable.

## Exact frozen-surface equilibrium

For the frozen electronic state `psi_s` and held phase:

1. `u_eq = -alpha_intra * population / k1`;
2. solve the periodic x-chain Poisson equation for `vx_eq` independently in Fourier space for every row;
3. solve the periodic y-chain Poisson equation for `vy_eq` independently for every column;
4. set the zero-frequency chain-shift modes to zero, matching the projected zero-mode convention.

Numerically verify that the Ehrenfest force evaluated at this equilibrium is zero within floating-point tolerance.

## Exact modal energy decomposition

Use orthonormal 2D FFTs of deviations from the frozen equilibrium.

For every `(qx,qy)`:

`E_u = 1/2 m_u |V_u|^2 + 1/2 k1 |Q_u|^2`

`E_vx = 1/2 m_v |V_vx|^2 + 1/2 k2 lambda_x(qx) |Q_vx|^2`

`E_vy = 1/2 m_v |V_vy|^2 + 1/2 k2 lambda_y(qy) |Q_vy|^2`

with `lambda(q)=4 sin^2(q/2)`.

Parseval closure must reproduce the corresponding real-space excitation energy above the frozen-surface minimum.

### Primary polarization result

Report at switch and over the 10 ps continuation:

- Holstein fraction `E_u / E_total`;
- longitudinal Peierls-x fraction `E_vx / E_total`;
- transverse Peierls-y fraction `E_vy / E_total`;
- total Peierls fraction `(E_vx+E_vy)/E_total`.

Classify the memory as:

- **Peierls dominated** if Peierls fraction >= 2/3;
- **Holstein dominated** if Holstein fraction >= 2/3;
- **mixed** otherwise.

This is a descriptive classification, not a numerical acceptance gate.

## Wave-vector distribution

For `vx`, integrate modal energy over `qy` to obtain `E_vx(qx)`. For `vy`, integrate over `qx` to obtain `E_vy(qy)`.

Report:

- top five occupied q sectors for each intermolecular polarization;
- wavelength in lattice sites where finite;
- analytic mode frequency;
- analytic group velocity in sites/ps;
- q-space participation ratio;
- fraction of energy in the lowest nonzero quarter of the Brillouin-zone magnitude.

The group velocities are harmonic model diagnostics, not fitted material velocities.

## Traveling-wave decomposition along x

For every positive non-special `qx`, decompose the complex `vx` normal coordinate into +x and -x traveling components using

`A_plus = (Q + i V/omega)/2`

`A_minus = (Q - i V/omega)/2`.

For the observed -x carrier hop:

- +x lattice propagation is retrograde/trailing;
- -x propagation is co-moving with the carrier.

Report:

- retrograde traveling `vx` energy;
- co-moving traveling `vx` energy;
- nondirectional/special-sector energy;
- retrograde fraction among the direction-resolved `vx` energy;
- energy-weighted group velocity of the retrograde sector.

Do not infer a transport coefficient from this decomposition.

## q-omega spectrum

From the 10 fs sampled velocity fields:

- apply a Hann temporal window;
- calculate temporal positive-frequency spectra;
- Fourier transform spatially;
- form `S_vx(qx,omega)` by summing over `qy`;
- form `S_vy(qy,omega)` by summing over `qx`;
- form spatially integrated `S_u(omega)`.

For occupied intermolecular q sectors, compare the dominant spectral ridge with the analytic harmonic dispersion.

Report frequency-bin width and the energy-weighted ridge error. A ridge agreement within one Fourier frequency bin is the intended validation target for sufficiently occupied/resolved sectors; unresolved sectors below one frequency bin are reported but excluded from the ridge gate.

## Numerical gates

Before physical interpretation:

1. pycompile passes;
2. focused IP1q/IP1o/IP1n/IP1m/D3 tests pass;
3. full pytest passes;
4. recurrence preflight passes for the complete drive + 10 ps frozen continuation;
5. the same first persistent natural x event is detected and used for branching;
6. held phase rates are zero;
7. electronic state remains bitwise unchanged;
8. frozen-surface energy drift passes the established size-aware tolerance;
9. exact frozen-surface equilibrium force residual is below `1e-12` in model force units;
10. real-space excitation energy and summed modal energy agree within `1e-9 eV` at every stored sample;
11. polarization-resolved modal energies remain individually conserved to `1e-8 eV` over the 10 ps frozen trajectory;
12. q-space zero-mode kinetic energy remains negligible;
13. requested lattice state/velocity trajectory and spectral arrays are finite and serialized.

The polarization classification and retrograde-mode fraction are scientific outputs, not numerical pass/fail gates.

## Interpretation guards

- IP1q analyzes the conservative frozen-electronic surface established by IP1o; it does not claim the same exact modal occupations in a finite-temperature material.
- The static electronic force shifts the equilibrium and remains present. IP1q isolates lattice dynamics around that fixed surface, not a carrier-removed lattice.
- A large retrograde traveling-mode fraction is a directional lattice-mode statement, not carrier mobility.
- The PBC held-phase twist is static. Because `psi` is frozen, it enters only through the fixed equilibrium force; subtracting the exact frozen-surface equilibrium removes the corresponding static displacement offset from the modal dynamics.
- No calibrated phonon lifetime is extracted from an undamped harmonic continuation.

## Decision after IP1q

If the modal decomposition is numerically closed, the isotropic-barrier branch will have a complete mechanism-level characterization of the T=0 autonomous post-hop memory: existence, causal storage in lattice phase space, polarization content, wave-vector content and propagation direction.

Only after this closure should the project return to finite-temperature dynamics and ask whether the same identified mode sectors predict or bias subsequent thermally activated carrier relocation.