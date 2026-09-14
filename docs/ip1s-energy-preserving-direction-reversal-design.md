# IP1s — energy-preserving Peierls direction-reversal causal control

## Motivation

IP1r establishes an exact distinction between global modal energy and local trailing current:

- direction-resolved vx energy is nearly balanced (~50.36% retrograde / 49.64% co-moving);
- the late local trailing x-current at d=1..4 is nevertheless ~94.9% attributable to the retrograde vx traveling component.

This motivates the next causal question:

> If the traveling direction of the x-Peierls phase-space excitation is reversed at fixed coordinates and fixed energy, does the subsequent electronic trajectory change?

This is a deterministic trajectory-level causal control. It does not estimate a hopping probability, hopping rate, mobility, activation barrier, or material relaxation time.

## Common pre-branch trajectory

Use the validated isotropic natural-hop protocol:

- 40x40 PBC;
- `J0y/J0x = 1`;
- initial field +10 mV/A along x;
- T=0;
- no thermostat;
- no IDC;
- D3 deterministic coupled dynamics;
- dt=0.2 fs;
- CF4-Lanczos, Krylov dimension 6;
- persistent-site criterion 50 fs, sampled every 2 fs.

Drive until the first persistent nearest-neighbor x event is accepted. The screened trajectory is expected to reproduce `820 -> 819 (-x)`, start near 2826 fs and acceptance near 2874 fs, but these values must be freshly detected.

At the acceptance time `ts`, switch both branches to the same held Peierls phase used by IP1n/IP1o, so both continuations have:

- identical static boundary twist;
- zero phase rate;
- zero external electric power.

## Branches

### A. Native released branch

Continue the accepted coupled state unchanged under fully coupled Ehrenfest dynamics with the held phase.

### B. Direction-reversed vx branch

At exactly `ts`:

- keep the electronic wavefunction identical;
- keep all lattice coordinates `u, vx, vy` identical;
- keep `u` and `vy` velocities identical;
- decompose the vx velocity into non-special (`0 < |qx| < pi`) and special (`qx=0` and Nyquist) Fourier sectors;
- reverse the sign of every non-special vx velocity coefficient;
- leave special vx velocity sectors unchanged.

At fixed vx coordinates this transformation exactly exchanges the +x/-x traveling amplitudes of every non-special harmonic vx mode:

`A_plus <-> A_minus`.

It therefore swaps retrograde and co-moving traveling energies while preserving the vx modal-energy spectrum.

## Intervention gates

Before propagation, require:

1. all lattice coordinates are bitwise unchanged;
2. electronic state is bitwise unchanged;
3. u and vy velocities are bitwise unchanged;
4. special vx velocity Fourier sectors are unchanged;
5. non-special vx velocity Fourier sectors are sign reversed;
6. total vx kinetic energy is preserved to <=1e-12 eV;
7. full matter energy is preserved to <=1e-12 eV;
8. global retrograde and co-moving vx energies are exchanged to <=1e-12 eV;
9. no artificial coordinate displacement is introduced.

The intervention is deliberately an instantaneous counterfactual velocity operation. Force continuity is not required because velocities are the manipulated variable; coordinates/electronic Hamiltonian remain continuous.

## Post-switch propagation

Continue each branch for **5 ps** under fully coupled Ehrenfest dynamics with the same held phase.

Sample:

- persistent-site tracker every 2 fs;
- selected electronic observables every 2 fs;
- energy/current diagnostics every 10 fs.

The recurrence preflight must cover the full pre-hop + 5 ps continuation interval.

## Primary electronic observables

For each branch record:

- all persistent events after the switch;
- first persistent x event after switch, if any;
- its transition-start and acceptance times relative to `ts`;
- its direction;
- populations of the pre-hop site A and post-hop site B;
- maximum-site index;
- inverse participation ratio;
- pairwise L1 distance between the two branch population distributions at matched samples.

The original IP1n released branch is expected to show rapid recrossing, but event times/directions must be freshly detected.

## Preregistered trajectory-level causal classification

Define `event_sequence_changed = true` if within the first 2 ps after `ts` any of the following holds:

- one branch has a persistent x event and the other does not;
- the first persistent x-event directions differ;
- the first persistent x-event transition-start times differ by at least 100 fs.

Define `electronic_state_diverged = true` if the population L1 distance reaches at least 0.25 at any sampled time within the first 2 ps.

Define

`directional_memory_changes_electronic_trajectory = event_sequence_changed AND electronic_state_diverged`.

This is a deterministic sensitivity criterion, not a statistical rate/probability statement.

Also report the first time at which L1 >=0.10 and L1 >=0.25.

## Secondary lattice-current diagnostics

Use the same B->C post-hop frame as IP1m-IP1r and report, for d=1..4:

- trailing RMS amplitude;
- late local current directionality;
- similarity between native and direction-reversed trailing-current vectors.

Confirm that the intervention reverses the early vx traveling attribution before nonlinear electron-lattice feedback modifies the branches.

## Numerical gates

Before physical interpretation:

1. pycompile passes;
2. focused IP1s/IP1r/IP1q/IP1o/IP1n/D3 tests pass;
3. full pytest passes;
4. recurrence preflight passes;
5. natural x event is detected;
6. intervention gates all pass;
7. both branches complete requested steps;
8. external work is zero to roundoff in both branches;
9. matter-energy conservation passes the size-aware tolerance in both branches;
10. electronic norms and projected zero modes remain controlled;
11. branch population/current arrays are finite and serialized.

Physical divergence is not a numerical acceptance gate.

## Interpretation guards

- Direction reversal is a numerical counterfactual, not an experimental pulse protocol.
- An altered deterministic trajectory demonstrates causal sensitivity to vx phase-space direction under this intervention; it does not provide a hopping rate or probability.
- Because u and vy are retained unchanged, the test isolates the traveling-direction content of vx rather than removing the dominant total Peierls energy reservoir.
- Because total energy and the vx modal spectrum are preserved, any trajectory difference cannot be attributed simply to an energy change.
- The held PBC twist is common to both branches and therefore cancels as a differential confounder.

## Decision after IP1s

If the preregistered causal classification passes, the next stage should determine which q sectors of the retrograde vx content control the electronic sensitivity using energy-preserving band-selective direction reversals.

If it fails, the long-lived retrograde local current should be interpreted as a persistent by-product of the hop rather than a demonstrated controller of subsequent electronic relocation.