# IP1t — post-recrossing energy-preserving Peierls direction control

## Motivation

IP1s passed all numerical gates but formally failed its preregistered primary causal criterion because the first post-release persistent x event was the same `+x` recrossing in both branches and its start time differed by only 36 fs (<100 fs), even though the population L1 distance later reached ~0.498. A strong secondary difference then appeared: the native branch returned `-x`, whereas the direction-reversed branch showed no second persistent x event during the 5 ps observation window.

That later difference was not part of the IP1s primary gate and must not be promoted post hoc.

IP1t prospectively asks the narrower causal question:

> After the common first post-release `+x` recrossing has already been accepted, does reversing the direction of the non-special x-Peierls traveling content change the immediate return/commitment dynamics?

This is a deterministic trajectory-level counterfactual. It does not estimate a hopping probability, rate, mobility, activation energy, or material lifetime.

## Common trajectory before the IP1t branch point

Use the validated isotropic protocol:

- 40x40 PBC;
- `J0y/J0x = 1`;
- initial field +10 mV/A along x;
- T=0;
- no thermostat or IDC;
- D3 deterministic dynamics;
- dt=0.2 fs;
- CF4-Lanczos, Krylov dimension 6;
- persistent-site criterion 50 fs, sampled every 2 fs.

1. Drive until the first persistent nearest-neighbor x event is accepted. Expected screened event: `820 -> 819 (-x)`, accepted near 2874 fs, but detect it freshly.
2. Freeze the Peierls phase at that acceptance time so the physical post-switch electric field and external power are zero.
3. Continue the **single common fully coupled released trajectory** until its first subsequent persistent nearest-neighbor x event is accepted. Expected screened event: `819 -> 820 (+x)`, accepted near 3072 fs, but detect it freshly.
4. The accepted state of this common `+x` recrossing defines the IP1t branch point `tr`.

No branch manipulation occurs before `tr`.

## Branches at `tr`

Let the common recrossing be `B -> A`, where `B=819` and `A=820` in the screened trajectory.

### Native branch

Continue the full coupled state unchanged under the same held Peierls phase.

### Direction-reversed branch

At exactly `tr`:

- keep the electronic wavefunction bitwise identical;
- keep all lattice coordinates bitwise identical;
- keep `u` and `vy` velocities bitwise identical;
- preserve qx=0 and Nyquist `vx` velocity sectors;
- reverse every non-special `vx` velocity Fourier coefficient.

As in IP1s, this exchanges `A_plus <-> A_minus` for every non-special x-Peierls harmonic mode while preserving the `vx` modal-energy spectrum, kinetic energy, and total matter energy.

## Primary return/commitment observable

Use a **1 ps commitment window** after `tr`.

For each branch find the first persistent nearest-neighbor x event whose transition start lies within `tr + 1000 fs`.

Define `direct_return = true` when that first x event is exactly the recrossing reversal `A -> B` (same two sites, opposite direction to the common `B -> A` event).

Define `return_commitment_changed = true` if any of the following holds:

1. exactly one branch has a first persistent x event in the 1 ps window;
2. both have first x events but only one is the direct `A -> B` return;
3. their first x-event directions differ;
4. both first events are direct returns but their transition-start times differ by at least **100 fs**.

This gate is fixed prospectively before IP1t execution.

## Electronic divergence gate

At matched 2 fs samples record the full population vector and its L1 distance between branches.

Define `electronic_state_diverged = true` if L1 reaches **0.25** within the first **2 ps** after `tr`.

Also report first times for L1 >=0.10 and >=0.25.

Primary causal classification:

`directional_memory_controls_recrossing_commitment = return_commitment_changed AND electronic_state_diverged`.

The classification is deterministic and trajectory-specific, not a rate/probability statement.

## Secondary observables

Report without using them to redefine the primary result:

- all persistent events in each branch during 2 ps;
- population of branch-point site A and previous site B;
- maximum-population site and IPR;
- residence/commitment duration on A before any persistent x relocation;
- early and late B-centered trailing-current vector cosine similarity;
- trailing RMS amplitude at d=1..4.

## Intervention and numerical gates

Before physical interpretation require:

1. pycompile passes;
2. focused IP1t/IP1s/IP1r/IP1q/IP1o/IP1n/D3 tests pass;
3. full pytest passes;
4. recurrence preflight covers the full drive + common released + 2 ps branch duration;
5. first natural x hop is detected;
6. common post-release x recrossing is detected and accepted before branching;
7. common held-phase propagation to `tr` has zero external work and conserves matter energy;
8. all lattice coordinates are unchanged by the branch intervention;
9. electronic state is unchanged;
10. u/vy velocities are unchanged;
11. special vx velocity sectors are unchanged;
12. non-special vx velocity sectors are sign reversed;
13. vx kinetic energy is preserved to <=1e-12 eV;
14. total matter energy is preserved to <=1e-12 eV;
15. retrograde/comoving vx energies are exchanged to <=1e-12 eV;
16. both post-branch trajectories complete requested steps;
17. external work is zero to roundoff in both branches;
18. matter-energy conservation, electronic norm, and projected zero modes pass established tolerances;
19. population/current arrays are finite and serializable.

Physical return/commitment divergence is not a numerical acceptance gate.

## Interpretation guards

- Direction reversal is an instantaneous numerical counterfactual, not an experimental pulse protocol.
- A positive result demonstrates deterministic sensitivity of the immediate recrossing/commitment trajectory to the x-Peierls phase-space direction at fixed coordinates and energy.
- It does not by itself establish a hopping rate, hopping probability, mobility, activation barrier, or finite-temperature mechanism.
- The common held PBC twist is identical in both branches and cancels as a differential confounder.
- IP1s remains formally negative under its own first-event preregistration regardless of the IP1t result.

## Decision after IP1t

If the preregistered IP1t causal gate passes, the next stage should use energy-preserving **q-band-selective** direction reversals to identify which long-wavelength vx sectors control the return/commitment sensitivity.

If it fails, the later IP1s event difference remains an unpromoted secondary observation and the retrograde current should not be claimed as a demonstrated controller of recrossing commitment.
