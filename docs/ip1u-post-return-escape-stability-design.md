# IP1u — post-return escape-stability causal control

## Motivation

IP1t formally fails its preregistered immediate-return criterion: both native and direction-reversed branches execute the direct `820 -> 819` return, with only a 60 fs timing difference. A secondary observation is that the direction-reversed branch later re-escapes `819 -> 820`, whereas the native branch does not within the observed window.

IP1u converts that secondary observation into a prospective test.

## Common trajectory and branch point

Use the validated isotropic protocol:
- 40x40 PBC
- `J0y/J0x = 1`
- initial +10 mV/A x field
- T=0
- no thermostat, no IDC
- D3 deterministic coupled dynamics
- dt=0.2 fs
- CF4-Lanczos m6
- persistent-site criterion 50 fs sampled every 2 fs

Procedure:
1. drive until the first natural `820 -> 819 (-x)` event is accepted;
2. freeze the Peierls phase exactly as in IP1n–IP1t so external power is zero;
3. continue one common released trajectory until the first direct `819 -> 820 (+x)` recrossing is accepted;
4. continue the same common trajectory until the subsequent direct return `820 -> 819 (-x)` is accepted;
5. only then create two branches from that identical state.

All event times must be freshly detected. Historical times are expectations only.

## Branches

### Native post-return branch
Continue the accepted common state unchanged.

### Direction-reversed post-return branch
At exactly the post-return acceptance time:
- retain all lattice coordinates;
- retain the complete electronic state;
- retain `u` and `vy` velocities;
- retain special `vx` Fourier sectors;
- sign-reverse every non-special `vx` velocity Fourier coefficient.

This exchanges the `+x/-x` traveling amplitudes at fixed coordinates and preserves the `vx` modal-energy spectrum and total matter energy.

## Primary question

Does the Peierls traveling direction control the stability of the returned state against the next `819 -> 820 (+x)` re-escape?

## Preregistered escape classification

Use a primary escape window of **1.5 ps** after the post-return branch point.

For each branch record the first persistent nearest-neighbor x event.
An event counts in this primary window only if its transition starts at or after the branch point and its **persistent acceptance is no later than the 1.5 ps cutoff**. An event started inside but accepted after the cutoff is excluded. Compare the transition-start times when both branches have direct re-escapes.

Define `reescape_status_changed = true` if any of:
- one branch has an x event in the window and the other does not;
- one branch's first x event is the direct re-escape `819 -> 820 (+x)` and the other's is not;
- both have direct re-escape but their transition-start times differ by >=100 fs.

Define `electronic_state_diverged = true` if population L1 reaches >=0.25 within **2 ps**.

Define:
`directional_memory_controls_post_return_escape = reescape_status_changed AND electronic_state_diverged`.

Also report first L1 >=0.10 and >=0.25.

## Secondary diagnostics

Report:
- all persistent x events over 3 ps;
- branch-site and previous-site populations;
- maximum-site index and IPR;
- trailing current d=1..4;
- early and late native/reversed current-vector cosine similarity;
- amplitudes of the trailing pattern.

## Numerical gates

Before physical interpretation require:
1. pycompile PASS;
2. focused IP1u/IP1t/IP1s/IP1r/IP1q/IP1o/IP1n/D3 tests PASS;
3. full pytest PASS;
4. recurrence preflight covers pre-history + 3 ps continuation;
5. first natural hop found;
6. common +x recrossing found;
7. common -x return found;
8. zero external work on common released and both post-return branches;
9. intervention preserves coordinates/electronic state and swaps non-special `vx` directions;
10. `vx` kinetic energy and total matter energy preserved <=1e-12 eV;
11. both branches complete requested steps;
12. size-aware matter-energy conservation PASS;
13. electronic norms and projected zero modes controlled;
14. finite serialized population/current diagnostics.

Physical divergence is not a numerical gate.

## Interpretation guards

- The direction reversal is a numerical counterfactual, not an experimental pulse.
- A positive result is deterministic sensitivity, not a hopping probability or rate.
- A negative result means no demonstrated control under this intervention/window; it does not prove the Peierls memory is dynamically irrelevant.
- No mobility, activation energy, transport coefficient, calibrated phonon lifetime, or field threshold is inferred.

## Decision after IP1u

If the preregistered escape classification passes, the next mechanistic stage may use q-band-selective energy-preserving reversals to identify which `vx` wave-number sectors control the post-return instability.

If it fails, close the isotropic causal-control line at the trajectory level: the retrograde Peierls wake is a persistent, dynamically coupled memory with measurable electronic sensitivity, but no validated control over discrete hop commitment under the tested interventions.
