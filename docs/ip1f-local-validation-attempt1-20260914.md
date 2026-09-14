# IP1f local validation — attempt 1 (2026-09-14)

## Outcome

The first local IP1f run did not reach the physical benchmark summary. The failure was a deterministic analysis bug in event-aligned coordinates for even periodic cells; it was not a failure of the Holstein-Peierls dynamics or of the numerical integrator.

The pre-benchmark gates were clean:

- pycompile: PASS;
- focused tests: PASS;
- full regression suite: 412 PASS;
- 40x40 phonon-recurrence preflight: PASS;
- stationary-carrier harmonic wrap estimate: 21.693046 ps for the 15 ps run;
- gamma_v = 0.002 fs^-1 is not fully overdamped.

The benchmark stopped after about 376 s with

```text
ValueError: wake profiles and s_axis must have identical shapes
```

inside `wake_window_metrics`.

## Root cause

`event_aligned_coordinates` first formed minimum-image x/y coordinates and then rotated them so the electronic hop pointed toward +s. On an even lattice there is a unique antipodal coordinate represented canonically as `-L/2`. For a negative hop direction, multiplication by -1 changed that point to `+L/2`.

For L=40, a -x or -y event could therefore produce longitudinal coordinates in `[-19, +20]`, while the declared canonical profile axis remained `[-20, +19]`. `np.bincount` consequently created 41 bins instead of 40, and the later wake metric correctly rejected the shape mismatch.

This is a PBC coordinate bookkeeping error only. It does not alter any propagated state, energy, current, thermostat, IDC event, or prior IP1e conclusion.

## Fix

The rotated longitudinal and transverse event coordinates are now re-wrapped to a canonical periodic interval after rotation. For even cells the convention is always `[-L/2, ..., L/2-1]`. `longitudinal_profile` also asserts that no coordinate escapes this range and that the resulting profile has exactly the periodic-axis length.

A dedicated 40x40 regression test now checks +x, -x, +y and -y hops, including exact profile length and scalar-sum conservation.

## Decision

Re-run IP1f from the corrected branch. No result from this failed attempt should be used for physical interpretation because the aggregate IP1f artifact was never written.
