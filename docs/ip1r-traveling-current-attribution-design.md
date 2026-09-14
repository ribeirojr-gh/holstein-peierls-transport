# IP1r — exact traveling-wave attribution of the local trailing x-current pattern

## Motivation

IP1q established two facts that must be kept distinct:

1. the **global autonomous lattice excitation** is 98.76% Peierls and is dominated energetically by `vy`;
2. the previously used trailing observable is the harmonic **x-current `jx`**, which depends directly on the `vx` chains.

IP1q also found that the global direction-resolved `vx` energy is nearly balanced between laboratory +x and -x traveling waves, even though earlier real-space analyses found a strong local trailing pattern behind a carrier that hopped -x.

An exploratory algebraic decomposition of the saved IP1q trajectory indicates that this is not contradictory: the late B-centered trailing current is phase-selected and is dominated locally by the retrograde traveling `vx` component despite the nearly 50/50 global energy reservoir.

IP1r converts that posthoc observation into an auditable deterministic decomposition.

## Scope

Use the same isotropic natural-hop protocol and frozen-electronic surface as IP1q:

- 40x40 PBC;
- initial field +10 mV/A along x;
- natural hop detected rather than hard-coded;
- branch at the accepted first x hop;
- held Peierls phase, zero phase rate;
- complete complex electronic state frozen;
- T=0, no bath, no IDC;
- dt=0.2 fs;
- frozen continuation: 5 ps;
- lattice state sampled every 10 fs.

Five picoseconds is sufficient for the local current-attribution question and remains far below the 40x40 ballistic recurrence time.

## Exact traveling-wave decomposition

Subtract the exact frozen-surface equilibrium from `vx`. For every sampled state, Fourier transform the excitation coordinate and velocity.

For each positive non-special `qx`, define

`A_plus = (Q_q + i V_q/omega_q)/2`

`A_minus = (Q_q - i V_q/omega_q)/2`.

Reconstruct real fields for

- laboratory +x traveling waves;
- laboratory -x traveling waves;
- qx=0/Nyquist special sectors.

For the observed carrier hop `dx=-1`, +x is retrograde and -x is co-moving. The mapping is reversed automatically if the detected carrier hop is +x.

The decomposition must reconstruct `vx - vx_eq` and `dot(vx)` to machine precision at every sampled time.

## Exact x-current decomposition

For a coordinate component `delta v_c` and its velocity `dot v_c`, define its isolated current on the common static equilibrium strain as

`j_c = -(K2/2) (dot v_c,i + dot v_c,i+1) * [(v_eq,i+1-v_eq,i) + (delta v_c,i+1-delta v_c,i)]`.

Compute isolated currents for retrograde, co-moving and special components.

The total harmonic x-current is evaluated from the complete `vx` field. Define the remaining exact cross/interference current as

`j_cross = j_full - j_retro - j_comoving - j_special`.

This definition includes all bilinear counterpropagating cross terms and gives an exact algebraic closure by construction. Verify the closure numerically rather than assuming it.

## B-centered trailing observable

Use the same fixed post-hop residence frame as IP1m-IP1o:

- source = hop target B;
- target = next nearest neighbor C in the carrier direction;
- +s follows carrier motion;
- trailing direction is -s;
- transverse strip `|p| <= 3`;
- primary boundaries d=1,2,3,4.

At each time form the four-component trailing outward-current vector for the full current and for every component.

## Attribution metrics

Evaluate an early window `0-1000 fs` after the switch and a late window `>=1000 fs` through the end of the 5 ps continuation.

For each component c calculate

`projection_c = sum_t J_c(t) dot J_full(t) / sum_t |J_full(t)|^2`.

Because the current decomposition is exact, the component projections must sum to one up to roundoff. This is the primary additive attribution metric.

Also report

- squared-norm ratio `sum |J_c|^2 / sum |J_full|^2`;
- mean and median cosine similarity of each component with the full vector;
- RMS trailing amplitude;
- boundary-resolved signed net transported energy for d=1..4;
- global direction-resolved `vx` energy split at the switch for comparison.

## Classification

This is an algebraic attribution stage, not an independent statistical hypothesis test. Use the following descriptive classification:

- local retrograde-dominated: retrograde projection >= 2/3;
- local co-moving-dominated: co-moving projection >= 2/3;
- otherwise: mixed/interference.

A result can simultaneously be **globally direction-balanced in vx energy** and **locally retrograde-dominated in trailing current**. These statements refer to different observables and must not be conflated.

## Numerical gates

Before interpretation:

1. pycompile passes;
2. focused IP1r/IP1q/IP1o/IP1n/D3 tests pass;
3. full pytest passes;
4. recurrence preflight passes for the driven + 5 ps frozen protocol;
5. first natural x hop is found and selected;
6. frozen electronic state remains bitwise unchanged;
7. held phase rates remain zero;
8. frozen-surface energy conservation passes;
9. equilibrium force residual remains controlled;
10. traveling fields reconstruct excitation coordinate and velocity with max error < 1e-12;
11. full `jx` is reconstructed from retrograde + co-moving + special + cross terms with max error < 1e-12 eV/fs;
12. additive projection closure differs from one by < 1e-10 in early and late windows;
13. all attribution diagnostics are finite.

The local dominance classification is not a numerical acceptance gate.

## Interpretation guards

- `vy` can dominate total stored energy while contributing no direct `jx`; do not call total-energy dominance a current attribution.
- A local retrograde-dominated trailing current does not imply globally retrograde-dominated `vx` energy.
- Cross/interference current is an exact bilinear remainder, not a separate normal mode.
- The held boundary twist remains part of the frozen surface.
- No mobility, hopping rate, activation energy, threshold field or calibrated phonon lifetime is inferred.

## Decision after IP1r

If the late local trailing pattern is retrograde-dominated while the global `vx` energy remains direction-balanced, the mechanism becomes: **balanced counterpropagating Peierls energy globally, but phase-selected retrograde localization in the carrier-relative trailing current observable**.

The next stage should then identify which low-|qx| sectors create that phase selection and whether the same modal phase structure predicts the location/timing of later natural carrier relocations in the fully coupled dynamics.
