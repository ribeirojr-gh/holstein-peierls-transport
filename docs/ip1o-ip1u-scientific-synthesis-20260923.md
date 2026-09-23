# IP1o–IP1u scientific synthesis — isotropic Peierls memory and causal controls

Date: 2026-09-23  
Branch: `isotropic-polaron-barrier`

## Purpose

This document consolidates the closed isotropic-memory sequence IP1o–IP1u and separates four logically distinct questions:

1. Does a natural carrier relocation leave a persistent lattice memory?
2. What modal degrees of freedom store that memory?
3. Which traveling components generate the local trailing x-current signature?
4. Does reversing those traveling components control a later discrete carrier relocation?

The answers are not interchangeable.

## 1. Existence of autonomous lattice memory — established

IP1o freezes the complete electronic wavefunction at the accepted natural hop while allowing the classical lattice to evolve conservatively on the corresponding static Ehrenfest surface.

Validated facts:
- natural event: `820 -> 819 (-x)`, start 2826 fs, accepted 2874 fs;
- no electronic redistribution is possible in the frozen branch;
- no external electrical work is performed after the switch;
- frozen-surface energy drift: `1.43043e-10 eV`;
- late memory bins: 10/10 above preregistered pre-hop background thresholds;
- frozen/full late mean amplitude ratio: **0.949160**;
- late current-vector cosine similarity: **0.989977**.

Supported claim:

> The natural isotropic relocation leaves a long-lived phase-structured lattice-current memory encoded in the classical lattice phase space. Dynamic electronic redistribution and continued electrical power are not required for its persistence over the validated ~2 ps window.

This is a statement about memory on a static electronic force surface, not about removing the carrier entirely.

## 2. Modal content of the autonomous memory — established

IP1q resolves the frozen-surface excitation relative to its exact shifted equilibrium.

Total autonomous excitation:
`E_exc = 0.011502317612246332 eV`.

Partition:
- Holstein `u`: **1.2378%**
- Peierls `vx`: **18.3523%**
- Peierls `vy`: **80.4099%**
- total Peierls: **98.7622%**

The excitation is therefore overwhelmingly intermolecular/Peierls.

The dominant Peierls sectors are long wavelength. In `vx`, 72.50% lies in the lowest nonzero quarter of the x Brillouin zone. In `vy`, 90.85% lies in the corresponding low-q quarter.

The harmonic q-omega ridges agree with the exact model dispersion to within one FFT-frequency bin for resolved sectors.

Supported claim:

> The autonomous post-hop memory is a long-wavelength, Peierls-dominated harmonic lattice excitation on the frozen electronic surface.

## 3. Global traveling energy versus local trailing x-current — established

IP1q shows that direction-resolved `vx` energy is nearly balanced:
- retrograde: **50.3558%**
- co-moving: **49.6442%**

Thus there is no globally retrograde-dominated `vx` energy reservoir.

IP1r performs an exact algebraic reconstruction of the local `jx` current from retrograde, co-moving, special, and interference components.

Late (+1 to +5 ps) additive attribution of the B-centered trailing d=1..4 current:
- retrograde: **94.8739%**
- co-moving: **3.8806%**
- cross/interference: **1.2455%**
- special sectors: negligible

Late retrograde attribution is robust across d=1..4 (~91.5–96.2%).

Supported claim:

> The long-lived local trailing x-current pattern is strongly selected by retrograde `vx` traveling phases even though the global direction-resolved `vx` energy is almost perfectly balanced.

This is a local-current attribution, not a statement that ~95% of total lattice energy propagates backward.

## 4. Does Peierls traveling direction control discrete electronic commitment? — not established

IP1s, IP1t, and IP1u apply energy-preserving `vx` traveling-direction reversals at progressively later prospectively defined branch points.

### IP1s — immediately after the natural hop

Primary gate: first persistent x event within 2 ps must change by presence/direction or >=100 fs in start time, AND population L1 must reach >=0.25.

Result:
- first post-switch event remains `819 -> 820 (+x)` in both branches;
- start-time difference: 36 fs;
- maximum L1 in first 2 ps: **0.497651**;
- primary physical result: **FAIL**.

Secondary: native later returns `820 -> 819`; reversed does not during the observed continuation.

### IP1t — after the common +x recrossing

Primary gate: immediate `820 -> 819` return commitment within 1 ps must change, AND L1 >=0.25.

Result:
- both branches return `820 -> 819 (-x)`;
- start-time difference: 60 fs;
- maximum L1 in first 2 ps: **0.339238**;
- primary physical result: **FAIL**.

Secondary: the reversed branch later re-escapes `819 -> 820`; native does not during the observed continuation.

### IP1u — after the common -x return

Primary gate: first re-escape within 1.5 ps must change, AND L1 >=0.25 within 2 ps.

Result:
- neither branch has a persistent x event in the primary window;
- neither branch has any persistent event over the complete 3 ps continuation;
- maximum L1 in first 2 ps: **0.10484056**;
- maximum L1 in the full 3 ps: **0.14053626**;
- primary physical result: **FAIL**.

The intervention itself is strong: early native/reversed trailing-current cosine is `-0.95157`, but no persistent carrier relocation follows.

## Hierarchy of supported statements

### Strongly supported

1. A natural isotropic carrier relocation leaves a persistent classical lattice phase-space memory.
2. That memory is overwhelmingly Peierls/intermolecular and predominantly long wavelength.
3. Total stored excitation is dominated by `vy`, while the local trailing `jx` observable is a `vx` phenomenon.
4. Global `vx` traveling energy is nearly direction-balanced.
5. The local trailing x-current is nevertheless robustly retrograde-dominated in exact traveling-wave attribution.
6. Energy-preserving phase-direction reversal measurably changes lattice-current structure and, in IP1s/IP1t, substantially changes later electronic populations.

### Not established

1. Retrograde Peierls memory does not have a validated causal control over the specific discrete hop/recrossing/re-escape commitments tested in IP1s–IP1u.
2. The deterministic trajectory sequence does not yield a hopping probability, hopping rate, mobility, activation barrier, threshold field, diffusion coefficient, or material phonon lifetime.
3. The three negative primary gates do not prove universal dynamical irrelevance of Peierls phase direction.

## Stop rule for IP1

The single-trajectory isotropic causal-control sequence ends at IP1u. No branch point, event window, or threshold will be moved again using this same trajectory in order to seek a positive outcome.

Any further causal transport study must be a separately preregistered ensemble study with independent initial conditions or parameter instances and prospective statistical handling.

## Recommended manuscript-level framing

A defensible central narrative is:

> Natural polaron relocation imprints a long-lived Peierls phase-space memory. Although the global x-Peierls energy is direction-balanced, phase selection makes the local trailing x-current strongly retrograde. Counterfactual reversal of that traveling phase changes later coupled electron-lattice evolution, but the present deterministic controls do not establish control of individual hopping commitment.

This distinction should be preserved in figures, abstract language, and conclusions.
