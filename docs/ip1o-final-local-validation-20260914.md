# IP1o final local validation — frozen-electronic-surface causal control

Date: 2026-09-14
Branch: `isotropic-polaron-barrier`
Validated commit: `3db0f98894aedab5aca17634088b68c86d1d347e`
Local artifact: `ip1o-local-validation/20260914T200102Z`

## Numerical closure

IP1o is numerically closed.

- overall local runner: PASS;
- failed numerical gates: none;
- pycompile: PASS;
- focused tests: PASS;
- full suite: **449/449 PASS**;
- 40x40 zero-damping PBC recurrence preflight: PASS;
- first persistent natural x event detected and selected correctly;
- held Peierls phase continuous at the switch: PASS;
- initial frozen-surface force exactly continuous with the fully coupled released force: PASS;
- branch states identical at creation: PASS;
- frozen electronic state bitwise unchanged through the complete continuation: PASS;
- frozen-branch external work: exactly 0 by construction and explicit numerical gate;
- frozen-surface energy conservation: PASS;
- fully coupled released energy conservation: PASS;
- projected zero modes: PASS in both branches;
- complete branch current trajectories serialized: PASS.

The validated working tree reports two unrelated untracked local helper files, `.git-run.sh.swp` and `git-run.sh`; neither changes the validated tracked source at the recorded commit.

## Protocol

- cell: 40x40 PBC;
- isotropic `J0y/J0x = 1.0`;
- initial field: +10 mV/A along x;
- T = 0 K;
- no thermostat;
- no IDC;
- dt = 0.2 fs;
- first natural persistent hop: 820 -> 819 (-x), transition start 2826 fs, accepted/switch 2874 fs;
- post-switch continuation: 2 ps;
- branch A: fully coupled, gauge-continuous zero-power held-phase continuation;
- branch B: identical held phase but complete electronic state frozen exactly at the switch while the classical lattice evolves conservatively on that fixed Ehrenfest surface;
- primary memory observable: RMS strength of the phase-structured trailing current vector at d=1..4;
- pre-hop same-trajectory field background used for the 95th-percentile robustness discriminator.

## Main result

The fully coupled released branch reproduces IP1n: two later electronic relocations occur, but its trailing pattern remains strongly separated from the pre-hop background through the entire late interval.

The decisive result is the frozen-electronic-surface branch. No electronic redistribution is possible there, yet the late pattern survives essentially unchanged.

### Frozen electronic surface

- electronic state bitwise unchanged: YES;
- late bins: 10/10 jointly above the background RMS and peak thresholds;
- latest passing bin center: 4824 fs = +1950 fs relative to switch;
- sustained pattern-memory gate: PASS;
- maximum frozen-surface energy drift: **1.43043e-10 eV**;
- no external work is performed after the switch.

### Comparison with the fully coupled released branch

- fully coupled late bins: 10/10 PASS;
- fully coupled released energy drift: 1.06572e-8 eV;
- frozen/full late mean amplitude ratio: **0.949160**;
- late mean cosine similarity of the four-component trailing-current vectors: **0.989977**.

The similarity is not restricted to one late peak. Across the continuation the two branch current patterns remain almost collinear. In the 1.0-1.5 ps interval the frozen amplitude is about 89% of the fully coupled value; in the 1.5-2.0 ps interval it returns to approximately the same mean amplitude as the fully coupled branch.

For the late interval, d=1, d=3 and d=4 are outward essentially all the time in both branches. d=2 remains the phase-sensitive component and changes sign. Therefore the memory remains a structured oscillatory current pattern and must not be rewritten as a monotonic backward flux through every boundary.

The early d3 -> d4 delay remains especially clean in the frozen branch: lag 616 fs, speed 1.62338 sites/ps, correlation 0.9999997, with both boundaries majority-outward. The d1 -> d2 gate remains boundary-limited and is not promoted as a transport result.

## Physical closure

IP1o provides strong causal evidence that the persistent post-hop response is encoded in the **classical lattice phase space** — the displacement/velocity state present after the natural hop — rather than requiring ongoing electronic redistribution or continued electrical power.

The strongest supported statement is:

> A natural isotropic polaron relocation leaves a long-lived, phase-structured lattice-current memory that persists for at least about 2 ps on a conservative frozen-electronic surface, with nearly the same spatial current-vector structure as the fully coupled zero-power continuation.

This is stronger than IP1n because later electronic hops are impossible in the frozen branch.

## Important guardrails

- The frozen electronic state still defines a static Ehrenfest force surface. IP1o shows that **dynamic electronic redistribution/backaction is not required**; it does not show what would happen if the carrier/electronic force were removed entirely.
- The held Peierls phase preserves the boundary twist inherited at switch. Both zero-power branches share this same static twist, so it cannot explain their difference, but IP1o is not a zero-twist PBC experiment.
- The trailing RMS is a norm of a phase-structured current vector, not a directional transport coefficient.
- Alternating signs across d=1..4 remain physically important.
- No mobility, hopping rate, activation energy, threshold field or calibrated phonon lifetime is extracted.

## Next stage

Proceed to **IP1q: mode-resolved decomposition of the autonomous frozen-surface memory**. The historical name `ip1p_phonon_recurrence_audit.py` is already occupied by the recurrence helper, so the new scientific stage skips `p` to avoid ambiguity.

The next calculation should rerun only the frozen-electronic-surface continuation while storing complete lattice displacement and velocity fields. Because the frozen electronic coupling enters the lattice equations linearly, it shifts the equilibrium but does not change the harmonic Hessian. Subtract the fixed-surface equilibrium and resolve the remaining autonomous response by polarization, wave vector and frequency. The primary questions are whether the persistent memory is dominated by the intermolecular Peierls coordinates or the intramolecular Holstein coordinate, which q sectors dominate, and whether the temporal spectral ridge follows the known harmonic dispersion.