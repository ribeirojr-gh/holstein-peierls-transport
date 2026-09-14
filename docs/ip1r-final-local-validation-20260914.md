# IP1r final local validation — 2026-09-14

## Validation artifact

User-supplied local artifact:

`ip1r-local-validation/20260914T214101Z`

Tracked branch state:

- branch: `isotropic-polaron-barrier`
- commit: `9ab68d7b67d9a09431cc28a0ce4e27a75399415a`
- local status contained only unrelated untracked `.git-run.sh.swp` and `git-run.sh` files.

## Numerical closure

IP1r is numerically closed.

- pycompile: PASS;
- focused pytest: PASS;
- full pytest: **458 passed**;
- recurrence preflight: PASS;
- source IP1q frozen trajectory: PASS;
- exact traveling-current attribution: PASS;
- failed gates: none.

The decomposition reconstructs the original lattice phase space/current to roundoff:

- maximum coordinate reconstruction error: `5.551115123125783e-17 A`;
- maximum velocity reconstruction error: `2.710505431213761e-20 A/fs`;
- maximum x-current reconstruction error: `3.3881317890172014e-21 eV/fs`;
- early projection closure error: `0`;
- late projection closure error: `2.220446049250313e-16`.

The source natural event remains the isotropic `820 -> 819 (-x)` hop, transition start `2826 fs`, accepted/switch time `2874 fs`.

## Global modal energy versus local current attribution

The global direction-resolved x-Peierls modal energy remains almost balanced:

- retrograde: `0.001062279912818175 eV`;
- co-moving: `0.0010472669657421807 eV`;
- retrograde fraction of direction-resolved vx energy: **0.5035583345**;
- direction-resolved fraction of vx energy: **0.9993413357**.

Thus there is no global retrograde-energy dominance in vx.

The local trailing x-current is very different.

### Early interval: switch to +1 ps

Additive projection onto the full trailing current vector at d=1..4:

- retrograde: **0.8268215195**;
- co-moving: `0.1936469869`;
- cross/interference: `-0.0204685063`;
- special sectors: negligible.

The negative cross contribution is destructive interference; all additive projections sum exactly to one.

### Late interval: +1 to +5 ps

Additive projection:

- retrograde: **0.9487385459**;
- co-moving: **0.0388061732**;
- cross/interference: **0.0124552808**;
- special sectors: negligible.

Retrograde squared-norm/full squared-norm ratio: **0.9177483614**.

The pipeline therefore reproduces the prior exploratory ~94.8% result as a preregistered, audited calculation.

## Robustness in time and distance

A post-validation sensitivity audit of the saved current decomposition shows that the late result is not caused by one boundary or one isolated time interval.

Late (>1 ps) retrograde additive projection by trailing boundary:

- d=1: **0.9146**;
- d=2: **0.9548**;
- d=3: **0.9616**;
- d=4: **0.9616**.

Across successive 1 ps windows the retrograde term is normally the dominant contribution (~0.83–0.98 for ordinary positive additive windows). In one interval the co-moving contribution is destructively aligned with the full current, so the additive retrograde projection exceeds one; this is permitted because projection fractions are an additive decomposition with interference, not probabilities.

## Scientific closure

IP1r resolves the apparent IP1q paradox.

1. The autonomous frozen memory is overwhelmingly Peierls in total energy, with most total energy stored in vy.
2. The x-Peierls traveling-wave energy itself is globally almost 50/50 retrograde/co-moving.
3. Nevertheless, the **local trailing x-current pattern behind the hopped carrier is strongly and robustly selected by the retrograde vx traveling component**.
4. Therefore global energy asymmetry and local current directionality are distinct observables.

The validated statement is:

> The long-lived local x-current memory left behind the natural isotropic hop is retrograde-dominated in its vx traveling-wave attribution (~94.9% over +1 to +5 ps), even though the global direction-resolved vx modal energy is nearly balanced.

This does **not** imply that ~95% of total lattice energy propagates backward. It does **not** define mobility, a hopping rate, an activation barrier, a threshold field, or a calibrated phonon lifetime.

## Next scientific question

The next step should return to the transport/barrier problem and ask causally whether the direction of the Peierls memory changes the subsequent electronic trajectory.

The appropriate intervention is an energy-preserving reversal of the non-special vx traveling direction at the post-hop switch:

- keep all lattice coordinates unchanged;
- keep u and vy velocities unchanged;
- keep vx q=0/Nyquist special velocity sectors unchanged;
- reverse the sign of every non-special vx velocity Fourier component, which exactly swaps the +x and -x traveling amplitudes at fixed coordinates;
- preserve vx modal energy and therefore total matter energy at the intervention;
- use the same held Peierls phase / zero external power in both comparison branches;
- then resume fully coupled Ehrenfest dynamics.

This isolates traveling-direction phase information without changing the vx energy spectrum.