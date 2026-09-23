# IP1u — final local validation (2026-09-23)

## Status and provenance

**Numerical validation: PASS. Preregistered primary physical classification: FAIL.**

User-supplied local validation package: `ip1u-local-validation/20260923T140533Z/`.

- Git branch: `isotropic-polaron-barrier`
- Validated source commit: `28e275bc599f41d558b58c68d45714511f21ce60`
- Local working tree: `M run.sh`. This local modification is disclosed; the uploaded package does not contain a patch for it, so it is not claimed to match the tracked root script byte-for-byte. The recorded runner commands and Git commit are consistent with the intended IP1u workflow.
- Full test suite: **477 passed in 7.76 s**
- Focused tests and pycompile: PASS
- Benchmark and phonon recurrence preflight: PASS
- Overall runner status: PASS; failed numerical gates: none.
- System: 40x40 isotropic PBC, +10 mV/A x during first-hop search, T=0, no bath/IDC, dt=0.2 fs, held phase for the zero-power released/common and post-return branches, CF4-Lanczos Krylov dimension 6.

## Verified common event history and intervention point

All event times are freshly detected:
1. Natural `820 -> 819 (-x)`: transition start **2826 fs**, accepted **2874 fs**.
2. Common released recrossing `819 -> 820 (+x)`: start **3024 fs**, accepted **3072 fs**.
3. Common released return `820 -> 819 (-x)`: start **3344 fs**, accepted **3392 fs**.

Both counterfactual branches are created from the identical fully coupled accepted state at **3392 fs**, after the common return.

At that instant, all lattice coordinates, the complete electronic state, `u`/`vy` velocities, and the special `vx` Fourier sectors are retained. Only the non-special `vx` velocity Fourier coefficients have their signs reversed. Consequently, the energy-resolved traveling components are exchanged without adding/removing matter energy.

- Special `vx` velocity sector mismatch: `9.41e-22 A/fs`.
- Non-special sector sign-reversal mismatch: `3.41e-21 A/fs`.
- `vx` kinetic-energy mismatch: **`2.17e-19 eV`**.
- Full matter-energy mismatch: **`0 eV`**.
- Retrograde/comoving energy exchange error: `4.34e-19 eV`.

Pre-intervention energy resolved along the last carrier hop direction (-x):
- retrograde: `0.0009573643883 eV`
- co-moving: `0.0008358002218 eV`.

After intervention, those values exchange to roundoff.

## Numerical conservation and recurrence

Common released segment:
- zero accumulated external work;
- maximum matter-energy drift `8.67e-9 eV`;
- maximum electronic norm error `4.13e-14`.

Post-return native branch (3 ps):
- zero accumulated external work;
- maximum matter-energy drift `1.10502e-8 eV`;
- maximum electronic norm error `8.22e-14`.

Post-return direction-reversed branch (3 ps):
- zero accumulated external work;
- maximum matter-energy drift `8.75001e-9 eV`;
- maximum electronic norm error `1.59e-13`.

All numerical gates, including projected zero modes and finite trajectory arrays, passed. Established size-aware matter-energy tolerance: `2e-4 eV`.

PBC harmonic maximum group velocity: `1.843909 sites/ps`; full stationary packet wrap: `21.693046 ps`. The planned final time at most 8 ps is safely earlier than this preflight bound. The stationary-packet estimate does not prove absence of all possible moving-carrier encounters, but no wrap is required to interpret the present window.

## Preregistered first re-escape decision

The prospective branch point is `t_b=3392 fs`. Primary first-x-event window: **1.5 ps**; qualifying events must *start and be persistently accepted* no later than `t_b+1500 fs`. Population L1 threshold: **0.25** within **2 ps**.

Observed:
- Native first persistent x event within primary window: **none**.
- Reversed first persistent x event within primary window: **none**.
- `reescape_status_changed = FALSE`.
- Maximum population L1 in first 2 ps: **0.10484056**.
- First `L1 >= 0.10`: **+1980 fs**.
- No `L1 >= 0.25` in the 2 ps classification window.
- `electronic_state_diverged = FALSE`.
- `directional_memory_controls_post_return_escape = FALSE`.

The **primary IP1u physical criterion formally FAILS**, irrespective of the numerical PASS.

## Complete 3 ps secondary observation

The full event lists are empty in **both** post-return branches. Thus no persistent x or y event is recorded during the 3 ps continuation, not merely within the primary 1.5 ps window.

Population L1 rises moderately after ~1.5 ps but never reaches the causal threshold:
- maximum L1 across full 3 ps: **0.14053626**, at **+2996 fs**;
- no sampled `L1 >= 0.25` anywhere in the full trajectory.

The intervention demonstrably perturbs the lattice current:
- early (0–0.5 ps) native/reversed trailing-current cosine mean: **-0.95157**;
- late (>=1 ps) cosine mean: **-0.06935**;
- native late trailing-amplitude mean: `2.69475e-7 eV/fs`;
- reversed late trailing-amplitude mean: `8.14370e-7 eV/fs`.

This verifies strong initial direction reversal without a corresponding persistent electronic relocation.

## Scientific closure and stop rule

IP1o–IP1r validate a long-lived, intrinsically classical Peierls phase-space memory and its strongly retrograde attribution to the **local trailing x-current**; this does **not** imply a retrograde majority of global lattice energy.

IP1s and IP1t found significant secondary electronic trajectory differences but failed their respective preregistered first-hop/return commitment gates. IP1u, designed prospectively around the post-return re-escape hypothesis, fails **both** its event-status and L1 thresholds despite a clean energy-preserving intervention.

Therefore **stop this isotropic deterministic causal-control sequence** rather than moving the branch point/window after observing another secondary difference. The established claim is a persistent Peierls-memory pattern, dynamically coupled to the electron, with demonstrated trajectory sensitivity in some interventions; **not** a validated causal control of discrete hop commitment under the three preregistered interventions tested.

A distinct future study, if pursued, must preregister a new physical question and independent ensemble of initial conditions/parameters, appropriate convergence checks, and statistical handling. No hopping rate, probability, mobility, activation barrier, transport coefficient, calibrated material phonon lifetime, or universal negative effect is inferred.

No pull request, merge into `main`, or GitHub Actions validation is implied by this local closure.
