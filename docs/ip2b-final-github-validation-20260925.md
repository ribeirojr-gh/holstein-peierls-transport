# IP2b — final GitHub Actions validation and scientific closure (2026-09-25)

## Status

**GitHub execution integrity: PASS.**  
**All 32 preregistered members valid: 32/32.**  
**Locked IP2b primary finite-design criterion: PASS.**

IP2b therefore supports a reproducible electronic-trajectory sensitivity to the energy-preserving non-special x-Peierls traveling-direction reversal **across the complete tested deterministic 32-state design**.

This is not a population probability statement and does not estimate a hopping rate, mobility, diffusion coefficient, activation energy, threshold field or material lifetime.

## GitHub Actions provenance

Workflow: `ip2b-production-paired-ensemble`  
Run ID: `36198826221`  
Run attempt: `1`  
Validated commit: `cddfb1d45f7b71f2bc4911c8304b75f7fee48ef5`  
Branch: `isotropic-polaron-barrier`  
Workflow conclusion: **success**

The workflow contained 34 jobs:
- one validation/preflight job;
- 32 independent member jobs, IDs 0..31;
- one final aggregate job.

All 34 jobs completed successfully.

The validation job reports:
- pycompile: PASS;
- focused IP2b/direction-reversal/ensemble/field/numerical tests: PASS;
- complete repository suite: **516 passed in 25.77 s**;
- 40x40 recurrence preflight: PASS;
- stationary harmonic full-wrap diagnostic: **21.693046 ps**;
- maximum IP2b planned horizon: **6 ps**.

GitHub artifacts include one validation artifact, all 32 member artifacts and one aggregate artifact, retained for 90 days.

## Frozen production protocol

- 40x40 isotropic PBC;
- deterministic 32-state Peierls velocity preparation grid;
- 16 exact antithetic pairs;
- preparation energy: **1.0e-5 eV**;
- T=0, no thermostat, no IDC;
- event-generation field: **+10 mV/A x**;
- dt: **0.10 fs**;
- CF4-Lanczos Krylov dimension 6;
- independent 200 fs field-free control;
- maximum driven event search: 4000 fs;
- first persistent event of any direction governs inclusion;
- population/event sampling: 2 fs;
- energy diagnostics: 10 fs;
- gauge-continuous held Peierls phase after accepted event;
- three 2000 fs zero-power continuations: native, reversed and native-replicate.

No member was replaced, excluded post hoc or selected from post-intervention behavior.

## Complete pre-intervention audit

All 32 members pass every pre-intervention criterion.

For every member:
- field-free control is clean;
- first persistent driven event is the same nearest-neighbor hop
  `820 -> 819 (-x)`;
- transition start: **2826 fs**;
- acceptance/branch time: **2874 fs**;
- no early preparation-associated relocation.

Across all 32 members:
- maximum field-free energy-drift range:
  `3.886e-16 .. 8.327e-16 eV`;
- driven pre-event energy-work residual range:
  `5.55069e-7 .. 5.56887e-7 eV`;
- driven electronic-norm error range:
  `2.769e-12 .. 3.433e-12`;
- projected zero-mode mean range:
  `5.90e-18 .. 2.55e-17`.

The strict `2e-6 eV` pre-event energy-work gate is therefore satisfied for every member.

## Intervention audit

Every member passes every intervention identity/conservation gate.

Across all members:
- held/driving phase mismatch at branch point: exactly zero;
- held/driving Hamiltonian mismatch: exactly zero;
- full matter-energy mismatch:
  `0 .. 5.55112e-17 eV`;
- vx kinetic-energy mismatch:
  `0 .. 1.08420e-18 eV`;
- traveling retrograde/comoving swap error:
  `2.16840e-19 .. 8.67362e-19 eV`;
- coordinates and electronic state are unchanged by intervention;
- u and vy velocities are unchanged;
- special qx sectors are preserved;
- all non-special vx velocity sectors are sign reversed;
- native-replicate state is an exact independent copy of native.

## Zero-power continuation audit

All three branches complete the full 2000 fs continuation for all 32 members.

External work is exactly zero in every continuation.

Maximum energy-drift ranges:
- native: `2.27945e-9 .. 2.88956e-9 eV`;
- reversed: `1.42295e-9 .. 1.67252e-9 eV`;
- native-replicate: identical to native.

Maximum branch norm errors are below `2.75e-13`.  
Maximum projected zero-mode means are below `3.11e-17`.

The native and native-replicate electronic population trajectories are identical at every saved sample for all 32 members:

`D_i^ctrl = 0` exactly for every member.

The aggregate ratio therefore uses the preregistered denominator floor `max(D_i^ctrl,1e-12)`. The resulting ~`5e11` ratios should be interpreted as intervention divergence being far above a zero deterministic reproducibility floor, not as a physically meaningful dimensionless amplification factor.

## Primary electronic-population result

For each member,

`D_i = max_{0 <= t-t_s <= 2 ps} sum_j |p_j^native - p_j^reversed|`.

Observed distribution:

- minimum `D_i`: **0.4954747243**;
- Q1: **0.4971888800**;
- median: **0.4976863456**;
- Q3: **0.4984731808**;
- maximum: **0.5002696855**;
- IQR: **0.0012843008**;
- finite-design standard deviation: ~`0.0010404`;
- members reaching `D_i >= 0.25`: **32/32 = 100%**;
- first `L1 >= 0.10`: 372, 380 or 384 fs depending on member;
- first `L1 >= 0.25`: **994 fs for all 32 members**.

The maximum `D_i` occurs at 1990 or 2000 fs in all members, near the fixed analysis-window edge. Therefore `D_i` is a **2 ps window-limited maximum**, not evidence that the native/reversed separation has saturated by 2 ps.

Antithetic partners remain very close descriptively:
- median absolute partner difference in `D_i`: ~`0.001462`;
- maximum absolute partner difference: ~`0.004795`.

No inferential statistics are assigned to those deterministic pairs.

## Locked primary gates

All five preregistered gates pass:

1. at least 24 valid members: **PASS, 32/32**;
2. median `D_i >= 0.25`: **PASS, 0.497686**;
3. >=60% reach `D_i >= 0.25`: **PASS, 100%**;
4. median `D_i/max(D_i^ctrl,1e-12) >= 100`: **PASS, ~4.97686e11**;
5. all 32 member IDs present or explicitly rejected: **PASS, all 32 valid**.

**Primary IP2b classification: PASS.**

## Secondary discrete-event result

The first post-branch persistent x event is a direct recrossing `819 -> 820 (+x)` in **both native and reversed branches for all 32 members**.

Thus:
- first-event presence disagreement: 0/32;
- first-event direction disagreement: 0/32;
- native direct-recross fraction: 32/32;
- reversed direct-recross fraction: 32/32.

However, reversal advances the first recrossing transition start by **36 or 38 fs** across the ensemble.

More importantly, the ordered x-event topology histories differ in **32/32 members**:
- native: `819 -> 820`, followed by `820 -> 819` within the 2 ps window;
- reversed: `819 -> 820` only within the 2 ps window.

Native second-return transition starts occur at 3344, 3354 or 3356 fs (accepted at 3392, 3402 or 3404 fs absolute trajectory time). This secondary topology-history result is consistent across the full deterministic design but remains secondary to the preregistered L1 endpoint.

## Pre-intervention modal covariates

Across the 32 accepted branch points:
- fixed-surface excitation total:
  `0.0114338 .. 0.0115802 eV`;
- vx excitation:
  `0.00210561 .. 0.00212674 eV`;
- vy excitation:
  `0.00916821 .. 0.00932938 eV`;
- retrograde fraction of direction-resolved vx:
  **0.49958 .. 0.50747**;
- vx lowest-nonzero-quarter fraction:
  **0.72144 .. 0.72970**;
- vy lowest-nonzero-quarter fraction:
  **0.90824 .. 0.90878**.

Thus the branch-point ensemble remains globally close to a 50/50 vx traveling-energy split while retaining long-wavelength Peierls character, consistent with the earlier IP1q distinction between global energy balance and local phase-selected response.

The instantaneous branch-point d=1..4 retrograde trailing-current projection ranges ~0.474 to ~0.498. This quantity is **not** the late-time ~95% local retrograde attribution established in IP1r, so the two must not be conflated.

Current-decomposition reconstruction error is <=`3.39e-21 eV/fs`.

## Independent artifact audit

The final aggregate JSON, aggregate NPZ and all 32 individual JSON/NPZ artifacts were independently cross-checked after download from GitHub Actions.

Confirmed:
- exactly member IDs 0..31;
- 16 antithetic pair IDs with +/- members;
- 32/32 explicit valid-member records;
- no rejection reasons;
- every individual `D_i` equals the maximum of its saved native/reversed L1 series;
- every individual `D_i^ctrl` equals the maximum of its saved native/native-replicate L1 series;
- all member continuations contain 1001 samples from 0 to 2000 fs;
- aggregate member-indexed `D_i` and `D_i^ctrl` arrays exactly match the independently read member artifacts;
- saved max-L1 population maps remain normalized to roundoff.

## Scientific conclusion

The energy-preserving reversal of the non-special x-Peierls velocity phase is not merely a single-trajectory sensitivity in this model. Under the frozen IP2 protocol it produces a highly reproducible change in subsequent electronic population evolution across all 32 tested deterministic low-energy Peierls preparations, while an exact native-replicate control remains indistinguishable.

The strongest defensible wording is:

> Across the complete tested deterministic preparation grid, reversing the non-special x-Peierls traveling phase at a matched post-hop state reproducibly changes the subsequent coupled electron-lattice trajectory, producing large electronic-population divergence relative to an exact deterministic reproducibility control.

The experiment also shows a consistent secondary change in the later x-event topology history, but it does **not** establish a population-level hopping probability or universal dynamical law.

## Relationship to IP1

IP1s–IP1u remain formally negative under their own preregistered discrete-event commitment gates and must not be rewritten as positive.

IP2b asks a different primary question: reproducible **trajectory sensitivity** across a prospective deterministic preparation design. It passes that criterion decisively.

Thus the combined record is internally consistent:
- persistent Peierls memory: established;
- local phase-selected retrograde current attribution: established;
- individual preregistered IP1 commitment-control gates: negative;
- finite-design ensemble electronic-trajectory sensitivity to Peierls-direction reversal: positive in IP2b.

## Next-step rule

The original IP2 design permits a separately preregistered IP2c q-band study after a positive IP2b result. IP2c would address *which fixed q bands* carry the causal sensitivity and is a mechanistic extension, not a prerequisite for accepting the IP2b result.

For the broader project workflow, IP2b also completes the major T=0 single-polaron ensemble causal-validation milestone. Any decision to perform IP2c before freezing the T=0 production release should be made prospectively and separately; IP2b thresholds and outcomes are now closed.
