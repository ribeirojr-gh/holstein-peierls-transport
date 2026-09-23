# IP2 — deterministic design ensemble study of Peierls-phase sensitivity

Date: 2026-09-23  
Status: preregistration / design only

## Why IP2 is a new study

IP1s–IP1u use one deterministic isotropic trajectory and prospectively move the intervention point after specific accepted events. All three preregistered discrete-event gates fail, although IP1s/IP1t show substantial secondary population sensitivity.

IP2 is intentionally not named IP1v. It starts a new inferential unit: a paired ensemble in which each initial condition produces a native and an energy-preserving Peierls-direction-reversed trajectory.

The objective is to determine whether the phase-direction intervention produces a reproducible **ensemble-level change in electronic trajectory sensitivity**. The first phase will not estimate material hopping rates or mobility.

## Primary scientific question

Across a preregistered deterministic design ensemble of physically matched post-hop states, does energy-preserving reversal of non-special `vx` traveling direction produce a systematic increase in subsequent electronic-state divergence relative to numerical/reproducibility controls?

## Phase IP2a — ensemble construction and numerical reproducibility

### IP2a-1 — construction-only validation

Before any field-driven feasibility calculation, validate the deterministic 32-member preparation family at all three candidate energies. This substage is restricted to:

- exact fixed added kinetic/matter energy;
- zero-mode removal;
- low-q Fourier support;
- exact sign-opposite pairing;
- zero ensemble-mean added velocity;
- distinctness of the 16 base preparations;
- unchanged lattice coordinates and electronic state at preparation.

**IP2a-1 must not propagate native/reversed outcomes and must not select the production perturbation energy.**

### IP2a-2 — pre-intervention event-generation calibration

The fixed calibration subset is **member IDs 0, 4, 8, 12, 16, 20, 24, 28** (four antithetic pairs) at **all three** candidate energies. The driven event-search horizon is **4000 fs**. Before each driven trial, an independent **200 fs field-free continuation** of the same prepared state must show no persistent relocation; it is discarded rather than carried into the driven trial. A driven event starting in the first **200 fs** is classified as preparation-associated early relocation and does not count toward feasibility.

A valid event-generation trial requires: first accepted persistent event is a nearest-neighbor x event within 4000 fs; no early event; field-free control clean; pre-event field-on energy/work residual <= **2e-6 eV** (and within the established size-aware bound), electronic norm error < **1e-10**, and projected zero-mode mean < **1e-10**. The 2e-6 eV pre-intervention work-balance bound is deliberately tighter than the generic extensive-system tolerance because the smallest injected lattice energy is only 1e-5 eV. Stop on the first persistent event even if it is y-directed; never skip earlier y events to seek an x event.

Choose the **smallest candidate energy** meeting: all eight pilot trials numerically stable and field-free clean, no early driven event, and at least **6 of 8** valid first-x-event trials. Ties are broken by lower energy. If none qualifies, do not select a production energy or begin IP2b. Record every pilot outcome, including rejected trials, before any native/reversed continuation.


Only after IP2a-1 passes, run the preregistered calibration subset across the three candidate energies. IP2a-2 may inspect only pre-intervention diagnostics: integration stability, first-x-event yield within the fixed search horizon, field-on numerical quality, and absence of immediate preparation-induced relocation. It may not create or compare native/reversed post-hop branches.


### Common physical model

Retain the validated isotropic model and integrator:
- 40x40 PBC
- `J0y/J0x = 1`
- T=0 baseline
- no thermostat / no IDC
- dt=0.2 fs
- CF4-Lanczos, Krylov dimension 6
- +10 mV/A x field only during event-generation stage
- gauge-continuous held Peierls phase after the accepted event
- no external power during paired continuation.

### Distinct deterministic initial-condition family

Generate a deterministic grid of **32 distinct low-energy lattice phase-space preparations** before the driven event search.

The ensemble must be specified without looking at post-intervention outcomes.

For each preparation:
1. start from the same converged static polaron;
2. perturb only the nonzero Peierls harmonic modes using a deterministic phase grid;
3. preserve zero-mode constraints exactly;
4. use paired +/- phase assignments so the ensemble has zero mean added displacement and velocity over the complete design;
5. keep the added lattice excitation energy fixed at a single preregistered value for all members;
6. do not condition membership on whether a later counterfactual produces a desired event.

The perturbation energy must be small relative to the validated IP1q autonomous excitation `0.0115023 eV`. Before production, IP2a will test candidate fixed energies `1e-5`, `3e-5`, and `1e-4 eV` solely for numerical/event-generation feasibility. The production value must be selected using pre-intervention diagnostics only: stable integration, event-generation yield, and absence of immediate artificial relocation at t=0. It must then be frozen for IP2b.

This calibration stage may not inspect native-versus-reversed outcome differences.

### Design-distinctness and statistical-inference guard

Translations of the exact same state are symmetry copies and do not count as distinct ensemble members. Distinct members must differ in Peierls phase-space coordinates/velocities before the field-driven event generation. The 32 preparations form a fixed deterministic phase grid with 16 antithetic (+/- velocity) pairs; they are **not independent random samples**. Treat ensemble summaries as finite-design descriptive statistics, not population probabilities or confidence intervals. A separate stochastic ensemble, if needed later, must specify a sampling distribution and count antithetic pairs as dependent clusters.

### Event inclusion

For each production ensemble member:
- run the field-driven trajectory up to a fixed maximum search time preregistered before IP2b;
- select the first persistent nearest-neighbor x event;
- accept the member if the event is found and all numerical gates pass;
- rejection reasons are numerical/event-generation reasons only and are recorded before any paired intervention.

No member may be discarded because its native/reversed result is small, null, inconvenient, or opposite to expectation.

## Phase IP2b — paired counterfactual ensemble

At each accepted first x event:
- create a native zero-power released branch;
- create a direction-reversed branch by sign reversing all non-special `vx` velocity Fourier coefficients at fixed coordinates/electronic state;
- preserve `u`, `vy`, special `vx`, total matter energy, and the `vx` modal-energy spectrum to established tolerances;
- propagate both branches for a fixed **2 ps** primary comparison window, with an optional preregistered 5 ps secondary observation window if recurrence permits for all members.

## Primary endpoint

For ensemble member i define

`D_i = max_{0 <= dt <= 2 ps} L1_i(dt)`

where `L1_i` is the matched native/reversed electronic population distance.

Primary ensemble statistic:
- median `D_i` across all accepted ensemble members.

Also report:
- interquartile range;
- fraction with `D_i >= 0.25`;
- paired time-to-first `L1 >= 0.10` and `>=0.25` when reached.

The `0.25` threshold is retained from IP1s–IP1u and is not tuned from IP2 outcomes.

## Negative-control endpoint

Each accepted state also receives a **replicate native branch** with no intervention but independent code-path instantiation. In deterministic arithmetic this pair should agree to roundoff/numerical tolerance.

Define

`D_i^ctrl = max L1(native, native-replicate)`.

Numerical/causal separation requires that intervention-induced divergence be well above this reproducibility floor.

## Primary hypothesis criterion

IP2b will be classified as evidence of reproducible trajectory sensitivity only if all of the following are met:

1. at least **24 of the preregistered 32** ensemble members yield valid accepted x events and paired continuations;
2. median `D_i >= 0.25`;
3. at least 60% of valid members reach `L1 >= 0.25` within 2 ps;
4. median `D_i / max(D_i^ctrl, 1e-12) >= 100`;
5. the full 32-member finite-design response distribution and member-level results are reported, preserving all 16 antithetic-pair identifiers and all prespecified exclusions; **no p-values, permutation/randomization claims, or population confidence intervals are assigned to this deterministic grid**.

The principal conclusion is reproducible sensitivity across the **tested deterministic design states** under the modeled intervention. It is not a population-level probability statement, a physical hopping rate, or a mobility measurement.

## Secondary discrete-event analysis

Prospectively record the first persistent x event in 2 ps for both branches.

Report, without promoting it above the primary L1 endpoint:
- same/different event presence;
- same/different direction;
- transition-start difference;
- direct recrossing fraction;
- event-history disagreement fraction.

Because IP1 showed that event labels can remain unchanged while populations diverge, event disagreement is secondary in IP2.

## Modal covariates

For every accepted branch point record, before intervention:
- total autonomous excitation estimate relative to the appropriate local/frozen reference when available;
- `u`, `vx`, `vy` energy partitions;
- direction-resolved `vx` retrograde/co-moving energies;
- low-q `vx` energy fractions using fixed bins inherited from IP1q;
- local d=1..4 trailing-current vector and retrograde attribution when computationally practical.

These covariates may be used to explain heterogeneity only after the primary ensemble result is fixed. Any q-band optimization becomes a separate IP2c preregistration.

## Numerical gates per member

Require:
- static preparation converged;
- electronic norm controlled;
- zero modes controlled;
- first x event detected within preregistered search horizon;
- field-on energy/work residual within established tolerance;
- held-phase rate zero;
- intervention coordinate/electronic-state identity;
- non-special `vx` sign reversal and special-sector preservation;
- `vx` kinetic and full matter energy mismatch <=1e-12 eV;
- native/reversed external work zero;
- native/reversed energy conservation within the established size-aware bound;
- recurrence preflight valid for the complete member horizon;
- all serialized observables finite.

A failed numerical gate rejects that member before physical aggregation and records the reason.

## Multiplicity / analysis discipline

- IP2b has one primary endpoint: the **finite-design distribution** of maximum population L1 in 2 ps.
- Event disagreement and modal correlations are secondary.
- q-band-specific interventions are not tested in IP2b.
- No branch point/window/threshold is changed after production outcomes are inspected.
- All accepted and rejected member identifiers and rejection reasons are serialized.

## Stop / advance rule

If the IP2b primary criterion fails, close the present Peierls-direction causal claim at the ensemble level.

If it passes, proceed to IP2c with a new preregistration that tests fixed q bands inherited from IP1q, using energy-preserving band-selective reversals. IP2c may ask *which* modes drive the ensemble sensitivity; it may not retroactively alter the IP2b criterion.

## Scope guard

Even a positive IP2b result supports only reproducible sensitivity across the fixed deterministic preparation grid to a Peierls phase-direction intervention in this model.

A physical hopping probability/rate requires a physically justified statistical ensemble (e.g. thermalized initial conditions and/or stochastic bath model), adequate independent sampling, censoring treatment, convergence with system size/time step, and a separately preregistered rate estimator.
