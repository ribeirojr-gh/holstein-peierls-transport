# IP2a-4 — prospective complete pilot calibration at dt=0.10 fs

Date: 2026-09-23  
Status: preregistered; **not yet locally validated**.

## Rationale and strict separation

IP2a-2 at dt=0.20 fs is formally unqualified: 24 valid first x events nevertheless failed the preregistered pre-event work-balance limit of `2e-6 eV`. IP2a-3, a distinct six-trajectory numerical diagnostic, observed approximately fourfold residual reduction on dt halving. Both fixed reference states meet the **unchanged** limit at dt=0.10 fs. This motivates a **new** complete pilot calibration using dt=0.10 fs.

IP2a-4 is not a reinterpretation of IP2a-2 and cannot be counted as IP2b evidence. It must not inspect a native/reversed paired trajectory, select an energy based on post-event behavior, or adjust its criteria after observing the outcomes.

## Locked protocol

Reuse the IP2a-1 deterministic velocity-only 32-state design, evaluating only the fixed antithetic-pair pilot IDs:

`0, 4, 8, 12, 16, 20, 24, 28`.

Evaluate each ID at each fixed candidate energy:

`1e-5, 3e-5, 1e-4 eV`.

System and integrator:
- 40x40 isotropic PBC;
- same static optimized initial polaron;
- `J0y/J0x=1`;
- +10 mV/A x for event generation only;
- zero-temperature deterministic dynamics, no thermostat, no IDC;
- **dt=0.10 fs** fixed;
- CF4-Lanczos Krylov dimension 6 fixed;
- 2 fs event sampling, 10 fs energy/work diagnostics fixed;
- 4000 fs driven search limit fixed;
- no post-event continuations.

For every trial:
1. prepare the fixed energy-matched velocity-only state;
2. independently propagate a **200 fs field-free control** and reject its feasibility if any persistent event is accepted, numerical gates fail, or the trial becomes nonfinite;
3. discard the control branch and restart field-driven evolution from the **original prepared state**, not the endpoint of the control;
4. stop at the **first** accepted persistent event, whether x/y/non-neighbor, or at 4000 fs if none;
5. reject event-generation feasibility if the earliest persistent event transition starts <200 fs, is not nearest-neighbor x, or is not accepted by 4000 fs.

Do not seek a later x event after an earlier y event.

## Numerical and eligibility criteria

The original fixed IP2a-2 bounds are retained:
- maximum field-free matter-energy drift <= **2e-6 eV** and size-aware tolerance;
- maximum field-on energy–work residual <= **2e-6 eV** and size-aware tolerance;
- electronic norm error < **1e-10**;
- projected lattice zero-mode mean < **1e-10**;
- no field-free persistent event;
- no early driven event.

A trial is a qualifying first-x event only if **every** numerical and event condition passes. Count excluded member IDs and explicit reasons; do not delete or replace any member.

## Prospective candidate selection

An energy qualifies only if:
- all **8/8** trials are numerically stable and field-free clean;
- none of the eight has an early driven event;
- at least **6/8** have a valid qualifying first persistent nearest-neighbor x event.

Select the **smallest qualifying energy**, with ascending-energy priority fixed before execution. If all candidates fail, select **no energy** and stop before IP2b.

## Numerical integrity versus physical selection

The executable must expose two separate results:

- **execution/numerical integrity**: pycompile, focused/full pytest, recurrence preflight, fixed protocol, complete serialized trials and finite audits;
- **energy selection**: eligibility under the locked 8/8 and >=6/8 criteria.

A runner may pass execution integrity while no energy qualifies. Conversely, no production energy may be selected if the runner's integrity gate fails.

## Reproducibility

Record source commit, branch, dirty/untracked status, software versions, thread environment, every command and log, all 24 member-level records, full maximum residuals, first-event times, accepted-event directions and classification reasons. Include JSON and NPZ summaries.

Retain the completed IP2a-2 FAIL and IP2a-3 numerical convergence PASS as separate records. No GitHub Actions/PR/merge is part of local validation.

## Downstream condition

Only after an IP2a-4 artifact passes both execution integrity and candidate-selection gates may the chosen energy be frozen for a *newly reviewed* IP2b implementation. Any claim of population-level statistical inference still requires a genuine stochastic sampling model; the existing 32-member antithetic grid supports finite-design descriptive analysis only.
