# IP2a-4 — final GitHub Actions validation (2026-09-25)

## Status

**Execution/numerical integrity: PASS.**  
**Production Peierls preparation energy selection: PASS.**

Selected production preparation energy:

`1.0e-5 eV`

This is the smallest qualifying preregistered candidate and is therefore selected by the locked IP2a-4 rule. The larger candidates `3.0e-5` and `1.0e-4 eV` also qualify, but are not selected.

The earlier IP2a-2 calibration at `dt=0.20 fs` remains formally failed. IP2a-3 remains a separate numerical refinement diagnostic. IP2a-4 is the first complete pilot that qualifies under the original `2e-6 eV` accuracy requirement.

## GitHub Actions provenance

Workflow: `ip2a4-refined-complete-calibration`  
Run ID: `36182788956`  
Run attempt: `1`  
Workflow conclusion: **success**  
Validated commit: `6a87f68bd4d4fa027ca929797d672438248c3f5b`  
Branch: `isotropic-polaron-barrier`  
Python: 3.12.14  
NumPy: 2.5.3  
SciPy: 1.18.1  
Threads: OPENBLAS/OMP/MKL = 1  
Artifact: `ip2a4-refined-complete-calibration-36182788956-1`  
Artifact SHA256 reported by GitHub upload step: `ab788f1f74379d0621d8656862bea07fea0a27054c95489065bd0f96f2e5bfd5`

The workflow provenance step records the exact `GITHUB_SHA`, branch ref, run ID, run attempt, and a clean `git status --porcelain` output. The artifact-internal helper reported `git_status: unknown`; the GitHub job log is therefore the authoritative repository-cleanliness provenance for this hosted run.

## Validation gates

- pycompile: PASS
- focused pytest: PASS
- full pytest: **508 passed**
- recurrence preflight: PASS
- planned driven search: 4.0 ps
- stationary harmonic full-wrap diagnostic: ~21.693 ps
- all 24 pilot records complete and finite: PASS
- static initial relaxation converged: PASS
- locked pilot IDs preserved: PASS
- locked candidate energies preserved: PASS
- `dt=0.10 fs`: PASS
- event sampling 2 fs / energy sampling 10 fs: PASS
- no native/reversed post-event branches in calibration: PASS
- NPZ/JSON record cross-check: exact PASS for event times, directions, residuals, candidate energies and member IDs

## Pilot outcomes

Fixed member IDs:

`0, 4, 8, 12, 16, 20, 24, 28`

Candidate energies:

`1e-5, 3e-5, 1e-4 eV`

All three candidates satisfy:
- 8/8 numerically stable trials;
- 8/8 clean 200 fs field-free controls;
- 0/8 early driven events;
- 8/8 valid first persistent nearest-neighbor x events.

Every accepted first event is `820 -> 819 (-x)`. At `1e-5 eV`, all eight start at 2826 fs and are accepted at 2874 fs. The higher-energy candidates show only modest timing shifts in a few members.

### Numerical residuals

Field-on maximum pre-event energy–work residual ranges:

- `1e-5 eV`: `5.555935692e-7` to `5.563707236e-7 eV`
- `3e-5 eV`: `5.553070317e-7` to `5.566530834e-7 eV`
- `1e-4 eV`: `5.547440337e-7` to `5.572014576e-7 eV`

All are comfortably below the unchanged `2.0e-6 eV` preregistered limit.

Maximum field-free matter-energy drift across all 24 controls is `2.386979503e-15 eV`.

Electronic norm and projected zero-mode errors remain well within their `1e-10` bounds.

## Locked selection decision

The preregistered rule selects the **smallest** candidate that has all 8/8 trials stable and clean, no early event, and at least 6/8 qualifying first x events.

All three candidates qualify. Therefore:

**Freeze `1.0e-5 eV` as the production Peierls preparation energy for IP2b.**

Do not promote the larger preparation energies for IP2b based on any later native/reversed outcome.

## Interpretation

IP2a-4 establishes only that the deterministic ensemble can be generated and driven to a qualifying branch point with controlled pre-intervention numerics using `dt=0.10 fs`.

It does not establish:
- a physical hopping rate or probability;
- mobility;
- activation energy;
- population-level statistical inference;
- causal control of post-hop commitment by Peierls direction.

Those questions remain downstream.

## IP2b is now unlocked

IP2b may now use:
- 40x40 isotropic PBC;
- `dt=0.10 fs`;
- CF4-Lanczos Krylov dimension 6;
- fixed preparation energy `1.0e-5 eV`;
- full deterministic 32-member antithetic design;
- first qualifying persistent x branch point;
- native, direction-reversed, and native-replicate control branches;
- primary 2 ps matched population-L1 endpoint under the existing IP2 design rules.

Before running the production 32-member counterfactual ensemble, implement and validate the IP2b intervention/continuation runner and freeze its complete numerical and physical acceptance gates in a dedicated preregistration.
