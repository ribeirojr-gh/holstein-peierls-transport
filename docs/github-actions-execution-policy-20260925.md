# GitHub Actions execution policy

Date: 2026-09-25

## Status

GitHub Actions is again the primary execution environment for numerical validation and production simulations in this project.

Local `run.sh` scripts remain mandatory as reproducibility/fallback entry points, but routine validation no longer requires the user to execute them locally and upload ZIP artifacts.

## Rules

1. Every scientific stage keeps a preregistered protocol in `docs/` before its production workflow is executed.
2. Stage-specific GitHub Actions workflows must pin the branch/commit provenance, Python version, thread environment and all command-line parameters.
3. Numerical simulation outputs, JSON/NPZ summaries, logs and validation reports are uploaded as workflow artifacts.
4. Heavy scientific workflows should use `workflow_dispatch` and/or narrowly scoped push triggers so ordinary documentation commits do not consume compute unnecessarily.
5. Production workflows must distinguish:
   - execution/numerical integrity;
   - scientific/physical acceptance criteria.
6. Failed physical gates are preserved as results and are not converted into PASS by changing thresholds post hoc.
7. GitHub-hosted execution does not remove the requirement for timestep, size, Krylov, recurrence and conservation audits appropriate to each physical sector.
8. The root `run.sh` remains self-contained enough to reproduce the same stage locally when needed.
9. No merge to `main` or publication release is implied by a successful Action run; those are separate explicit decisions.
10. Uploaded artifacts should normally be retained for 90 days during active validation. Publication-relevant final artifacts must be archived in the project release/data structure before expiry.

## Current stage

The first resumed GitHub-hosted simulation is IP2a-4, the full refined-dt pre-intervention calibration at `dt=0.10 fs`.
