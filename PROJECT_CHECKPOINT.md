# Holstein–Peierls project checkpoint

Updated: 2026-09-30  
Branch: `s3-paper1-parameter-campaign`  
Comparison implementation/report commit: `e235dcd`

## Current state

Paper 1 static bipolaron S3 has completed its 20x20 production map and targeted
40x40 finite-size checks. A reproducible cross-size comparison is now available
from `scripts/compare_s3_finite_size.py`.

- 16 shared parameter points: all 16 have matching observable topology at both
  sizes, converged required branches, and pass the linear-Peierls gate;
- 10 adjacent topology-transition brackets in the targeted subset: all 10 show
  the same endpoint topology sequence at 20x20 and 40x40;
- these are finite sampled intervals, not interpolated or exact boundary
  locations;
- 32 other 20x20 production points were deliberately not selected for the
  targeted 40x40 campaign; this is not a failed or incomplete campaign;
- 40x40 checks used the observed topology branch plus same-cell separated
  branch, not the full five-seed ensemble.

## Current decision and next action

The formal 20x20/40x40 comparison and explicit sampled finite-size statuses are
complete. Because ten intervals remain too coarse for well-localized Paper-1
transition values, the next stage is a frozen 40x40 midpoint campaign: 10 points
and 28 branches. It is a targeted competing-root check, not a global five-seed
search. After those results, assess whether seven new midpoint coordinates need
20x20 counterparts before promoting size-dependent claims, then proceed to the
Paper-1 S4 data freeze.

## Canonical artifacts

- comparison implementation: `scripts/compare_s3_finite_size.py`;
- comparison report: `docs/s3-paper1-bipolaron-finite-size-comparison-20260930.md`;
- next frozen manifest:
  `configs/s3-paper1-bipolaron-boundary-midpoints-40x40-v1.json`;
- midpoint campaign design:
  `docs/s3-paper1-bipolaron-boundary-midpoints-40x40-design-20260930.md`;
- comparison JSON/CSV outputs: under the 40x40 run directory's `comparison/`;
- 40x40 run directory:
  `s3-local-runs/s3-paper1-bipolaron-finite-size-40x40-v1/20260928T183000Z`;
- 20x20 run directory:
  `s3-local-runs/s3-paper1-bipolaron-production-20x20-v1/20260928T150000Z`;
- previous S3 result report:
  `docs/s3-paper1-bipolaron-finite-size-40x40-results-20260930.md`;
- execution habit: `docs/project-checkpoint-practice.md`.

The 40x40 directory timestamp label is inconsistent with its provenance start
time (`2026-09-28T20:51:53Z`); the provenance record is authoritative.

## Google Drive

Campaign folder:
https://drive.google.com/drive/folders/1sLQuIJGFDZoqc9rx9CSTtz6tOEXYwp0y

- frozen 40x40 manifest: `1CX2D-wSDc-TDcZW7OD4I9nbCfU8P7l5I`;
- 40x40 final raw archive: `1OT04-4gbzPcf0iibnbXKLQvXj8iq3CCf`;
- 40x40 result report: `1dvoAJ4QGVOHib6Sk0RJPAzC8hQcMSiQn`;
- 20x20/40x40 comparison report: `1x1_CLThljFLwYvp4aJW7SaTZURD59wYC`;
- reproducible 20x20+40x40 comparison archive: `1qqGo1beNtPzux_FNKdOcynWUpEzT3JAv`.

The target folder was listed after upload and all artifacts above were visible.
