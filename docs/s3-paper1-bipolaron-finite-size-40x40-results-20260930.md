# S3 Paper-1 bipolaron 40x40 finite-size results (2026-09-30)

## Execution outcome

**Execution PASS:** all 32/32 targeted branch tasks completed for 16/16 points;
there were no failed tasks. All required branches at every point met the locked
stationarity criteria (maximum coordinate update `<1e-8 A`, gradient
`<1e-6 eV/A`). Every selected minimum passed the linear-Peierls filter
`max |Delta t_mu|/|J_mu| <= 0.25`.

This is a targeted size check, not a global topology search: each point contains
the 20x20 observed bound-topology seed and a same-cell separated seed. No claim
is made for unmeasured seed basins or unselected parameter values.

## Provenance

- campaign: `s3-paper1-bipolaron-finite-size-40x40-v1`;
- frozen code commit: `35285bbc6b90d32be462321d7b0ee0a037e74f37`;
- run started: `2026-09-28T20:51:53Z`;
- Python 3.12.14, NumPy 2.5.3, SciPy 1.18.1;
- `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`;
- periodic isotropic square lattice, `Jx=Jy=0.0575 eV`;
- local run directory: `s3-local-runs/s3-paper1-bipolaron-finite-size-40x40-v1/20260928T183000Z`.

The run directory's timestamp label is inconsistent with its actual start time;
the provenance timestamp above is authoritative. This naming discrepancy does
not affect the frozen manifest, task IDs, or branch results.

## 20x20 comparison

All 16 targeted 40x40 observable classifications exactly match their 20x20
counterparts. Binding relative to the same-cell separated branch changes by at
most `0.2872 meV` in absolute value across these points. Thus the sampled
onsite, axial, diagonal, and separated classifications are stable under this
size check. This is not sufficient to promote unsampled boundaries or to claim
global finite-size convergence.

The reproducible join of the two campaign summaries marks all 16 paired points
as `stable_at_sampled_point` and finds 10 matching sampled topology-transition
intervals. See
`docs/s3-paper1-bipolaron-finite-size-comparison-20260930.md`; these intervals
remain brackets, not localized boundary values.

The resulting 40x40 classification counts are:

| Classification | Points |
| --- | ---: |
| separated | 6 |
| axial, robust | 3 |
| axial, marginal | 1 |
| diagonal, marginal | 5 |
| onsite, marginal | 1 |

There are no robust onsite or diagonal points in this targeted sample.

## Artifact integrity

- manifest as executed: SHA-256 `473decd30206619ee76ca61290e44f157530cc3b2bca8e531b79608f22bb0be5`;
- summary: SHA-256 `cee2c3ee9122dfadff7ec0c4470d779ed78d96335b741f467b24d5e071d4cf48`;
- points CSV: SHA-256 `7878ecde2de4ca8c790722e4faafd8f424298d650eca1264d88d46fd69711a1c`.

The canonical config manifest has SHA-256
`a2e338a3c470718cde39d4396d14c74ebe97bdc744a7fd5dbd4a118b23f2f1ba`; the
executed copy is semantically identical JSON serialized by the local runner.
