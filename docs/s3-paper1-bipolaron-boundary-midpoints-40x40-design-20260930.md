# S3 targeted 40x40 transition-midpoint campaign (2026-09-30)

## Purpose and frozen scope

The matched 20x20/40x40 comparison found ten sampled intervals containing the
same topology changes at both sizes. The intervals are too wide to quote as
well-localized transition locations. This campaign evaluates the arithmetic
midpoint of each interval at 40x40: ten points and 28 branch relaxations.

At an axial-to-diagonal midpoint, both axial orientations, the diagonal seed,
and the separated seed are retained (four branches). At a diagonal-to-separated
midpoint, diagonal and separated branches are retained. The onsite-to-separated
midpoint retains onsite and separated branches. This is a targeted competing-root
check, not a global five-seed search.

| g | U (eV) | V1 midpoint (eV) | Competing topology interval |
| ---: | ---: | ---: | --- |
| 0.9 | 0.75 | 0.004 | axial / diagonal |
| 0.9 | 0.75 | 0.024 | diagonal / separated |
| 1.0 | 0.525 | 0.280 | onsite / separated |
| 1.0 | 0.75 | 0.010 | axial / diagonal |
| 1.0 | 0.75 | 0.028 | diagonal / separated |
| 1.0 | 1.0 | 0.002 | axial / diagonal |
| 1.0 | 1.0 | 0.010 | diagonal / separated |
| 1.1 | 0.75 | 0.028 | diagonal / separated |
| 1.1 | 1.0 | 0.004 | axial / diagonal |
| 1.1 | 1.0 | 0.012 | diagonal / separated |

The three midpoint coordinates that already occur in the 20x20 production map
can be directly size-paired. The other seven are new 40x40 samples and have no
same-coordinate 20x20 result; they refine the 40x40 map but do not by themselves
establish a matched finite-size status. Any quantitative boundary claim remains
subject to strict convergence and the linear-Peierls gate at the selected
minimum.

Frozen manifest: `configs/s3-paper1-bipolaron-boundary-midpoints-40x40-v1.json`.
