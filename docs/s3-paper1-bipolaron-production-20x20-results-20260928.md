# S3 Paper-1 bipolaron 20x20 production map (2026-09-28)

## Execution classification

**Execution PASS:** all 240/240 branch tasks completed and all 48/48 parameter
points contain the full five-seed set.  Every branch passed the locked
structural convergence gate (`max update < 1e-8 A`, gradient
`< 1e-6 eV/A`).

**Linear-Peierls quantitative gate:** 42/48 points PASS; six are excluded
from quantitative interpretation because a promoted minimum exceeds
`max |Delta t_mu|/|J_mu| = 0.25`.  No 40x40 boundary checks have yet been run,
so all finite-size promotion statuses remain pending.

## Provenance

- campaign: `s3-paper1-bipolaron-production-20x20-v1`;
- frozen manifest and solver commit: `a4bb8333d9bb791033dc97f24c2751a5999c0086`;
- Python 3.12.14, NumPy 2.5.3, SciPy 1.18.1;
- `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`;
- periodic isotropic square lattice, `Jx=Jy=0.0575 eV`;
- 48 points x 5 independently initialized branches.

## Aggregate topology counts

| 20x20 classification | points |
| --- | ---: |
| separated | 12 |
| onsite, robust | 7 |
| axial, robust | 9 |
| axial, marginal (<5 meV) | 8 |
| diagonal, marginal (<5 meV) | 11 |
| onsite, marginal (<5 meV) | 1 |

Every point classified as marginal has a same-cell separated-branch energy
difference below 5 meV.  The diagonal minima are especially shallow and
require 40x40 confirmation before any stability claim.

## Linear-Peierls exclusions

Six fully converged points fail only the declared Peierls modulation filter:

- `g=0.9`, `U=0.525 eV`, `V1={0, 0.004, 0.008, 0.016} eV`;
  maximum directional ratios range from 0.2522 to 0.2564;
- `g=1.1`, `U=0.750 eV`, `V1={0, 0.004} eV`;
  maximum ratios are 0.2870 and 0.2885.

These results remain in the raw archive and must not be used for quantitative
phase boundaries under the current linear-coupling criterion.

## Screen-to-20x20 topology summary

The 20x20 map resolves several trends from the 12x12 screen:

- At `g=1.0, U=0.525 eV`, onsite pairing remains robust through `V1=0.16 eV`,
  is marginal at `0.24 eV`, and is separated at `0.32 eV`.
- At `g=1.0, U=0.75 eV`, the axial minimum is robust at `V1=0` and `0.004 eV`,
  marginal axial at `0.008 eV`, marginal diagonal at `0.016 eV`, then separated
  at `0.04 eV` and above.
- At `g=1.0, U=1.0 eV`, axial and diagonal minima remain marginal through
  `V1=0.008 eV`, with separation by `0.016 eV`.
- At `g=1.1, U=0.525 eV`, onsite pairing remains robust throughout the tested
  range `V1=0.08--0.32 eV`; its dissociation boundary is outside this map.
- At `g=1.1, U=1.0 eV`, the sequence is robust axial at zero V1, marginal
  axial at 2--4 meV, marginal diagonal at 8 meV, and separated at 16 meV.
- The low-coupling `g=0.8` control remains separated.

At `g=0.9, U=0.525 eV`, the 20x20 axial states exceed the linear-Peierls limit
even though the 12x12 screen had placed them just inside it.  This is a
finite-size sensitivity of the validity filter and excludes that row from
quantitative interpretation.  At `g=0.9, U=0.75 eV`, axial-to-diagonal and
diagonal-to-separated changes persist, but all bound minima are marginal.

## 40x40 refinement targets

The next finite-size stage should focus on:

1. `g=1.0, U=0.525 eV`, the onsite-to-separated bracket `V1=0.24--0.32 eV`;
2. `g=1.0, U=0.75 eV`, axial/diagonal/separated points spanning
   `V1=0.004--0.04 eV`;
3. `g=1.0, U=1.0 eV`, the shallow axial/diagonal/separated window
   `V1=0--0.016 eV`;
4. `g=1.1, U=1.0 eV`, the axial/diagonal/separated window
   `V1=0--0.016 eV`;
5. `g=0.9, U=0.75 eV`, as a marginal-binding size control;
6. `g=1.1, U=0.75 eV`, only for the in-window diagonal-to-separated region;
   the two low-V1 axial points remain outside the linear-Peierls gate.

No boundary is promoted until its competing observable topologies, binding
sign, strict convergence, and linear-Peierls gate are checked at 40x40.

## Artifact integrity

- `manifest.json` SHA-256:
  `e1b18ac8d5ca6c8c4168f5a7c5a04687f3e558fdbc0ef55e2e45ae1f2d420c01`;
- `summary.json` SHA-256:
  `e49b712c1ae0c38d9a56ab8fbf835016652c82e1f198af1417abd0e6ec4f02a8`;
- `points.csv` SHA-256:
  `c555f58f4ac5284aa1d2fd8e30cbed5aadd4a58510310d3cb8a91c5b8af5ad1a`.
