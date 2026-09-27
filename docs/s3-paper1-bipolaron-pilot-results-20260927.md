# S3 Paper-1 bipolaron pilot results (2026-09-27)

## Classification

**Pilot workflow PASS.**  All 60 independently relaxed branches completed,
all 12 parameter points contain the five preregistered seeds, every branch
satisfied the strict structural convergence gates, and every promoted state
remained inside the conservative linear-Peierls window.

These 8x8 results validate the manifest/checkpoint/aggregation workflow.  They
are not Paper-1 phase-boundary data; finite-size status remains pending for
every point.

## Provenance

- campaign: `s3-paper1-bipolaron-pilot-v1`;
- aggregate commit: `940bf5a16fc484210c2e85b36a45cf5d0c4a4adf`;
- branch solver introduced at commit: `f51fe2b`;
- Python 3.12.14;
- NumPy 2.5.3;
- SciPy 1.18.1;
- OpenBLAS/OMP/MKL thread counts fixed to 1;
- full repository suite before the final aggregate: 552 passed.

## Pilot grid

- isotropic periodic 8x8 lattice;
- `Jx = Jy = 0.0575 eV`;
- `U = {0.525, 1.000} eV`;
- `V1 = {0.000, 0.016, 0.320} eV`;
- global coupling multiplier `g = {0.8, 1.0}`;
- onsite, intersite-x, intersite-y, diagonal, and separated seeds at every
  point.

## Observable-classified minima

| g | U (eV) | V1 (eV) | classification | binding vs separated seed (meV) | status |
| ---: | ---: | ---: | --- | ---: | --- |
| 0.8 | 0.525 | 0.000 | separated | 27.917 | separated topology; binding not assigned |
| 0.8 | 0.525 | 0.016 | separated | 27.366 | separated topology; binding not assigned |
| 0.8 | 0.525 | 0.320 | separated | 22.899 | separated topology; binding not assigned |
| 0.8 | 1.000 | 0.000 | separated | 27.315 | separated topology; binding not assigned |
| 0.8 | 1.000 | 0.016 | separated | 26.833 | separated topology; binding not assigned |
| 0.8 | 1.000 | 0.320 | separated | 22.786 | separated topology; binding not assigned |
| 1.0 | 0.525 | 0.000 | onsite | 21.839 | robust at pilot size |
| 1.0 | 0.525 | 0.016 | onsite | 19.998 | robust at pilot size |
| 1.0 | 0.525 | 0.320 | separated | 0.000 | separated |
| 1.0 | 1.000 | 0.000 | marginal axial | 3.255 | below 5 meV robust-binding gate |
| 1.0 | 1.000 | 0.016 | separated | 0.000 | separated |
| 1.0 | 1.000 | 0.320 | separated | 0.000 | separated |

The six `g=0.8` minima have low local pair probability and mean separations
between 3.30 and 3.56 sites.  Their energy differences from the particular
separated-seed stationary branch are therefore retained as small-cell
root/seed diagnostics, not interpreted as pair binding.

For the isotropic Hamiltonian, x- and y-oriented nearest-neighbour solutions
are symmetry-equivalent and are reported as one axial phase.  Seed names
remain stored as provenance only.

## Pilot conclusions

1. The resumable local execution path and deterministic task matrix work.
2. Observable classification prevents seed labels from becoming phase labels.
3. `g=0.8` is a useful weak-coupling separated control.
4. At `g=1.0`, the grid samples onsite, axial-marginal, and separated regimes.
5. The coarse `V1` values intentionally do not resolve the known narrow
   axial/diagonal window near `U=1 eV`; the next screen must add low-meV `V1`
   points.
6. No boundary or critical value may be promoted until 20x20 refinement and
   40x40 boundary checks pass.

## Artifact integrity

- frozen run `manifest.json` SHA-256:
  `1b3c4f6f4c7602511eb3952c4c03b80913ba3880d07246fc7626b8a9a68b6b5c`;
- `summary.json` SHA-256:
  `97d2edac1a5397439b0ada37db258c2561942c2d0d1ab4c9437b7f2172f56cdb`;
- `points.csv` SHA-256:
  `bc1abb31ad612a69e0b2cb74cafbfd22926b9c84da3690676a46cf50430e422a`;
- complete ZIP SHA-256:
  `6a43fef0d429c638ff8180737d4912c468b45fe0dc0310a0cec6afc7c6239175`.
