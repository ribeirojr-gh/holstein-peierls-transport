# S3 Paper-1 bipolaron 12x12 screen results (2026-09-27)

## Classification

**Screen execution PASS.**  All 300 independently relaxed branches completed,
all 60 parameter points contain the five frozen seeds, and every branch met
the strict structural convergence criteria.

**Quantitative-at-size filter: 58/60 points PASS.**  Two axial points exceeded
the conservative linear-Peierls limit and are retained as qualitative
topology observations only.  The entire 12x12 screen remains nonproduction:
finite-size status is pending and no Paper-1 boundary is promoted here.

## Provenance

- campaign: `s3-paper1-bipolaron-screen-v1`;
- commit: `9701de9`;
- Python 3.12.14, NumPy 2.5.3, SciPy 1.18.1;
- OpenBLAS/OMP/MKL thread counts fixed to 1;
- 553 repository tests passed before execution;
- periodic isotropic 12x12 lattice;
- 4 coupling strengths x 3 U values x 5 V1 values x 5 seeds.

## Aggregate topology counts

| classification | points |
| --- | ---: |
| separated | 33 |
| onsite, robust | 9 |
| axial, robust | 8 |
| axial, marginal (<5 meV) | 5 |
| diagonal, marginal (<5 meV) | 5 |

The two linear-Peierls failures are:

- `g=1.1, U=0.75 eV, V1=0`: axial, maximum modulation ratio `0.26806`;
- `g=1.1, U=0.75 eV, V1=0.004 eV`: axial, maximum modulation ratio `0.26941`.

Both remain fully converged.  They fail only the declared quantitative
linear-coupling filter.

## Observable topology map

Within each row, entries follow
`V1 = {0, 0.004, 0.016, 0.080, 0.320} eV`.

| g | U (eV) | classifications in increasing V1 |
| ---: | ---: | --- |
| 0.8 | 0.525 | separated, separated, separated, separated, separated |
| 0.8 | 0.750 | separated, separated, separated, separated, separated |
| 0.8 | 1.000 | separated, separated, separated, separated, separated |
| 0.9 | 0.525 | axial, axial, marginal axial, separated, separated |
| 0.9 | 0.750 | axial, marginal axial, marginal diagonal, separated, separated |
| 0.9 | 1.000 | marginal axial, marginal diagonal, separated, separated, separated |
| 1.0 | 0.525 | onsite, onsite, onsite, onsite, separated |
| 1.0 | 0.750 | axial, axial, marginal diagonal, separated, separated |
| 1.0 | 1.000 | marginal axial, marginal diagonal, separated, separated, separated |
| 1.1 | 0.525 | onsite, onsite, onsite, onsite, onsite |
| 1.1 | 0.750 | axial*, axial*, marginal diagonal, separated, separated |
| 1.1 | 1.000 | axial, marginal axial, separated, separated, separated |

`*` outside the linear-Peierls quantitative window.

## Scientific reading at screen level

1. `g=0.8` is consistently a separated weak-coupling control across the
   screened U/V plane.
2. Increasing coupling produces onsite binding at low U and axial binding at
   larger U.
3. Small positive V1 converts axial states into diagonal states before
   separation over several U/g rows, consistent with the previously validated
   isotropic topology.
4. The diagonal points are weakly bound on 12x12 and require finite-size
   refinement before any stability claim.
5. The strong-coupling low-U onsite region persists to `V1=0.32 eV` at
   `g=1.1`, so its dissociation boundary lies outside this screen.
6. The `g=1.1, U=0.75` axial results demonstrate why the Peierls filter must be
   applied branch by branch rather than inferred from input coupling alone.

## Production-selection decision

The 20x20 stage should retain `g={0.9,1.0,1.1}` and all three U values, while
focusing V1 resolution on observed topology changes.  The `g=0.8` plane need
not be repeated exhaustively; one finite-size separated control is sufficient.
Every diagonal candidate and every topology/binding bracket must be retained
for 20x20 refinement.  Only surviving brackets then advance to 40x40.

## Artifact integrity

- frozen `manifest.json` SHA-256:
  `2d82bec53172cb2a321f8709c2fda9c878bf1457553ebbfafbdbdd9232a70343`;
- `summary.json` SHA-256:
  `0e8bbb45a4e8bb71558f3bd6d74cd2f5826ae11dc3e82226c631892b2de8eaf6`;
- `points.csv` SHA-256:
  `83c2ecf7c6b41e73c34f9e115189f5929e22c2c0000bea489215ab37251f577d`;
- complete ZIP SHA-256:
  `4ca48963b829c56d4123853fb5e9fd64873b65affc432320c64fb65ccfe21a91`.
