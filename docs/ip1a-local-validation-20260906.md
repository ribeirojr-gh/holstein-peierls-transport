# IP1a local validation — zero-field thermal hopping screen

Date: 2026-09-06

## Status

**Numerical validation: PASS / CLOSED.**

**Physical interpretation: promising mechanistic screen, but persistent dominant-site changes are not yet accepted as fully dressed polaron hops.**

The local validation artifact was produced from a ZIP checkout, so `git_commit`, `git_branch`, and `git_status` were reported as `unknown`. This is an accepted provenance limitation of the current local-ZIP workflow and does not affect the numerical diagnostics below.

## Environment and protocol

- Python 3.12.3, NumPy 2.5.2, SciPy 1.18.1, WSL2 x86_64.
- BLAS/OpenMP/MKL thread counts fixed to one.
- 20x20 periodic lattice.
- BAOAB + CF4-Lanczos (`m=6`) + IDC-BM, zero external field.
- `dt = 0.2 fs`, total time `20 ps`, burn-in `2 ps`.
- IDC interval `t_d = 180 fs` remains a numerical control, not a material calibration.
- Four lattice/decoherence seed pairs per condition.
- Isotropic scan: `J0y/J0x = 1.0`, T = 100, 200, 300, 400, 500 K.
- Anisotropic reference: `J0y/J0x = 0.15`, T = 300 K.
- Residence-site sampling every 2 fs; candidate-site persistence 20 fs.

## Regression and numerical closure

- IP1a focused tests: **36 passed**.
- Full suite: **372 passed**.
- 24/24 requested trajectories completed.
- Exact IDC event count satisfied for every trajectory.
- All static polaron relaxations converged.
- Lattice temperatures tracked their targets accurately.
- Maximum zero-field generalized-energy residual: approximately `2.65e-5 eV` over the complete temperature screen.
- Maximum electronic norm error: approximately `2.37e-13`.
- Maximum projected zero-mode magnitude: approximately `2.66e-17`.
- All eight pre-registered numerical gates passed.

## Observed persistent nearest-neighbour transitions

| J0y/J0x | T [K] | NN transitions | pooled rate [ps^-1] | trajectories with >=1 NN transition | IDC-associated persistent fraction |
|---:|---:|---:|---:|---:|---:|
| 1.00 | 100 | 92 | 1.27778 | 1.00 | 0.000 |
| 1.00 | 200 | 125 | 1.73611 | 1.00 | 0.049 |
| 1.00 | 300 | 137 | 1.90278 | 1.00 | 0.066 |
| 1.00 | 400 | 169 | 2.34722 | 1.00 | 0.079 |
| 1.00 | 500 | 187 | 2.59722 | 1.00 | 0.082 |
| 0.15 | 300 | 139 | 1.93056 | 1.00 | 0.054 |

The isotropic persistent-transition rate increases monotonically with temperature. A purely descriptive pooled Arrhenius fit to these five rates gives an apparent scale of about 7 meV, but **this is not accepted as an activation energy** because the event definition has not yet been validated against translation of the lattice-distortion cloud and the independent-seed uncertainty is large.

## Directional structure

The isotropic runs show both x and y nearest-neighbour transitions at every temperature, with no imposed field. The 300 K anisotropic reference is strongly easy-axis weighted: 112 accepted x transitions (`+x` + `-x`) versus 27 accepted y transitions (`+y` + `-y`). This is qualitatively consistent with the static barrier anisotropy and is an encouraging mechanistic regression.

Only a small minority of persistent transitions start exactly at an IDC sampling time. The association fraction stays below about 8.3% across the isotropic scan, so the observed site changes are not simply identical to instantaneous IDC dominant-site changes. This does **not** prove that the IDC model is dynamically irrelevant.

## Transient local transfer anisotropy

The mean local transfer-anisotropy diagnostic increases with temperature in the isotropic system:

- 100 K: baseline 0.4026, pre-hop 0.5153;
- 200 K: baseline 0.5553, pre-hop 0.5983;
- 300 K: baseline 0.6348, pre-hop 0.6372;
- 400 K: baseline 0.6470, pre-hop 0.6647;
- 500 K: baseline 0.6773, pre-hop 0.6984.

The prospective hop direction has a positive mean transfer bias at all isotropic temperatures (roughly 0.27–0.30). This is compatible with thermally generated local anisotropy selecting a direction, but the current screen lacks a matched null/rank test, so it is not yet causal evidence.

## Why IP1a does not yet establish physical hopping

Several diagnostics require a stricter second-stage validation:

1. The persistent tracker follows the **dominant molecular population**, not the center of the full lattice deformation.
2. At high temperature the number of persistent non-nearest-neighbour dominant-site transitions becomes large (11, 37, 74, 70, 154 for the isotropic 100–500 K scan), indicating that a broad/delocalized electronic cloud can move its maximum without necessarily translating as a dressed quasiparticle.
3. Typical accepted residence intervals can be hundreds of femtoseconds, considerably shorter than the ~6.8 ps collective-period diagnostic found for the isotropic frozen translation coordinate in IP0b. The two quantities are not identical, but the mismatch is large enough that it must be tested explicitly rather than explained away.
4. The 100 K seed-to-seed NN rate varies strongly (about 0.22–2.17 ps^-1), so four 20 ps trajectories are a screen, not a converged kinetic measurement.

Accordingly, IP1a closes as a **successful numerical and mechanistic screen**, not as evidence for a final isotropic hopping rate, diffusion coefficient, mobility, or activation energy.

## Decision for IP1b

IP1b will test whether the electronic residence-site transitions correspond to translation of the **dressed polaron**, using a periodic matched filter built from the relaxed lattice-distortion template. The template will use the physically relevant fields `u`, `Delta_x vx`, and `Delta_y vy`, with elastic-energy weighting. The next screen will:

- track electronic and lattice-distortion centers independently;
- test persistence sensitivity (20, 50, and 100 fs) on the same trajectories;
- measure electronic/lattice transition lags instead of assuming simultaneous motion;
- classify electronic-only versus lattice-followed (dressed) transitions;
- quantify whether the future hop bond is preferentially the strongest local bond before the event, with a matched directional/rank diagnostic;
- retain zero-field energy, norm, temperature, zero-mode, and IDC-association checks.

No nonzero dressed-hop count is a PASS requirement. If the IP1a events collapse under the dressed-polaron criterion, that is a valid physical result and will identify dominant-site flicker rather than transport.
