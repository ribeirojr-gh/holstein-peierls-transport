# S3 20x20-to-40x40 comparison (2026-09-30)

## Outcome

The reproducible comparison joins the frozen 20x20 production summary and the
targeted 40x40 summary by `(coupling_scale, U, V1)`. The comparison script
reports each shared point, each sampled transition interval, branch coverage,
strict convergence, the linear-Peierls gate, and the binding-energy change.

- 16 points are present at both sizes; all 16 are `stable_at_sampled_point`;
- no paired point changes observable topology;
- all 10 adjacent topology-transition brackets represented in the targeted
  40x40 slice set have the same endpoint topology sequence at both sizes;
- all bracket endpoints meet convergence and linear-Peierls gates;
- 32 other 20x20 points were intentionally not selected for this targeted
  40x40 campaign.

`stable_at_sampled_point` is deliberately narrower than a claim of global
finite-size convergence. `same_sampled_transition_bracket` means that the same
topology change lies between the same sampled endpoints at both sizes. It does
not locate the transition within that interval or establish an entire phase
boundary.

## Matched transition intervals

| g | U (eV) | V1 interval (eV) | Sampled topology change |
| ---: | ---: | ---: | --- |
| 0.9 | 0.75 | 0.000–0.008 | axial → diagonal |
| 0.9 | 0.75 | 0.008–0.040 | diagonal → separated |
| 1.0 | 0.525 | 0.240–0.320 | onsite → separated |
| 1.0 | 0.75 | 0.004–0.016 | axial → diagonal |
| 1.0 | 0.75 | 0.016–0.040 | diagonal → separated |
| 1.0 | 1.0 | 0.000–0.004 | axial → diagonal |
| 1.0 | 1.0 | 0.004–0.016 | diagonal → separated |
| 1.1 | 0.75 | 0.016–0.040 | diagonal → separated |
| 1.1 | 1.0 | 0.000–0.008 | axial → diagonal |
| 1.1 | 1.0 | 0.008–0.016 | diagonal → separated |

At 20x20 each point retains the complete five-seed ensemble. At 40x40 the
campaign intentionally repeated only the 20x20 observed topology seed and the
same-cell separated seed. Thus the comparison validates the selected competing
branches at these samples but is not an independent five-seed root search at
40x40.

## Reproduction and data

Run `scripts/compare_s3_finite_size.py` with the production and finite-size
`summary.json` files. It writes `finite_size_comparison.json`,
`finite_size_points.csv`, and `transition_brackets.csv` under the supplied
output directory. The generated files and their hashes are kept with the raw
run archive in Google Drive; generated run data are not committed to Git.
