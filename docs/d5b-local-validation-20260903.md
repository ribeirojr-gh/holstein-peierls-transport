# D5b local validation — fixed-interval instantaneous decoherence

Local validation artifact: `d5b-local-validation/20260903T205408Z`.

Environment reported by the ZIP runner:

- Python 3.12.3
- NumPy 2.5.2
- SciPy 1.18.1
- Linux / WSL2
- `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`
- Git metadata unavailable because the source was downloaded as a GitHub ZIP.

## Numerical gates

- focused D5a+D5b tests: PASS
- full pytest suite: **279 passed**
- 20x20 D5b ensemble: PASS
- no failed runner gates
- scientific ensemble runtime: 967.24 s

The runner PASS means that the implementation, regression checks and benchmark execution succeeded. It does not by itself select a physical decoherence scheme.

## Control

- lattice: 20x20
- temperature: 300 K
- `gamma_u = gamma_v = 0.01 fs^-1`
- timestep: 0.2 fs
- duration: 10 ps
- burn-in: 2 ps
- zero-mode policy: `project`
- electronic propagator between IDC events: CF4-Lanczos, `m=6`
- decoherence interval: 100 fs (numerical control only)
- four independent lattice seeds for each IDC scheme
- independent RNG streams for lattice and electronic-collapse events

## Ensemble results

| Scheme | <T> [K] | pre-heating coordinate | late pre-heating | slope [ps^-1] | TV to canonical | ground population | canonical ground | expected post-heating | max generalized-energy residual [eV] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IDC-DP | 298.856 | 0.01456 | 0.02015 | 0.00450 | 0.25450 | 0.78574 | 0.83342 | 0.01456 | 1.400e-5 |
| IDC-BM | 298.816 | -0.00311 | 0.00045 | 0.00116 | 0.14866 | 0.89646 | 0.88754 | -0.00441 | 1.419e-5 |
| IDC-MA | 298.804 | 0.00645 | 0.00447 | 0.00283 | 0.20702 | 0.83494 | 0.85648 | 0.00366 | 1.433e-5 |

All three schemes retain the validated lattice temperature and electronic norm. The generalized energy accounting

`Delta E_matter ~= Q_lattice + Q_electronic_environment`

remains at the D4 integration-error scale.

At `t_d = 100 fs`, IDC-BM is the strongest candidate: it gives the smallest canonical-distance metric, essentially zero late heating coordinate, and the closest ground-manifold population to the instantaneous canonical reference. IDC-MA is second; IDC-DP removes phase coherence but is less effective at energy relaxation.

## Important interpretation of beta_eff

The ensemble mean of the snapshot-wise nonlinear quantity `beta_eff/beta_bath` is about 2 for all three IDC variants even when the mean energy-based heating coordinate is near zero. This is not a direct contradiction: averaging an inverse temperature inferred separately from a fluctuating instantaneous spectrum is nonlinear and can be dominated by cold excursions. It must therefore not be used alone to select the production scheme.

The next gate uses more robust quantities: mean/absolute/RMS heating coordinates, canonical total-variation distance, ground-manifold mismatch, early-to-late drift, collapse energy exchange and lattice temperature.

## Decision

D5b validates the IDC implementation but does **not** close D5. The 100 fs decoherence interval is a numerical control and cannot be promoted to a material parameter. D5c must measure sensitivity to the decoherence interval before selecting BM or MA for downstream field-driven transport.
