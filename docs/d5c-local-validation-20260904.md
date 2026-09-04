# D5c local validation and scheme selection — 2026-09-04

## Provenance

The D5c sensitivity sweep was executed locally from a GitHub ZIP snapshot of branch `d5-electronic-thermalization-decoherence`. Because Git metadata are not present in the ZIP, the runner records `git_commit`, `git_branch`, and `git_status` as `unknown`. The branch head that produced the downloadable snapshot before this run was `3af62dfe38d9dcd3ee0ffaa3b119bf17a02f46b0`.

Environment recorded by the runner:

- Python 3.12.3
- NumPy 2.5.2
- SciPy 1.18.1
- Linux / WSL2 x86_64
- `OPENBLAS_NUM_THREADS=1`
- `OMP_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`

## Numerical gates

All requested gates passed:

- D5c focused tests: 16 passed
- full pytest suite: 281 passed
- 20x20 decoherence-interval sweep: completed successfully
- 40 finite-temperature trajectories: BM and MA, 5 decoherence intervals, 4 lattice seeds

The interval sweep used 6 ps trajectories, 2 ps burn-in, 300 K, `dt=0.2 fs`, projected intermolecular zero modes, and CF4-Lanczos with Krylov dimension 6.

## Sensitivity results

| scheme | td [fs] | <T> [K] | mean heating | late heating | slope [ps^-1] | TV canonical | ground mismatch | expected post heating | Qe rate [eV/ps] | max balance residual [eV] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BM | 50 | 297.820 | -0.0052 | -0.0086 | -0.0008 | 0.1279 | 0.0329 | -0.0058 | -2.3719e-3 | 1.359e-5 |
| BM | 100 | 298.624 | -0.0071 | -0.0113 | -0.0018 | 0.1222 | 0.0370 | -0.0076 | -2.4411e-3 | 1.388e-5 |
| BM | 180 | 297.238 | -0.0063 | -0.0103 | -0.0020 | 0.1214 | 0.0246 | -0.0075 | -3.7700e-3 | 1.381e-5 |
| BM | 250 | 299.718 | -0.0063 | -0.0107 | -0.0009 | 0.1236 | 0.0645 | -0.0083 | -4.1478e-3 | 1.387e-5 |
| BM | 500 | 300.585 | -0.0000 | 0.0010 | 0.0010 | 0.0979 | 0.0669 | -0.0055 | -5.7552e-3 | 1.338e-5 |
| MA | 50 | 297.817 | -0.0072 | -0.0095 | -0.0012 | 0.1219 | 0.0425 | -0.0075 | -2.7297e-3 | 1.357e-5 |
| MA | 100 | 298.624 | -0.0071 | -0.0113 | -0.0018 | 0.1222 | 0.0370 | -0.0075 | -2.4411e-3 | 1.388e-5 |
| MA | 180 | 297.224 | -0.0042 | -0.0082 | -0.0023 | 0.1503 | 0.0199 | -0.0060 | -2.5961e-3 | 1.382e-5 |
| MA | 250 | 299.718 | -0.0063 | -0.0107 | -0.0009 | 0.1236 | 0.0645 | -0.0083 | -4.1478e-3 | 1.387e-5 |
| MA | 500 | 300.585 | -0.0000 | 0.0010 | 0.0010 | 0.0979 | 0.0669 | -0.0053 | -5.7552e-3 | 1.338e-5 |

## Interpretation

Both energy-relaxing IDC schemes suppress the systematic electronic overheating seen in coherent Ehrenfest D5a over the full tested interval range. The classical lattice remains compatible with the validated 300 K D4 bath, the generalized energy residual stays near `1.4e-5 eV`, and the electronic norm remains at roundoff scale.

BM is selected as the reference D5 production candidate because it is at least as close to the canonical reference as MA across most of the sweep and is clearly better in canonical TV distance at `td=180 fs`. MA remains a supported comparison model; the data do not justify claiming that BM is universally superior.

Several BM/MA trajectories are numerically identical for the tested seeds and intervals even though the collapse probability laws differ. This occurs when the stochastic draw selects the same adiabatic state under both probability vectors. The expected post-collapse heating coordinate still reveals the underlying probability-law difference. Therefore this finite ensemble must not be interpreted as evidence that BM and MA are mathematically equivalent.

No material-specific decoherence time has been determined. `td` remains an explicit phenomenological model parameter.

## D5d closure choice

For the final 10 ps closure gate, use IDC-BM with `td=180 fs` as a **numerical reference control**, not a calibrated material parameter. This is a mid-range interval with near-zero heating, no positive late-time drift, low canonical TV distance, and the smallest BM ground-manifold mismatch among the 50–250 fs controls. The already validated `td=100 fs` point remains an additional reference.
