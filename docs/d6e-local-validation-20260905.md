# D6e local IDC screening validation — 2026-09-05

## Scope

D6e screened instantaneous-decoherence controls for the finite-temperature pair sectors at one common **numerical-control** interval, `t_d = 100 fs`. The control was 4x4, 300 K, `gamma_u = gamma_v = 0.01 fs^-1`, `dt = 0.2 fs`, 4 ps total time with 1 ps burn-in, four stochastic lattice seeds per sector/scheme, projected intermolecular zero modes, and CF4-Lanczos with Krylov dimension 8.

The bipolaron collapse basis was restricted to the symmetric spatial singlet sector. The distinguishable electron-hole exciton used the full ordered pair space. No field, mobility extraction, or material-specific decoherence-time claim was made.

## Numerical closure

Local validation passed completely:

- 37 focused D6 tests passed;
- full test suite: **318 passed**;
- all 24 numerical D6e gates passed;
- maximum generalized balance residual remained below `1.54e-6 eV`;
- maximum norm error was `2.22e-16`;
- maximum pair-sector constraint error was `2.22e-15`;
- the lattice remained near 300 K for every sector/scheme ensemble.

Because the validation was run from a downloaded ZIP, git commit/branch metadata inside the artifact are `unknown`; provenance is recorded manually by this document and repository history.

## Physical screening results

| sector | scheme | mean pre-heating | late heating | slope [ps^-1] | TV canonical | ground mismatch | electronic exchange rate [eV/ps] |
|---|---|---:|---:|---:|---:|---:|---:|
| bipolaron | DP | 4.98e-6 | 7.17e-8 | -2.66e-6 | 3.56e-5 | 3.48e-5 | +3.91e-3 |
| bipolaron | BM | 4.73e-6 | -3.18e-7 | -1.13e-6 | 3.56e-5 | 3.49e-5 | -1.94e-2 |
| bipolaron | MA | 1.66e-2 | 1.05e-3 | -1.17e-2 | 1.60e-1 | 1.60e-1 | -2.80e-2 |
| exciton | DP | 1.83e-6 | 1.84e-6 | +5.51e-7 | 4.75e-6 | 4.71e-6 | -4.68e-3 |
| exciton | BM | 1.39e-6 | 1.68e-6 | +3.76e-7 | 3.06e-6 | 3.06e-6 | -1.15e-2 |
| exciton | MA | 1.39e-6 | 1.68e-6 | +3.76e-7 | 3.06e-6 | 3.06e-6 | -1.15e-2 |

The coherent D6d controls showed substantial progressive heating, whereas D6e at 100 fs suppresses it almost completely for DP and BM in both sectors.

### Bipolaron

DP and BM are essentially indistinguishable in the canonical-distance diagnostics at 100 fs. DP achieves this without an explicit Boltzmann reweighting and with a much smaller net electronic-environment energy sink. MA is not competitive at this control because one seed developed a large excited-state excursion (`mean TV canonical ~= 0.638`, `ground mismatch ~= 0.638`), producing a large ensemble variance.

### Exciton

All three schemes are close to the instantaneous canonical reference at 100 fs. BM and MA produced identical trajectories for this control because the same random draws selected the same adiabatic states. DP is only slightly farther from canonical and again requires no thermal reweighting.

## D6e decision

D6e does **not** select a production decoherence model or a material-specific `t_d`.

For the next sensitivity stage, retain **DP and BM** as the primary candidates and deprioritize MA. DP is especially important because it is the least thermodynamically prescriptive extension: at 100 fs, phase destruction alone is already sufficient to arrest the pair-sector Ehrenfest overheating. BM remains the detailed-balance candidate and must be compared across a nontrivial interval range.

D6f will therefore sweep `t_d` for DP and BM separately in the bipolaron and exciton sectors. The preferred model, if any, must remain stable across a range of intervals rather than only at the 100 fs screening point.
