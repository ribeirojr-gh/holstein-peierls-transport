# D6 final pair-dynamics closure — local validation 2026-09-05

## Status

D6 is **CLOSED** for the validated zero-field finite-temperature pair-dynamics scope.

The final D6g local gate completed successfully for the selected pair electronic-decoherence controls:

- singlet bipolaron: IDC-BM;
- distinguishable electron-hole exciton: IDC-DP;
- decoherence interval `t_d = 100 fs` used strictly as a numerical reference control, not as a calibrated material parameter.

The control is a 4x4 lattice at 300 K, `gamma_u = gamma_v = 0.01 fs^-1`, `dt = 0.2 fs`, 10 ps total duration, 2 ps burn-in, four stochastic seeds per sector, projected intermolecular zero modes, and CF4-Lanczos with Krylov dimension 8.

## Local provenance

Archive timestamp: `20260905T193355Z`.

Environment:

- Python 3.12.3;
- NumPy 2.5.2;
- SciPy 1.18.1;
- WSL2 x86_64;
- `OPENBLAS_NUM_THREADS=1`;
- `OMP_NUM_THREADS=1`;
- `MKL_NUM_THREADS=1`.

Because the user executed from a GitHub ZIP tree without `.git`, the local archive reports `git_commit`, `git_branch`, and `git_status` as `unknown`. This is the known ZIP-provenance limitation. The validation package was generated from the D6 branch immediately after the pre-registered D6g protocol commit.

## Regression gates

All requested gates passed:

- D6 focused pytest: 39/39 passed;
- full pytest: 320 passed;
- D0a frozen-propagator smoke benchmark: passed;
- S0 singlet onsite/bond_x/bond_y strict relaxations: passed;
- S0 triplet onsite/bond_x/bond_y strict relaxations: passed;
- D6g pair final-closure benchmark: passed;
- failed gates: none.

## Final D6g physical/numerical results

| sector | selected IDC | mean T [K] | mean pre-heating | late pre-heating | heating slope [ps^-1] | TV to canonical | ground-manifold mismatch | expected post-heating | electronic exchange rate [eV/ps] | max generalized balance residual [eV] |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| singlet bipolaron | BM | 293.908 | 3.63739e-06 | 2.50845e-06 | -5.17081e-07 | 1.77101e-05 | 1.74295e-05 | -8.62117e-07 | -7.81647e-03 | 9.61567e-07 |
| distinguishable e-h exciton | DP | 293.467 | 1.48281e-06 | 1.41320e-06 | -1.23542e-07 | 3.48813e-06 | 3.47376e-06 | 1.48281e-06 | -1.88580e-03 | 7.97944e-07 |

Additional constraint maxima:

- bipolaron norm error: `2.22e-16`;
- bipolaron sector-constraint error: `1.78e-15`;
- exciton norm error: `2.22e-16`;
- exciton sector-constraint error: `6.66e-16`.

All 20 pre-registered D6g closure checks passed: temperature, mean heating, late heating, heating slope, canonical TV distance, ground-manifold mismatch, expected post-collapse heating, generalized energy balance, electronic norm, and sector constraints for both sectors.

## D6 model selection established by D6d–D6g

The coherent D6c Ehrenfest controls overheat electronically in both pair sectors even while the lattice bath remains correctly thermalized. D6e/D6f therefore tested explicit instantaneous-decoherence controls rather than transferring the one-polaron D5 choice blindly.

The interval sweep established distinct reference schemes:

- **singlet bipolaron: IDC-BM**. DP is excellent at short intervals but loses robustness for larger intervals; BM remains robust over the tested 50–500 fs range.
- **distinguishable e-h exciton: IDC-DP**. DP remains close to the canonical reference over the full tested 50–500 fs range, so the additional Boltzmann reweighting of BM is not required for the reference model.

This is a model- and control-specific result. It does not establish universal superiority of BM or DP, and `t_d` remains a phenomenological model parameter.

## Frozen D6 reference architecture

For the validated zero-field pair problem:

1. electronic propagation is complex-safe and matrix-free in the ordered `N^2` representation;
2. the bipolaron remains restricted to the symmetric spatial singlet sector;
3. the e-h exciton remains a distinguishable, spin-blind direct-interaction reference and must not be labeled singlet/triplet;
4. pair forces are obtained from propagated complex one-body reduced density matrices;
5. the lattice uses BAOAB with exact Ornstein-Uhlenbeck thermostatting and explicit projected intermolecular zero modes;
6. generalized energy accounting includes both lattice-bath heat and stochastic electronic-environment exchange;
7. no hidden state normalization, velocity rescaling, or energy repair is used in validation;
8. `dt = 0.2 fs`, Krylov dimension 8, bath friction 0.01 fs^-1, and `t_d = 100 fs` are validated numerical controls for this specific 4x4 closure, not universal/material parameters.

## Scope boundary

D6 does **not** establish field-driven finite-temperature pair transport, steady-state mobility, a material-specific decoherence time, or GPU/parallel performance. Those require separate downstream validation.

In particular, any future field-driven calculation with an electronic-decoherence model must validate the complete generalized balance

`Delta E_matter ~= Q_lattice + Q_electronic_environment + W_field`

before mobility or transport coefficients are interpreted.
