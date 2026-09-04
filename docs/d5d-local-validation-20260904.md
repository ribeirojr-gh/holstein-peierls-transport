# D5d final local validation and D5 closure

## Scope

This document records the final local D5 closure run performed from the ZIP snapshot of branch `d5-electronic-thermalization-decoherence` on 2026-09-04 UTC. Because a GitHub ZIP has no `.git` metadata, the local runner reports `git_commit`, `git_branch`, and `git_status` as `unknown`. The validated branch state immediately before this report was `f3bb4376d6a02f53f55b040d2c677e26b70a7953`.

The run used Python 3.12.3, NumPy 2.5.2, SciPy 1.18.1 under Linux/WSL2, with `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`.

## Numerical/regression gates

All requested gates passed:

- focused D5 tests: PASS;
- full `pytest`: PASS;
- D0a frozen-H propagator control: PASS;
- six strict S0 relaxations: PASS (`singlet/triplet x onsite/bond_x/bond_y`);
- final D5d 20x20, 300 K, 10 ps IDC-BM closure ensemble: PASS.

No GitHub Actions run was used because hosted Actions minutes were intentionally avoided during the local-validation period.

## Final D5d physical control

The closure control used:

- lattice: 20 x 20;
- temperature: 300 K;
- `dt = 0.2 fs`;
- trajectory length: 10 ps;
- burn-in: 2 ps;
- four independent lattice seeds;
- projected intermolecular collective zero modes;
- D4 BAOAB lattice bath;
- CF4-Lanczos electronic propagation with Krylov dimension 6;
- IDC-BM electronic relaxation/decoherence;
- `t_d = 180 fs` as an explicit **numerical control**, not a material-calibrated parameter.

Aggregate post-burn diagnostics (176 samples):

- mean lattice temperature: `297.877164 K`;
- mean pre-collapse heating coordinate: `0.002264`;
- late-window heating coordinate: `0.012617`;
- heating-coordinate slope: `0.003035 ps^-1`;
- mean TV distance to instantaneous canonical occupations: `0.173967`;
- mean absolute ground-manifold mismatch: `0.047616`;
- expected post-collapse heating coordinate: `-0.001280`;
- electronic-environment exchange rate: `-9.875599e-3 eV/ps`;
- maximum generalized energy residual: `1.458118e-5 eV`;
- maximum electronic norm error: `2.282619e-13`.

All nine pre-registered closure checks passed:

1. lattice temperature;
2. mean heating coordinate;
3. late-window heating coordinate;
4. heating slope;
5. canonical TV distance;
6. ground-manifold population mismatch;
7. expected post-collapse heating coordinate;
8. generalized energy balance;
9. electronic norm.

`beta_eff/beta_bath` remains a secondary nonlinear diagnostic and is not used alone as an equilibrium criterion.

## Scientific decision

D5 is closed for the one-carrier zero-field finite-temperature control.

The validated conclusions are:

1. coherent Ehrenfest alone exhibits systematic electronic overheating for the tested 20x20/300 K control;
2. the D4 lattice thermostat itself remains correctly thermalized, so the failure is electronic rather than a lattice-bath artifact;
3. explicit instantaneous-decoherence controls remove the severe heating trend;
4. IDC-BM is retained as the **reference electronic relaxation/decoherence scheme** for downstream controlled comparisons;
5. IDC-MA remains available as an independent control;
6. no material-specific decoherence time has been determined — `t_d` must remain an explicit model parameter unless independently calibrated or derived;
7. D5 does not establish steady-state mobility or other transport coefficients.

The next roadmap checkpoint is D6: bipolaron and exciton dynamics using the validated matrix-free propagator architecture. Frozen-H pair-sector validation must precede coupled pair-lattice production dynamics.
