# D6g final pair-dynamics closure protocol

## Selected reference controls

D6f supports a sector-specific choice:

- symmetric singlet bipolaron: BM instantaneous decoherence;
- distinguishable electron-hole exciton: DP instantaneous decoherence.

Both use `t_d = 100 fs` only as a common numerical-control interval for the final closure. The value is not a calibrated material property and must remain explicit in downstream calculations.

## Final ensemble

The D6g benchmark uses:

- 4x4 lattice;
- 300 K classical bath;
- gamma_u = gamma_v = 0.01 fs^-1;
- dt = 0.2 fs;
- 10 ps trajectories;
- 2 ps burn-in;
- four stochastic seeds per sector;
- CF4-Lanczos with m = 8;
- projected intermolecular zero modes;
- zero electric field.

## Pre-registered physical/numerical closure criteria

For each selected sector independently:

1. ensemble mean lattice temperature must remain in [240, 360] K;
2. absolute ensemble mean pre-collapse heating coordinate < 0.02;
3. absolute late-window heating coordinate < 0.04;
4. absolute heating-coordinate slope < 0.01 ps^-1;
5. ensemble mean TV distance to the instantaneous canonical distribution < 0.05;
6. ensemble mean absolute ground-manifold population mismatch < 0.05;
7. absolute mean expected post-collapse heating coordinate < 0.02;
8. maximum generalized energy-balance residual < 1e-4 eV;
9. maximum electronic norm error < 1e-10;
10. maximum sector-constraint error < 1e-10.

These thresholds are deliberately broader than the D6f 4 ps observations so that the 10 ps closure is not a re-test against fitted tolerances. A PASS establishes numerical/physical closure for the selected zero-field control model only. It does not establish mobility, field-driven detailed balance, or a material-specific decoherence time.

## Repository closure

The local D6g runner additionally reruns:

- focused D6 tests;
- the full pytest suite;
- D0a frozen-propagator benchmark;
- six S0 strict 4x4 relaxation jobs (singlet/triplet x onsite/bond_x/bond_y).

D6 is not declared complete unless both the 10 ps pair benchmark and these repository-wide gates pass.
