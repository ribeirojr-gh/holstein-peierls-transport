# D6b coupled zero-temperature pair dynamics — closure

## Status

D6b is numerically closed for the registered 4x4 control.

The local validation artifact `d6b-local-validation/20260904T230038Z/` reported:

- focused D6 tests: 15/15 passed;
- full test suite: 296 passed;
- benchmark closure: PASS;
- no failed gates.

The local run came from a ZIP checkout, so `git_commit`, `git_branch`, and `git_status` were recorded as `unknown`. The artifact was supplied after downloading branch `d6-pair-dynamics` at repository head `304b427893dc2070c8d9dc67479f6fb02f2b30e7`; this manual association is provenance metadata, not a machine-verifiable git record inside the artifact.

## Reference control

The benchmark used a 4x4 lattice, Hilbert dimension 256 in each pair sector, explicit lattice masses 75000 and 150000 eV fs^2/A^2, a 4 fs tightened DOP853 reference, fixed-step CF4-Lanczos/velocity-Verlet at dt = 0.2, 0.1, 0.05 fs, and a 100 fs stability run at dt = 0.2 fs.

No thermostat, decoherence, electric field, or transport fit is part of D6b.

## Bipolaron results

At dt = 0.2 fs:

- phase-aligned electronic error: 5.8882581e-8;
- lattice max error: 7.6939956e-10 A;
- total-energy change after 4 fs: 5.8299116e-10 eV;
- norm error: 1.3322676e-15.

Halving the time step reduced the electronic error approximately by factors of four:

- dt 0.2 fs: 5.8883e-8;
- dt 0.1 fs: 1.4720e-8;
- dt 0.05 fs: 3.6800e-9.

The 100 fs stability run gave:

- max |dE| = 1.2820467e-7 eV;
- max norm error = 2.7755576e-15;
- max exchange/RDM-sector constraint error = 1.3322683e-15;
- max lattice excursion = 0.0274147 A.

## Exciton results

At dt = 0.2 fs:

- phase-aligned electronic error: 5.0989445e-7;
- lattice max error: 8.1501453e-10 A;
- total-energy change after 4 fs: -2.3809454e-10 eV;
- norm error: 2.2204460e-16.

Time-step convergence was again approximately second order for the coupled split:

- dt 0.2 fs: 5.0989e-7;
- dt 0.1 fs: 1.2747e-7;
- dt 0.05 fs: 3.1868e-8.

The 100 fs stability run gave:

- max |dE| = 2.5169191e-7 eV;
- final dE = -9.3460840e-8 eV;
- max norm error = 2.2204460e-15;
- max RDM trace constraint error = 4.4409518e-16;
- max lattice excursion = 0.0779921 A.

## Krylov decision

For the tested dt = 0.2 fs control, m = 6, 8, and 12 gave indistinguishable coupled errors within the time-discretization floor. Therefore m = 8 remains the conservative D6 reference for the next moving-lattice gates, but it is not a universal production setting and must be rechecked when temperature, field, lattice size, or interaction strength changes substantially.

## Closure checks

All pre-registered D6b checks passed in both sectors:

- time-step improvement;
- m=8 accuracy;
- energy stability;
- norm preservation;
- sector/RDM constraints;
- bounded lattice motion.

## Scientific interpretation

D6b validates deterministic zero-temperature coupled bipolaron and distinguishable electron-hole dynamics for the registered control. It does not validate finite-temperature equilibrium, electronic decoherence, field-driven transport, material-specific pair interactions, or mobilities.

The next checkpoint is D6c: transfer the already validated D4 BAOAB lattice bath to both pair sectors while preserving explicit zero-mode policy, bath heat accounting, pair-sector constraints, and the matrix-free complex-safe electronic propagator. D6c remains coherent Ehrenfest pair dynamics; any electronic thermalization/decoherence correction is a later gate and must be diagnosed rather than assumed.
