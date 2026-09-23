# IP2a-1 — final local validation (2026-09-23)

## Provenance and numerical status

User-supplied artifact: `ip2a-local-validation/20260923T145733Z/`.
Validated branch: `isotropic-polaron-barrier`.
Validated source commit: `6d1dcdebbc58c4a202d866db2ac813fe0965d34a`.

- Runner overall: **PASS**, no failed gates.
- Pycompile: PASS.
- Focused pytest: PASS.
- Full pytest: **487 passed**.
- Ensemble construction experiment: PASS.
- Python 3.12.3; NumPy 2.5.3; SciPy 1.18.1; BLAS/OpenMP/MKL threads pinned to 1.

Local Git status contains `M run.sh` and untracked `run.sh.backup`. These are disclosed; the validation logs show the intended IP2a experiment commands, but no claim is made that the locally edited root runner matches the tracked file byte-for-byte.

## Construction audit

The archive contains complete float64 `vx` and `vy` velocity arrays for 32 members at each of three fixed candidate energies: `1e-5`, `3e-5`, and `1e-4 eV`, plus the unchanged baseline electronic population.

Each of the 32 members uses the same converged static 40x40 isotropic polaron, unchanged coordinates and electronic wavefunction, zero `u` velocity, and low-q Peierls velocity-only perturbations. The 16 pairs have exactly opposite `vx` and `vy` perturbations. Added energy is split exactly equally between `vx` and `vy`.

Reported construction invariants:
- maximum added kinetic-energy error: `4.06576e-20 eV`;
- maximum matter-energy increment error: `2.55092e-17 eV`;
- maximum zero-mode mean speed: `1.01644e-22 A/fs`;
- maximum sign-opposite pair mismatch: **0 A/fs**;
- minimum Euclidean distance between distinct base preparations: `1.28625e-5 A/fs`;
- no Fourier support outside the preregistered first four nonzero x/y sectors.

Independent audit of the serialized arrays recomputed their kinetic energies with `m2 = 1.5e5 eV fs^2/A^2`. The maximum errors for the three candidates are respectively `5.08e-21`, `1.69e-20`, and `4.07e-20 eV`. Opposite pairs cancel exactly, the energy is 50/50 between `vx`/`vy`, and the largest forbidden FFT coefficient is `8.28e-21` in the highest-energy array. Initial electronic population sums to 1 and is maximal at site 820.

A periodic-translation check found the closest pair of *distinct* preparations (among all 32) remains separated after independent x/y lattice translations, with a minimum 1D-profile Euclidean distance `3.976e-7 A/fs` in the lowest-energy candidate. This is a deterministic distinctness check, not a proof of statistical independence.

## Interpretation and methodological correction

**IP2a-1 is closed numerically.** It validates only ensemble construction, not event-generation yield, causal effects, diffusion, rates, or statistical generalization.

The 32 states are a fixed **deterministic design ensemble**, with 16 exactly antithetic pairs. They are not 32 independent random observations. Accordingly, a paired sign/permutation p-value or population confidence interval is not justified solely by this design. Before IP2b begins, the preregistration must replace unsupported inferential-statistics language with finite-design descriptive endpoints or specify and validate a genuine sampling/randomization scheme.

## Next substage

Proceed to IP2a-2: a pre-intervention, fixed-pilot feasibility experiment across the three energies. Select production energy only from numerical stability, early no-field control, and field-driven first-x-event yield. Do not generate or inspect native/reversed post-event outcomes during calibration.
