# TP2b local validation — 2026-09-05

## Scope

This document records the completed local execution of the original TP2b paired-field convergence protocol. The run was started before the transport interpretation was revised toward an explicit hopping-mechanism analysis. Therefore, the execution remains a valid numerical/statistical diagnostic, but its previous `mobility_ready` logic is no longer used as the sole physical decision rule for the project.

The executed control was:

- one-polaron Holstein–Peierls dynamics;
- 20×20 periodic lattice;
- 300 K;
- BAOAB lattice bath;
- `gamma_u = gamma_v = 0.01 fs^-1`;
- field-aware IDC-BM;
- `t_d = 180 fs` (numerical control, not material-calibrated);
- `dt = 0.2 fs`;
- CF4-Lanczos with Krylov dimension 6;
- projected intermolecular zero modes;
- paired common-random-number `+E/-E` trajectories;
- fields ±0.5, ±1.0 and ±2.0 mV/Å;
- eight seed groups;
- 12 ps total time with 2 ps burn-in;
- cumulative checkpoints at 6, 9 and 12 ps.

The local artifact was produced under Python 3.12.3 on WSL2 with NumPy 2.5.2 and SciPy 1.18.1, with `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`. Git metadata was unavailable because the user executed from a downloaded ZIP; this provenance caveat is accepted.

## Numerical validation

All numerical gates passed.

- focused TP0/TP1/TP2/field/bath/IDC tests: 39/39 PASS;
- full repository pytest: 339 passed;
- compile gate: PASS;
- 48/48 trajectories completed;
- mean lattice temperature: 299.5207 K;
- maximum generalized energy-balance residual: 1.5339×10^-5 eV;
- maximum electronic norm error: 2.4580×10^-13;
- maximum projected zero-mode mean: 1.4988×10^-17;
- maximum direct field-work vs `-E Δx` disagreement: 3.6397×10^-14 eV;
- maximum instantaneous power/velocity identity error: 0 within recorded precision;
- benchmark elapsed time: 5584.45 s (~93.1 min).

Thus TP2b is numerically closed as a valid execution of the original protocol.

## Temporal response diagnostics

### 6 ps checkpoint (4 ps post-burn)

- ensemble mobility diagnostic: 0.0328203 cm²/Vs;
- 95% seed-level CI: [0.0020927, 0.0635478] cm²/Vs;
- through-origin R²: 0.7880;
- maximum fractional linearity residual: 0.3797;
- maximum even/odd mean ratio: 1.6403.

### 9 ps checkpoint (7 ps post-burn)

- ensemble mobility diagnostic: 0.0407168 cm²/Vs;
- 95% seed-level CI: [0.0120352, 0.0693985] cm²/Vs;
- through-origin R²: 0.9573;
- maximum fractional linearity residual: 0.2288;
- maximum even/odd mean ratio: 0.7484.

### 12 ps checkpoint (10 ps post-burn)

- ensemble mobility diagnostic: 0.0284509 cm²/Vs;
- 95% seed-level CI: [0.00979335, 0.0471084] cm²/Vs;
- through-origin R²: 0.9557;
- maximum fractional linearity residual: 0.1750;
- maximum even/odd mean ratio: 0.6889;
- fractional mobility change between 9 and 12 ps: 0.4311.

All eight final seed-level all-field mobility estimates were positive:

`0.01166, 0.01580, 0.03069, 0.03499, 0.03931, 0.00582, 0.07500, 0.01433 cm²/Vs`.

The broad seed-to-seed spread remains physically significant.

## Field-resolved 12 ps diagnostics

The final paired odd response gave the following field-specific mobility-like diagnostics:

- 0.5 mV/Å: 0.04756 cm²/Vs, 95% CI [-0.00289, 0.09801];
- 1.0 mV/Å: 0.01849 cm²/Vs, 95% CI [0.00217, 0.03482];
- 2.0 mV/Å: 0.02975 cm²/Vs, 95% CI [0.00671, 0.05278].

The corresponding odd mean velocities were approximately:

- 0.5 mV/Å: -2.378×10^-4 Å/fs;
- 1.0 mV/Å: -1.849×10^-4 Å/fs;
- 2.0 mV/Å: -5.949×10^-4 Å/fs.

The low-field-only (0.5 and 1.0 mV/Å) regression produced 0.02431 cm²/Vs with 95% CI [0.01413, 0.03449], but its through-origin R² was only 0.8138 and the maximum even/odd mean ratio was 1.1689.

## Original readiness result

Under the pre-registered original TP2b readiness logic, the following checks passed:

- positive 95% seed-level mobility CI;
- ensemble through-origin R² ≥ 0.90;
- maximum fractional linearity residual ≤ 0.25;
- low-field vs all-field mobility difference ≤ 25%;
- at least 75% positive seed-level mobilities.

The following did not converge:

- maximum mean even/odd ratio ≤ 0.50;
- mobility change between 9 and 12 ps ≤ 25%.

Therefore the original `mobility_ready` flag was false.

## Revised physical interpretation

After the user clarified that the relevant transport mechanism is persistent nearest-neighbour hopping of a dressed polaron and that isotropic systems may remain strongly pinned because the charge must reorganize the intra-site and both intermolecular distortions, the project interpretation changed.

The current data should therefore **not** be promoted to a production mobility, even though the 12 ps paired-field statistics are more stable than TP2a. A non-zero integrated current can contain intra-polaron oscillatory response, breathing, or transient redistribution that does not correspond to a persistent site-to-site hop. Conversely, a mobility compatible with zero in an isotropic, strongly dressed regime is not automatically a convergence failure.

The field dependence is also not sufficiently monotonic to justify forcing a simple linear-response picture. In particular, the 2.0 mV/Å odd velocity is substantially larger in magnitude than the 1.0 mV/Å value, while the 0.5 mV/Å estimate remains noisy. This is compatible with the need to explicitly resolve hopping events and possible field-assisted onset rather than infer mechanism from a global linear fit.

## Decision

1. TP2b numerical execution: **PASS / CLOSED**.
2. TP2b production mobility claim: **NOT ESTABLISHED**.
3. The former mobility-readiness protocol is retained only for provenance and is superseded as the primary physical decision rule.
4. The next transport analysis must resolve persistent hops, residence times, direction-resolved rates, IPR/participation changes, and lattice-distortion reorganization.
5. In parallel, the `isotropic-polaron-barrier` branch will investigate the static translation barrier and its dependence on anisotropy before finite-temperature isotropic hopping is attempted.

No material-calibrated mobility, threshold field, activation energy, or isotropic transport claim is made from this TP2b run alone.
