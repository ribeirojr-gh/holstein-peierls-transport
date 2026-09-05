# TP2 paired-field linear-response and mobility design

## Purpose

TP2 begins the statistical transport phase after TP1 established a PBC-safe particle current and unwrapped displacement. The purpose is to determine whether a finite-field drift can be separated from stochastic noise and whether a field interval exists in which the response is linear.

TP2 is deliberately staged:

- **TP2a** — paired `+E/-E` screening and validation of the statistical estimator;
- **TP2b** — once a candidate linear interval is identified, increase trajectory length and ensemble size and test time/ensemble convergence;
- **TP2c** — only after TP2b, report a production mobility with uncertainty for a clearly identified numerical/material parameter set.

A TP2a numerical PASS is therefore not itself a mobility claim.

## Why field reversal is mandatory

Short stochastic trajectories exhibit seed-dependent drift even when the field response is weak. For each seed and field magnitude, TP2a therefore runs the same stochastic streams for `+E` and `-E` and forms

`v_odd(E) = [v(+E) - v(-E)] / 2`,

`v_even(E) = [v(+E) + v(-E)] / 2`.

The odd component is the field-driven estimator. The even component is a direct diagnostic of finite-time stochastic bias and nonlinear/even contamination.

Using common random-number seeds for the paired trajectories does not force the trajectories to remain identical: field-dependent propagation and IDC state selection can make them diverge. It only provides a controlled pairing and can reduce variance.

## Electron-like mobility convention

The validated Peierls phase has the electron-like power convention

`P_field = - E dot v_particle`.

For a positive scalar electron mobility,

`v_x = -mu_e E_x`.

With velocity in angstrom/fs and field in V/angstrom,

`mu_e [cm^2/(V s)] = -0.1 * v_x[angstrom/fs] / E_x[V/angstrom]`.

Equivalently, if the field is supplied in mV/angstrom,

`mu_e [cm^2/(V s)] = -100 * v_x[angstrom/fs] / E_x[mV/angstrom]`.

This sign convention is explicit and belongs to the electron-like field coupling already validated in D1/TP1. A future hole model must not silently reuse the same sign.

## Seed-level estimator

Field magnitudes for TP2a are fitted separately for each stochastic seed. For seed `s`, the through-origin odd-response slope is

`s_s = sum_E E * v_odd,s(E) / sum_E E^2`.

The seed mobility is

`mu_s = -0.1 s_s`

when `E` is in V/angstrom.

The ensemble mobility diagnostic is the arithmetic mean of the independent seed mobilities. Its uncertainty is reported as a Student-t 95% confidence interval across seeds. This avoids treating multiple field values from the same seed as independent observations.

Field-specific paired mobility values are also retained. They are useful for detecting field dependence but are not pooled as independent samples.

## TP2a numerical control

The first screening uses

- `20 x 20` periodic lattice;
- `T = 300 K`;
- `gamma_u = gamma_v = 0.01 fs^-1`;
- field magnitudes `0.5, 1.0, 2.0 mV/angstrom`, each with both signs;
- `dt = 0.2 fs`;
- total time `6 ps`;
- burn-in `2 ps`;
- IDC-BM with `t_d = 180 fs`;
- CF4-Lanczos, `m = 6`;
- projected intermolecular zero modes; and
- four independent seeds.

The field values and `t_d` are numerical screening controls, not material-calibrated quantities.

There are `3 field magnitudes x 2 signs x 4 seeds = 24` trajectories.

## Drift integration

For each deterministic/thermal propagation step, the TP1 velocity is evaluated at the start of the step and at the pre-IDC end of the step. The step displacement is integrated trapezoidally.

An IDC event occurs at the endpoint and has zero duration, so the post-collapse velocity is not incorrectly half-weighted into the interval that precedes the event. The collapsed state becomes the starting state for the next interval.

As an independent identity check for a purely x-directed field,

`W_field = -E_x * Delta x`.

This should remain at roundoff/integration precision because both quantities use the validated TP1/D3 conventions.

## Pre-registered TP2a closure gates

The following are numerical/implementation gates:

1. all focused TP0/TP1/TP2a tests pass;
2. the full regression suite passes;
3. every paired trajectory finishes with the expected IDC event count;
4. ensemble mean lattice temperature lies in `240--360 K`;
5. maximum generalized energy residual remains below `1e-4 eV`;
6. maximum electronic norm error remains below `1e-10`;
7. maximum projected zero-mode mean remains below `1e-12`;
8. all paired odd/even velocities and mobility diagnostics are finite;
9. maximum `|W_field + E_x Delta x|` remains below `1e-10 eV`.

These gates validate the estimator and the dynamics. They do **not** force a noisy or nonlinear physical response to look linear.

## Pre-registered TP2a response diagnostics

After the numerical gates pass, the response is classified using, but not forced to satisfy, the following diagnostics:

- sign of the ensemble odd response at each field;
- 95% confidence interval of the seed-level mobility;
- field-specific mobility means and confidence intervals;
- through-origin fit of ensemble-mean `v_odd(E)`;
- coefficient of determination for that fit;
- maximum fractional deviation from the through-origin linear fit, reported only when the fitted signal is non-negligible;
- magnitude of the even component relative to the odd component and its sampling uncertainty.

A candidate linear interval requires mutually compatible field-specific mobilities and no systematic growth of nonlinear residuals. Statistical significance requires a confidence interval narrow enough to resolve the response from zero. If either condition is absent, TP2b must increase trajectory time, ensemble size, lower the field, or some combination thereof.

## Claims prohibited at TP2a

TP2a must not be described as

- a converged mobility;
- a material prediction;
- a steady-state transport calculation;
- evidence that `20 x 20`, `6 ps`, four seeds, `m=6`, `dt=0.2 fs`, or `t_d=180 fs` are universally converged;
- a hole mobility; or
- a justification for GPU/threading choices.

The result of TP2a is a statistically controlled screening decision that determines how TP2b should be designed.
