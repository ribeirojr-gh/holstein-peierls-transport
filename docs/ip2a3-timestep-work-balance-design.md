# IP2a-3 — preregistered driven energy/work time-step diagnostic

Date: 2026-09-23  
Status: prospective numerical diagnostic; IP2a-2 selection remains **failed**.

## Reason and separation from IP2b

Every IP2a-2 pilot found a valid first persistent x event and passed field-free, norm and zero-mode controls, but every field-on residual exceeded the fixed `2e-6 eV` calibration bound. The observed residual is ~`2.22e-6 eV` at `dt=0.2 fs` and is almost insensitive to the added Peierls energy.

This study tests whether reducing integration time step improves that *numerical* error. It does not reclassify the completed IP2a-2 calibration and cannot select a production Peierls energy. It creates no post-event native/reversed branches.

## Locked reference calculations

Use the same 40x40 isotropic static polaron and +10 mV/A x field used in IP2a-2. Compare two fixed initial states:

1. unperturbed converged static polaron with zero initial lattice velocity;
2. preregistered deterministic ensemble member 0 with `1e-5 eV` added Peierls velocity-only excitation.

For each initial state run the first-event generation procedure with:
- `dt=0.2, 0.1, 0.05 fs` in that order;
- 2 fs event sampling and 10 fs energy diagnostics;
- CF4-Lanczos, Krylov dimension 6;
- at most 4000 fs of driven evolution, stopping at the first accepted persistent event, **whether x or y**;
- same initial field gauge and same trapezoidal step-work accumulation as IP2a-2;
- the same `2e-6 eV` accuracy bound.

Do not change noise seeds, initial velocity phase, field, lattice size, Krylov dimension, sample interval or event criterion.

## Recorded quantities

For each of six trials:
- exact state label, target perturbation energy, dt and intended maximum search time;
- first event, its transition-start and accepted time if present;
- maximum pre-event `|E_matter(t)-E_matter(0)-W_field(t)|`;
- maximum electronic norm error, projected zero-mode mean, accumulated work;
- elapsed runtime;
- reproducibility check for the original IP2a-2 member 0 `dt=0.2` result.

Report successive residual reduction factors and effective log2 convergence orders. Do not infer a formal asymptotic convergence order if errors are not in the asymptotic regime.

## Prospective numerical decision

Separate **execution integrity** from **timestep diagnostic**:

- execution integrity requires pycompile, focused/full pytest, recurrence preflight, fixed input protocol, complete finite records, no post-event branches and all six trials reaching the first persistent x event before the fixed cutoff;
- numerical refinement evidence requires residuals to decrease strictly on each halving of dt for each reference state, and at least one tighter dt to meet the **original** `2e-6 eV` bound for both states;
- choose the **largest of the two tighter dt values** meeting the bound for both reference states, but only if both states exhibit monotonic residual reduction across all three dt values.

This dt is merely a candidate for a newly preregistered complete pilot calibration, not a production integrator and not a production perturbation-energy selection.

If none meets the diagnostic, do not expand the original threshold. Investigate time integration, work quadrature, measurement synchronization and gauge implementation separately before any further calibration.

## Stop rule and interpretation

IP2a-2 remains a completed strict failure regardless of the IP2a-3 outcome. A numerically successful IP2a-3 may motivate a **newly registered IP2a-4** with a complete fixed pilot at the refined dt. Only that new calibration, if it meets its own unchanged `2e-6 eV` work-balance and >=6/8 valid-first-x rules, can unlock IP2b.

The deterministic phase grid does not provide 32 statistically independent random observations. Do not infer a physical hopping rate or probability from the six diagnostic trajectories.
