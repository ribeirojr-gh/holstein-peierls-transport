# IP2a-3 — final local numerical validation (2026-09-23)

## Provenance and status

Local artifact: `ip2a3-local-validation/20260923T155518Z/`.

- Validated source commit: `d6632ce5669fedf6c31195adb55613c74431c73b`
- Branch: `isotropic-polaron-barrier`
- Local git status: only untracked `run.sh.backup`; tracked working tree clean.
- Pycompile: PASS.
- Focused pytest: PASS.
- Complete test suite: **500 passed in 7.86 s**.
- PBC recurrence preflight: PASS; 4 ps planned versus ~21.693 ps stationary-packet wrap diagnostic.
- All six fixed-reference experiment records: complete, finite, accepted nearest-neighbor x event.
- Numerical execution gates: PASS.
- Preregistered time-step refinement diagnostic: **PASS**.
- **Original IP2a-2 production-energy calibration: FAIL, unchanged.**

## Six fixed pre-event trials

The unperturbed 40x40 isotropic polaron and prepared member 0 with `1e-5 eV` added Peierls kinetic energy were each driven with +10 mV/A x until their first accepted persistent event, using 2 fs event samples, 10 fs energy samples and CF4-Lanczos Krylov dimension 6.

| Reference | dt (fs) | Maximum pre-event energy–work residual (eV) |
| --- | ---: | ---: |
| Unperturbed | 0.20 | 2.224654963834793e-6 |
| Unperturbed | 0.10 | 5.559831909103703e-7 |
| Unperturbed | 0.05 | 1.389845232279210e-7 |
| Member 0, 1e-5 eV | 0.20 | 2.223957749940535e-6 |
| Member 0, 1e-5 eV | 0.10 | 5.558089229535344e-7 |
| Member 0, 1e-5 eV | 0.05 | 1.389409583405464e-7 |

In both references the residual decreases strictly by approximately 4x per step halving, giving effective log2 orders ~2.0001–2.0005 over the tested range. This is **observed numerical convergence of the combined integrator/work balance**, not proof of any particular error source in isolation.

All six runs detect the same first persistent event, `820 -> 819 (-x)`, start 2826 fs and accepted 2874 fs (roundoff ~5e-13 fs at the smallest step). Electronic norm errors remain below 7e-12, and projected zero-mode means remain below 2.4e-17. There are no native/reversed continuations.

The reference `dt=0.2 fs` member-0 residual reproduces the original IP2a-2 result exactly to the serialized precision. The NPZ and JSON arrays of dt, residual and acceptance time agree elementwise.

## Locked numerical decision

Original accuracy bound: `2.0e-6 eV`, **not changed**.

Both `dt=0.10` and `0.05 fs` satisfy it for both references. In accord with the IP2a-3 prospective selection rule, choose the **largest tightened step `dt=0.10 fs` as a candidate for a new complete calibration**.

This does **not** qualify any of the three Peierls preparation energies, does not retroactively turn IP2a-2 into a pass, and does not establish physical control of a later event.

## Next locked stage: IP2a-4

Rerun the complete fixed eight-member pilot at all three original energies using `dt=0.10 fs` and otherwise unchanged IP2a-2 criteria:
- pilot IDs `0,4,8,12,16,20,24,28`;
- candidate energies `1e-5, 3e-5, 1e-4 eV`;
- separate 200 fs field-free control per trial, discarded before the driven search;
- +10 mV/A x, 4000 fs driven first-event search;
- exclude trials with any pre-200 fs driven transition start or an earlier y event;
- field-free drift and field-on work-balance residual each <= `2e-6 eV` and the established size-aware bounds;
- norm <`1e-10`, projected zero modes <`1e-10`;
- qualifying energy needs all 8/8 numerically clean, no early events, and >=6/8 valid first persistent x events;
- choose the **smallest qualifying candidate energy**; choose none if all fail.

IP2a-4 must be preregistered in a separate document and create no post-event counterfactual branches. IP2b cannot start before the full pilot is independently validated.
