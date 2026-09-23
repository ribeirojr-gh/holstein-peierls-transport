# IP2a-2 — final local audit: calibration gate not met (2026-09-23)

## Provenance and execution

User-supplied artifact: `ip2a2-local-validation/20260923T151804Z/`.

- Validated source commit: `0f34d983f33983d8f36dc9a3ec96f23a458e4d50`
- Branch: `isotropic-polaron-barrier`
- Local status: `M run.sh` and untracked `run.sh.backup`. The runner command list in the artifact matches the intended stage, but a byte-identical claim about the locally modified `run.sh` is not made.
- Pycompile: PASS
- Focused pytest: PASS
- Full pytest: **495 passed**
- Harmonic PBC recurrence preflight: PASS; planned 4 ps versus stationary full-wrap estimate 21.693 ps
- Pre-intervention calibration script: exited normally; all serialization and stage-integrity checks: PASS
- Numerical/experiment runner overall: PASS, no execution-level failed gates
- Production energy selection: **FAIL — no candidate qualifies**.

**Runner PASS is not candidate qualification.** The stringent per-trial pre-event work-balance threshold was not achieved. This is a deliberate two-level status, not a contradiction.

## Fixed pilot and first events

The preregistered member IDs `0,4,8,12,16,20,24,28` were tested at each of `1e-5`, `3e-5`, and `1e-4 eV`. For every preparation, the 200 fs independent field-free control had no persistent event and a tiny matter-energy drift. None of the driven first events started before the 200 fs early-event cutoff.

**All 24/24 driven first accepted persistent events were valid nearest-neighbor x hops `820 -> 819 (-x)`** before 4000 fs. In most cases the transition started at 2826 fs and was accepted at 2874 fs. The largest tested energy shows modest onset shifts, including 2792 fs and 2838 fs in individual members. These events are observed, but **none counts toward the preregistered qualifying trial total** because the pre-event numerical criterion failed.

## Decisive work-balance gate

The locked IP2a-2 threshold was

`maximum |(E_matter(t)-E_matter(0)) - W_field(0,t)| <= 2.0e-6 eV`.

Observed ranges by candidate:

| Added Peierls energy (eV) | Clean no-field controls | Raw valid first x events | Pre-event maximum energy–work residual (eV) | Strict qualifying events | Candidate qualifies |
| ---: | ---: | ---: | ---: | ---: | --- |
| `1e-5` | 8/8 | 8/8 | `2.22309596e-6` to `2.22620561e-6` | 0/8 | No |
| `3e-5` | 8/8 | 8/8 | `2.22194943e-6` to `2.22733542e-6` | 0/8 | No |
| `1e-4` | 8/8 | 8/8 | `2.21969669e-6` to `2.22952965e-6` | 0/8 | No |

Across all 24 trials, norm and projected zero-mode diagnostics were comfortably within their preregistered bounds (`1e-10` each). Field-free energy drift was of order `1e-15` to `1e-14 eV`. The strong near-constancy of the field-on work residual across perturbation energies and antithetic partners is consistent with a common integration/work-quadrature error, but that mechanism has **not yet been established** by convergence testing.

The saved `.npz` event-start and residual arrays were independently cross-checked against all 24 JSON trial records. No missing or reclassified trial is needed to obtain the above result.

## Fixed decision

The pre-registered selection rule requires all eight pilot trials to pass numerical/field-free/early-event criteria and at least six qualifying first-x events. **None of the three candidate energies meets this rule. No production energy is selected and IP2b must not start under the existing calibration.**

Do not redefine the `2e-6 eV` gate after seeing the results, count the 24 raw events as qualified, or silently relabel this calibration PASS.

## Independent next diagnostic: IP2a-3

Study the time-step convergence of the **pre-event** energy/work residual without native/reversed branches, using the fixed reference unperturbed polaron and the fixed IP2a member 0 at `1e-5 eV`. Compare `dt=0.2, 0.1, 0.05 fs` at otherwise unchanged field, 40x40 lattice, Krylov dimension 6, sample/energy intervals and maximum search horizon.

This is **not** a re-analysis that can flip the IP2a-2 decision. If tighter time steps restore the strict original numerical accuracy, a new prospective production-energy calibration must be separately registered and run on all candidate energies/members using that finalized integrator.

A failure of the strict *calibration* gate does not negate previously validated broader size-aware D3 conservation results, nor does it imply the observed first hops were physical false positives. It only blocks this specific high-accuracy production protocol.
