# IP2b — production paired counterfactual ensemble preregistration

Date: 2026-09-25  
Status: prospective production design; no IP2b outcomes have been inspected.

## Scientific question

Across the complete fixed 32-member deterministic Peierls preparation grid, does an energy-preserving reversal of the non-special x-Peierls traveling direction at the first accepted persistent x hop produce reproducible post-hop electronic-population divergence relative to a deterministic native-replicate control?

This is a finite-design causal-sensitivity experiment in the model. It is not a random-sample estimate of a hopping probability, mobility, diffusion constant, activation energy, or material lifetime.

## Frozen upstream choices

IP2a-1 validated the deterministic 32-member velocity-only construction. IP2a-4 selected, prospectively and solely from pre-intervention feasibility:

- preparation energy: **1.0e-5 eV**;
- integration step: **0.10 fs**;
- 40x40 isotropic PBC;
- +10 mV/A x event-generation field;
- CF4-Lanczos, Krylov dimension 6;
- 2 fs event/population sampling;
- 10 fs energy diagnostics;
- 4000 fs maximum driven event-search horizon.

These values are fixed for IP2b and may not be changed after member outcomes are viewed.

## Complete deterministic design

Run all member IDs `0..31`, preserving the original 16 antithetic pair IDs and signs. No member may be replaced by another preparation.

For every member, before any intervention:

1. start from the same converged 40x40 isotropic static polaron;
2. add exactly the fixed `1e-5 eV` velocity-only Peierls preparation;
3. run an independent 200 fs field-free control from that prepared state;
4. discard the control endpoint and restart the driven trial from the original prepared state;
5. apply +10 mV/A x and stop at the **first accepted persistent event of any direction**, or at 4000 fs if none appears;
6. accept a member for paired continuation only if that first event is a nearest-neighbor x event, its transition start is >=200 fs, and every pre-intervention numerical gate passes.

Do not skip an earlier y/non-neighbor event in order to find a later x event.

## Pre-intervention numerical gates

Per member require:

- static relaxation converged;
- field-free control has no persistent relocation in 200 fs;
- field-free maximum matter-energy drift <= `2e-6 eV` and within the established size-aware tolerance;
- first driven event accepted by 4000 fs;
- event transition start >=200 fs;
- first event is nearest-neighbor x;
- maximum driven pre-event energy-work residual <= `2e-6 eV` and within the established size-aware tolerance;
- electronic norm error < `1e-10`;
- projected zero-mode mean < `1e-10`;
- all diagnostics finite.

A member that fails here is rejected **before** native/reversed outcomes exist and retains an explicit rejection reason.

## Branch point and gauge-continuous release

At the accepted first x event time `t_s`:

- replace the driving field by the gauge-continuous held Peierls phase;
- require phase continuity < `1e-14` and Hamiltonian continuity < `1e-13 eV`;
- require held phase rates < `1e-15 fs^-1`, so external power is zero.

Instantiate three independent continuations from the same accepted state:

1. **native** — unchanged state under the held phase;
2. **reversed** — identical lattice coordinates and electronic state, identical `u` and `vy` velocities, but sign-reverse every non-special `vx` Fourier velocity sector;
3. **native-replicate** — a separately instantiated exact copy of native propagated through the same numerical code path.

The native-replicate branch is a deterministic numerical reproducibility control, not an independent physical sample.

## Intervention gates

Before propagation require:

- lattice coordinates native/reversed identical exactly;
- electronic state identical exactly;
- `u` and `vy` velocities identical exactly;
- special qx=0/Nyquist vx sectors unchanged to <`1e-15 A/fs`;
- non-special vx sectors sign reversed to <`1e-15 A/fs`;
- vx kinetic-energy mismatch <=`1e-12 eV`;
- full matter-energy mismatch <=`1e-12 eV`;
- retrograde/comoving traveling vx energies exchanged to <=`1e-12 eV`.

Any failed intervention gate invalidates the member before physical aggregation.

## Primary continuation

Propagate native, reversed and native-replicate for exactly **2000 fs** after `t_s`, with:

- dt=0.10 fs;
- held Peierls phase, zero phase rate;
- CF4-Lanczos Krylov dimension 6;
- population/event sampling every 2 fs;
- energy diagnostics every 10 fs.

No optional 5 ps secondary continuation is part of this production IP2b run.

The complete member horizon is at most 6 ps (4 ps search + 2 ps continuation), below the established ~21.693 ps stationary harmonic full-wrap recurrence diagnostic. The recurrence estimate remains a preflight bound, not a guarantee for a moving carrier.

## Branch numerical gates

For all three branches require:

- exactly 2000 fs propagated;
- accumulated external work magnitude <`1e-12 eV`;
- maximum matter-energy drift within the established 40x40 size-aware tolerance;
- electronic norm error <`1e-10`;
- projected zero-mode mean <`1e-10`;
- all serialized population/event diagnostics finite.

For native/native-replicate, record but do not pre-impose a physical divergence threshold. Their measured difference defines the numerical control floor.

## Primary member endpoints

At the fixed 2 fs samples define

`L1_i(t) = sum_j |p_j^native(t) - p_j^reversed(t)|`

and

`D_i = max_{0 <= t-t_s <= 2000 fs} L1_i(t)`.

Also define the deterministic control

`D_i^ctrl = max_t sum_j |p_j^native(t) - p_j^native-replicate(t)|`.

Record:
- `D_i`;
- `D_i^ctrl`;
- first sampled time reaching L1 >=0.10;
- first sampled time reaching L1 >=0.25;
- electronic population maps at the sample where `D_i` is maximal;
- max-site and IPR time series.

The 0.10 and 0.25 thresholds are inherited from the preregistered IP1/IP2 design and are not tuned from IP2b outcomes.

## Primary finite-design aggregation

Aggregate only members passing every numerical/pre-intervention/intervention/branch gate. Report all 32 member records, including explicit rejection reasons.

Use NumPy's standard linear percentile convention for quartiles.

Report:
- valid-member count;
- median `D_i`;
- Q1/Q3 and IQR of `D_i`;
- fraction of valid members with `D_i >= 0.25`;
- distribution of first L1>=0.10 and >=0.25 times;
- per-member ratio `R_i = D_i / max(D_i^ctrl, 1e-12)`;
- median `R_i`;
- results by the 16 antithetic preparation-pair IDs, descriptively only.

**No p-values, randomization/permutation tests or population confidence intervals are assigned to this deterministic grid.**

## Locked primary evidence criterion

Classify IP2b as evidence of reproducible trajectory sensitivity across the tested deterministic design states only if **all** are true:

1. at least **24/32** members are valid;
2. median `D_i >= 0.25`;
3. at least **60%** of valid members reach `D_i >= 0.25` within 2 ps;
4. median across valid members of `R_i = D_i/max(D_i^ctrl,1e-12)` is >= **100**;
5. all 32 preregistered member IDs are represented in the aggregate output with either complete valid results or an explicit predeclared rejection reason.

Failure of any one criterion means the primary IP2b claim fails. Do not redefine thresholds or remove inconvenient valid members.

## Secondary discrete-event analysis

Within the same 2 ps branch window, prospectively record every persistent nearest-neighbor x event.

For native and reversed report:
- first x-event presence;
- first x-event direction;
- absolute first transition-start difference when both exist;
- direct recross indicator: first x event returns from the post-hop site to the original pre-hop site;
- ordered x-event topology history represented as `(source_site,target_site)` pairs.

A member has secondary **event-history disagreement** when native and reversed ordered x-event topology sequences differ. Timing difference alone is reported separately and does not alter topology-history equality.

These are secondary and may not override the primary L1 classification.

## Pre-intervention modal covariates

At each valid branch point, before intervention, record descriptively:

- exact fixed-surface excitation energies `u`, `vx`, `vy`, total;
- `vx` retrograde, co-moving and special energies;
- retrograde fraction of direction-resolved `vx`;
- fixed low-q fractions for `vx` and `vy`, using the IP1q lowest nonzero quarter masks `0 < |q| <= pi/4`;
- branch-point d=1..4 local trailing x-current vector and the additive retrograde projection attribution when numerically defined.

These covariates are explanatory only. No q-band or covariate-dependent subgroup is promoted to a primary test in IP2b.

## Storage and GitHub execution

Run member calculations as a GitHub Actions matrix over IDs 0..31, with fail-fast disabled. Every member uploads JSON + NPZ even when rejected before continuation. A separate aggregation job downloads all member artifacts, verifies member completeness and runs the locked aggregate classifier.

The workflow must upload:
- validation/test logs;
- all 32 member JSON/NPZ artifacts;
- aggregate JSON/NPZ/Markdown;
- exact commit/run provenance.

A successful GitHub workflow is **not** itself a positive scientific result; execution integrity and the primary physical criterion are reported separately.

## Stop / advance rule

- If IP2b primary criterion **fails**, close the present Peierls-direction causal-control line at the deterministic ensemble level. Do not launch q-band rescue tests on the same outcomes.
- If IP2b primary criterion **passes**, IP2c may be separately preregistered to test fixed q-band interventions inherited from IP1q.
- Regardless of outcome, this stage does not estimate transport coefficients or physical hopping probabilities.
