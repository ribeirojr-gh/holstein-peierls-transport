# Holstein–Peierls extended-model program: validation and six-paper roadmap

Date: 2026-09-23  
Status: agreed research sequence; scope and numerical acceptance gates, not an assertion that future calculations have passed.  
Working branch: `isotropic-polaron-barrier`. Keep the ongoing IP2a-2 runner unchanged until its local artifact is audited.

## Program architecture

Treat the project as one reproducible computational platform with separate physical sectors and versioned production releases, not as one all-purpose solver whose tests automatically validate every sector.

- **P (one carrier):** Holstein–Peierls polaron; at fixed particle number one, ordinary two-body electron–electron U and V contributions vanish unless explicitly defined backgrounds or other physics are added.
- **BP (two equal charges):** correlated two-particle bipolaron; onsite U, offsite V and possibly screened long-range repulsion; spin symmetry and reduced-density-matrix normalization must be explicit.
- **X (electron–hole):** distinguishable two-body reference with electron–hole attraction. It does not acquire a physical singlet–triplet splitting merely by assigning labels.
- **SA (spin-adapted/multiconfigurational):** neutral-reference, exchange-consistent open-shell singlet and triplet theory with explicit state-specific orbital optimization, spin purity and lattice forces. Do not call the entire two-body reference X solver MCHF.
- **DA (donor–acceptor bilayer):** explicit two-layer electronic and structural model with intra-/interlayer hopping, offsets, screening and state/channel projectors.

A validated single-polaron T=0 integrator does not by itself certify correlated BP or spin-adapted X dynamics. Conversely, a deterministic model-grid response is not an experimentally calibrated diffusion coefficient or hopping probability.

## Gate 1 — freeze T=0 baseline and production infrastructure

Finish the ongoing IP2a-2 *pre-intervention* calibration. If its preregistered energy-selection rule finds no qualifying candidate, report that honestly and do not begin IP2b under an arbitrarily chosen energy.

Where feasible, finish the IP2b 32-member deterministic paired design using its existing frozen criteria. IP2b asks about trajectory sensitivity, **not** numerical validity of the one-polaron propagator: a physically negative IP2b does not invalidate otherwise converged D3/D4 code.

Prepare an immutable **T0 one-polaron production tag** only after:
1. algorithmic checks: norm, energy/work balance, gauge-continuous field release, Peierls phases, zero modes, event detection, PBC recurrence;
2. reproducibility and numerical convergence: dt, Krylov dimension, spatial size, checkpoint/restart equivalence, local `run.sh` and fully recorded parameter manifest;
3. physically distinct zero-field and field-driven reference cases;
4. all accepted/rejected test instances and known limitations documented;
5. raw trajectories, derived observables and figure-generation scripts tied to commit/parameter hashes.

The tag is a one-polaron T=0 baseline, **not** a certification for U/V-correlated BP or singlet/triplet dynamics. Keep IP1o–IP1u claims bounded: persistent Peierls memory and local retrograde-current attribution are established; none of IP1s/t/u met its preregistered discrete-event causal commitment gate.

## Paper 1 — static extended Holstein–Peierls model

**Focus:** new stationary electronic sectors and interactions, not dynamics. Use an extended Holstein–Peierls Hamiltonian with explicitly differentiated carrier sectors and RPROP-based lattice relaxation where that is the implemented stationary solver.

Scientific cases:
- single polaron static reference;
- correlated singlet bipolaron with U, short-range V and optional controlled screened tail; onsite/intersite/diagonal/separated stationary minima;
- electron–hole reference exciton (Frenkel/local, charge transfer, separated configurations);
- separately validated state-specific spin-adapted/MCHF singlet and triplet excitons, with exchange and neutral-background subtraction.

Important method boundary: the existing reference X pair solver is spin blind; its raw pair energies are **not** singlet/triplet energies. The existing S0 spin-adapted relaxation benchmark uses orbital optimization and an FFT-Hessian/preconditioned structural optimizer with line search, **not** legacy component-wise RPROP. Describe the stationary methods accurately instead of claiming that every MCHF stationary result comes from RPROP. A unified RPROP interface is optional only if independently implemented and benchmarked.

Required physical checks:
- exact small-cell U/V limits and direct-Hamiltonian energies; pair symmetry and electron/hole RDM traces;
- spin purity `<S^2>`, exchange-driven singlet/triplet splitting and its zero-exchange limit;
- neutral-reference energy and structural-force conventions;
- analytic versus finite-difference gradients; fully converged electronic and structural optimizations;
- multiple independent seeds/root tracking, finite-size/bound-state checks and the linear-Peierls regime;
- separate binding, relaxation and excitation-energy conventions; no material optical gap without a calibrated reference.

Deliverable: a static Hamiltonian and benchmark paper with reproducible U–V phase/topology maps and spin-resolved stationary examples only after the spin-adapted sector passes production-level gates.

## Paper 2 — zero-temperature electric-field dynamics of polarons and bipolarons

**Focus:** coherent/semiclassical field-driven T=0 response and carrier stability with and without U,V as physically applicable. The one-carrier baseline should show explicitly that ordinary two-body U,V do not affect an isolated one-particle sector; U,V govern correlated BP response.

Build on validated one-carrier field dynamics. Audit/reuse existing matrix-free D6 pair propagator and pair-coupled modules rather than rewriting them. For BP require:
- complex-valued pair propagation versus exact small-system reference; exchange symmetry and trace-two one-body RDM;
- moving-lattice pair forces and field-coupling sign for two equal charges;
- field-on energy/work balance; held-phase zero-power controls;
- dt/Krylov/size convergence, pair binding and separation observables, center-of-charge displacement;
- stable/broken-pair classification preregistered from pair correlations, with field-free controls.

Do **not** extract a steady mobility, hopping rate or activation energy from a single deterministic trajectory. The IP1 Peierls-memory results can inform mechanism, but their failed commitment gates must remain explicit.

## Paper 3 — thermal diffusion at zero applied field

**Focus:** temperature-driven spreading and stability of polarons, bipolarons and excitons without external field. A **single low-field illustrative control** may compare thermal and weak field responses, but must not contaminate the E=0 diffusion estimator.

Use/validate BAOAB/Langevin lattice bath and any electronic decoherence/thermalization module separately for P, BP and X/SA. Do not assume that the spin-blind reference X dynamics provides singlet/triplet resolved exciton diffusion. Add state-specific spin treatment if those labels appear in paper claims; otherwise label the reference exciton explicitly as spin blind.

Require:
- correct fluctuation–dissipation balance, temperature/energy distributions, collective zero modes and thermostat-only controls;
- independent thermal initial conditions/noise realizations and reproducible random seeds;
- center-of-mass mean-square displacement versus time, adequately identified linear window, finite-size/PBC unwrapping, statistical uncertainty and convergence;
- distinguish diffusion of the bound composite from separated constituent charges; condition on surviving carriers or report censoring explicitly;
- low-field comparator: separate drift from diffusion and report relative changes in carrier stability.

Do not infer exciton radiative yield or spin conversion without explicit corresponding physics.

## Paper 4 — joint E–T–U–V stability landscape

**Focus:** interaction/thermal/field competition rather than replotting Paper 2 field-only transport or Paper 3 field-free diffusion. Use a preregistered, tractable factorial or fractional-factorial design over field, T, U,V and distinguish physical sectors.

Primary outcomes should be sector-appropriate binding/separation and survival metrics, localization, structural memory and redistribution; rates/lifetimes only after a defined ensemble estimator, observation horizon, right-censoring and size/time convergence.

Document effect interactions (E×T, U×V, etc.) from matched controls, not just separate trends. Exclude unvalidated high-field regimes or parameter sets outside the model's linear-Peierls/screening validity.

## Paper 5 — bilayer donor–acceptor exciton dissociation

**Focus:** a genuinely new two-layer DA model, not a relabeling of a 2D monolayer. Specify:
- donor and acceptor HOMO/LUMO reference energies and band offsets;
- separate intralayer and interlayer hopping and electron–phonon couplings;
- layer-dependent lattice coordinates, interface geometry and e–h screened Coulomb/exchange interactions;
- spin-adapted singlet/triplet sectors as appropriate; absent spin–orbit coupling, no spontaneous singlet–triplet conversion;
- transparent channel projectors for LE (local exciton), interfacial CT and CS (charge-separated) configurations, including orthogonality/overlap and consistent interlayer-distance criteria.

Validate the decoupled-layer, zero-offset, vanishing-exchange and tiny-cell exact limits before production trajectories. Track populations and fluxes between LE/CT/CS **without double counting**; separate channel projection yield from measured external quantum efficiency. Exciton dissociation into CS must be distinguished from coherent amplitude transfer alone by explicit separation/survival criteria and finite-size convergence.

## Paper sequence and release gates

| Order | Program gate | Publication deliverable | Dependency |
| --- | --- | --- | --- |
| 1 | T0 one-polaron release | Validated computational baseline (not necessarily a standalone paper) | IP2a-2 audit; optional preregistered IP2b; convergence and production runner |
| 2 | Static extended sectors | Paper 1 — static U/V/BP/X and spin-adapted singlet/triplet | Audit existing static implementations; finish SA production validation |
| 3 | T0 correlated pair dynamics | Paper 2 — field-driven P/BP, U/V | D6 pair dynamics plus field/work and convergence gates |
| 4 | Thermal P/BP/X dynamics | Paper 3 — E=0 diffusion and one low-E comparator | Thermostat, ensemble, unwrapping, spin/charge-sector gates |
| 5 | Joint coupled stability | Paper 4 — E–T–U–V stability | Matched field/thermal interaction design and estimators |
| 6 | DA bilayer | Paper 5 — LE/CT/CS dissociation | Separate bilayer Hamiltonian, exchange, projectors, exact controls |

The user's six numbered items correspond to **one code-validation milestone and five research papers**. Paper numbers here reflect that accounting; do not inadvertently call the first code milestone “Paper 1.”

## Immediate execution policy (while IP2a-2 is running)

1. Do **not** change `run.sh`, physics kernels or the frozen IP2a-2 calibration protocol mid-run.
2. Wait for and audit the user-supplied local IP2a-2 ZIP, distinguishing numerical PASS from candidate energy selection.
3. Freeze the T0 single-polaron release gate and independently audit readiness of static BP, reference X, SA singlet/triplet and D6 dynamics.
4. Only then decide which IP2b cases are needed for the field-dynamics paper and which correlated pair gates must be added.
5. Every executable stage must provide a self-contained root `run.sh`; user executes locally and uploads the stage validation ZIP. Do not consume GitHub Actions quota and do not merge into `main` until authorized.
