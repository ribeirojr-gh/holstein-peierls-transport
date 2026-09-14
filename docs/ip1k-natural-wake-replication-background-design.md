# IP1k — natural-hop wake replication and matched-background audit

## Purpose

IP1j establishes the **existence** of a backward-propagating intermolecular energy-current branch during the first natural anisotropic field-driven carrier relocation. The branch is weak at d=2, carrying only about 1% of the positive outward energy over the full 1.5 ps window, while the forward side dominates strongly.

IP1k therefore asks the next necessary question:

> Is the weak retrograde branch reproducibly associated with persistent carrier relocations, or can the same signature be produced by continuous field-driven lattice oscillations during the same pre-hop residence state?

This stage is deliberately an event/background discrimination experiment. It does not estimate mobility, hopping rates, activation barriers, threshold fields or material phonon lifetimes.

## Production controls

Rerun only the two IP1j controls:

- cell: 40x40 PBC;
- `J0y/J0x = 1.0` and `0.15`;
- field: +10 mV/A along x;
- T = 0 K;
- no thermostat;
- no IDC;
- deterministic D3 coupled field dynamics;
- `dt = 0.2 fs`;
- final time = 5 ps;
- harmonic intermolecular current sampled every 2 fs;
- energy-work balance sampled every 10 fs;
- persistent residence criterion: 50 fs;
- projected uniform intermolecular zero modes;
- CF4-Lanczos, Krylov dimension 6.

The field amplitude remains a numerical protocol control, not a material-calibrated transport threshold.

## Real-event windows

Analyze **every persistent nearest-neighbor x relocation** for which a complete local window fits inside the trajectory and does not contain another persistent x relocation.

For one event starting at `t0`:

- baseline: `t0 - 700 fs` to `t0 - 100 fs`;
- post-event analysis: `t0` to `t0 + 800 fs`;
- event source -> target defines +s;
- transverse strip: `|p| <= 3` sites;
- d=1 and d=2 fixed boundaries are used for the primary packet test.

A real event is excluded if either its baseline or post window contains another persistent x-event start.

The 800 fs post window is long enough to capture the IP1j anisotropic backward d1->d2 delay near 584 fs while remaining shorter than the 1612 fs first-to-second anisotropic event separation observed in IP1i/IP1j.

## Residence-matched pseudo-event background

For each real event, construct deterministic pseudo-event times from event-free parts of the **same trajectory and the same pre-hop persistent-x residence interval**.

Candidate pseudo-event centers are sampled every **20 fs** and must satisfy:

1. at least 700 fs of trajectory exists before the candidate;
2. at least 800 fs exists after it;
3. the complete `[-700,+800] fs` window contains no persistent x-event start;
4. the candidate occurs after the previous persistent x relocation, if one exists, and before the real event under analysis;
5. the same source-centered event frame and carrier-direction convention used by the corresponding real event is retained.

The residence restriction matters. It prevents, for example, comparing the second anisotropic hop from site 819 with a pseudo window centered on the earlier site-820 residence. For the second anisotropic event, the 1612 fs inter-event spacing leaves only about 112 fs of complete event-free pseudo-center support once the full `[-700,+800] fs` window is enforced. A 20 fs cadence provides at least five matched controls in that interval.

For a pseudo-event at time `tp`, use exactly the same baseline subtraction and d=1/d=2 boundary analysis as for the real event. This creates an empirical null distribution for continuous field-driven lattice oscillations in the **same carrier residence basin**, without introducing a different Hamiltonian or an arbitrary no-field control.

Pseudo-event windows overlap and are therefore correlated samples. They are used for empirical background percentiles, not formal independent-sample p-values.

## Primary observables

For each real and pseudo event record:

1. d=2 backward positive outward energy over 0-800 fs;
2. d=2 forward positive outward energy over 0-800 fs;
3. d=2 signed directionality
   `D2 = (E_back^+ - E_front^+) / (E_back^+ + E_front^+)`;
4. maximum backward outward flux at d=2;
5. maximum forward outward flux at d=2;
6. backward d1->d2 normalized delay correlation;
7. forward d1->d2 normalized delay correlation;
8. inferred packet speed for any accepted delay.

## Packet-delay gate

The primary backward delay search is restricted to `300-700 fs`.

This range contains the robust IP1j anisotropic maximum at 584 fs and intentionally excludes the long-lag aliases found in the post-upload sensitivity audit.

A backward branch is packet-qualified when:

- correlation >= 0.80;
- fitted lag is not at the search boundary;
- inferred speed <= `1.05 * v_harmonic,max`.

The same diagnostic is computed for the forward branch, but forward propagation is not required for a retrograde-event conclusion.

## Event-associated amplitude gate

For each real event compare the two backward-amplitude observables with its own residence-matched pseudo-event background:

- d=2 backward positive outward energy;
- d=2 peak backward outward flux.

Record the empirical percentile of the real value in each pseudo distribution.

A real backward packet is called **background-separated** only when:

- it passes the packet-delay gate;
- its backward positive energy is above the 95th percentile of matched pseudo windows;
- its backward peak outward flux is above the 95th percentile of matched pseudo windows.

The 95th-percentile rule is a preregistered robustness discriminator, not a formal frequentist significance level because pseudo windows overlap in time.

## Replication decision

### Anisotropic control

The natural retrograde-wake interpretation is promoted from single-event evidence to **replicated event-associated evidence** only if at least two complete anisotropic persistent x relocations are available and at least two independently analyzed real events are background-separated by the above rule.

If only the first event passes, IP1j remains a valid existence result but the wake is not considered replicated.

### Isotropic control

The isotropic result remains diagnostic. IP1j showed that its backward classification is threshold-sensitive because the unconstrained optimum lies only 6 fs below the first speed-admissible lag. IP1k must therefore report isotropic real-event/background percentiles without translating a failed binary delay gate into an absence claim.

## Numerical gates

Before physical interpretation:

1. pycompile passes;
2. focused IP1k/IP1j/IP1h/IP1f/D3/TP1 tests pass;
3. full pytest passes;
4. 40x40 zero-damping phonon-recurrence preflight passes for 5 ps;
5. static relaxations converge;
6. complete requested dynamics steps are executed;
7. size-aware field-work balance passes;
8. electronic norm error remains below the established tolerance;
9. projected intermolecular zero modes remain controlled;
10. all real and pseudo metrics are finite;
11. at least one complete real event is available in each condition;
12. **every complete real event has at least five residence-matched pseudo-event windows**.

Physical packet/background outcomes are not numerical acceptance gates.

## Interpretation guards

- The electric field remains on in real and pseudo windows.
- Pseudo-event samples from one deterministic trajectory are not statistically independent.
- The 20 fs cadence is chosen to obtain several controls in the short second anisotropic residence interval; overlapping pseudo windows do not increase the number of independent trajectories.
- Baseline subtraction does not uniquely separate bound polaron dressing from free normal modes.
- d1->d2 delay is a packet-propagation diagnostic, not a unique material phonon group velocity.
- A weak but background-separated retrograde component may coexist with strongly forward-dominated total radiation.
- No transport coefficient or material parameter is extracted in IP1k.

## Decision after IP1k

If the anisotropic backward branch is replicated and background-separated, the next stage may investigate how its relative amplitude evolves with distance and field strength. If it is not background-separated, the correct conclusion is that IP1j demonstrated only a weak single-event component whose causal association with hopping remains unresolved; long-range or transport-mechanism claims must then be deferred.
