# IP1l — inter-hop wake memory and incremental second-hop radiation

## Motivation

IP1k did not satisfy its preregistered two-event background-separation rule. The first anisotropic natural hop was fully background-separated, whereas the second hop occurred while the residence-matched pseudo-event windows already contained a large, coherent retrograde current left after the first hop.

The second event therefore cannot be interpreted using a neutral event-free background. IP1l asks a different, causal question:

> Does the first natural hop leave a retrograde lattice-energy wake that persists through the inter-hop residence, and does the second hop launch an additional backward pulse on top of that inherited wake?

This stage preserves the negative IP1k replication decision. It does not redefine the IP1k criterion after seeing the result.

## Production trajectory

Rerun only the anisotropic IP1i/IP1j/IP1k control:

- cell: 40x40 PBC;
- `J0y/J0x = 0.15`;
- field: +10 mV/A along x;
- T = 0 K;
- no thermostat;
- no IDC;
- deterministic D3 coupled field dynamics;
- dt = 0.2 fs;
- final time = 5 ps;
- harmonic intermolecular x-current stored every 2 fs;
- energy-work balance sampled every 10 fs;
- persistent nearest-neighbor residence criterion: 50 fs;
- projected uniform intermolecular zero modes;
- CF4-Lanczos, Krylov dimension 6.

Save the complete sampled current trajectory so that all later memory analyses can be rerun posthoc without repeating dynamics.

The expected screened trajectory contains two persistent -x relocations near 2496 and 4108 fs. Their exact detected times must be taken from the new run rather than hard-coded.

## Common inter-hop frame

Let the first two persistent x events be

- event 1: site A -> site B at `t1`;
- event 2: site B -> site C at `t2`.

Use the second-event source B and target C to define the longitudinal axis. Carrier motion is +s; the retrograde/trailing direction is -s.

This frame is fixed in the laboratory lattice during the complete B-residence interval `t1 < t < t2`. Therefore a backward d=2 boundary probes lattice energy two sites behind the carrier while the carrier resides at B.

Primary boundaries: d=1 and d=2. d=3 is diagnostic only because the available inter-hop time is short compared with the packet travel time.

## Field-only reference before the first hop

The no-memory reference is the same spatial B-centered frame before event 1.

Use:

- reference baseline: `t1 - 700 fs` to `t1 - 100 fs`.

This interval precedes any persistent x hop and therefore contains the deterministic field-driven lattice background but not a wake from an earlier carrier relocation.

Subtract the mean longitudinal current profile of this reference interval from all subsequent profiles used for the inter-hop memory audit.

## Inter-hop memory observable

At the backward d=2 boundary, define the outward current after reference subtraction. Quantify the first-hop wake using non-overlapping 100 fs bins from `t1` to `t2 - 100 fs`.

For each bin record:

1. positive backward transported energy;
2. signed net backward transported energy;
3. maximum backward outward flux;
4. mean backward outward flux.

Construct an empirical field-only background from all possible 100 fs windows inside the 600 fs pre-first reference interval, sampled every 20 fs.

For each inter-hop 100 fs bin record the percentile of its positive backward energy and peak flux relative to this pre-first field-only background.

### Wake-memory gate

The first-hop wake is considered to persist into the late inter-hop residence when, after the expected d=2 arrival time,

- at least one 100 fs bin whose center lies more than 1000 fs after `t1` has backward positive energy above the 95th percentile of the pre-first field-only distribution; and
- at least one such late bin has backward peak flux above the 95th percentile of that distribution.

Also report the latest bin center satisfying both conditions and the fraction of post-arrival inter-hop bins satisfying both conditions.

These percentiles are robustness discriminators, not formal independent-sample p-values.

## Packet propagation from the first hop

Using the same reference-subtracted fixed B-centered frame, determine the d=1 -> d=2 backward delay in a window beginning at event 1 and ending before event 2.

Search lags in 300-700 fs. A propagation branch is accepted when:

- correlation >= 0.80;
- fitted lag is internal to the search interval;
- inferred speed <= `1.05 * v_harmonic,max`.

This tests whether the memory observable is connected to a propagating packet rather than only to a static current offset.

## Incremental radiation from the second hop

The second hop starts from a lattice state that already contains first-hop memory. Therefore its incremental signal must be referenced to the **immediate pre-second wake state**, not to the pre-first field-only state.

Use:

- local second-event baseline: `t2 - 700 fs` to `t2 - 100 fs`;
- second-event post window: `t2` to `t2 + 800 fs`.

Subtract the mean longitudinal profile of the local second-event baseline and compute:

1. backward d=2 positive excess energy;
2. backward d=2 peak excess flux;
3. backward d=1 -> d=2 delay correlation and packet speed;
4. corresponding forward quantities for directionality context.

Construct a local fluctuation reference from 100 fs windows wholly inside the second-event baseline, sampled every 20 fs.

### Incremental second-hop gate

An additional retrograde pulse is supported when:

- the baseline-subtracted second-event backward d=1 -> d=2 packet passes the propagation gate;
- its post-event backward peak excess flux exceeds the 95th percentile of local pre-second 100 fs peak fluctuations;
- its positive excess energy is finite and positive.

Integrated post-event energy is **not** required to exceed the inherited wake energy. This is intentional: IP1k showed that the inherited wake itself can carry more integrated backward energy than the second event's locally baseline-subtracted increment.

## Secondary quantities

Report without using as acceptance gates:

- forward/backward positive-energy directionality in each inter-hop bin;
- ratio of second-event excess peak to the maximum local pre-second fluctuation peak;
- absolute backward energy carried during the late residence;
- event separation `t2 - t1`;
- d=1/d=2 phase coherence over the late residence;
- diagnostic d=3 flux when finite and interpretable.

## Numerical gates

Before physical interpretation:

1. pycompile passes;
2. focused IP1l/IP1k/IP1j/IP1h/D3/TP1 tests pass;
3. full pytest passes;
4. 40x40 zero-damping PBC recurrence preflight passes for the 5 ps trajectory;
5. exactly the expected two persistent x relocations are present in the anisotropic trajectory;
6. static relaxation converges;
7. all requested dynamics steps execute;
8. size-aware field-work balance passes;
9. electronic norm and projected zero modes remain controlled;
10. all memory and incremental-pulse observables are finite;
11. complete current trajectory is serialized for posthoc reanalysis.

The physical memory and incremental-pulse gates are not numerical acceptance gates.

## Interpretation guards

- IP1k remains formally non-replicated under its original joint criterion.
- IP1l tests memory and incremental radiation; it does not retroactively change the IP1k background definition.
- The pre-first reference is a deterministic field-only background in the chosen spatial frame, not a separate Hamiltonian.
- The local pre-second reference deliberately contains inherited first-hop memory.
- d1 -> d2 speeds are flux-packet propagation speeds, not unique normal-mode material group velocities.
- No mobility, hopping rate, activation energy, threshold field or material phonon lifetime is extracted.

## Decision after IP1l

If both the late-memory gate and the incremental second-hop gate pass, the working mechanism becomes a **history-dependent hopping process with a persistent retrograde lattice wake plus renewed radiation at subsequent relocations**. The next stage should then quantify how the wake amplitude decays with time/distance and whether it modifies the probability or timing of the next hop.

If the memory gate passes but the incremental second-hop gate fails, the dominant picture is persistent first-hop radiation without clear renewed backward emission at the second hop.

If the memory gate fails, the large IP1k pseudo-event background must instead be treated as a baseline-construction artifact or nonlocal field-driven oscillation, and the wake interpretation must be reconsidered.
