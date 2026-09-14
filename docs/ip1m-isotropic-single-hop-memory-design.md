# IP1m — isotropic single-hop wake persistence and spatial-envelope control

## Motivation

IP1l closes both preregistered physical gates in the anisotropic 10 mV/A trajectory: a retrograde first-hop wake persists deep into the 1612 fs residence and the second hop launches a renewed backward pulse on top of that inherited state.

The late anisotropic current is strongly non-monotonic, however, and the second hop is already approaching while the largest late backward signal develops. The cleanest next control is therefore the natural **isotropic** 10 mV/A trajectory identified by IP1i/IP1j/IP1k. It contains one persistent x relocation near 2.8 ps and no second persistent hop before 5 ps.

IP1m asks:

> Does a natural isotropic hop leave a long-lived retrograde lattice-energy memory when no second persistent relocation occurs during the remaining observation time?

A positive result would show that an approaching second hop is not required for the late-memory phenomenon and would return the mechanism analysis to the central isotropic-polaron problem.

## Production trajectory

Rerun only the isotropic control:

- cell: 40x40 PBC;
- `J0y/J0x = 1.0`;
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

The expected screened trajectory contains exactly one persistent -x relocation near 2826 fs. The detected event time must be taken from the new run rather than hard-coded.

Save the complete sampled current trajectory for posthoc reanalysis.

## Post-hop residence frame

Let the single event be A -> B at `t1`, with nearest-neighbor displacement `dx = +/-1` and `dy = 0`.

Construct a fixed post-hop residence axis B -> C, where C is the next nearest neighbor in the same x direction. The carrier therefore resides at the frame origin B after the event; +s follows carrier motion and -s is the trailing/retrograde direction.

This convention matches the B-residence frame used in IP1l and prevents a source-centered frame from drifting one lattice site behind the post-hop carrier.

Primary backward boundaries: d=1, d=2 and d=3. d=4 is diagnostic because the available 2.17 ps post-hop interval approaches the harmonic travel time to four sites.

## Field-only reference

Use the same B-centered spatial frame before the first hop.

Reference interval:

- `t1 - 700 fs` to `t1 - 100 fs`.

This contains the deterministic field-driven lattice background but no prior persistent x relocation. Subtract its mean longitudinal current profile from all post-hop profiles.

For each distance d=1..4, construct empirical field-only distributions from 100 fs windows inside the reference interval, sampled every 20 fs.

Pseudo/background windows are correlated and are used only for robustness percentiles, not formal p-values.

## First-hop packet test

At d=1 and d=2, compute the backward normalized-delay correlation in the first 800 fs after `t1`.

Search lags in 300-700 fs. A packet is qualified when:

- correlation >= 0.80;
- the best lag is internal to the search interval;
- inferred speed <= `1.05 * v_harmonic,max`.

This is the same refined natural-event gate used in IP1k/IP1l. The earlier IP1j binary isotropic result remains unchanged; IP1m is a new B-residence-frame calculation.

## Long-lived memory test

Partition the post-hop interval from `t1` to `final_time - 100 fs` into non-overlapping 100 fs bins.

For the backward d=2 boundary record per bin:

1. positive outward energy;
2. signed net outward energy;
3. maximum outward flux;
4. mean outward flux;
5. percentile of positive energy relative to the pre-hop field background;
6. percentile of peak flux relative to the same background.

A bin is jointly background-separated when both amplitude percentiles exceed 95.

### Late-memory gate

`late_memory_present = true` when at least one jointly separated bin has center more than 1000 fs after `t1`.

### Sustained late-memory gate

`sustained_late_memory = true` only when:

- at least three late bins are jointly background-separated; and
- at least one jointly separated bin has center >= `t1 + 1500 fs`.

Also report the number/fraction of late passing bins and the latest passing center.

The stronger sustained gate is the criterion used to promote the interpretation from a late fluctuation to a long-lived single-hop memory.

## Spatial-envelope diagnostic

At d=1,2,3 and diagnostic d=4:

- compute 100 fs post-hop bins with the same field-background subtraction;
- identify the first bin whose positive outward energy and peak outward flux both exceed their distance-matched 95th-percentile backgrounds;
- record first-separated-bin center, maximum post-hop positive energy, maximum peak flux and latest separated-bin center.

If d=1..3 all have ordered finite first-arrival centers, fit arrival time versus distance and report the corresponding envelope speed. This is a coarse wake-envelope diagnostic, not a unique normal-mode group velocity and not a numerical acceptance gate.

## Numerical gates

Before physical interpretation:

1. pycompile passes;
2. focused IP1m/IP1l/IP1k/IP1j/IP1h/D3/TP1 tests pass;
3. full pytest passes;
4. 40x40 zero-damping recurrence preflight passes for 5 ps;
5. static relaxation converges;
6. all requested integration steps execute;
7. size-aware field-work balance passes;
8. electronic norm and projected intermolecular zero modes remain controlled;
9. exactly one persistent nearest-neighbor x event is present;
10. all packet, memory and spatial-envelope scalars are finite where defined;
11. the complete sampled current trajectory is serialized.

Physical packet/memory outcomes are not numerical acceptance gates.

## Interpretation guards

- The absence of a second persistent hop removes second-hop-event contamination but does not turn the calculation into a field-free phonon-lifetime measurement; the field remains on.
- The pre-hop reference subtracts the same deterministic field background, but nonlinear interaction between the post-hop lattice state and the field can remain.
- A delayed oscillatory signal is lattice memory even if it is not one compact packet continuously attached to the carrier.
- Spatial-envelope speeds are diagnostic propagation scales, not calibrated material group velocities.
- No mobility, hopping rate, activation energy, threshold field or transport coefficient is extracted.

## Decision after IP1m

If the first-hop packet and sustained late-memory gates pass, long-lived natural retrograde lattice memory is established in the isotropic one-hop trajectory without any subsequent persistent relocation. The next stage should then quantify whether near-carrier wake amplitude leads or follows incipient electronic relocation, before attempting any causal claim about hopping timing.

If the packet passes but sustained late memory fails, the natural isotropic system supports retrograde emission but not the long-lived memory observed in the anisotropic two-hop sequence.

If the packet itself fails under the refined B-residence protocol, the isotropic retrograde branch remains threshold-sensitive and should not be promoted beyond the coherent but ambiguous IP1j/IP1k evidence.