# IP1h — fixed-boundary lattice-radiation flux reanalysis

## Purpose

IP1h is a no-dynamics posthoc reanalysis of the completed IP1g deterministic quench trajectories. Its purpose is to establish whether a propagating lattice-radiation packet can be tracked toward `-s` and/or `+s` using a quantity with a direct continuity interpretation: the harmonic intermolecular energy current crossing a fixed longitudinal boundary.

This stage does not infer a natural hopping rate, activation energy, diffusion coefficient, mobility, material phonon lifetime, or a unique normal-mode group velocity.

## Why IP1h is needed

The original IP1f/IP1g half-space current summary sums local current over an extended region. Such a sum is not the flux through a surface and can mix packet extent with propagation direction. Also, the positive-excess centroid used in IP1g can move because the profile reshapes, even when no single packet travels at that fitted slope.

IP1h therefore supersedes those two quantities for propagation claims.

## Fixed-boundary convention

The hop/launch direction is `+s`. For a boundary at distance `d` sites from the launch center:

- backward outward flux is `-j_parallel(s=-d,t)`;
- forward outward flux is `+j_parallel(s=+d,t)`.

The positive outward energy crossing a boundary is

`E_out^+(d) = integral max(J_out(d,t),0) dt`,

while the signed net transported energy is

`E_out^net(d) = integral J_out(d,t) dt`.

At equal distance the directionality index is

`D(d) = [E_back^+(d) - E_front^+(d)]/[E_back^+(d) + E_front^+(d)]`.

Positive `D` means more lattice energy crossed the backward boundary.

## Propagation-delay diagnostic

Boundaries `d=2,3,4,5,6` are used. For each side, the outward-current trace at `d+1` is cross-correlated with the trace at `d`. The best positive lag within 300–1000 fs gives a one-site packet delay.

The corresponding speed is

`v_packet = 1 site / lag`.

Each pair is screened against two conditions:

1. normalized overlap correlation >= 0.80;
2. speed <= 1.05 times the harmonic maximum group velocity.

The second condition is a physical consistency screen, not a fitted material parameter. A packet that fails it is not used in the validated median speed, but its failure is not a numerical failure of the trajectory.

## Expected interpretation

A consistent sequence of positive lags with high correlation and speeds below the harmonic maximum establishes a real propagating lattice-current packet on that side of the launch. If both sides pass, the impulse radiates bilaterally; directionality is then decided by the boundary-integrated energies rather than by the mere existence of a packet.

Because IP1g imposes the electronic relocation, even a strong backward packet remains a mechanistic radiation diagnostic. A final claim about carrier-generated backward phonons requires the same observables in a genuinely moving carrier trajectory, preferably field-driven and before PBC recurrence.
