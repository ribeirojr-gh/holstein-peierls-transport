# IP1c retrospective phonon-recurrence audit

Date: 2026-09-06

## Motivation

A prior-model observation was added after IP1c: dynamic lattice excitations can be emitted opposite to the carrier and later wrap through periodic boundaries, colliding with the carrier in a finite cell.  IP1c used 20x20 PBC and 20 ps trajectories, so its full production interval extends beyond the conservative stationary-carrier harmonic full-wrap scale of about 10.8465 ps for the current K2 and M2.

The original IP1c artifact was therefore re-analyzed without rerunning dynamics.  For the +/-500 fs event-conditioned window, a conservative early subset retained only events with transition start <= 10.3465 ps, so the complete post-event edge occurs before 10.8465 ps.

This cutoff is a stationary-carrier ballistic diagnostic, not proof of absence of finite-size interactions for a moving carrier.

## Early-subset comparison

| J0y/J0x | T [K] | all complete events | early events | <Delta Q_L> early | lattice sign reversal early | current + fraction +/-100 fs early | future bond top-1 -100:-80 fs early | future bond top-1 -20:0 fs early |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | 100 | 84 | 38 | 0.0925 | 0.289 | 0.921 | 0.745 | 0.636 |
| 1.00 | 300 | 110 | 48 | 0.1614 | 0.292 | 0.958 | 0.770 | 0.725 |
| 1.00 | 500 | 142 | 60 | 0.1521 | 0.233 | 0.883 | 0.641 | 0.655 |
| 0.15 | 300 | 131 | 55 | 0.1430 | 0.255 | 0.982 | 0.775 | 0.890 |

The central IP1c qualitative signatures remain present in the early subset: the lattice source-target coordinate shifts toward the electronic target on average; the +/-100 fs TP1 current is positive along the accepted electronic relocation for about 88–98% of early events; and the future hop bond is already top-ranked in roughly 64–78% of the -100:-80 fs samples, well above the naive four-bond 0.25 reference.

Some pooled fractions change between early and late subsets, so the 20 ps aggregate numbers should not be treated as finite-size-clean kinetic observables.  In particular, IP1c remains a mechanistic screen, not a hopping-rate, activation-energy, diffusion, or mobility result.

## Langevin caveat

The current thermal runs use gamma_v=0.01 fs^-1.  For the harmonic intermolecular lattice omega_max is about 3.6878e-3 fs^-1, so gamma_v > 2 omega_max and all non-zero harmonic intermolecular modes are overdamped in the linear limit.  This makes a coherent ballistic wrapped phonon less likely in the current production protocol, but gamma is a numerical bath parameter rather than a calibrated material phonon lifetime.  Strong damping may suppress both the finite-size recurrence artifact and genuine physical phonon memory.

## Decision

Do not discard IP1a–IP1c, but do not use their late-time 20 ps event counts for kinetics.  IP1d is shortened to 10 ps and now includes an explicit IP1p recurrence preflight.  Before any long-time transport coefficient is inferred, perform lattice-size and damping sensitivity controls.
