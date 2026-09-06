# IP1p — periodic-boundary phonon recurrence audit

## Motivation

Previous simulations of this model showed that dynamic lattice excitations can be emitted opposite to the carrier motion and, under periodic boundary conditions, later return and collide with the carrier.  Such a recollision is a finite-cell artifact for sufficiently long trajectories.  Earlier IP1a–IP1c analyses treated electronic transport observables correctly under PBC but did not promote lattice-wave recurrence to an explicit validation gate.

IP1p is inserted before IP1d so that long-time mechanistic conclusions are not drawn from a trajectory window that can contain a wrapped lattice excitation.

## Harmonic propagation scale

Away from the carrier, the intermolecular coordinates obey

M_v q_ddot_n = -K_2 (2 q_n - q_{n-1} - q_{n+1}).

The undamped dispersion is

omega(k) = 2 sqrt(K_2/M_v) |sin(k/2)|,

so the maximum undamped group velocity is

v_g,max = sqrt(K_2/M_v) sites/fs.

For the current K_2=0.51 eV/A^2 and M_2=1.5e11 eV as^2/A^2 (=1.5e5 eV fs^2/A^2),

v_g,max = 1.8439e-3 sites/fs = 1.8439 sites/ps.

With a=3 A this is about 5.5317 A/ps.  A 20-site periodic direction therefore has a stationary-carrier full-wrap scale of about 10.8465 ps.  A 40-site direction doubles that scale to about 21.6930 ps.

This is a protocol diagnostic, not a rigorous collision time.  A moving carrier can alter the relative encounter time, wave packets are dispersive, and not every emitted mode travels at v_g,max.

## Langevin damping

The production thermal dynamics currently use gamma_v=0.01 fs^-1.  The largest harmonic intermolecular angular frequency is

omega_max = 2 sqrt(K_2/M_v) = 3.6878e-3 fs^-1.

For q_ddot + gamma q_dot + omega_0^2 q = 0, all harmonic modes are overdamped when gamma >= 2 omega_max.  Here 0.01 > 7.3756e-3 fs^-1, so the current numerical bath overdamps all non-zero harmonic intermolecular modes in the linear limit.

This strongly reduces the likelihood of a coherent ballistic phonon packet surviving a full 20-site traversal.  However, gamma is a numerical thermostat parameter rather than a calibrated material phonon lifetime.  Therefore the bath must not be used as the sole justification for ignoring recurrence: it can suppress both the finite-size artifact and real phonon memory.

## Protocol decision before IP1d

IP1d is a local precursor-mechanism test, not a kinetic-rate calculation.  Its production trajectory is therefore shortened from 20 ps to 10 ps on the 20x20 cell.  With a +/-500 fs event window, accepted complete events occur no later than 9.5 ps, before the 10.8465 ps stationary-carrier wrap scale.

This is a conservative practical improvement, not proof that all moving-carrier finite-size effects are absent.  Before any long-time hopping rate, diffusion coefficient or mobility is extracted, the project must add at least one explicit finite-size/damping sensitivity control, preferably a larger cell (e.g. 40x40) and a physically motivated gamma/equilibration study.

## Numerical checks

IP1p unit tests verify the harmonic dispersion scales, linear scaling of the wrap time with cell length, the current overdamped classification, and the event-window cutoff construction.  No physical hopping or precursor threshold is imposed.
