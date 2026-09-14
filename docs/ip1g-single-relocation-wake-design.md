# IP1g — controlled single-relocation lattice-radiation design

## Motivation

IP1f numerically validated the carrier-centered lattice-energy and harmonic Peierls energy-current observables and found a suggestive early retrograde energy-current bias in the weakly damped isotropic case. A causal backward-propagating packet could not yet be claimed because every thermal event lay within 100 fs of an IDC event and no complete 2 ps wake window was isolated from neighbouring nearest-neighbour electronic relocations.

IP1g removes those two confounders by replacing natural thermal hopping with one controlled charge-relocation quench.

## Protocol

1. Relax a zero-temperature one-polaron state on a 40x40 periodic lattice.
2. At t=0, translate only the electronic wavefunction by one site toward +x.
3. Leave the relaxed lattice distortion and zero lattice velocity unchanged.
4. Propagate deterministic zero-field Ehrenfest dynamics for 5 ps with dt=0.2 fs and CF4-Lanczos (m=6).
5. No thermostat, stochastic force, IDC or external field is present after the quench.
6. Compare J0y/J0x = 1.0 and 0.15.

The quench is intentionally non-equilibrium. It is a clean re-dressing/radiation impulse experiment, not a natural hopping event.

## Observables

The validated IP1f observables are reused without changing the dynamics:

- local classical lattice-energy density;
- longitudinal harmonic Peierls energy current;
- front/back excess lattice energy relative to the t=0 lattice profile;
- backward and forward outward energy flux;
- longitudinal versus transverse positive excess intermolecular energy;
- centroid of positive excess energy behind the imposed relocation;
- source and target electronic populations;
- deterministic total-energy conservation, norm and intermolecular zero modes.

All coordinates are rotated so the imposed charge relocation is +s. Therefore negative longitudinal energy current and negative wake-centroid slope are retrograde relative to the launch direction.

## Interpretation hierarchy

A positive backward outward flux is the primary directional diagnostic. A moving centroid with negative slope is secondary and should be called a descriptive wake velocity unless a clean space-time ridge and spectral dispersion are subsequently demonstrated.

If IP1g produces a clean backward packet, a later spectral stage can test whether its `(q, omega)` support follows the harmonic Peierls dispersion and whether a Cherenkov-like phase-matching interpretation is justified. No such label is used in IP1g.

## Numerical gates

- full test suite PASS;
- 40x40 recurrence preflight PASS for the 5 ps observation window;
- all static relaxations converge;
- deterministic total-energy residual obeys the existing size-aware extensive gate;
- electronic norm error < 1e-10;
- uniform intermolecular coordinate/velocity modes remain < 1e-10;
- wake diagnostics remain finite.

Physical wake direction, transverse fraction and centroid slopes are never numerical PASS gates.
