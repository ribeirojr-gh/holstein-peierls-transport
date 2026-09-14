# IP1i — field-driven transport screen before wake analysis

## Purpose

IP1h proved that the controlled IP1g relocation launches propagating lattice-energy packets and that the anisotropic control develops a strongly retrograde long-range packet. The missing step is to establish a **self-consistent moving-polaron protocol** before attaching the wake diagnostic to natural field-driven dynamics.

IP1i is therefore deliberately a transport-screen stage, not a wake stage. It asks which deterministic zero-temperature field/anisotropy controls generate an unambiguous persistent carrier displacement on a 40x40 lattice while preserving the already validated D3 energy-work accounting.

No mobility, hopping rate, activation energy, diffusion coefficient, or material field threshold is inferred.

## Protocol

- lattice: 40x40 PBC;
- anisotropy ratios J0y/J0x = 1.0 and 0.15;
- field along +x: 2, 5 and 10 mV/A;
- T = 0 K;
- thermostat: none;
- IDC/decoherence: none;
- dt = 0.2 fs;
- final time = 5 ps;
- TP1 probability-current displacement integrated every time step;
- persistent residence tracker sampled every 2 fs with 50 fs persistence;
- projected uniform vx/vy zero modes in the initial state;
- CF4-Lanczos, Krylov dimension 6.

The fields are numerical screening controls and are not material calibrated.

## Numerical gates

Each trajectory must satisfy:

1. converged static initial polaron;
2. complete requested propagation;
3. electronic norm error < 1e-10;
4. projected intermolecular zero-mode mean < 1e-10;
5. D3 generalized work balance within the size-scaled numerical tolerance;
6. finite TP1 displacement and persistent-event diagnostics.

Failure to move is **not** a numerical failure. Transport qualification is recorded separately.

## Transport qualification

For each trajectory record:

- total TP1 displacement x/y;
- mean x velocity over the complete run;
- persistent nearest-neighbour event count;
- first-event time and direction;
- fraction of nearest-neighbour events in +/-x versus +/-y;
- median gap between nearest-neighbour events;
- longest quiet interval after a nearest-neighbour event;
- dominant-site net minimum-image displacement as a secondary residence diagnostic.

A trajectory is marked `transport_qualified` for the next wake stage when all of the following hold:

- at least two persistent nearest-neighbour events;
- |TP1 displacement_x| >= 6 A (two molecular spacings);
- >= 70% of persistent nearest-neighbour events lie along x;
- the sign of the net persistent x displacement agrees with the TP1 displacement sign.

These thresholds select a mechanically interpretable moving control; they do not define a transport coefficient.

## Decision after IP1i

The next stage will use the weakest field that qualifies robustly for the anisotropic control, and any isotropic field that qualifies without introducing obviously different high-field dynamics. The fixed-boundary lattice-energy flux diagnostic will then be evaluated relative to the actual carrier direction.
