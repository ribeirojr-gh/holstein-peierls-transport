# IP0a — frozen one-site translation-barrier diagnostic

## Scientific question

The earlier anisotropic Holstein-Peierls calculations found a mobile quasi-1D/intermediate polaron but an essentially immobile two-dimensional polaron as the transfer network approached isotropy. The current hypothesis is that the apparent immobility reflects a large lattice-reorganization/pinning barrier rather than a fundamental prohibition of transport.

IP0a asks a deliberately limited first question:

> Which terms of the existing Holstein-Peierls Hamiltonian make a one-site translation of the relaxed polaron expensive, and how do those terms change as `J0y/J0x` approaches unity?

## Why IP0a is not yet the activation barrier

The relaxed lattice distortion is translated by exactly one nearest-neighbour site. The classical coordinates between the two equivalent endpoints are then **linearly interpolated**. At every image, only the electronic ground state is re-solved.

Therefore the maximum energy along this path is an **upper-bound frozen-path diagnostic barrier**. It is not:

- a minimum-energy-path (MEP) barrier;
- a Peierls-Nabarro barrier obtained from a constrained relaxation;
- a finite-temperature free-energy barrier;
- a hopping activation energy;
- a mobility.

A constrained/NEB-like calculation is reserved for IP0b after IP0a identifies a stable path representation and the dominant energy terms.

## Frozen model parameters

Except for `J0y`, IP0a retains the current static one-polaron defaults:

- `J0x = 0.100 eV`;
- `alpha_intra = 3.0 eV/A`;
- `alpha_interx = alpha_intery = 0.4 eV/A`;
- `K1 = 16.51 eV/A^2`;
- `K2 = 0.51 eV/A^2`;
- 20x20 periodic lattice;
- sparse electronic ground-state solver;
- optimized static gradient;
- full three-coordinate static convergence.

The transfer anisotropy scan is

`J0y/J0x = 0.15, 0.30, 0.50, 0.70, 1.00`.

This isolates the change in bare-transfer anisotropy. It does **not** claim that a real material can be made isotropic by changing only one parameter.

## Two translation directions

For every relaxed static state, two one-site paths are evaluated:

- `+x`;
- `+y`.

For the isotropic endpoint (`J0y/J0x = 1`) the two profiles should be physically equivalent in the ideal square model, apart from numerical/localization branch effects. For anisotropic controls they need not be equivalent.

No equality or monotonic trend is imposed as a numerical gate; those are physical diagnostics to inspect after execution.

## Energy decomposition

At every image, the physical energy is decomposed into

1. intramolecular elastic energy;
2. x intermolecular elastic energy;
3. y intermolecular elastic energy;
4. Holstein coupling energy;
5. bare x transfer contribution;
6. bare y transfer contribution;
7. x Peierls coupling contribution;
8. y Peierls coupling contribution.

The sum is checked against the direct electronic eigenvalue plus lattice energy. At the highest-energy image, IP0a records the change of every component relative to the average of the two equivalent endpoints.

This is intended to distinguish, for example, whether an increasing translation cost is primarily associated with the intramolecular deformation, the two-dimensional intermolecular distortion, loss of electronic delocalization energy, or cancellation among several terms.

## Charge diagnostics

For each static endpoint and each path image IP0a records:

- IPR;
- participation number;
- maximum molecular population;
- source and target populations;
- `n_source - n_target`;
- site of maximum population.

The actual maximum-density site of the relaxed endpoint is used as the source diagnostic, rather than assuming that the sparse self-consistent relaxation remained exactly on the initially seeded molecular index.

## Numerical gates

IP0a requires only implementation consistency:

1. every static relaxation reports convergence;
2. translated endpoints have equal physical energy within `1e-8 eV`;
3. the explicit energy decomposition reproduces the electronic eigenvalue within `1e-9 eV`;
4. electronic norm error remains below `1e-10`;
5. all path barriers are finite and non-negative within numerical tolerance.

No barrier magnitude, monotonic anisotropy trend, localization threshold, saddle location, or x/y relation is pre-registered as a physical PASS criterion.

## Decision after IP0a

If the numerical gates pass, the result is inspected manually.

The main questions are:

- Does the frozen-path barrier grow as the model approaches isotropy?
- Is the isotropic profile qualitatively different from the anisotropic one?
- Which energy terms dominate the additional cost?
- Does the charge expand/share between neighbouring sites near the highest-energy image?
- Are x and y barriers equivalent at `J0y/J0x = 1`?
- Does the static isotropic solution remain localized enough that a translated-polaron barrier is meaningful?

IP0b will then replace the frozen interpolation with constrained relaxation / minimum-energy-path machinery. Only after a relaxed barrier is established will a temperature scan test whether thermally activated hopping over that barrier occurs.
