# IP0b local validation and physical interpretation — 2026-09-06

## Status

IP0b is numerically CLOSED.

The local validation archive `20260906T100601Z` reports:

- Python 3.12.3 on WSL2;
- NumPy 2.5.2, SciPy 1.18.1;
- `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`;
- ZIP provenance, therefore git commit/branch/status recorded as `unknown` locally;
- pycompile: PASS;
- focused tests: 19/19 PASS;
- full test suite: 351 passed;
- IP0b benchmark: PASS;
- all seven numerical checks: PASS.

The scope remains a path-specific lattice reaction-coordinate inertia and midpoint electronic spectrum. It is not a complete quasiparticle effective mass, MEP, hopping rate, activation energy, or mobility.

## Numerical integrity

Aggregate extrema from the archive:

- minimum collective mass metric: 8.5642543e4 eV fs^2;
- maximum collective mass metric: 4.9623796e5 eV fs^2;
- midpoint first-gap range: 56.034–352.795 meV;
- endpoint curvature range: 0.169922–0.993780 eV along the dimensionless frozen translation coordinate;
- maximum endpoint energy mismatch: 3.33e-16 eV;
- maximum explicit-decomposition error: 9.99e-16 eV;
- maximum electronic norm error: 6.66e-16.

## Main physical result

For translation along the easy `+x` direction, approaching transfer isotropy increases the path-specific lattice inertia substantially while leaving the frozen-path barrier nearly unchanged:

| J0y/J0x | barrier +x [meV] | mass metric +x [eV fs^2] | harmonic scale [THz] | period [ps] |
|---:|---:|---:|---:|---:|
| 0.15 | 13.046 | 8.564e4 | 0.2406 | 4.157 |
| 0.30 | 12.497 | 8.983e4 | 0.2300 | 4.349 |
| 0.50 | 11.311 | 9.903e4 | 0.2085 | 4.797 |
| 0.70 | 11.356 | 1.584e5 | 0.1649 | 6.063 |
| 1.00 | 12.014 | 2.110e5 | 0.1467 | 6.816 |

Thus, from ratio 0.15 to 1.00, the +x inertia grows by about 2.46x and the local harmonic attempt-frequency scale falls by about 39%, while the frozen barrier stays near 11–13 meV.

This supports the idea that the loss of transport on approaching isotropy cannot be explained by a growing static potential barrier alone. A dynamically heavier lattice reorganization coordinate is a plausible contributor.

However, the result is only partial support: the change in this one path-specific prefactor scale is not yet sufficient to explain a complete transport shutdown, and it must not be called the full polaron effective mass.

## Redistribution of the lattice inertia

The mass composition changes strongly with anisotropy.

For +x translation at J0y/J0x=0.15, about 95.7% of the metric comes from the `vx` field and only about 3.1% from `vy`. At isotropy, the same +x path contains only about 17.2% `vx` but about 82.3% `vy` contribution. The intramolecular `u` contribution is small in this particular geometric mass norm.

The isotropic +x and +y results are symmetry-equivalent within numerical precision, with the `vx` and `vy` fractions interchanged. This is an important internal symmetry check.

The large transverse contribution confirms that isotropic translation reorganizes lattice distortions in both directions even when the net hop is along only one axis.

## Midpoint electronic spectrum

The IP0a suspicion that asymmetric midpoint populations at strong anisotropy could be caused by a nearly degenerate lowest electronic doublet is not supported.

The first adiabatic electronic gap at the frozen midpoint is:

- +x, ratio 0.15: 352.795 meV;
- +x, ratio 0.30: 325.526 meV;
- +x, ratio 0.50: 291.306 meV;
- +x, ratio 0.70: 252.395 meV;
- isotropic: 211.509 meV.

These are not near-degenerate levels. The asymmetric source/target populations seen at the anisotropic frozen midpoint are therefore a property of the frozen interpolated Hamiltonian/path, not an eigensolver choice inside an almost degenerate ground-state subspace.

For the +y paths the midpoint source/target populations are already symmetric over the scan, while the first gap rises from about 56 meV at ratio 0.15 to about 212 meV at isotropy.

## Scientific interpretation

IP0a + IP0b imply:

1. the isotropic system does not acquire a large frozen one-site potential barrier;
2. the easy-axis lattice reaction coordinate becomes substantially heavier as isotropy is approached;
3. the transverse Peierls distortion becomes a dominant part of an x-directed translation at isotropy;
4. the frozen midpoint is not generally the relaxed hopping transition configuration;
5. electronic near-degeneracy is not the origin of the anisotropic midpoint asymmetry.

The next required calculation is therefore a transversely relaxed/constrained translation profile. It must allow all lattice degrees of freedom orthogonal to a controlled translation coordinate to relax, while retaining the two exactly equivalent translated endpoints. This will determine whether the approximately 12 meV frozen isotropic barrier survives relaxation and will provide a better reaction path for a subsequent finite-temperature hopping study.

## Claims explicitly not made

IP0b does not establish:

- a complete effective mass tensor;
- a Peierls-Nabarro or MEP activation barrier;
- an Arrhenius prefactor;
- a hopping rate;
- thermally activated isotropic transport;
- a mobility.

Those remain downstream validation targets.