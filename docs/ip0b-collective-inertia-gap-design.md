# IP0b — collective translation inertia and midpoint electronic gap

## Motivation from IP0a

IP0a passed all numerical gates and produced a surprising physical result: the frozen +x one-site translation barrier remains approximately 11--13 meV across `J0y/J0x = 0.15 ... 1.00`. The +y barrier decreases from approximately 102 meV in the strongly anisotropic control to approximately 12 meV at isotropy. Therefore the previously observed immobility of the isotropic 2D polaron is not explained by a larger static frozen-path potential barrier.

This motivates a direct test of the earlier **effective-mass / lattice-inertia** interpretation.

## Scientific questions

IP0b asks:

1. How much mass-weighted lattice deformation must move when the polaron is translated by one site?
2. Does the relative contribution of `u`, `vx`, and `vy` change as isotropy is approached?
3. Does the harmonic frequency scale of the collective translation coordinate become smaller toward isotropy?
4. Is the lowest electronic state nearly degenerate with the first excited state at the symmetric path midpoint, indicating a weak avoided crossing or branch-selection issue?

## Reaction-coordinate mass metric

For a path

`q(s) = q_A + s (q_B-q_A)`, `0 <= s <= 1`,

the lattice kinetic energy associated with motion only along this path is

`T = 1/2 * Lambda * (ds/dt)^2`,

with

`Lambda = M_u sum(Delta u)^2 + M_v sum(Delta vx)^2 + M_v sum(Delta vy)^2`.

`Lambda` has units eV fs^2 and is reported together with its three components. The validated legacy masses are converted exactly as in the production dynamics:

- `M_u = 75000 eV fs^2/A^2`
- `M_v = 150000 eV fs^2/A^2`.

This is a **path-specific lattice collective inertia**, not a complete polaron band effective mass. It intentionally excludes any additional electronic quantum metric contribution.

## Frozen-path harmonic scale

IP0b repeats the IP0a frozen translation profile with 21 images. A local quadratic fit to the first/last five images estimates the endpoint curvature `d^2E/ds^2`.

When the fitted mean endpoint curvature is positive, IP0b reports

`omega_coll = sqrt(curvature/Lambda)`

and `nu_coll = omega_coll/(2*pi)`.

This is a harmonic timescale of the chosen frozen coordinate. It is **not** a Kramers/Marcus hopping prefactor and no hopping rate is inferred from it.

## Midpoint electronic spectrum

At `s=0.5`, the lowest four eigenvalues of the dense 20x20 one-particle Hamiltonian are calculated with a deterministic Hermitian eigensolver. IP0b records in particular

`Delta_01 = E_1 - E_0`.

This addresses an IP0a warning: for low-anisotropy +x paths the sparse ground-state solution at the symmetric midpoint can remain localized on one of two equivalent sites. An extremely small `Delta_01` would show that this is a near-degenerate electronic doublet rather than a true asymmetric lattice energy profile.

The midpoint gap is not a Landau-Zener transition probability or a nonadiabatic hopping rate.

## Frozen controls

- 20x20 periodic lattice
- `J0x = 0.100 eV`
- `J0y/J0x = 0.15, 0.30, 0.50, 0.70, 1.00`
- `alpha_intra = 3.0 eV/A`
- `alpha_interx = alpha_intery = 0.4 eV/A`
- `K1 = 16.51 eV/A^2`
- `K2 = 0.51 eV/A^2`
- sparse solver for static relaxation and path ground states
- dense lowest-four-state solver only at the path midpoint
- optimized static gradient
- full static convergence
- translation directions `+x` and `+y`
- 21 frozen-path images
- five-point endpoint quadratic fit.

## Numerical gates

IP0b requires:

1. every static relaxation converges;
2. all mass metrics are finite and positive;
3. all midpoint first gaps are finite and non-negative within `1e-12 eV` numerical tolerance;
4. all fitted endpoint curvatures are finite (their sign is a physical/path diagnostic, not a numerical PASS condition);
5. translated endpoint energy mismatch remains below `1e-8 eV`;
6. explicit energy decomposition error remains below `1e-9 eV`;
7. electronic norm error remains below `1e-10`.

No monotonic inertia trend, frequency trend, midpoint-gap trend, barrier magnitude, or isotropic/anisotropic conclusion is required for PASS.

## Decision after IP0b

If the mass metric grows strongly and the collective harmonic scale decreases toward isotropy, the effective-inertia hypothesis gains direct support. The component decomposition will identify whether the added inertia comes mainly from the intramolecular deformation, transverse intermolecular distortion, or both.

If inertia does not grow appreciably, the next priority becomes the electronic avoided-crossing/nonadiabatic structure and then a constrained MEP/free-energy analysis.

If both a shallow potential barrier and a moderate collective inertia are found, then the absence of hopping in earlier few-ps simulations is more likely a limitation of dynamical sampling or the semiclassical propagation/decoherence treatment than a true static pinning mechanism.

No MEP, finite-temperature activation energy, hopping rate, mobility, or material calibration is part of IP0b.
