# Static hopping-isotropy validation

## Motivation

The validated bipolaron calculations up to the extended-Hubbard `U + V1` stage use a strongly anisotropic hopping set,

`Jx = 0.100 eV`, `Jy = 0.015 eV`,

so `Jy/Jx = 0.15`. This anisotropy is essential for transport, but it also affects the stationary kinetic energy, localization length, orientation selection, and competition between onsite, intersite, and separated two-particle states. The isotropic stationary limit must therefore be validated before the extended-Hubbard branch is merged.

## Separation of anisotropy and bandwidth

For the square nearest-neighbour model the rigid-band width is

`W = 4 * (Jx + Jy)`.

The primary anisotropy trajectory keeps

`Jx + Jy = 0.115 eV`

fixed while varying

`eta = Jy/Jx`.

The pilot values are

`eta = 0.15, 0.25, 0.50, 0.75, 1.00`.

For each eta,

`Jx = 0.115 / (1 + eta)` and `Jy = eta * Jx`.

Thus `eta = 0.15` reproduces the current hopping pair and `eta = 1` gives the bandwidth-matched isotropic control

`Jx = Jy = 0.0575 eV`.

A second endpoint control uses

`Jx = Jy = 0.100 eV`,

which is isotropic but has a larger total bandwidth. This distinguishes the effect of isotropy from the effect of changing the kinetic-energy scale.

## Two isotropic controls

Control A isolates hopping isotropy:

- `Jx = Jy = 0.0575 eV`;
- `alpha_x = 0.10 eV/A`;
- `alpha_y = 0.12 eV/A`.

Control B makes the x/y Hamiltonian fully isotropic:

- `Jx = Jy = 0.0575 eV`;
- `alpha_x = alpha_y = 0.10 eV/A`.

For a square cell with isotropic `V1`, Control B must be invariant under a 90-degree rotation. In particular, x- and y-oriented pair seeds must be energetically degenerate within numerical precision after the corresponding rotation.

## Competing branches

The exploratory scan relaxes five lattice seeds independently:

- onsite;
- intersite-x;
- intersite-y;
- diagonal `(1,1)`;
- separated.

The diagonal seed is included because nearest-neighbour `V1` penalizes axial first neighbours but not a diagonal pair directly, and isotropy removes the strong x-direction selection present in the baseline model.

Final states are labeled from pair probabilities rather than seed names. In addition to onsite and nearest-neighbour probabilities, the experiment records the probability for minimum-image diagonal separation `|dx| = |dy| = 1`.

## Pilot calculation

The first stage uses `10x10`, `V1 = 0`, and both `U = 0.525` and `1.000 eV`. The primary trajectory uses fixed `Jx + Jy = 0.115 eV`. The two isotropic endpoints described above are evaluated separately, including the high-bandwidth isotropic control.

Recorded observables include total and binding energies, `P_onsite`, `P_NN,x`, `P_NN,y`, diagonal probability, mean and RMS pair separation, one-body IPR, Peierls hopping-modulation ratios, iteration count, and residual gradient.

## Decision rule

The 10x10 scan determines whether isotropy changes the phase topology and identifies the relevant competing branches. Only the resulting isotropic boundaries or representative states will be promoted to 20x20 and then 40x40 strict validation. No full 40x40 anisotropy map is planned unless the pilot reveals additional phases that require it.

The extended-Hubbard PR remains open until this isotropy study is complete.
