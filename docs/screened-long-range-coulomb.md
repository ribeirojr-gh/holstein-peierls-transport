# Screened long-range Coulomb model

## Scope

This document defines the first controlled long-range repulsion extension of the static singlet bipolaron solver. It is intentionally model-general and does not assign pentacene-specific lattice spacings or dielectric constants by default.

The stable `v0.4.0a1` interaction sector is recovered exactly when `long_range_coulomb = False`.

## Interaction definition

The onsite interaction remains the independent Hubbard parameter

`V_ii = U`.

For distinct sites, the opt-in continuum tail is

`V_ij = e^2 / (4 pi epsilon_0 epsilon_r r_ij)`,

with

`e^2 / (4 pi epsilon_0) = 14.3996454784255 eV A`.

The physical distance is constructed from explicit lattice spacings,

`r_ij = sqrt[(a_x dx_ij)^2 + (a_y dy_ij)^2]`,

where `dx_ij` and `dy_ij` are minimum-image separations on the finite periodic lattice. Enabling the model therefore requires positive explicit values for `a_x`, `a_y`, and `epsilon_r`; no implicit unit lattice or vacuum dielectric is allowed.

## Short-range replacement and double counting

Continuum dielectric screening is least reliable at molecular-contact distances. The generalized-Hubbard literature also treats onsite and short-range intermolecular matrix elements as effective screened parameters rather than as literal point-charge Coulomb values.

Accordingly, if `nearest_neighbor_v` is nonzero while the long-range tail is enabled, the four cardinal nearest-neighbour values are **replaced** by `V1`:

`V_NN = V1`,

not

`V_NN = V1 + V_Coulomb`.

This makes `V1` a short-range screened override and avoids implicit double counting. If `V1 = 0`, the continuum expression is used at nearest-neighbour distance as well.

The current scalar `V1` overrides x and y nearest neighbours equally. Direction-dependent short-range parameters can be added later if material-specific screening requires them.

## Periodic-boundary convention

The first implementation uses a **minimum-image finite-cell** Coulomb model. It is not an Ewald sum.

A fully periodic Ewald sum for two equal-sign carriers is not uniquely defined without an explicit charge-neutralization convention. Introducing a uniform compensating background would create a different physical model and cell-dependent terms. That choice is therefore deferred rather than hidden inside the interaction routine.

The consequence is that two carriers placed at the largest available separation in a finite cell still experience a positive residual repulsion of order `1/L`. This residual must not be interpreted as a binding contribution.

## Dissociation reference

With a long-range tail, the preferred infinite-separation reference is

`E_diss = 2 E_polaron`,

where `E_polaron` is the relaxed one-carrier energy for the same one-particle lattice parameters.

A two-particle `separated` branch in a finite periodic cell remains useful as a finite-size diagnostic, but its energy contains the residual minimum-image Coulomb repulsion. The difference

`E_separated(L) - 2 E_polaron`

should decrease toward zero with increasing cell size. This becomes an explicit validation target for the long-range model.

## Frozen-distance approximation

In this first stage, `r_ij` is built from equilibrium lattice spacings and integer minimum-image offsets. The Peierls coordinates `vx` and `vy` modify transfer integrals but do not modify Coulomb distances directly. Therefore the long-range interaction changes structural forces only indirectly through the correlated electronic ground state; it adds no explicit electrostatic force term to the current lattice gradient.

This approximation deliberately separates validation of the electronic long-range interaction from the later question of direct Coulomb-lattice forces. A state-dependent distance model of the form

`R_i - R_j = (a_x dx, a_y dy) + (v_i - v_j)`

would require additional analytic Coulomb contributions to the `vx` and `vy` gradients and will be considered only after the frozen-distance model is numerically characterized.

## Literature context

Electronic polarization is a major contribution to charge energetics in molecular crystals. Tsiper and Soos reported strong solid-state polarization in pentacene and an optical dielectric tensor for the neutral crystal (Phys. Rev. B 68, 085301, 2003; DOI 10.1103/PhysRevB.68.085301). Ha, Qi, and Kahn extracted an approximate relative permittivity near 2.8 from STM line profiles in doped pentacene films (Chem. Phys. Lett. 495, 212-217, 2010; DOI 10.1016/j.cplett.2010.06.085).

These values establish the relevant screening scale but are **not** installed as model defaults. Real-space screening treatments for organic molecular crystals emphasize that molecular point-polarizability/continuum approximations become unreliable at nearest-neighbour contact and that short-range Coulomb matrix elements should be treated as screened effective parameters (Cano-Cortes et al., Physica B 405, S185-S187, 2010; DOI 10.1016/j.physb.2009.12.079).

## Validation sequence

Before any release or material claim, the long-range branch must pass the following sequence:

1. analytic matrix-element tests for x, y, diagonal, and wrapped minimum-image pairs;
2. exact recovery of the `v0.4` `U + V1` behavior when the tail is disabled;
3. finite-difference structural-gradient checks in the frozen-distance approximation;
4. finite-size verification that `E_separated(L) - 2 E_polaron -> 0`;
5. small-cell scans in dielectric strength with multiple pair branches;
6. promotion of only physically controlled boundaries to 20x20 and 40x40 strict validation;
7. separate assessment of whether direct Coulomb-lattice forces are needed before a material-specific interpretation.
