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

## Interpretation of screening controls

The first parameter scans are model controls, not material fits. The most transparent interaction scales are

`Vx = 14.3996454784255 / (epsilon_r a_x)` eV,

`Vy = 14.3996454784255 / (epsilon_r a_y)` eV,

and

`Vdiag = 14.3996454784255 / (epsilon_r sqrt(a_x^2 + a_y^2))` eV.

These quantities should be reported alongside `a_x`, `a_y`, and `epsilon_r`. In particular, comparison with the validated `U + V1` phase boundaries should be made through these actual energy scales rather than by treating `epsilon_r` alone as a universal coupling parameter.

## Validated control geometry

The numerical validation below uses an intentionally generic isotropic control,

`Jx = Jy = 0.0575 eV`,

`alpha_x = alpha_y = 0.10 eV/A`,

`a_x = a_y = 7.0 A`,

with `V1 = 0`. The choice `a = 7 A` is a convenient molecular-lattice length scale, not a pentacene fit. The results in this section therefore validate the numerical and qualitative response of the long-range model rather than a material-specific dielectric constant.

All promoted large-cell branch calculations use the strict convergence criteria

`max structural update < 1e-8 A`,

`max structural gradient < 1e-6 eV/A`,

and `eigsh` tolerance `1e-11`.

## Validation of the dissociation reference

The finite-cell separated branch was calculated for `L = 10, 20, 40` at `epsilon_r = 10` and `100`. The residual excess above the infinite-separation reference is:

| L | epsilon_r=10 | epsilon_r=100 |
|---:|---:|---:|
| 10 | 29.473 meV | 2.948 meV |
| 20 | 14.640 meV | 1.464 meV |
| 40 | 7.29634 meV | 0.729635 meV |

The data simultaneously show

`E_separated(L) - 2 E_polaron ~ 1 / (epsilon_r L)`.

For a square cell with the two localized carriers at maximum diagonal separation, `r_max ~= a L / sqrt(2)`, so the point-charge prediction is

`epsilon_r L [E_separated - 2 E_polaron] -> 14.3996454784255 sqrt(2) / a`.

For `a = 7 A`, the analytic coefficient is `2.90917 eV`. The 40x40 numerical value is about `2.919 eV`, agreeing at the approximately 0.3% level. This validates both the finite-size scaling and the absolute minimum-image Coulomb scale. Consequently, `2 E_polaron`, not the finite-cell separated branch, is the authoritative dissociation reference.

The corresponding GitHub Actions size-scaling run is `33338074979`.

## Binding boundaries for the pure 1/r tail

Small-cell exploratory scans identify two relevant bound branches for the control geometry: an onsite bipolaron at `U = 0.525 eV` and an axial intersite bipolaron at `U = 1.0 eV`.

The promoted critical screening values obtained from the scans and strict calculations are:

| U (eV) | bound branch | 10x10 | 20x20 strict | 40x40 strict |
|---:|---|---:|---:|---:|
| 0.525 | onsite | ~6.824 | 6.95468 | 6.89070 |
| 1.000 | intersite axial | ~571.2 | 524.02 | 493.908 |

The final 40x40 narrow brackets, using the binding convention `E_bind = 2 E_polaron - E_pair`, are:

- `U = 0.525`, onsite: `E_bind = -0.0598028 meV` at `epsilon_r = 6.86` and `+0.0570784 meV` at `epsilon_r = 6.92`, giving `epsilon_c = 6.89070` by linear interpolation.
- `U = 1.0`, intersite-x: `E_bind = -0.0310847 meV` at `epsilon_r = 490` and `+0.0484589 meV` at `epsilon_r = 500`, giving `epsilon_c = 493.908` by linear interpolation.

All four narrow-bracket calculations satisfy the strict structural criteria. Around the onsite crossing, `max |Delta t|/J` is approximately `0.096` in both directions. Around the axial crossing, the primary x modulation is approximately `0.184` and the transverse y modulation approximately `0.081`, both within the conservative linear-Peierls working range.

For `a = 7 A`, the final 40x40 critical screenings correspond to a cardinal point-charge scale `Vx` of approximately `0.2985 eV` for the onsite boundary and `4.165 meV` for the axial boundary. These are control-model energy scales, not material parameters.

The strict 20x20 run is `33361841218`; the broad strict 40x40 run is `33362002931`; the final narrow 40x40 refinement is `33367133004`.

## Absence of the diagonal intermediate phase for the pure tail

The `U + V1` model in `v0.4.0a1` supports an intermediate diagonal bipolaron in the fully isotropic system because a cardinal `V1` penalizes axial nearest-neighbour configurations while leaving the diagonal separation unpenalized.

The pure `1/r` tail changes this topology. It penalizes the diagonal configuration as well. A strict 40x40 diagonal branch at `U = 1.0`, `epsilon_r = 540` remains unbound by `1.78865 meV` relative to `2 E_polaron`, while the axial branch at the same screening is bound by `0.337179 meV`. Its diagonal probability is about `0.880`, confirming that the calculation genuinely follows the diagonal local minimum rather than relaxing into the axial state.

Thus, within the validated control model, there is no diagonal rescue phase near the axial dissociation boundary for the pure continuum tail.

## Physical interpretation and limitation

The control calculations expose two very different robustness scales. Near the final 40x40 crossings, the equivalent nearest-neighbour continuum repulsion is approximately `0.299 eV` for the onsite `U = 0.525 eV` state but only approximately `4.17 meV` for the axial `U = 1.0 eV` state. The onsite pair is protected because its dominant probability remains on the same site, where the independent Hubbard `U` rather than the offsite continuum tail applies. The intersite pair is directly exposed to intermolecular repulsion and is correspondingly fragile.

These values must not be read as material Coulomb parameters. A molecular-contact point-charge expression is precisely where continuum screening is least controlled. The more physical next model is therefore an effective short-range `V1` combined with the continuum tail at longer separation, with `V1` replacing rather than adding to the continuum nearest-neighbour value. Sensitivity to the short-range replacement range should be assessed before making a pentacene-specific claim.

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

Items 1-6 are satisfied for the pure-tail control geometry described above. Item 7 remains intentionally open and is not required for interpreting the present frozen-distance model as a numerical control.
