# Nearest-neighbour repulsion validation

## Scope

This development stage extends the validated static singlet bipolaron model from onsite Hubbard repulsion `U` to an extended-Hubbard interaction with a positive nearest-neighbour term `V1`.

The interaction energy is

`E_int = U * P_onsite + V1 * P_NN`.

The exact parameter derivative is therefore

`dE/dV1 = P_NN`,

which is used as a regression test. The periodic nearest-neighbour graph uses the same minimum-image square lattice as the hopping model.

`V1` is presently a fixed electronic parameter. It does not depend explicitly on the instantaneous lattice coordinates, so there is no extra direct classical force. Structural gradients still change self-consistently because the correlated ground state changes with `V1`.

The physical scans use the validated anisotropic reference set `alpha_x = 0.10 eV/A`, `alpha_y = 0.12 eV/A`, with `U = 0.525` and `1.000 eV`. Onsite, intersite-x, intersite-y, and separated seeds were relaxed independently in the exploratory scans; final-state labels are assigned from pair observables rather than seed names. The finite-cell binding reference is the separated branch in the same cell.

## Numerical validation

The interaction implementation is covered by periodic-neighbour, synthetic-state energy, and Hellmann-Feynman tests. Structural gradients at finite `V1` are checked against finite differences. The final 40x40 boundary calculations use the strict criteria

- maximum coordinate update `< 1e-8 A`,
- maximum structural gradient `< 1e-6 eV/A`,
- electronic eigensolver tolerance `1e-11`.

All final boundary jobs converged. Residual gradients are of order `2e-8` to `3e-8 eV/A`, well below the declared limit.

## Final 40x40 phase boundaries

### U = 1.000 eV: intersite-x to separated

At `V1 = 7.5 meV`, the intersite-x branch is lower than the separated branch by `0.353380 meV`. At `V1 = 8.0 meV`, it is higher by `0.076254 meV`. Linear interpolation gives

`V1_c = 7.911257 meV`.

Thus the 40x40 boundary is

`intersite-x bipolaron -> separated polarons` at `V1 ~= 7.91 meV`.

The intersite-x state remains well inside the conservative linear-Peierls diagnostic near this boundary, with `max |Delta t_x|/J_x ~= 0.160`.

### U = 0.525 eV: intersite-x to onsite

At `V1 = 30.0 meV`,

`E_onsite - E_x = +0.111989 meV`,

so the intersite-x solution is still lower. At `V1 = 30.5 meV`,

`E_onsite - E_x = -0.187878 meV`,

so the onsite solution is lower. Linear interpolation gives

`V1_c = 30.186731 meV`.

The two competing solutions remain structurally distinct at `30.5 meV`:

- intersite-x: `P_onsite = 0.13964`, `P_NN = 0.77946`, `mean_r = 0.9430`;
- onsite: `P_onsite = 0.80932`, `P_NN = 0.17963`, `mean_r = 0.2012`.

This is therefore a genuine crossing between two different relaxed bipolaron structures, not a relabeling of one minimum.

The intersite-x branch at this boundary has `max |Delta t_x|/J_x ~= 0.261`, slightly above the project's conservative 25% working cutoff. The crossing is numerically resolved, but its quantitative material interpretation should therefore be treated as borderline with respect to the linear Peierls approximation.

### U = 0.525 eV: onsite to separated

At `V1 = 184 meV`, the onsite branch is lower than the separated branch by `0.137484 meV`. At `V1 = 186 meV`, it is higher by `0.053554 meV`. Linear interpolation gives

`V1_c = 185.439333 meV`.

Thus the 40x40 sequence at this value of `U` is

`intersite-x bipolaron -> onsite bipolaron -> separated polarons`,

with boundaries at approximately `30.19 meV` and `185.44 meV`.

## Finite-size trend

The exploratory 10x10 and refined 20x20 calculations establish the same phase topology. The most weakly bound boundary shows the largest finite-size drift, while the onsite dissociation boundary is already nearly converged by 20x20. The final 40x40 values used for interpretation are:

| U (eV) | Boundary | 10x10 (meV) | 20x20 (meV) | 40x40 (meV) |
| ---: | --- | ---: | ---: | ---: |
| 1.000 | intersite-x -> separated | ~6.44 | ~7.40 | 7.9113 |
| 0.525 | intersite-x -> onsite | ~27.5 | ~29.26 | 30.1867 |
| 0.525 | onsite -> separated | ~184 | ~184.9 | 185.4393 |

The monotonic size trend and preservation of the phase ordering support the interpretation that these are physical boundaries of the present finite-parameter model rather than small-cell artifacts.

## Physical interpretation

Nearest-neighbour repulsion directly penalizes the intersite bipolaron because `dE/dV1 = P_NN`. For `U = 1.000 eV`, onsite double occupation is too costly to provide an alternative bound structure, so a small `V1` dissociates the intersite pair. For `U = 0.525 eV`, the onsite configuration remains competitive because `U` lies close to the local Holstein atomic pairing scale `A^2/K1 ~= 0.545 eV`. Increasing `V1` first drives an intersite-to-onsite structural transition and only at much larger `V1` destroys the onsite pair.

These results establish a nontrivial three-regime extended-Hubbard phase structure in the current adiabatic Holstein-Peierls model.

## Limitations and next step

This validation concerns a fixed nearest-neighbour electronic repulsion on the effective square lattice. It is not yet a material-specific Coulomb model. A long-range screened interaction is deliberately deferred until explicit molecular spacings and dielectric screening are added. The legacy `3.5 A` factor used for velocity conversion is not treated as sufficient evidence for a Coulomb geometry.

The next Coulomb stage should therefore introduce explicit `a_x`, `a_y`, and dielectric screening parameters and define the offsite interaction separately from the onsite Hubbard `U`, avoiding implicit double counting.
