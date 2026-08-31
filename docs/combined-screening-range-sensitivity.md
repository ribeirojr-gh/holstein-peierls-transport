# Combined short-range screening: range-sensitivity study

## Scope

This study quantifies how the static bipolaron dissociation boundary changes when the continuum screened Coulomb interaction is replaced by a stronger effective screening model over progressively larger short-range shells. It follows the shell-resolved interaction interface introduced after `v0.5.0a1`.

The study is intentionally model-general. It does not assign pentacene-specific screened matrix elements.

## Control model

The stationary lattice parameters are the bandwidth-matched isotropic control used in the long-range validation:

- `Jx = Jy = 0.0575 eV`;
- `alpha_x = alpha_y = 0.10 eV/angstrom`;
- `a_x = a_y = 7.0 angstrom`;
- onsite `U = 0.525 eV` or `1.000 eV`.

The continuum offsite interaction is

`V_cont(r) = 14.3996454784255 / (epsilon_r r) eV`.

For this sensitivity study, every explicitly replaced short-range shell is assigned

`V_short(r) = eta V_cont(r)`

with the common dimensionless control `eta = 0.75`. Thus the shell treatment changes only the short-range screening strength while preserving the same `1/epsilon_r` scaling and geometric distance dependence within each replaced shell.

The compared interaction ranges are:

- `R0`: pure continuum tail, no short-range replacement;
- `R1`: cardinal nearest neighbours `(1,0)` and `(0,1)`;
- `R2`: `R1` plus the first diagonal shell `(1,1)`;
- `R3`: `R2` plus the second axial shell `(2,0)` and `(0,2)`.

The authoritative dissociation reference remains

`E_diss = 2 E_polaron`,

and the binding convention is

`E_bind = 2 E_polaron - E_pair`,

so positive `E_bind` denotes a bound bipolaron.

## 10x10 exploratory scan

All branch searches converged. Linear interpolation across the nearest sign-changing `E_bind` bracket gives:

| U (eV) | R0 | R1 | R2 | R3 |
|---:|---:|---:|---:|---:|
| 0.525 | 6.82898 | 5.16244 | 5.13164 | 5.12407 |
| 1.000 | 572.055 | 437.780 | 431.952 | 430.041 |

The pure-continuum controls reproduce the previously established 10x10 long-range boundaries. The dominant change comes from replacing the cardinal nearest-neighbour shell. Extending the replacement to the diagonal shell produces a smaller correction, and extending it through the second axial shell produces a still smaller correction.

## Strict 20x20 brackets

The 20x20 promotion used:

- maximum structural update `< 1e-8 angstrom`;
- maximum structural gradient `< 1e-6 eV/angstrom`;
- `eigsh` tolerance `1e-11`.

All searched branches converged. The bounding points and interpolated critical dielectric constants are:

### U = 0.525 eV

| model | lower epsilon | E_bind lower (meV) | upper epsilon | E_bind upper (meV) | epsilon_c |
|---|---:|---:|---:|---:|---:|
| R1 | 5.0 | -0.496807 | 5.6 | +0.997412 | 5.19949 |
| R2 | 5.0 | -0.419686 | 5.6 | +1.077737 | 5.16816 |
| R3 | 5.0 | -0.400321 | 5.6 | +1.097616 | 5.16035 |

The minimum is the onsite branch throughout these brackets. Competing localized axial/diagonal states are far above the onsite minimum in this regime.

### U = 1.000 eV

| model | lower epsilon | E_bind lower (meV) | upper epsilon | E_bind upper (meV) | epsilon_c |
|---|---:|---:|---:|---:|---:|
| R1 | 370 | -0.266275 | 430 | +0.296550 | 398.386 |
| R2 | 370 | -0.208942 | 430 | +0.345860 | 392.596 |
| R3 | 370 | -0.190175 | 430 | +0.361992 | 390.665 |

The minimum is the axial intersite branch. The diagonal branch remains metastable and approximately `1.9-2.7 meV` above the axial branch over these brackets, so no diagonal rescue phase appears at 20x20.

## Finite-size trend from 10x10 to 20x20

For `U = 0.525 eV`, the critical values move upward by only about `0.7%` from 10x10 to 20x20 for R1-R3. For `U = 1.000 eV`, they move downward by about `9%`, consistent with the stronger finite-size sensitivity already observed for the weakly bound axial state in the pure-continuum study.

The shell-range hierarchy itself is stable with size:

- first-shell replacement accounts for the overwhelming majority of the shift from the pure continuum result;
- the R1 -> R2 correction is clearly smaller;
- the R2 -> R3 correction is smaller again.

At 20x20, the total R1 -> R3 change is approximately `0.0391` in `epsilon_c` for `U = 0.525 eV` and `7.72` for `U = 1.000 eV`.

## Normalized screening interpretation

Because every replaced shell is scaled by the same factor `eta`, a useful diagnostic is the ratio of the combined-model boundary to the pure-continuum boundary at the same cell size.

For R3,

- at 10x10, `epsilon_c(R3) / epsilon_c(R0) = 0.75034` for `U = 0.525 eV` and `0.75175` for `U = 1.000 eV`;
- at 20x20, using the independently validated pure-continuum values `6.95468` and `524.02`, the ratios are `0.74200` and `0.74552`, respectively.

These values are close to the imposed `eta = 0.75`. This is physically natural: once the explicitly screened region covers the shells carrying most of the localized pair probability, the dissociation threshold is controlled primarily by the effective short-range scale `eta / epsilon_r`; the unscreened remainder of the continuum tail supplies the residual departure from exact proportionality. The 40x40 calculation tests whether this interpretation persists at larger size.

## 40x40 promotion strategy

The final large-cell refinement uses only the branches required by the 20x20 energy landscape:

- for `U = 0.525 eV`, the onsite branch is followed against the authoritative `2 E_polaron` reference;
- for `U = 1.000 eV`, both axial and diagonal localized branches are recalculated so that the phase topology is rechecked at large size;
- the finite-cell separated branch is not used to define dissociation and therefore is not required for the critical boundary.

The strict 40x40 refinement is performed for R1-R3 with narrow bracketing intervals. Its results will be appended below after completion.
