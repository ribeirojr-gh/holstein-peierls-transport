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

## Strict 40x40 refinement

The final large-cell refinement used the same strict structural and eigensolver criteria. For `U = 0.525 eV`, the onsite branch alone was followed because the 20x20 landscape places competing localized branches far above it. For `U = 1.000 eV`, both the axial and diagonal localized branches were recalculated to recheck the phase topology.

All 18 large-cell relaxations converged. Maximum final structural updates were below `1e-8 angstrom`; maximum structural gradients were of order `1e-8 eV/angstrom`.

### U = 0.525 eV

| model | lower epsilon | E_bind lower (meV) | upper epsilon | E_bind upper (meV) | epsilon_c |
|---|---:|---:|---:|---:|---:|
| R1 | 5.10 | -0.278563 | 5.25 | +0.109325 | 5.207723 |
| R2 | 5.10 | -0.199895 | 5.25 | +0.188867 | 5.177128 |
| R3 | 5.10 | -0.180034 | 5.25 | +0.208876 | 5.169438 |

The minimum remains onsite throughout. At the bracket endpoints its onsite probability is approximately `0.954-0.955`, confirming that the critical state remains a strongly local pair.

### U = 1.000 eV

| model | lower epsilon | E_bind lower (meV) | upper epsilon | E_bind upper (meV) | epsilon_c |
|---|---:|---:|---:|---:|---:|
| R1 | 350 | -0.315174 | 390 | +0.121918 | 378.843 |
| R2 | 350 | -0.254007 | 390 | +0.176792 | 373.585 |
| R3 | 350 | -0.233809 | 390 | +0.194904 | 371.815 |

The axial intersite branch remains the minimum throughout the 40x40 brackets. Its nearest-neighbour-x probability is approximately `0.881`. The explicitly followed diagonal branch remains unbound and about `2.0-2.9 meV` above the axial state over the bracket, so there is no diagonal rescue phase at the large-cell dissociation boundary.

## Size and shell-range convergence

The interpolated boundaries across cell sizes are:

| U (eV) | size | R1 | R2 | R3 |
|---:|---:|---:|---:|---:|
| 0.525 | 10 | 5.16244 | 5.13164 | 5.12407 |
| 0.525 | 20 | 5.19949 | 5.16816 | 5.16035 |
| 0.525 | 40 | 5.20772 | 5.17713 | 5.16944 |
| 1.000 | 10 | 437.780 | 431.952 | 430.041 |
| 1.000 | 20 | 398.386 | 392.596 | 390.665 |
| 1.000 | 40 | 378.843 | 373.585 | 371.815 |

For `U = 0.525 eV`, finite-size drift is already small by 20x20. For `U = 1.000 eV`, the weakly bound axial state retains the stronger finite-size sensitivity expected from the pure-continuum study, but the ordering of R1, R2, and R3 is unchanged at every size.

The shell-range hierarchy is strongly convergent at 40x40:

- `R1 -> R2` changes `epsilon_c` by `0.03060` for `U = 0.525 eV` and `5.258` for `U = 1.000 eV`;
- `R2 -> R3` changes it by only `0.00769` and `1.770`, respectively.

Thus the cardinal nearest-neighbour shell accounts for the overwhelming majority of the short-range-screening correction. The diagonal shell gives a smaller secondary contribution, and the second axial shell gives a smaller correction again. A fourth shell is not justified for the present generic control because the range sequence already shows a clear hierarchy and no competing phase is emerging.

## Normalized screening interpretation

A useful diagnostic is the ratio of the combined-model boundary to the corresponding pure-continuum boundary. Using the previously validated 40x40 pure-tail values `epsilon_c = 6.89070` for `U = 0.525 eV` and `493.908` for `U = 1.000 eV`,

- `epsilon_c(R3) / epsilon_c(R0) = 0.75021` for `U = 0.525 eV`;
- `epsilon_c(R3) / epsilon_c(R0) = 0.75280` for `U = 1.000 eV`.

Both are extremely close to the imposed `eta = 0.75`. Once the explicitly screened region covers the shells carrying most of the localized-pair probability, the dissociation threshold is therefore controlled primarily by the effective short-range scale `eta / epsilon_r`; the remaining continuum tail supplies only a residual correction.

## Conclusion and next model stage

The shell-resolved combined interaction is numerically stable, preserves the validated pure-tail limits, and provides a controlled bridge between a continuum dielectric description and effective short-range screened matrix elements. The generic isotropic control establishes that three short-range shell classes are sufficient for the present sensitivity analysis; extending to `R4` would add computational cost without a demonstrated physical need.

The next stage should therefore be material-specific rather than a longer generic shell expansion. Short-range interaction matrix elements, dielectric response, crystal geometry, transfer integrals, and electron-phonon couplings should be obtained from traceable literature or electronic-structure calculations for a specific molecular semiconductor before constructing material-specific bipolaron phase diagrams.

Direct Coulomb-Peierls force terms remain outside this frozen-distance validation and should be introduced separately if material-specific interpretation shows that geometry-dependent Coulomb forces are quantitatively required.
