# Static exciton reference validation

## Purpose

This document validates the generic electron-hole exciton solver before any
material-specific parameterization or time propagation is introduced.  The
control calculation intentionally keeps the parameter scale already used to
consolidate the static polaron framework:

- `J0x = 0.100 eV`;
- `J0y = 0.015 eV`;
- `alpha_1^e = alpha_1^h = 3.0 eV/A`;
- `alpha_2x^e = alpha_2x^h = 0.4 eV/A`;
- `alpha_2y^e = alpha_2y^h = 0.4 eV/A`;
- `K1 = 16.51 eV/A^2`;
- `K2 = 0.51 eV/A^2`.

An onsite electron-hole attraction magnitude of `0.525 eV` is used only as a
numerical control scale.  It is not a material fit.  The dissociation reference
is always the sum of two independently relaxed control polarons,

`E_diss = E_e-pol + E_h-pol = 2 E_pol`

for this equal-carrier benchmark, and

`E_bind^X = E_diss - E_X`.

Positive binding therefore means a bound relaxed electron-hole state.

## Unit and analytic gates

The test suite verifies, independently of the numerical branch scans:

- the noninteracting pair energy equals the sum of electron and hole band
  minima;
- both one-body reduced density matrices have trace one;
- the atomic strong-onsite limit gives an onsite Frenkel pair;
- the atomic nearest-neighbour limit gives a one-site charge-transfer pair;
- equal electron/hole parameters produce equal carrier densities;
- `u`, `vx`, and `vy` Hellmann-Feynman gradients agree with central finite
  differences;
- the short-range/continuum attraction replacement semantics do not double
  count interactions;
- strict RPROP relaxation satisfies both the coordinate-update and structural
  gradient criteria;
- branch selection never promotes a non-converged result and treats numerical
  energy ties deterministically.

The full repository suite is green after these gates.

## Finite-cell reference sequence

A reproducible benchmark is implemented in
`experiments/exciton_reference_branch_benchmark.py`.  The same control is run on
6x6, 10x10, and the standard 20x20 lattice.  The table reports the lowest
Frenkel-like basin reached in each cell.

| lattice | E_pol (eV) | E_X (eV) | E_bind^X (eV) | P_onsite | IPR_e = IPR_h | <r_eh> (sites) |
|---|---:|---:|---:|---:|---:|---:|
| 6x6 | -0.374053052041 | -1.667271043345 | 0.919164939263 | 0.93626177 | 0.93495288 | 0.06454875 |
| 10x10 | -0.391013197899 | -1.680725664320 | 0.898699268521 | 0.90238686 | 0.89940754 | 0.09953660 |
| 20x20 | -0.405459057756 | -1.695788941148 | 0.884870825635 | 0.86042936 | 0.85456165 | 0.14360021 |

The control exciton remains strongly bound and predominantly onsite as the cell
is enlarged.  The gradual reduction of onsite probability/IPR and binding
energy shows that the larger cells permit a broader relative-coordinate tail;
there is no loss of the bound solution at the standard 20x20 size.

These values characterize this generic control Hamiltonian only.  They should
not be interpreted as a prediction for a particular molecular crystal.

## Strict 20x20 branch topology

All five requested seeds were relaxed independently on the 20x20 periodic
lattice.  The results are:

| seed | E_X (eV) | E_bind^X (eV) | P_onsite | IPR_e | <r_eh> (sites) | iterations |
|---|---:|---:|---:|---:|---:|---:|
| Frenkel | -1.695788941148 | 0.884870825635 | 0.86042936 | 0.85456165 | 0.14360021 | 399 |
| CT-x | -1.695788941148 | 0.884870825635 | 0.86042930 | 0.85456158 | 0.14360028 | 418 |
| CT-y | -1.695788941148 | 0.884870825635 | 0.86042934 | 0.85456163 | 0.14360024 | 547 |
| diagonal | -1.282521888704 | 0.471603773192 | 0.38872768 | 0.24487645 | 0.69258720 | 359 |
| separated | -1.695788941148 | 0.884870825635 | 0.86042937 | 0.85456165 | 0.14360021 | 407 |

Frenkel, CT-x, CT-y, and separated initializations reach the same final minimum
to approximately `1e-14 eV`.  They are therefore different paths into one
physical basin, not four distinct ground states.  The branch manager uses a
`1e-10 eV` energy-tie tolerance and records all such seed degeneracies while
selecting the first requested seed as the canonical label.

The diagonal initialization instead remains in a distinct stationary state,
approximately `0.413267 eV` above the Frenkel-like minimum.  It is more weakly
localized and has a substantially larger electron-hole separation, so it is a
useful metastable excitonic control state for later dynamics.

## Strict convergence at 20x20

Every branch satisfies the promoted static criteria:

- maximum final coordinate update: below `1.0e-8 A`;
- maximum final structural gradient: below `1.0e-6 eV/A`;
- eigensolver tolerance: `1.0e-11`.

Numerically, the largest final update among the five branches is
`9.93e-9 A` and the largest final gradient is only `2.52e-8 eV/A`, almost two
orders of magnitude below the required gradient threshold.

## Intermediate-cell metastability

The 10x10 scan retains additional strictly stationary CT-x and CT-y basins:

- CT-x: `E_X = -1.602234924626 eV`, `P_onsite = 0.59670`;
- CT-y: `E_X = -1.480880257219 eV`, `P_onsite = 0.58092`;
- diagonal: `E_X = -1.248066603038 eV`, `P_onsite = 0.39939`.

At 20x20 the axial CT seeds instead relax into the Frenkel-like basin, while the
diagonal basin survives.  This is a useful warning for future parameter maps:
metastable branch structure can depend on finite-cell geometry even when each
individual optimization is strictly converged.

## Conclusion

The static framework now generates, on the same generic Holstein-Peierls lattice
used for the polaron and bipolaron milestones:

1. a converged bound exciton ground-state basin;
2. separately normalized electron and hole densities;
3. a distinct metastable excitonic basin;
4. a consistent dissociation/binding reference to two relaxed polarons;
5. strict, finite-difference-validated shared-lattice forces.

This completes the physical prerequisites for moving next to a unified dynamics
layer for polaron, bipolaron, and exciton.  Material-specific parameter fitting
and phase-diagram scans remain deliberately deferred.
