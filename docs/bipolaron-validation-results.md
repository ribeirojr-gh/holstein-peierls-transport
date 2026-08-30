# Static bipolaron validation results

## Scope

This document consolidates the validation of the experimental static singlet bipolaron implementation based on the adiabatic Holstein-Peierls-Hubbard model. The purpose is to distinguish numerical validation from physical interpretation before integration into the main development line.

The implementation uses a correlated two-particle wavefunction `Psi(i,j)`, a one-particle reduced density matrix with trace 2, matrix-free electronic operators, periodic boundaries, and RPROP relaxation of the intramolecular and intermolecular lattice coordinates.

## Validation status

**Overall status: PASS for the static singlet bipolaron solver in the validated parameter domain.**

The following checks were completed:

- rigid-lattice noninteracting energy and density-factorization limit;
- singlet exchange symmetry and normalization;
- reduced-density-matrix trace and Hermiticity;
- onsite Hubbard interaction expectation value;
- analytic lattice gradients against finite differences;
- atomic Holstein limit and the exact threshold `U_c = A^2 / K1`;
- finite-hopping Holstein-Hubbard branch competition;
- Peierls-only limit with `A = 0`;
- simultaneous Holstein-Peierls relaxation;
- branch classification by final observables rather than initial seed labels;
- finite-size comparison through 20x20 and 40x40 cells;
- strict 40x40 convergence with the production experimental tolerances.

## Convergence criteria

The final 40x40 validation used the solver tolerances directly:

- maximum RPROP update: `1e-8 angstrom`;
- maximum lattice-gradient component: `1e-6 eV/angstrom`;
- sparse eigensolver tolerance: `1e-11`.

All four final 40x40 branches converged within 1200 RPROP iterations. The final residual gradients were of order `3e-8 eV/angstrom`, substantially below the declared threshold.

### RPROP convention

The two-particle solver uses a non-backtracking RPROP variant: when a gradient component changes sign, its step size is reduced and that coordinate receives no update on that iteration. This differs deliberately from the archived single-polaron compatibility path, which rolls back the previous coordinate step on a sign reversal. Consequently, two-particle and legacy one-particle iteration counts or optimization trajectories should not be compared directly. Validation is based on stationary energies, residual gradients, pair observables, analytic limits, and finite-size behavior rather than reproducing the historical RPROP path.

Several scripts in `experiments/` retain looser tolerances because they document exploratory scans used to identify branches and parameter windows. The authoritative large-cell benchmark is `experiments/bipolaron_branch_benchmark_seeded.py`, which uses the strict tolerances listed above and fails explicitly if a branch does not converge.

## Reference parameter set

The finite-size validation reported below used:

- `Jx = 0.100 eV`;
- `Jy = 0.015 eV`;
- `A = 3.0 eV/angstrom`;
- `K1 = 16.51 eV/angstrom^2`;
- `alpha_x = 0.10 eV/angstrom`;
- `alpha_y = 0.12 eV/angstrom`;
- onsite Hubbard repulsion `U` as indicated below.

The model is strongly anisotropic because `Jx >> Jy`.

## Atomic Holstein benchmark

In the atomic Holstein limit,

`Delta_BP = A^2 / K1 - U`,

so the exact onsite-pair threshold is

`U_c = A^2 / K1`.

For the reference input,

`U_c ~= 0.545 eV`.

The numerical solver reproduces this benchmark as the hopping is taken toward zero. This validates the factor-of-two structure of the reduced density matrix and the lattice force for double occupation.

## Holstein-Hubbard branch structure

With Peierls coupling disabled, the numerical solutions show the qualitative sequence

`onsite bipolaron -> intersite-x bipolaron -> weak/marginal pair -> separated polarons`

as `U` increases.

The intersite-x branch is preferred over the intersite-y branch over a broad interval because of the transfer-integral anisotropy. Close to the dissociation boundary, sub-meV energy differences should be treated as marginal rather than as a distinct robust phase unless independently confirmed by size scaling.

## Controlled Peierls domain

A conservative working criterion was introduced to avoid quantitative interpretation when the linear hopping modulation becomes too large:

`max |Delta t_mu| / |J_mu| <= approximately 0.25`.

This is a diagnostic criterion, not a mathematical singularity of the Hamiltonian.

For the present parameter set, the conservative quantitative domain identified in the scans is approximately

- `0 <= alpha_x <= 0.10 eV/angstrom`;
- `0 <= alpha_y <= 0.12 eV/angstrom`.

The x direction is physically dominant. In the 12x12 transition scans, the interpolated onsite-to-intersite-x crossing was approximately

| alpha_x (eV/angstrom) | U_c at alpha_y = 0 (eV) | U_c at alpha_y = 0.12 (eV) |
|---:|---:|---:|
| 0.00 | 0.508785 | 0.508933 |
| 0.05 | 0.506732 | 0.506878 |
| 0.10 | 0.499360 | 0.499500 |

Thus `alpha_x = 0.10 eV/angstrom` shifts the structural crossing downward by about 9.4 meV, while the corresponding effect of `alpha_y` is only about 0.15 meV. The 10x10-to-12x12 variation of these crossings is below about 0.5 meV.

## Peierls-only limit

With `A = 0`, no robust bound bipolaron was found in the controlled linear regime. Up to `alpha_x = 0.30 eV/angstrom`, the relaxed Peierls distortion returns to zero and the two-particle state remains unbound. A strongly bound intersite solution appears at `alpha_x = 0.40 eV/angstrom` for `U = 0`, but only together with `max |Delta t_x| / Jx ~= 5.26`, far outside the linear regime used for quantitative interpretation.

Within the validated domain, the present semiclassical model therefore supports the conclusion:

**Peierls coupling alone does not produce a robust bipolaron; rather, the bound state is Holstein-assisted and its stability and geometry are modulated by Peierls coupling, predominantly along x.**

## Final 40x40 strict-tolerance validation

The final large-cell comparison used the `intersite_x` and `separated` branches at `U = 0.525` and `1.000 eV`.

| U (eV) | branch | total energy (eV) | P_onsite | P_NN,x | mean separation | one-body IPR | max |Delta t_x|/Jx | max |Delta t_y|/Jy | final max gradient (eV/angstrom) | final max update (angstrom) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.525 | intersite_x | -0.671340983931729 | 0.131118 | 0.794061 | 0.942890 | 0.453555 | 0.254830 | 0.138786 | 3.32e-8 | 9.69e-9 |
| 0.525 | separated | -0.627428659405481 | ~0 | ~0 | 28.158168 | 0.415611 | 0.072652 | 0.116473 | 2.79e-8 | 9.39e-9 |
| 1.000 | intersite_x | -0.634241741090195 | 0.044676 | 0.861401 | 1.049916 | 0.447571 | 0.159109 | 0.143463 | 2.77e-8 | 9.00e-9 |
| 1.000 | separated | -0.627428659405365 | ~0 | ~0 | 28.158168 | 0.415611 | 0.072652 | 0.116474 | 2.65e-8 | 8.73e-9 |

The corresponding binding energies are

- `U = 0.525 eV`: `Delta_BP = 43.9123 meV`;
- `U = 1.000 eV`: `Delta_BP = 6.81308 meV`.

The `U = 0.525 eV` point has `max |Delta t_x| / Jx = 0.2548`, slightly above the conservative 0.25 working cutoff. It is therefore retained as a robust numerical bound state but should be described as borderline for quantitative interpretation within the linear Peierls approximation. The `U = 1.000 eV` point is comfortably inside the controlled regime.

## Finite-size conclusion

The same parameter points were previously evaluated on 20x20 cells. The binding energies changed only modestly on increasing the cell to 40x40:

- `U = 0.525 eV`: approximately `43.06 -> 43.91 meV`;
- `U = 1.000 eV`: approximately `6.40 -> 6.81 meV`.

More importantly, the bound-state separation remains of order one lattice spacing, while the separated reference expands to a mean separation of about 28 sites in the 40x40 cell. The nearest-neighbour-x probability also remains high (`~0.79-0.86`).

Therefore the intersite-x bipolaron identified in this parameter region is not a periodic-image or finite-cell localization artifact.

## Physical interpretation supported by the validation

Within the tested adiabatic two-dimensional molecular-crystal model:

1. Holstein coupling supplies the primary lattice-mediated attraction required for pairing in the controlled regime.
2. Hubbard repulsion drives a structural crossover from onsite to nearest-neighbour pairing.
3. Strong hopping anisotropy selects the x-oriented intersite pair.
4. Peierls-x coupling stabilizes that intersite-x geometry and shifts the onsite-to-intersite boundary to lower U.
5. Peierls-y coupling is a secondary correction for this strongly anisotropic reference set.
6. The bound intersite-x state survives 20x20-to-40x40 size scaling and strict numerical convergence tests.

## Limitations and claims that should not yet be made

These calculations do not yet establish a material-specific phase diagram for pentacene. The current parameter set should be treated as a molecular-crystal / oligoacene-like reference until HOMO/LUMO, Coulomb, transfer-integral, and electron-phonon parameters are tied consistently to a particular material.

The current Coulomb sector includes onsite Hubbard U as the principal repulsion. A realistic long-range screened interaction should be added before quantitative material-level claims about bipolaron stability are made.

The present solver covers the singlet sector only. Triplet/antisymmetric two-particle states are not yet implemented.

No conclusion should be drawn from parameter points for which the linear Peierls hopping modulation becomes order unity.

## Reproducibility

The final strict large-cell run is GitHub Actions run `33282853902` on commit

`364017f852b3f7f4df23e00c719916ea5d261221`.

The four final branch artifacts were stored by GitHub Actions. The normal pytest workflow for the same commit also completed successfully.

## Integration recommendation

The static singlet Holstein-Peierls-Hubbard bipolaron implementation is ready for code review and integration as an experimental two-particle module. It should remain clearly separated from the validated one-polaron production path, and its material-specific interpretation should remain conservative until realistic Coulomb and material parameterization are added.
