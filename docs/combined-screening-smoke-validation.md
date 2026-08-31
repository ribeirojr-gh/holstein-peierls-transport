# Combined-screening implementation smoke validation

## Scope

This record validates the first shell-resolved short-range replacement
implementation before large-cell sensitivity scans. The calculations are
intentionally small and use generic control values. They are **not** promoted
phase-boundary results and are not a material-specific parameterization.

The validation was run on a `6x6` isotropic lattice with

- `Jx = Jy = 0.0575 eV`;
- `alpha_x = alpha_y = 0.10 eV/angstrom`;
- `a_x = a_y = 7.0 angstrom`;
- `U = 0.525 eV`;
- `epsilon_r = 10`;
- onsite, intersite-x, diagonal, and separated seeds.

For the pure continuum model, the first-shell continuum values are

- `Vx = Vy = 0.2057092211 eV`;
- `Vdiag = 0.1454583852 eV`.

The replacement values below were chosen only to exercise progressively larger
short-range regions. They are not fitted molecular-crystal parameters.

## Results

| Model | Cardinal value | Diagonal value | Additional replacements | Best state | E_bind vs 2E_polaron |
|---|---:|---:|---|---|---:|
| pure continuum | 0.205709 eV | 0.145458 eV | none | onsite | 4.72244 meV |
| cardinal replacement | 0.160000 eV | 0.145458 eV | none | onsite | 7.34443 meV |
| cardinal + diagonal | 0.160000 eV | 0.110000 eV | none | onsite | 7.42053 meV |
| extended | 0.160000 eV | 0.110000 eV | `(2,0)=0.08 eV`, `(0,2)=0.08 eV` | onsite | 7.43610 meV |

All four seeded branches converged in every model. Across the recorded branch
calculations the final structural gradient remained below
`8.0e-7 eV/angstrom`, consistent with the requested convergence criterion.

The onsite probabilities of the four best states were approximately
`0.9440`, `0.9358`, `0.9353`, and `0.9352`, respectively. The small incremental
change from the diagonal and second-axial replacements is therefore consistent
with the low probability weight carried by those shells in this particular
onsite-dominated state.

## Interpretation limit

The finite-cell separated branch lies about `49.55 meV` above the authoritative
`2 E_polaron` reference in this `6x6` cell because of the residual
minimum-image Coulomb interaction. Consequently these calculations must not be
used to infer a dissociation threshold or a thermodynamic phase boundary.

Their purpose is narrower:

1. the shell parser and parameter validation work in the full relaxation path;
2. selected shells are modified while the continuum tail remains active beyond
   them;
3. all relevant branch seeds continue to converge;
4. the energy response is continuous and qualitatively ordered with the amount
   of repulsion removed from short range;
5. the frozen-distance structural-gradient assumption remains numerically
   consistent when shell overrides are active.

The next scientific validation should therefore move to larger cells and scan
near the already established `v0.5.0a1` dissociation regions, using
`2 E_polaron` as the reference and promoting only size-converged trends.

## Reproducibility

The temporary GitHub Actions smoke run was `33393364587`, evaluated at commit
`37c0152d53a9b3acb93ad1bf82060282f0e8b510`. Its artifact contained the full
CSV and JSON branch/minimum outputs for the four model variants.
