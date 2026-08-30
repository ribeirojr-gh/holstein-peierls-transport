# Static hopping-isotropy validation

## Motivation

The validated extended-Hubbard calculations originally used the strongly anisotropic hopping set

`Jx = 0.100 eV`, `Jy = 0.015 eV`,

so `eta = Jy/Jx = 0.15`. Because hopping anisotropy changes stationary kinetic energy, localization, pair orientation, and competition between onsite, intersite, diagonal, and separated states, the isotropic stationary limit was validated before merging the extended-Hubbard branch.

## Separating anisotropy from bandwidth

For the square nearest-neighbour model,

`W = 4 * (Jx + Jy)`.

The primary trajectory therefore keeps

`Jx + Jy = 0.115 eV`

fixed while varying

`eta = Jy/Jx`,

with

`Jx = 0.115 / (1 + eta)` and `Jy = eta * Jx`.

Thus `eta = 0.15` reproduces the baseline hopping pair and `eta = 1` gives the bandwidth-matched isotropic control

`Jx = Jy = 0.0575 eV`.

A separate high-bandwidth control used `Jx = Jy = 0.100 eV`. In that control the two-particle state dissociated for both tested Hubbard values, demonstrating that hopping isotropy and total bandwidth must not be conflated.

## Isotropic controls and symmetry

Control A isolates hopping isotropy:

- `Jx = Jy = 0.0575 eV`;
- `alpha_x = 0.10 eV/A`;
- `alpha_y = 0.12 eV/A`.

Control B makes the x/y Hamiltonian fully isotropic:

- `Jx = Jy = 0.0575 eV`;
- `alpha_x = alpha_y = 0.10 eV/A`;
- isotropic nearest-neighbour `V1`.

For Control B, a 90-degree rotation must exchange the x- and y-oriented pair states without changing the energy. The explicit regression test satisfies this degeneracy to numerical precision.

The branch search includes onsite, intersite-x, intersite-y, diagonal `(1,1)`, and separated seeds. Final states are identified from pair probabilities rather than from the seed label.

## Fixed-bandwidth anisotropy trajectory

At `V1 = 0`, the 10x10 scan showed that the bipolaron survives removal of hopping anisotropy when the total bandwidth is held fixed.

For `U = 0.525 eV`, the preferred structure changes from intersite-x to onsite. The 10x10 crossing was `eta_c ~= 0.79396`. Strict 20x20 refinement brackets the crossing between `eta = 0.80` and `0.82`, giving

`eta_c(20x20) ~= 0.81320`.

For `U = 1.000 eV`, the pair remains intersite but changes orientation from x to y because `alpha_y > alpha_x`. The 10x10 crossing was `eta_orient ~= 0.87455`. Strict 20x20 refinement brackets it between `eta = 0.85` and `0.87`, giving

`eta_orient(20x20) ~= 0.85946`.

These two eta crossings are numerically resolved but are not promoted to 40x40 quantitative boundaries. Near them, at least one competing branch exceeds the project's conservative linear-Peierls working criterion `max |Delta t|/J <= 0.25`: the `U = 0.525 eV` intersite-x branch reaches approximately `0.31`, and the `U = 1.000 eV` intersite-y branch reaches approximately `0.27`. They are therefore retained as qualitative evidence that anisotropy changes stationary phase selection, not as controlled material-scale critical values.

## Fully isotropic V1 phase topology

The fully isotropic Hamiltonian uses

`Jx = Jy = 0.0575 eV`, `alpha_x = alpha_y = 0.10 eV/A`.

This regime remains inside the conservative linear-Peierls window at all final boundaries and reveals a phase topology absent from the strongly anisotropic baseline.

### U = 1.000 eV: axial -> diagonal -> separated

At small `V1`, the lowest bipolaron is an axial nearest-neighbour state. The x- and y-oriented axial solutions are exactly degenerate by rotational symmetry. Increasing `V1` penalizes nearest neighbours and stabilizes a diagonal `(1,1)` bipolaron before eventual dissociation.

The finite-size critical values are:

| Boundary | 10x10 | 20x20 | 40x40 |
| --- | ---: | ---: | ---: |
| axial -> diagonal | 3.1047 meV | 3.46893 meV | **3.65887 meV** |
| diagonal -> separated | 14.8980 meV | 15.60854 meV | **15.97119 meV** |

For the final 40x40 axial/diagonal bracket,

- at `V1 = 3.3 meV`, `E_diag - E_axial = +0.298065 meV`;
- at `V1 = 3.7 meV`, `E_diag - E_axial = -0.034161 meV`.

Linear interpolation gives

`V1_c(axial -> diagonal, 40x40) = 3.6588699 meV`.

The diagonal state is structurally distinct: near the boundary its probability for minimum-image diagonal separation is approximately `0.881`, while the axial state has nearest-neighbour probability approximately `0.888`.

For diagonal dissociation,

- at `V1 = 15.0 meV`, `E_diag - E_sep = -0.053308 meV`;
- at `V1 = 16.5 meV`, `E_diag - E_sep = +0.029025 meV`.

Linear interpolation gives

`V1_c(diagonal -> separated, 40x40) = 15.9711938 meV`.

### U = 0.525 eV: onsite -> separated

At the bandwidth-matched fully isotropic endpoint the low-V1 bipolaron is onsite. The finite-size dissociation boundary is

| 10x10 | 20x20 | 40x40 |
| ---: | ---: | ---: |
| 311.47 meV | 309.50483 meV | **308.57480 meV** |

For the final 40x40 bracket,

- at `V1 = 307 meV`, `E_onsite - E_sep = -0.068977 meV`;
- at `V1 = 312 meV`, `E_onsite - E_sep = +0.150027 meV`.

Linear interpolation gives

`V1_c(onsite -> separated, 40x40) = 308.5747950 meV`.

The onsite state at this boundary has `P_onsite ~= 0.953`, whereas the separated reference has negligible local pair probability and a mean periodic separation of approximately `28.20` lattice sites.

## Strict 40x40 numerical validation

All 12 final branch calculations converged using the same strict criteria as the validated two-particle solver:

- maximum coordinate update `< 1e-8 A`;
- maximum structural gradient `< 1e-6 eV/A`;
- eigensolver tolerance `1e-11`.

The final residual gradients are of order `2-3e-8 eV/A`, and final coordinate updates are below `1e-8 A`.

The largest Peierls modulation ratio among the competitive 40x40 isotropic branches is approximately `0.184` for the axial state. The diagonal state is approximately `0.081`, the onsite state approximately `0.096`, and the separated reference approximately `0.075`. All are below the conservative 25% working cutoff.

## Physical conclusion

The isotropy study changes the interpretation of the extended-Hubbard results in three ways.

1. Removing hopping anisotropy at fixed bandwidth does not automatically destroy bipolaron binding.
2. Residual anisotropy in the Peierls coupling can select pair orientation even when `Jx = Jy`.
3. In the fully isotropic Hamiltonian, nearest-neighbour repulsion produces a stable diagonal bipolaron phase between the axial and separated phases for `U = 1 eV`.

Therefore hopping anisotropy is not only a future transport parameter: it changes the stationary energy landscape and phase topology. The high-bandwidth isotropic control further shows that bandwidth changes can destroy binding independently of isotropy.

## Scope and limitations

The eta-crossing values are retained as qualitative stationary-physics diagnostics because they approach or exceed the conservative linear-Peierls window. The fully isotropic `V1` boundaries are quantitatively controlled under the project's current working criterion and have been validated through 40x40.

The model still uses a fixed nearest-neighbour electronic repulsion `V1`; it is not yet a material-specific screened long-range Coulomb interaction. Explicit lattice spacings and dielectric screening remain deferred to a subsequent branch.

## Reproducibility

The final strict isotropic 40x40 calculation is GitHub Actions run `33328137691`, head commit `d5514a4abc6150f24b13c00e6738f250e6f708eb`. The strict 20x20 refinement is run `33327923389`. Exploratory workflow files are removed after harvesting the artifacts; the experiment scripts and this results document remain in the repository.
