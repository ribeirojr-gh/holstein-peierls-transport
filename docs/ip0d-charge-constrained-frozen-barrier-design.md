# IP0d — full-charge-cloud constrained frozen translation diagnostic

## Motivation

IP0c showed that fixing only the global projection of the classical lattice coordinates does not force the electronic polaron to move. In the strongly anisotropic `+y` control, for example, the geometrically constrained midpoint had nearly endpoint energy while the charge remained on the source molecule. The coordinate therefore admitted a nonphysical bypass and IP0c was superseded.

IP0d anchors the reaction coordinate directly to the **electronic localization**.

## Charge-transfer order parameter

Let `A` be a relaxed polaron and `B` its exact one-site lattice translation. Their one-electron endpoint densities are `n_A` and `n_B`. Define the diagonal order-parameter weights

`w_i = n_A(i) - n_B(i)`.

The weights are rescaled so that the endpoint expectations differ by exactly two. After subtracting the recorded offset, the centered order parameter is therefore

- `+1` at endpoint A;
- `-1` at endpoint B.

This full-density construction is preferred over a two-site population difference because an intermediate-anisotropy polaron can already be distributed over more than one molecule.

For image fraction `s`, the target centered coordinate is `1 - 2s`.

## Lagrange-multiplier electronic constraint

At fixed lattice coordinates `q`, solve

`H_lambda(q) = H(q) + lambda W`,

where `W = diag(w)`. The lowest state is recalculated while a scalar root solve adjusts `lambda` until the target expectation `<W>` is achieved.

The physical electronic energy reported by IP0d is always

`<psi_lambda | H(q) | psi_lambda>`,

not the biased auxiliary eigenvalue. The numerical identity

`epsilon_lambda = E_physical + lambda <W>`

is explicitly checked.

For one electron this is the constrained minimum for the chosen linear density order parameter, subject to numerical eigensolver/root tolerances.

## Lattice path

IP0d deliberately keeps the same linearly interpolated endpoint lattice path as IP0a. Only the electronic state is constrained.

This separation answers a narrow question first: how much energy is required to move the **charge cloud itself** along the frozen lattice interpolation?

The result is not yet:

- a transversely relaxed path;
- a NEB/string minimum-energy path;
- a finite-temperature free-energy barrier;
- a hopping activation energy;
- a hopping rate;
- a mobility.

If IP0d is numerically stable, a later stage can combine the same electronic constraint with lattice relaxation without allowing the charge to fall back into an endpoint basin.

## Control scan

- periodic `20x20` lattice;
- `J0x = 0.100 eV`;
- `J0y/J0x = 0.15, 0.50, 1.00`;
- model-default Holstein/Peierls couplings and elastic constants;
- `+x` and `+y` translations;
- 7 lattice images including endpoints and midpoint;
- sparse electronic solver;
- no electric field;
- no thermostat;
- no IDC.

## Numerical gates

IP0d requires:

1. all static endpoint relaxations converge;
2. normalized endpoint order-parameter span equals 2 within `1e-9`;
3. all target charge coordinates are achieved within `1e-8`;
4. electronic norm error remains below `1e-10`;
5. biased/unbiased energy identity error remains below `1e-9 eV`;
6. translated endpoint physical energies agree within `1e-8 eV`;
7. the constrained physical state never lies below the unconstrained ground state at the same lattice by more than `1e-8 eV`;
8. all reported constrained barriers are finite and non-negative within numerical tolerance.

No barrier magnitude, anisotropy trend, midpoint population, or x/y equality is pre-registered as a physical pass condition.

## Decision after IP0d

If the numerical gates pass, inspect:

- whether the IP0c near-zero bypass disappears when charge motion is enforced;
- whether the isotropic charge-constrained barrier remains small compared with `k_B T` near room temperature;
- how much extra energy the charge constraint adds over the ordinary adiabatic frozen path;
- whether the midpoint state is genuinely delocalized/shared;
- whether the isotropic `+x/+y` results agree within numerical symmetry;
- whether a charge-constrained lattice relaxation is worthwhile before beginning the finite-temperature hopping study.

IP0c remains in the repository as a documented negative methodological result and should not be rerun merely with larger iteration limits.