# TP1 — periodic transport observables

## Purpose

TP0 closed the combined field + BAOAB + IDC energy bookkeeping. TP1 now defines transport observables that remain well posed under periodic boundary conditions (PBC). No mobility is extracted in this checkpoint.

A naive expectation value of the Cartesian site coordinate is not an acceptable transport coordinate on a periodic lattice: a localized carrier crossing a cell boundary would appear to jump discontinuously from the last site to the first. TP1 therefore uses the probability flux through oriented nearest-neighbour bonds as the primary observable.

## Oriented bond probability current

For a normalized one-particle state and a Hermitian hopping matrix element from site `i` to its +x neighbour `j`,

`H_ij = t_x(i) exp(i phi_x)`,

the probability current oriented from `i` to `j` is

`J_i->j = -(2/hbar) Im[ psi_i^* H_ij psi_j ]`.

The analogous expression is used for +y bonds. Units are probability per fs. With this sign convention the site continuity equation is

`d rho_i/dt = J_(i-x)->i + J_(i-y)->i - J_i->(i+x) - J_i->(i+y)`.

The summed oriented flux gives the particle velocity

`v_x = a_x sum_i J_i->(i+x)`,

`v_y = a_y sum_i J_i->(i+y)`.

The accumulated transport displacement is the time integral of this velocity. This quantity is unwrapped by construction and remains continuous across periodic boundaries.

## Field-work identity

The existing field convention is the electron-like Peierls phase

`phi_x(t) = -a_x E_x (t-t0)/hbar`,

and similarly in y. Consequently the independently computed field power and the transport velocity obey

`P_field = -E_x v_x - E_y v_y`

in eV/fs when E is in V/angstrom and v in angstrom/fs.

This identity is a central TP1 gate because it ties the new current observable directly to the already validated D3/TP0 energy bookkeeping without introducing a second field convention.

## Periodic center diagnostic

TP1 also provides a circular first-moment center for localized states:

`z_x = sum rho(x,y) exp(i 2 pi x/N_x)`

and similarly in y. The angle gives a periodic center, while `|z|` measures whether the circular mean is well defined. A nearly uniform/delocalized state has `|z| ~= 0`, in which case the center is reported as undefined rather than returning a meaningless Cartesian average.

This circular center is a localization/visualization diagnostic only. The time-integrated bond flux is the transport coordinate.

## TP1 validation gates

The following gates are pre-registered before local execution:

1. bond-current continuity agrees with the direct Schrodinger population derivative to `1e-12 fs^-1` on a deterministic complex state;
2. all transport observables are invariant under a global electronic phase;
3. the zero-current result is exact for a localized basis state;
4. a uniform-lattice plane wave matches the analytic group velocity;
5. the field-work relation `P_field = -E dot v` agrees to `1e-12 eV/fs`;
6. the circular center handles a packet straddling a periodic boundary without jumping to the cell middle;
7. a uniform state is flagged as having an undefined circular center;
8. trapezoidal displacement accumulation is exact for constant velocity;
9. on the TP0 finite-temperature + field + IDC control, the field work obtained by integrating `-E dot v` agrees with the existing TP0 `W_field` to `1e-10 eV` per trajectory;
10. TP0 numerical stability gates remain satisfied during the transport-observable benchmark.

## Stochastic benchmark

The TP1 benchmark reuses the TP0 numerical control:

- 4 x 4 lattice;
- 300 K;
- `gamma_u = gamma_v = 0.01 fs^-1`;
- `E_x = 2 mV/angstrom`;
- `dt = 0.2 fs`;
- 4 ps total time;
- 1 ps burn-in;
- IDC-BM with `t_d = 180 fs`;
- CF4-Lanczos with Krylov dimension 6;
- projected intermolecular zero modes;
- four stochastic seeds.

For each trajectory TP1 records the unwrapped x/y displacement and the post-burn-in finite-time mean particle velocity. These values are diagnostics only. Four short 4 x 4 trajectories are insufficient to establish a steady state, linear response, or mobility.

## What TP1 does not claim

TP1 does not define a macroscopic current density because the present lattice model has no physical sample thickness/cross-sectional area layer. It does not infer mobility from `v/E`, does not claim a steady drift regime, and does not select production field strengths. Those questions belong to TP2, where field reversal, field-magnitude sweeps, longer trajectories, ensemble statistics, finite-size checks, and uncertainty estimates will be required.
