# Nearest-neighbour repulsion validation

This development stage extends the static singlet bipolaron model from onsite Hubbard repulsion `U` to an extended-Hubbard interaction with a positive nearest-neighbour term `V1`.

The interaction energy is

`E_int = U * P_onsite + V1 * P_NN`.

The exact parameter derivative is therefore `dE/dV1 = P_NN`, which is used as a regression test. The periodic nearest-neighbour graph uses the same minimum-image square lattice as the hopping model.

`V1` is presently a fixed electronic parameter. It does not depend explicitly on the instantaneous lattice coordinates, so there is no extra direct classical force. Structural gradients still change self-consistently because the correlated ground state changes with `V1`.

The first physical scan uses the validated anisotropic reference set with `alpha_x = 0.10 eV/A` and `alpha_y = 0.12 eV/A`, and the two strict large-cell reference values `U = 0.525` and `1.000 eV`. The initial grid is `V1 = 0, 5, 10, 20, 40, 60, 80 meV`, with onsite, intersite-x, intersite-y, and separated seeds relaxed independently.

The finite-cell binding reference remains the separated branch in the same cell. Final-state labels are assigned from pair observables rather than seed names.

A long-range screened interaction is deliberately deferred until explicit molecular spacings and dielectric screening are added to the model. The legacy 3.5 A factor used for velocity conversion is not treated as sufficient evidence for a material-specific Coulomb geometry.
