"""Exact x/y covariance checks for the canonical isotropic S0 control."""

import numpy as np

from holstein_peierls.hamiltonian import build_dense_hamiltonian
from holstein_peierls.lattice import LatticeState
from holstein_peierls.spin_adapted import (
    IsotropicControlParameters,
    IsotropicRelaxationSeed,
    density_density_control_interaction,
    isotropic_relaxation_seed,
    isotropic_staggered_site_energies,
)


def _transpose_xy(state: LatticeState) -> LatticeState:
    """Exchange x and y, including the corresponding Peierls coordinates."""
    return LatticeState(
        u=np.asarray(state.u.T, dtype=np.float64).copy(),
        vx=np.asarray(state.vy.T, dtype=np.float64).copy(),
        vy=np.asarray(state.vx.T, dtype=np.float64).copy(),
    )


def _xy_site_permutation(size: int) -> np.ndarray:
    """Return old-site indices in the C-order basis after x/y exchange."""
    return np.arange(size * size, dtype=np.int64).reshape(size, size).T.ravel()


def test_bond_x_and_bond_y_seeds_are_exact_xy_partners() -> None:
    parameters = IsotropicControlParameters().to_polaron_parameters(nx=4, ny=4)
    bond_x = isotropic_relaxation_seed(parameters, IsotropicRelaxationSeed.BOND_X)
    bond_y = isotropic_relaxation_seed(parameters, IsotropicRelaxationSeed.BOND_Y)
    transposed = _transpose_xy(bond_x)

    np.testing.assert_array_equal(transposed.u, bond_y.u)
    np.testing.assert_array_equal(transposed.vx, bond_y.vx)
    np.testing.assert_array_equal(transposed.vy, bond_y.vy)


def test_gapped_isotropic_hamiltonian_and_interaction_are_xy_covariant() -> None:
    control = IsotropicControlParameters()
    parameters = control.to_polaron_parameters(nx=4, ny=4)
    rng = np.random.default_rng(20260902)
    physical = LatticeState(
        u=rng.normal(scale=1.0e-2, size=(4, 4)),
        vx=rng.normal(scale=1.0e-2, size=(4, 4)),
        vy=rng.normal(scale=1.0e-2, size=(4, 4)),
    )
    partner = _transpose_xy(physical)

    checkerboard = isotropic_staggered_site_energies(parameters, 2.0)
    electronic = LatticeState(
        u=physical.u + checkerboard / parameters.alpha_intra,
        vx=physical.vx.copy(),
        vy=physical.vy.copy(),
    )
    electronic_partner = LatticeState(
        u=partner.u + checkerboard / parameters.alpha_intra,
        vx=partner.vx.copy(),
        vy=partner.vy.copy(),
    )

    permutation = _xy_site_permutation(4)
    hamiltonian = build_dense_hamiltonian(electronic, parameters)
    partner_hamiltonian = build_dense_hamiltonian(electronic_partner, parameters)
    expected_hamiltonian = hamiltonian[np.ix_(permutation, permutation)]
    np.testing.assert_allclose(
        partner_hamiltonian,
        expected_hamiltonian,
        rtol=0.0,
        atol=2.0e-15,
    )

    interaction = density_density_control_interaction(
        parameters,
        onsite_u=0.525,
        nearest_neighbor_v=0.08,
    )
    expected_interaction = interaction[np.ix_(permutation, permutation)]
    np.testing.assert_array_equal(interaction, expected_interaction)
