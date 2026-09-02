from types import SimpleNamespace

import numpy as np
import pytest

from holstein_peierls.hamiltonian import build_dense_hamiltonian
from holstein_peierls.lattice import LatticeState
from holstein_peierls.spin_adapted import (
    IsotropicControlParameters,
    IsotropicRelaxationSeed,
    half_filled_n_closed,
    harmonic_lattice_newton_direction,
    isotropic_relaxation_seed,
    isotropic_staggered_site_energies,
)


def test_half_filled_n_closed_matches_one_pi_electron_per_site() -> None:
    assert half_filled_n_closed(4) == 1
    assert half_filled_n_closed(16) == 7
    assert half_filled_n_closed(400) == 199
    with pytest.raises(ValueError, match="even N"):
        half_filled_n_closed(9)


def test_onsite_seed_has_zero_uniform_holstein_component() -> None:
    parameters = IsotropicControlParameters().to_polaron_parameters(nx=6, ny=6)
    seed = isotropic_relaxation_seed(
        parameters,
        IsotropicRelaxationSeed.ONSITE,
        amplitude=2.0e-3,
    )
    assert np.isclose(np.sum(seed.u), 0.0, atol=1.0e-15)
    assert np.max(np.abs(seed.u)) == pytest.approx(2.0e-3)
    assert np.count_nonzero(seed.vx) == 0
    assert np.count_nonzero(seed.vy) == 0


def test_x_and_y_bond_seeds_are_exact_isotropic_partners() -> None:
    parameters = IsotropicControlParameters().to_polaron_parameters(nx=6, ny=6)
    x_seed = isotropic_relaxation_seed(parameters, "bond_x", amplitude=1.5e-3)
    y_seed = isotropic_relaxation_seed(parameters, "bond_y", amplitude=1.5e-3)

    assert np.count_nonzero(x_seed.vx) == 2
    assert np.count_nonzero(x_seed.vy) == 0
    assert np.count_nonzero(y_seed.vx) == 0
    assert np.count_nonzero(y_seed.vy) == 2
    np.testing.assert_allclose(
        np.sort(x_seed.vx[x_seed.vx != 0.0]),
        np.sort(y_seed.vy[y_seed.vy != 0.0]),
        atol=0.0,
    )
    assert np.isclose(np.sum(x_seed.vx), 0.0, atol=1.0e-15)
    assert np.isclose(np.sum(y_seed.vy), 0.0, atol=1.0e-15)


def test_checkerboard_control_opens_the_requested_half_filled_gap() -> None:
    parameters = IsotropicControlParameters().to_polaron_parameters(nx=4, ny=4)
    requested_gap = 0.8
    site_energy = isotropic_staggered_site_energies(parameters, requested_gap)

    assert set(np.unique(site_energy)) == {-0.4, 0.4}
    np.testing.assert_allclose(site_energy, site_energy.T, atol=0.0)

    electronic_lattice = LatticeState(
        u=site_energy / parameters.alpha_intra,
        vx=np.zeros((parameters.ny, parameters.nx)),
        vy=np.zeros((parameters.ny, parameters.nx)),
    )
    eigenvalues = np.linalg.eigvalsh(
        build_dense_hamiltonian(electronic_lattice, parameters)
    )
    half = parameters.n_sites // 2
    numerical_gap = eigenvalues[half] - eigenvalues[half - 1]
    assert numerical_gap == pytest.approx(requested_gap, abs=2.0e-12)


def test_checkerboard_control_rejects_incompatible_periodic_cells() -> None:
    parameters = IsotropicControlParameters().to_polaron_parameters(nx=5, ny=4)
    with pytest.raises(ValueError, match="even nx and ny"):
        isotropic_staggered_site_energies(parameters, 0.8)


def test_harmonic_newton_direction_inverts_the_lattice_hessian() -> None:
    parameters = IsotropicControlParameters().to_polaron_parameters(nx=6, ny=4)
    rng = np.random.default_rng(17)

    expected_u = rng.normal(size=(parameters.ny, parameters.nx))
    expected_vx = rng.normal(size=(parameters.ny, parameters.nx))
    expected_vx -= np.mean(expected_vx, axis=1, keepdims=True)
    expected_vy = rng.normal(size=(parameters.ny, parameters.nx))
    expected_vy -= np.mean(expected_vy, axis=0, keepdims=True)

    gradient_u = -parameters.k1 * expected_u
    gradient_vx = -parameters.k2 * (
        2.0 * expected_vx
        - np.roll(expected_vx, shift=1, axis=1)
        - np.roll(expected_vx, shift=-1, axis=1)
    )
    gradient_vy = -parameters.k2 * (
        2.0 * expected_vy
        - np.roll(expected_vy, shift=1, axis=0)
        - np.roll(expected_vy, shift=-1, axis=0)
    )
    gradient = SimpleNamespace(
        u=gradient_u,
        vx=gradient_vx,
        vy=gradient_vy,
    )

    direction = harmonic_lattice_newton_direction(parameters, gradient)
    np.testing.assert_allclose(direction.u, expected_u, atol=2.0e-12)
    np.testing.assert_allclose(direction.vx, expected_vx, atol=2.0e-12)
    np.testing.assert_allclose(direction.vy, expected_vy, atol=2.0e-12)
    assert np.max(np.abs(np.mean(direction.vx, axis=1))) < 1.0e-13
    assert np.max(np.abs(np.mean(direction.vy, axis=0))) < 1.0e-13
