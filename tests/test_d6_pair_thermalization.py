from __future__ import annotations

import numpy as np
import pytest

from holstein_peierls.dynamics.pair_coupled import MovingPairHamiltonianFactory
from holstein_peierls.dynamics.pair_frozen import (
    bipolaron_frozen_action,
    dense_hamiltonian_from_action,
    exciton_frozen_action,
)
from holstein_peierls.dynamics.pair_thermalization import (
    diagnose_pair_thermalization,
    instantaneous_pair_spectrum,
    symmetric_pair_basis,
)
from holstein_peierls.exciton.parameters import ExcitonParameters
from holstein_peierls.lattice import LatticeState
from holstein_peierls.two_particle.parameters import BipolaronParameters


def _lattice() -> LatticeState:
    y, x = np.indices((3, 3), dtype=np.float64)
    return LatticeState(
        u=6.0e-3 * np.cos(0.5 * x + 0.3 * y),
        vx=2.0e-3 * np.sin(0.7 * x - 0.2 * y),
        vy=2.5e-3 * np.cos(0.4 * x + 0.6 * y),
    )


def _bip() -> BipolaronParameters:
    return BipolaronParameters(
        nx=3,
        ny=3,
        pair_position=5,
        hubbard_u=0.22,
        nearest_neighbor_v=0.04,
    )


def _exc() -> ExcitonParameters:
    return ExcitonParameters(
        nx=3,
        ny=3,
        exciton_position=5,
        electron_j0x=0.100,
        electron_j0y=0.015,
        hole_j0x=0.082,
        hole_j0y=0.021,
        electron_alpha_intra=3.0,
        hole_alpha_intra=2.4,
        electron_alpha_interx=0.4,
        electron_alpha_intery=0.4,
        hole_alpha_interx=0.31,
        hole_alpha_intery=0.28,
        onsite_attraction=0.30,
        nearest_neighbor_attraction=0.05,
    )


def test_symmetric_pair_basis_is_orthonormal_and_has_correct_dimension() -> None:
    n = 7
    basis = symmetric_pair_basis(n)
    assert basis.shape == (n * n, n * (n + 1) // 2)
    np.testing.assert_allclose(
        basis.conj().T @ basis,
        np.eye(basis.shape[1]),
        rtol=0.0,
        atol=2.0e-15,
    )
    for column in range(basis.shape[1]):
        matrix = basis[:, column].reshape((n, n), order="C")
        np.testing.assert_allclose(matrix, matrix.T, rtol=0.0, atol=0.0)


def test_bipolaron_spectrum_is_restricted_to_symmetric_singlet_sector() -> None:
    lattice = _lattice()
    parameters = _bip()
    action = bipolaron_frozen_action(lattice, parameters)
    basis = symmetric_pair_basis(parameters.n_sites)
    dense = dense_hamiltonian_from_action(action)
    restricted = basis.conj().T @ dense @ basis
    expected_values = np.linalg.eigvalsh(restricted)

    rng = np.random.default_rng(12)
    coordinates = rng.normal(size=basis.shape[1]) + 1.0j * rng.normal(size=basis.shape[1])
    coordinates /= np.linalg.norm(coordinates)
    state = basis @ coordinates
    spectrum = instantaneous_pair_spectrum(
        lattice, "bipolaron", parameters, state
    )
    assert spectrum.full_ordered_dimension == parameters.n_sites**2
    assert spectrum.physical_sector_dimension == parameters.n_sites * (parameters.n_sites + 1) // 2
    np.testing.assert_allclose(spectrum.eigenvalues_eV, expected_values, rtol=0.0, atol=2.0e-12)


def test_bipolaron_diagnostic_rejects_antisymmetric_contamination() -> None:
    lattice = _lattice()
    parameters = _bip()
    matrix = np.zeros((parameters.n_sites, parameters.n_sites), dtype=np.complex128)
    matrix[1, 2] = 1.0 / np.sqrt(2.0)
    matrix[2, 1] = -1.0 / np.sqrt(2.0)
    with pytest.raises(ValueError, match="symmetric spatial singlet"):
        instantaneous_pair_spectrum(
            lattice,
            "bipolaron",
            parameters,
            matrix.ravel(order="C"),
        )


def test_bipolaron_ground_adiabatic_state_has_unit_ground_population() -> None:
    lattice = _lattice()
    parameters = _bip()
    basis = symmetric_pair_basis(parameters.n_sites)
    action = bipolaron_frozen_action(lattice, parameters)
    dense = dense_hamiltonian_from_action(action)
    restricted = basis.conj().T @ dense @ basis
    _, vectors = np.linalg.eigh(restricted)
    state = basis @ vectors[:, 0]
    snapshot = diagnose_pair_thermalization(
        lattice,
        "bipolaron",
        parameters,
        state,
        300.0,
    )
    assert snapshot.ground_manifold_population > 1.0 - 2.0e-12
    assert snapshot.physical_sector_dimension == 45
    np.testing.assert_allclose(
        np.sum(snapshot.adiabatic_occupations), 1.0, rtol=0.0, atol=2.0e-15
    )


def test_exciton_diagnostic_uses_full_ordered_pair_space() -> None:
    lattice = _lattice()
    parameters = _exc()
    action = exciton_frozen_action(lattice, parameters)
    dense = dense_hamiltonian_from_action(action)
    _, vectors = np.linalg.eigh(dense)
    state = vectors[:, 0]
    snapshot = diagnose_pair_thermalization(
        lattice,
        "exciton",
        parameters,
        state,
        300.0,
    )
    assert snapshot.full_ordered_dimension == parameters.n_sites**2
    assert snapshot.physical_sector_dimension == parameters.n_sites**2
    assert snapshot.ground_manifold_population > 1.0 - 2.0e-12
    assert snapshot.tv_to_canonical >= 0.0
    assert snapshot.tv_to_uniform >= 0.0


def test_pair_diagnostics_do_not_modify_state_or_factory_counter() -> None:
    lattice = _lattice()
    parameters = _exc()
    factory = MovingPairHamiltonianFactory("exciton", parameters)
    rng = np.random.default_rng(44)
    state = rng.normal(size=parameters.n_sites**2) + 1.0j * rng.normal(size=parameters.n_sites**2)
    state = np.asarray(state / np.linalg.norm(state), dtype=np.complex128)
    before = state.copy()
    diagnose_pair_thermalization(
        lattice,
        "exciton",
        parameters,
        state,
        300.0,
        factory=factory,
    )
    np.testing.assert_array_equal(state, before)
