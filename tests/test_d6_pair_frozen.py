from __future__ import annotations

import numpy as np

from holstein_peierls.dynamics.frozen import (
    exact_spectral_step,
    lanczos_exponential_step,
)
from holstein_peierls.dynamics.pair_frozen import (
    bipolaron_exchange_symmetry_error,
    bipolaron_frozen_action,
    bipolaron_one_body_density_matrix,
    dense_hamiltonian_from_action,
    electronic_energy_expectation,
    exciton_frozen_action,
    exciton_one_body_density_matrices,
    normalized_pair_state,
    symmetrized_bipolaron_state,
)
from holstein_peierls.exciton.parameters import ExcitonParameters
from holstein_peierls.exciton.solver import exciton_linear_operator
from holstein_peierls.lattice import LatticeState
from holstein_peierls.two_particle.parameters import BipolaronParameters
from holstein_peierls.two_particle.peierls import two_particle_linear_operator


def _lattice() -> LatticeState:
    y, x = np.indices((3, 3), dtype=np.float64)
    return LatticeState(
        u=1.0e-2 * np.cos(0.7 * x + 0.4 * y),
        vx=4.0e-3 * np.sin(0.9 * x - 0.2 * y),
        vy=3.0e-3 * np.cos(0.3 * x + 0.8 * y),
    )


def _bipolaron_parameters() -> BipolaronParameters:
    return BipolaronParameters(
        nx=3,
        ny=3,
        pair_position=5,
        hubbard_u=0.22,
        nearest_neighbor_v=0.04,
    )


def _exciton_parameters() -> ExcitonParameters:
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


def test_bipolaron_complex_action_agrees_with_static_operator_on_real_vectors() -> None:
    lattice = _lattice()
    parameters = _bipolaron_parameters()
    dynamic = bipolaron_frozen_action(lattice, parameters)
    static = two_particle_linear_operator(lattice, parameters)
    rng = np.random.default_rng(12)
    vector = rng.normal(size=dynamic.dimension)
    expected = np.asarray(static @ vector, dtype=np.float64)
    actual = dynamic(vector)
    np.testing.assert_allclose(actual.real, expected, rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(actual.imag, 0.0, rtol=0.0, atol=0.0)


def test_exciton_complex_action_agrees_with_static_operator_on_real_vectors() -> None:
    lattice = _lattice()
    parameters = _exciton_parameters()
    dynamic = exciton_frozen_action(lattice, parameters)
    static = exciton_linear_operator(lattice, parameters)
    rng = np.random.default_rng(19)
    vector = rng.normal(size=dynamic.dimension)
    expected = np.asarray(static @ vector, dtype=np.float64)
    actual = dynamic(vector)
    np.testing.assert_allclose(actual.real, expected, rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(actual.imag, 0.0, rtol=0.0, atol=0.0)


def test_pair_actions_preserve_complex_amplitudes() -> None:
    lattice = _lattice()
    rng = np.random.default_rng(7)
    for action in (
        bipolaron_frozen_action(lattice, _bipolaron_parameters()),
        exciton_frozen_action(lattice, _exciton_parameters()),
    ):
        state = rng.normal(size=action.dimension) + 1.0j * rng.normal(size=action.dimension)
        applied = action(state)
        assert np.linalg.norm(applied.imag) > 1.0e-8
        assert applied.dtype == np.complex128


def test_small_dense_pair_hamiltonians_are_hermitian() -> None:
    lattice = _lattice()
    for action in (
        bipolaron_frozen_action(lattice, _bipolaron_parameters()),
        exciton_frozen_action(lattice, _exciton_parameters()),
    ):
        dense = dense_hamiltonian_from_action(action)
        np.testing.assert_allclose(dense, dense.conj().T, rtol=0.0, atol=2.0e-14)
        assert action.applications == 0


def test_bipolaron_lanczos_matches_exact_and_preserves_exchange_symmetry() -> None:
    lattice = _lattice()
    action = bipolaron_frozen_action(lattice, _bipolaron_parameters())
    rng = np.random.default_rng(23)
    initial = symmetrized_bipolaron_state(
        rng.normal(size=action.dimension) + 1.0j * rng.normal(size=action.dimension),
        action.n_sites,
    )
    dense = dense_hamiltonian_from_action(action)
    reference = exact_spectral_step(dense, initial, 0.2).state
    propagated = lanczos_exponential_step(
        action,
        initial,
        0.2,
        krylov_dimension=30,
    ).state
    np.testing.assert_allclose(propagated, reference, rtol=0.0, atol=2.0e-12)
    assert abs(np.linalg.norm(propagated) - 1.0) < 2.0e-13
    assert bipolaron_exchange_symmetry_error(propagated, action.n_sites) < 2.0e-13


def test_exciton_lanczos_matches_exact_without_exchange_projection() -> None:
    lattice = _lattice()
    action = exciton_frozen_action(lattice, _exciton_parameters())
    initial_matrix = np.zeros((action.n_sites, action.n_sites), dtype=np.complex128)
    initial_matrix[1, 6] = 1.0
    initial = normalized_pair_state(initial_matrix, action.n_sites)
    dense = dense_hamiltonian_from_action(action)
    reference = exact_spectral_step(dense, initial, 0.2).state
    propagated = lanczos_exponential_step(
        action,
        initial,
        0.2,
        krylov_dimension=30,
    ).state
    np.testing.assert_allclose(propagated, reference, rtol=0.0, atol=2.0e-12)
    psi = propagated.reshape((action.n_sites, action.n_sites), order="C")
    assert np.max(np.abs(psi - psi.T)) > 1.0e-3
    assert abs(np.linalg.norm(propagated) - 1.0) < 2.0e-13


def test_complex_pair_reduced_density_matrix_traces() -> None:
    rng = np.random.default_rng(31)
    n = 9
    bip = symmetrized_bipolaron_state(
        rng.normal(size=n * n) + 1.0j * rng.normal(size=n * n), n
    )
    gamma_b = bipolaron_one_body_density_matrix(bip, n)
    np.testing.assert_allclose(np.trace(gamma_b), 2.0, rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(gamma_b, gamma_b.conj().T, rtol=0.0, atol=2.0e-14)

    exc = normalized_pair_state(
        rng.normal(size=n * n) + 1.0j * rng.normal(size=n * n), n
    )
    gamma_e, gamma_h = exciton_one_body_density_matrices(exc, n)
    np.testing.assert_allclose(np.trace(gamma_e), 1.0, rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(np.trace(gamma_h), 1.0, rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(gamma_e, gamma_e.conj().T, rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(gamma_h, gamma_h.conj().T, rtol=0.0, atol=2.0e-14)


def test_frozen_pair_energy_expectation_is_real_and_phase_invariant() -> None:
    lattice = _lattice()
    action = exciton_frozen_action(lattice, _exciton_parameters())
    rng = np.random.default_rng(44)
    state = normalized_pair_state(
        rng.normal(size=action.dimension) + 1.0j * rng.normal(size=action.dimension),
        action.n_sites,
    )
    energy = electronic_energy_expectation(action, state)
    phased = electronic_energy_expectation(action, np.exp(0.713j) * state)
    np.testing.assert_allclose(phased, energy, rtol=0.0, atol=2.0e-14)
