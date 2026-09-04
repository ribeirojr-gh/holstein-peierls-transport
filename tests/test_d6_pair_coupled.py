from __future__ import annotations

import numpy as np

from holstein_peierls.dynamics.ehrenfest import LatticeVelocity
from holstein_peierls.dynamics.pair_coupled import (
    MovingPairHamiltonianFactory,
    PairCoupledState,
    PairLatticeMasses,
    integrate_pair_coupled_dop853,
    integrate_pair_coupled_verlet,
    pair_dynamic_total_energy,
    pair_ehrenfest_gradient,
    pair_electronic_energy,
    pair_lattice_energy,
)
from holstein_peierls.dynamics.pair_frozen import (
    bipolaron_exchange_symmetry_error,
    bipolaron_one_body_density_matrix,
    exciton_one_body_density_matrices,
    normalized_pair_state,
    symmetrized_bipolaron_state,
)
from holstein_peierls.dynamics.time_dependent import compare_time_dependent_states
from holstein_peierls.exciton.parameters import ExcitonParameters
from holstein_peierls.exciton.solver import solve_exciton_ground_state
from holstein_peierls.lattice import LatticeState
from holstein_peierls.two_particle.parameters import BipolaronParameters
from holstein_peierls.two_particle.peierls import solve_holstein_peierls_ground_state


def _lattice() -> LatticeState:
    y, x = np.indices((3, 3), dtype=np.float64)
    return LatticeState(
        u=8.0e-3 * np.cos(0.7 * x + 0.4 * y),
        vx=3.0e-3 * np.sin(0.9 * x - 0.2 * y),
        vy=2.5e-3 * np.cos(0.3 * x + 0.8 * y),
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


def _masses() -> PairLatticeMasses:
    return PairLatticeMasses.from_legacy_as2(7.5e10, 1.5e11)


def _fixed_state_energy(lattice, sector, parameters, state) -> float:
    intra, inter = pair_lattice_energy(lattice, parameters)
    electronic = pair_electronic_energy(lattice, sector, parameters, state)
    return intra + inter + electronic


def _perturbed(lattice: LatticeState, field: str, index: tuple[int, int], delta: float):
    result = lattice.copy()
    getattr(result, field)[index] += delta
    return result


def test_bipolaron_complex_ehrenfest_gradient_matches_finite_difference() -> None:
    lattice = _lattice()
    parameters = _bipolaron_parameters()
    rng = np.random.default_rng(61)
    state = symmetrized_bipolaron_state(
        rng.normal(size=81) + 1.0j * rng.normal(size=81), 9
    )
    gradient = pair_ehrenfest_gradient(lattice, "bipolaron", parameters, state)
    eps = 1.0e-6
    for field, index in (("u", (1, 1)), ("vx", (1, 0)), ("vy", (0, 2))):
        plus = _fixed_state_energy(
            _perturbed(lattice, field, index, eps), "bipolaron", parameters, state
        )
        minus = _fixed_state_energy(
            _perturbed(lattice, field, index, -eps), "bipolaron", parameters, state
        )
        finite_difference = (plus - minus) / (2.0 * eps)
        np.testing.assert_allclose(
            getattr(gradient, field)[index], finite_difference, rtol=2.0e-6, atol=2.0e-8
        )


def test_exciton_complex_ehrenfest_gradient_matches_finite_difference() -> None:
    lattice = _lattice()
    parameters = _exciton_parameters()
    rng = np.random.default_rng(62)
    state = normalized_pair_state(
        rng.normal(size=81) + 1.0j * rng.normal(size=81), 9
    )
    gradient = pair_ehrenfest_gradient(lattice, "exciton", parameters, state)
    eps = 1.0e-6
    for field, index in (("u", (1, 1)), ("vx", (2, 1)), ("vy", (1, 2))):
        plus = _fixed_state_energy(
            _perturbed(lattice, field, index, eps), "exciton", parameters, state
        )
        minus = _fixed_state_energy(
            _perturbed(lattice, field, index, -eps), "exciton", parameters, state
        )
        finite_difference = (plus - minus) / (2.0 * eps)
        np.testing.assert_allclose(
            getattr(gradient, field)[index], finite_difference, rtol=2.0e-6, atol=2.0e-8
        )


def _initial_bipolaron_state() -> tuple[PairCoupledState, BipolaronParameters]:
    lattice = _lattice()
    parameters = _bipolaron_parameters()
    ground = solve_holstein_peierls_ground_state(lattice, parameters)
    velocity = LatticeVelocity.zeros(parameters.ny, parameters.nx)
    velocity.u[1, 1] = 1.5e-4
    return PairCoupledState(
        lattice,
        velocity,
        np.asarray(ground.wavefunction, dtype=np.complex128).ravel(order="C"),
    ), parameters


def _initial_exciton_state() -> tuple[PairCoupledState, ExcitonParameters]:
    lattice = _lattice()
    parameters = _exciton_parameters()
    ground = solve_exciton_ground_state(lattice, parameters)
    velocity = LatticeVelocity.zeros(parameters.ny, parameters.nx)
    velocity.u[1, 1] = 1.5e-4
    return PairCoupledState(
        lattice,
        velocity,
        np.asarray(ground.wavefunction, dtype=np.complex128).ravel(order="C"),
    ), parameters


def test_bipolaron_coupled_step_preserves_norm_symmetry_and_rdm_trace() -> None:
    initial, parameters = _initial_bipolaron_state()
    result = integrate_pair_coupled_verlet(
        initial,
        "bipolaron",
        parameters,
        _masses(),
        dt_fs=0.1,
        steps=10,
        krylov_dimension=8,
    )
    state = result.state.electronic_state
    assert abs(np.linalg.norm(state) - 1.0) < 5.0e-12
    assert bipolaron_exchange_symmetry_error(state, parameters.n_sites) < 5.0e-12
    gamma = bipolaron_one_body_density_matrix(state, parameters.n_sites)
    assert abs(np.trace(gamma) - 2.0) < 5.0e-12


def test_exciton_coupled_step_preserves_norm_and_rdm_traces() -> None:
    initial, parameters = _initial_exciton_state()
    result = integrate_pair_coupled_verlet(
        initial,
        "exciton",
        parameters,
        _masses(),
        dt_fs=0.1,
        steps=10,
        krylov_dimension=8,
    )
    state = result.state.electronic_state
    assert abs(np.linalg.norm(state) - 1.0) < 5.0e-12
    gamma_e, gamma_h = exciton_one_body_density_matrices(state, parameters.n_sites)
    assert abs(np.trace(gamma_e) - 1.0) < 5.0e-12
    assert abs(np.trace(gamma_h) - 1.0) < 5.0e-12


def test_pair_split_agrees_with_tight_dop853_reference() -> None:
    masses = _masses()
    for sector, constructor in (
        ("bipolaron", _initial_bipolaron_state),
        ("exciton", _initial_exciton_state),
    ):
        initial, parameters = constructor()
        split = integrate_pair_coupled_verlet(
            initial,
            sector,
            parameters,
            masses,
            dt_fs=0.1,
            steps=4,
            krylov_dimension=8,
        )
        reference = integrate_pair_coupled_dop853(
            initial,
            sector,
            parameters,
            masses,
            final_time_fs=0.4,
            rtol=2.0e-11,
            atol=2.0e-13,
            max_step_fs=0.02,
        )
        assert reference.success
        metrics = compare_time_dependent_states(
            reference.state.electronic_state, split.state.electronic_state
        )
        assert metrics.phase_aligned_state_error < 2.0e-6
        assert np.max(np.abs(split.state.lattice.u - reference.state.lattice.u)) < 2.0e-7
        assert np.max(np.abs(split.state.velocity.u - reference.state.velocity.u)) < 2.0e-7


def test_pair_total_energy_is_stable_over_short_zero_temperature_control() -> None:
    masses = _masses()
    for sector, constructor in (
        ("bipolaron", _initial_bipolaron_state),
        ("exciton", _initial_exciton_state),
    ):
        initial, parameters = constructor()
        factory = MovingPairHamiltonianFactory(sector, parameters)
        initial_energy = pair_dynamic_total_energy(
            initial, sector, parameters, masses, factory=factory
        ).total
        result = integrate_pair_coupled_verlet(
            initial,
            sector,
            parameters,
            masses,
            dt_fs=0.05,
            steps=20,
            krylov_dimension=8,
        )
        final_energy = pair_dynamic_total_energy(
            result.state, sector, parameters, masses, factory=factory
        ).total
        assert abs(final_energy - initial_energy) < 2.0e-6


def test_pair_mass_conversion_matches_legacy_d2_units() -> None:
    masses = _masses()
    np.testing.assert_allclose(masses.intramolecular, 7.5e4, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(masses.intermolecular, 1.5e5, rtol=0.0, atol=0.0)
