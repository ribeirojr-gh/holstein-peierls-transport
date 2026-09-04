from __future__ import annotations

import numpy as np
import pytest

from holstein_peierls.dynamics.ehrenfest import LatticeVelocity
from holstein_peierls.dynamics.langevin import LangevinBath
from holstein_peierls.dynamics.pair_coupled import (
    MovingPairHamiltonianFactory,
    PairCoupledState,
    PairLatticeMasses,
    pair_coupled_verlet_step,
)
from holstein_peierls.dynamics.pair_frozen import (
    bipolaron_exchange_symmetry_error,
    bipolaron_one_body_density_matrix,
    exciton_one_body_density_matrices,
    normalized_pair_state,
    symmetrized_bipolaron_state,
)
from holstein_peierls.dynamics.pair_thermal import (
    pair_coupled_baoab_step,
    pair_generalized_energy_residual_eV,
    pair_kinetic_temperature_K,
    pair_thermostatted_degrees_of_freedom,
    pair_zero_mode_means,
    project_pair_zero_modes,
)
from holstein_peierls.exciton.parameters import ExcitonParameters
from holstein_peierls.lattice import LatticeState
from holstein_peierls.two_particle.parameters import BipolaronParameters


def _masses() -> PairLatticeMasses:
    return PairLatticeMasses(75000.0, 150000.0)


def _lattice(size: int = 3) -> LatticeState:
    y, x = np.indices((size, size), dtype=np.float64)
    return LatticeState(
        5.0e-3 * np.cos(0.4 * x + 0.7 * y),
        2.0e-3 * np.sin(0.5 * x - 0.2 * y),
        2.5e-3 * np.cos(0.3 * x + 0.6 * y),
    )


def _parameters(sector: str, size: int = 3):
    center = (size // 2) * size + size // 2 + 1
    if sector == "bipolaron":
        return BipolaronParameters(
            nx=size,
            ny=size,
            pair_position=center,
            hubbard_u=0.22,
            nearest_neighbor_v=0.04,
        )
    return ExcitonParameters(
        nx=size,
        ny=size,
        exciton_position=center,
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


def _state(sector: str, seed: int = 17) -> tuple[PairCoupledState, object]:
    parameters = _parameters(sector)
    n = parameters.n_sites
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=n * n) + 1.0j * rng.normal(size=n * n)
    if sector == "bipolaron":
        psi = symmetrized_bipolaron_state(raw, n)
    else:
        psi = normalized_pair_state(raw, n)
    velocity = LatticeVelocity(
        2.0e-5 * rng.normal(size=(3, 3)),
        2.0e-5 * rng.normal(size=(3, 3)),
        2.0e-5 * rng.normal(size=(3, 3)),
    )
    return PairCoupledState(_lattice(), velocity, psi), parameters


def _assert_states_close(first: PairCoupledState, second: PairCoupledState) -> None:
    for a, b in (
        (first.lattice.u, second.lattice.u),
        (first.lattice.vx, second.lattice.vx),
        (first.lattice.vy, second.lattice.vy),
        (first.velocity.u, second.velocity.u),
        (first.velocity.vx, second.velocity.vx),
        (first.velocity.vy, second.velocity.vy),
        (first.electronic_state, second.electronic_state),
    ):
        np.testing.assert_allclose(a, b, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("sector", ["bipolaron", "exciton"])
def test_frictionless_retain_reduces_exactly_to_d6b_without_rng_consumption(
    sector: str,
) -> None:
    state, parameters = _state(sector)
    masses = _masses()
    factory = MovingPairHamiltonianFactory(sector, parameters)
    expected, expected_eval, expected_apply = pair_coupled_verlet_step(
        state,
        sector,
        parameters,
        masses,
        0.2,
        krylov_dimension=8,
        factory=factory,
    )
    rng = np.random.default_rng(12345)
    control = np.random.default_rng(12345)
    actual = pair_coupled_baoab_step(
        state,
        sector,
        parameters,
        masses,
        LangevinBath(300.0, 0.0, 0.0),
        rng,
        0.0,
        0.2,
        zero_mode_policy="retain",
        krylov_dimension=8,
        factory=factory,
    )
    _assert_states_close(actual.state, expected)
    assert actual.bath_heat_eV == 0.0
    assert actual.hamiltonian_evaluations == expected_eval
    assert actual.hamiltonian_applications == expected_apply
    np.testing.assert_array_equal(rng.standard_normal(8), control.standard_normal(8))


def test_project_policy_requires_explicitly_projected_initial_state() -> None:
    state, parameters = _state("bipolaron")
    with pytest.raises(ValueError, match="initially projected"):
        pair_coupled_baoab_step(
            state,
            "bipolaron",
            parameters,
            _masses(),
            LangevinBath(300.0, 0.01, 0.01),
            np.random.default_rng(1),
            0.0,
            0.2,
            zero_mode_policy="project",
        )


@pytest.mark.parametrize("sector", ["bipolaron", "exciton"])
def test_projected_finite_temperature_step_preserves_zero_modes_and_constraints(
    sector: str,
) -> None:
    raw_state, parameters = _state(sector, 29)
    state = project_pair_zero_modes(raw_state)
    result = pair_coupled_baoab_step(
        state,
        sector,
        parameters,
        _masses(),
        LangevinBath(300.0, 0.01, 0.01),
        np.random.default_rng(777),
        0.0,
        0.2,
        zero_mode_policy="project",
        krylov_dimension=8,
    )
    assert max(abs(value) for value in pair_zero_mode_means(result.state).values()) < 2.0e-18
    assert abs(np.linalg.norm(result.state.electronic_state) - 1.0) < 2.0e-13
    if sector == "bipolaron":
        assert bipolaron_exchange_symmetry_error(
            result.state.electronic_state, parameters.n_sites
        ) < 2.0e-13
        gamma = bipolaron_one_body_density_matrix(
            result.state.electronic_state, parameters.n_sites
        )
        assert abs(complex(np.trace(gamma)) - 2.0) < 2.0e-13
    else:
        gamma_e, gamma_h = exciton_one_body_density_matrices(
            result.state.electronic_state, parameters.n_sites
        )
        assert abs(complex(np.trace(gamma_e)) - 1.0) < 2.0e-13
        assert abs(complex(np.trace(gamma_h)) - 1.0) < 2.0e-13


def test_projected_degrees_of_freedom_and_kinetic_temperature() -> None:
    parameters = _parameters("exciton")
    masses = _masses()
    dof = pair_thermostatted_degrees_of_freedom(parameters, "project")
    assert dof == 3 * parameters.n_sites - 2 == 25
    target = 300.0
    kinetic = 0.5 * dof * 8.617333262145e-5 * target
    velocity = LatticeVelocity.zeros(parameters.ny, parameters.nx)
    velocity.u.flat[0] = np.sqrt(2.0 * kinetic / masses.intramolecular)
    calculated = pair_kinetic_temperature_K(
        velocity, masses, degrees_of_freedom=dof
    )
    np.testing.assert_allclose(calculated, target, rtol=0.0, atol=2.0e-12)


def test_finite_temperature_seeded_reproducibility() -> None:
    raw_state, parameters = _state("exciton", 31)
    state = project_pair_zero_modes(raw_state)
    bath = LangevinBath(300.0, 0.01, 0.01)
    kwargs = dict(
        state=state,
        sector="exciton",
        parameters=parameters,
        masses=_masses(),
        bath=bath,
        time_fs=0.0,
        dt_fs=0.2,
        zero_mode_policy="project",
        krylov_dimension=8,
    )
    first = pair_coupled_baoab_step(rng=np.random.default_rng(2026), **kwargs)
    second = pair_coupled_baoab_step(rng=np.random.default_rng(2026), **kwargs)
    _assert_states_close(first.state, second.state)
    assert first.bath_heat_eV == second.bath_heat_eV


@pytest.mark.parametrize("sector", ["bipolaron", "exciton"])
def test_single_step_generalized_energy_residual_is_small(sector: str) -> None:
    raw_state, parameters = _state(sector, 43)
    state = project_pair_zero_modes(raw_state)
    result = pair_coupled_baoab_step(
        state,
        sector,
        parameters,
        _masses(),
        LangevinBath(300.0, 0.01, 0.01),
        np.random.default_rng(99),
        0.0,
        0.05,
        zero_mode_policy="project",
        krylov_dimension=8,
    )
    residual = pair_generalized_energy_residual_eV(
        state,
        result.state,
        sector,
        parameters,
        _masses(),
        result.bath_heat_eV,
    )
    assert abs(residual) < 5.0e-7
