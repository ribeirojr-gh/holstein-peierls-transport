"""Validation gates for the deterministic D2 Ehrenfest primitives."""

from __future__ import annotations

import numpy as np

from holstein_peierls.dynamics.ehrenfest import (
    LatticeVelocity,
    dynamic_total_energy,
    ehrenfest_gradient,
    electronic_energy_expectation,
    lattice_kinetic_energy,
    lattice_masses_fs,
    legacy_mass_to_fs,
)
from holstein_peierls.electronic import solve_ground_state
from holstein_peierls.energy import lattice_energy
from holstein_peierls.gradients import energy_gradient
from holstein_peierls.hamiltonian import build_dense_hamiltonian
from holstein_peierls.lattice import LatticeState
from holstein_peierls.parameters import StaticPolaronParameters


def _control_problem() -> tuple[LatticeState, StaticPolaronParameters, np.ndarray]:
    parameters = StaticPolaronParameters(nx=3, ny=3, polaron_position=5)
    rng = np.random.default_rng(20260903)
    state = LatticeState(
        u=0.03 * rng.normal(size=(3, 3)),
        vx=0.04 * rng.normal(size=(3, 3)),
        vy=0.04 * rng.normal(size=(3, 3)),
    )
    psi = rng.normal(size=9) + 1.0j * rng.normal(size=9)
    psi = np.asarray(psi / np.linalg.norm(psi), dtype=np.complex128)
    return state, parameters, psi


def _fixed_state_energy(
    state: LatticeState,
    parameters: StaticPolaronParameters,
    psi: np.ndarray,
) -> float:
    intra, inter = lattice_energy(state, parameters)
    return intra + inter + electronic_energy_expectation(state, parameters, psi)


def _finite_difference_component(
    state: LatticeState,
    parameters: StaticPolaronParameters,
    psi: np.ndarray,
    field: str,
    index: tuple[int, int],
    step: float = 1.0e-6,
) -> float:
    plus = state.copy()
    minus = state.copy()
    getattr(plus, field)[index] += step
    getattr(minus, field)[index] -= step
    return (
        _fixed_state_energy(plus, parameters, psi)
        - _fixed_state_energy(minus, parameters, psi)
    ) / (2.0 * step)


def test_complex_ehrenfest_gradient_matches_finite_differences() -> None:
    state, parameters, psi = _control_problem()
    gradient = ehrenfest_gradient(state, parameters, psi)

    for field, index in (
        ("u", (1, 1)),
        ("vx", (0, 2)),
        ("vy", (2, 0)),
    ):
        numerical = _finite_difference_component(
            state,
            parameters,
            psi,
            field,
            index,
        )
        analytical = float(getattr(gradient, field)[index])
        assert np.isclose(analytical, numerical, rtol=2.0e-7, atol=2.0e-9)


def test_complex_gradient_reduces_to_static_optimized_gradient() -> None:
    state, parameters, _ = _control_problem()
    ground = solve_ground_state(state, parameters, solver="dense_lowest")
    static, _ = energy_gradient(
        state,
        parameters,
        ground_state=ground,
        mode="optimized",
    )
    dynamic = ehrenfest_gradient(state, parameters, ground.wavefunction)

    np.testing.assert_allclose(dynamic.u, static.u, rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(dynamic.vx, static.vx, rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(dynamic.vy, static.vy, rtol=0.0, atol=2.0e-14)


def test_electronic_energy_uses_propagated_state_expectation() -> None:
    state, parameters, psi = _control_problem()
    hamiltonian = build_dense_hamiltonian(state, parameters)
    direct = float(np.real(np.vdot(psi, hamiltonian @ psi) / np.vdot(psi, psi)))
    observed = electronic_energy_expectation(state, parameters, psi)
    assert np.isclose(observed, direct, rtol=0.0, atol=2.0e-14)


def test_mass_conversion_matches_attosecond_to_femtosecond_units() -> None:
    parameters = StaticPolaronParameters()
    mass_u, mass_v = lattice_masses_fs(parameters)
    assert legacy_mass_to_fs(7.5e10) == 7.5e4
    assert legacy_mass_to_fs(1.5e11) == 1.5e5
    assert mass_u == 7.5e4
    assert mass_v == 1.5e5

    # F/M in A/as^2 converted to A/fs^2 gains 10^6, exactly cancelling
    # M_as -> M_fs / 10^6 in the denominator.
    force = 0.25
    acceleration_as = force / parameters.m1
    acceleration_fs_from_as = acceleration_as * 1.0e6
    acceleration_fs = force / mass_u
    assert np.isclose(acceleration_fs, acceleration_fs_from_as, rtol=0.0, atol=1.0e-18)


def test_lattice_kinetic_and_total_energy_breakdown() -> None:
    state, parameters, psi = _control_problem()
    velocity = LatticeVelocity.zeros(parameters.ny, parameters.nx)
    velocity.u[1, 1] = 2.0e-3
    velocity.vx[0, 0] = -1.0e-3
    velocity.vy[2, 2] = 0.5e-3

    mass_u, mass_v = lattice_masses_fs(parameters)
    expected_kinetic = (
        0.5 * mass_u * (2.0e-3) ** 2
        + 0.5 * mass_v * ((1.0e-3) ** 2 + (0.5e-3) ** 2)
    )
    assert np.isclose(
        lattice_kinetic_energy(velocity, parameters),
        expected_kinetic,
        rtol=0.0,
        atol=1.0e-15,
    )

    breakdown = dynamic_total_energy(state, velocity, parameters, psi)
    intra, inter = lattice_energy(state, parameters)
    assert np.isclose(breakdown.intramolecular_lattice, intra)
    assert np.isclose(breakdown.intermolecular_lattice, inter)
    assert np.isclose(breakdown.lattice_kinetic, expected_kinetic)
    assert np.isclose(
        breakdown.electronic,
        electronic_energy_expectation(state, parameters, psi),
    )
    assert np.isclose(
        breakdown.total,
        intra + inter + expected_kinetic + breakdown.electronic,
    )
