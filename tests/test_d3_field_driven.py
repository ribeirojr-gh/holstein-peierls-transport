"""Validation gates for D3 field-driven zero-temperature dynamics."""

from __future__ import annotations

import numpy as np

from holstein_peierls.dynamics.coupled import (
    CoupledEhrenfestState,
    coupled_verlet_step,
)
from holstein_peierls.dynamics.driven import (
    coupled_field_verlet_step,
    field_dynamic_total_energy,
    field_ehrenfest_gradient,
    field_electronic_energy_expectation,
    field_power,
    integrate_coupled_field_verlet,
)
from holstein_peierls.dynamics.ehrenfest import (
    LatticeVelocity,
    ehrenfest_gradient,
    electronic_energy_expectation,
)
from holstein_peierls.dynamics.field import UniformElectricField2D
from holstein_peierls.energy import lattice_energy
from holstein_peierls.lattice import LatticeState
from holstein_peierls.parameters import StaticPolaronParameters


def _control():
    parameters = StaticPolaronParameters(
        nx=4,
        ny=4,
        polaron_position=6,
    )
    rng = np.random.default_rng(314159)
    lattice = LatticeState(
        0.01 * rng.normal(size=(4, 4)),
        0.01 * rng.normal(size=(4, 4)),
        0.01 * rng.normal(size=(4, 4)),
    )
    psi = rng.normal(size=16) + 1.0j * rng.normal(size=16)
    psi = np.asarray(psi / np.linalg.norm(psi), dtype=np.complex128)
    velocity = LatticeVelocity.zeros(4, 4)
    return parameters, lattice, velocity, psi


def _matter_potential_energy(
    lattice,
    parameters,
    psi,
    field,
    time_fs,
):
    intra, inter = lattice_energy(lattice, parameters)
    return intra + inter + field_electronic_energy_expectation(
        lattice,
        parameters,
        psi,
        field,
        time_fs,
    )


def test_zero_field_reduces_to_d2_energy_gradient_and_step():
    parameters, lattice, velocity, psi = _control()
    field = UniformElectricField2D()
    time_fs = 1.234
    dt_fs = 0.05

    assert np.isclose(
        field_electronic_energy_expectation(
            lattice, parameters, psi, field, time_fs
        ),
        electronic_energy_expectation(lattice, parameters, psi),
        rtol=0.0,
        atol=1.0e-14,
    )

    driven_gradient = field_ehrenfest_gradient(
        lattice, parameters, psi, field, time_fs
    )
    d2_gradient = ehrenfest_gradient(lattice, parameters, psi)
    for driven, reference in (
        (driven_gradient.u, d2_gradient.u),
        (driven_gradient.vx, d2_gradient.vx),
        (driven_gradient.vy, d2_gradient.vy),
    ):
        assert np.allclose(driven, reference, rtol=0.0, atol=1.0e-14)

    assert field_power(lattice, parameters, psi, field, time_fs) == 0.0

    state = CoupledEhrenfestState(lattice, velocity, psi)
    d2_state, d2_eval, d2_apply = coupled_verlet_step(
        state,
        parameters,
        dt_fs,
        electronic_method="cfm4_lanczos",
        krylov_dimension=6,
    )
    d3_state, work, d3_eval, d3_apply = coupled_field_verlet_step(
        state,
        parameters,
        field,
        time_fs,
        dt_fs,
        electronic_method="cfm4_lanczos",
        krylov_dimension=6,
    )
    assert work == 0.0
    assert d3_eval == d2_eval
    assert d3_apply == d2_apply
    assert np.allclose(d3_state.electronic_state, d2_state.electronic_state, atol=2.0e-13)
    for driven, reference in (
        (d3_state.lattice.u, d2_state.lattice.u),
        (d3_state.lattice.vx, d2_state.lattice.vx),
        (d3_state.lattice.vy, d2_state.lattice.vy),
        (d3_state.velocity.u, d2_state.velocity.u),
        (d3_state.velocity.vx, d2_state.velocity.vx),
        (d3_state.velocity.vy, d2_state.velocity.vy),
    ):
        assert np.allclose(driven, reference, rtol=0.0, atol=2.0e-13)


def test_field_ehrenfest_gradient_matches_central_finite_differences():
    parameters, lattice, _, psi = _control()
    field = UniformElectricField2D.from_millivolt_per_angstrom(
        3.0,
        0.37,
        ax_angstrom=3.0,
        ay_angstrom=3.5,
    )
    time_fs = 4.2
    analytic = field_ehrenfest_gradient(lattice, parameters, psi, field, time_fs)
    epsilon = 1.0e-6

    for name, index in (("u", (1, 2)), ("vx", (2, 1)), ("vy", (0, 3))):
        plus = lattice.copy()
        minus = lattice.copy()
        getattr(plus, name)[index] += epsilon
        getattr(minus, name)[index] -= epsilon
        numeric = (
            _matter_potential_energy(plus, parameters, psi, field, time_fs)
            - _matter_potential_energy(minus, parameters, psi, field, time_fs)
        ) / (2.0 * epsilon)
        assert np.isclose(
            getattr(analytic, name)[index],
            numeric,
            rtol=2.0e-7,
            atol=2.0e-8,
        )


def test_field_power_matches_explicit_time_derivative_of_energy():
    parameters, lattice, _, psi = _control()
    field = UniformElectricField2D.from_millivolt_per_angstrom(
        4.0,
        -0.23,
        ax_angstrom=3.1,
        ay_angstrom=3.4,
    )
    time_fs = 2.7
    epsilon = 1.0e-4
    numeric = (
        field_electronic_energy_expectation(
            lattice, parameters, psi, field, time_fs + epsilon
        )
        - field_electronic_energy_expectation(
            lattice, parameters, psi, field, time_fs - epsilon
        )
    ) / (2.0 * epsilon)
    analytic = field_power(lattice, parameters, psi, field, time_fs)
    assert np.isclose(analytic, numeric, rtol=2.0e-7, atol=2.0e-10)


def test_cf4_field_driven_step_preserves_electronic_norm():
    parameters, lattice, velocity, psi = _control()
    field = UniformElectricField2D.from_millivolt_per_angstrom(2.0, 0.2)
    state = CoupledEhrenfestState(lattice, velocity, psi)
    result = integrate_coupled_field_verlet(
        state,
        parameters,
        field,
        dt_fs=0.05,
        steps=20,
        electronic_method="cfm4_lanczos",
        krylov_dimension=6,
    )
    assert abs(np.linalg.norm(result.state.electronic_state) - np.linalg.norm(psi)) < 2.0e-13


def test_matter_energy_minus_field_work_residual_converges_with_step_halving():
    parameters, lattice, velocity, psi = _control()
    field = UniformElectricField2D.from_millivolt_per_angstrom(5.0, 0.41)
    initial = CoupledEhrenfestState(lattice, velocity, psi)
    initial_energy = field_dynamic_total_energy(
        lattice,
        velocity,
        parameters,
        psi,
        field,
        0.0,
    ).total
    final_time_fs = 0.4
    residuals = []

    for dt_fs in (0.1, 0.05, 0.025):
        steps = int(round(final_time_fs / dt_fs))
        result = integrate_coupled_field_verlet(
            initial,
            parameters,
            field,
            dt_fs=dt_fs,
            steps=steps,
            electronic_method="cfm4_lanczos",
            krylov_dimension=6,
        )
        final_energy = field_dynamic_total_energy(
            result.state.lattice,
            result.state.velocity,
            parameters,
            result.state.electronic_state,
            field,
            final_time_fs,
        ).total
        residuals.append(abs((final_energy - initial_energy) - result.field_work_eV))

    assert residuals[1] < 0.4 * residuals[0]
    assert residuals[2] < 0.4 * residuals[1]
