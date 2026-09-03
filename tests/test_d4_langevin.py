"""Validation gates for the D4 finite-temperature lattice bath."""

from __future__ import annotations

import numpy as np

from holstein_peierls.dynamics.classical import velocity_verlet_step
from holstein_peierls.dynamics.ehrenfest import LatticeVelocity, lattice_masses_fs
from holstein_peierls.dynamics.langevin import (
    BOLTZMANN_EV_PER_K,
    LangevinBath,
    baoab_step,
    kinetic_temperature_K,
    ou_parameters,
    ou_velocity_step,
)
from holstein_peierls.gradients import LatticeGradient
from holstein_peierls.lattice import LatticeState
from holstein_peierls.parameters import StaticPolaronParameters


def _control(nx: int = 4, ny: int = 4) -> StaticPolaronParameters:
    return StaticPolaronParameters(
        nx=nx,
        ny=ny,
        polaron_position=1,
    )


def _harmonic_force(parameters: StaticPolaronParameters):
    def force(state: LatticeState) -> LatticeGradient:
        # Independent u oscillators; vx/vy are deliberately force-free in this
        # thermostat-unit control and their bath friction is set to zero.
        zeros = np.zeros_like(state.u)
        return LatticeGradient(-parameters.k1 * state.u, zeros.copy(), zeros.copy())

    return force


def test_ou_coefficients_satisfy_exact_fluctuation_dissipation_variance():
    parameters = _control()
    mass_u, _ = lattice_masses_fs(parameters)
    temperature = 300.0
    gamma = 0.0125
    dt = 0.2
    coefficients = ou_parameters(
        gamma_per_fs=gamma,
        temperature_K=temperature,
        mass_eV_fs2_per_A2=mass_u,
        dt_fs=dt,
    )
    expected_c = np.exp(-gamma * dt)
    expected_variance = (
        1.0 - expected_c**2
    ) * BOLTZMANN_EV_PER_K * temperature / mass_u
    assert np.isclose(coefficients.damping_factor, expected_c, rtol=0.0, atol=2.0e-16)
    assert np.isclose(
        coefficients.velocity_sigma**2,
        expected_variance,
        rtol=2.0e-15,
        atol=0.0,
    )


def test_frictionless_ou_is_identity_and_does_not_consume_rng_state():
    parameters = _control()
    rng_values = np.random.default_rng(10)
    velocity = LatticeVelocity(
        rng_values.normal(size=(4, 4)),
        rng_values.normal(size=(4, 4)),
        rng_values.normal(size=(4, 4)),
    )
    bath = LangevinBath(temperature_K=300.0, gamma_u_per_fs=0.0, gamma_v_per_fs=0.0)

    rng = np.random.default_rng(12345)
    expected_next = np.random.default_rng(12345).standard_normal()
    transformed = ou_velocity_step(velocity, parameters, bath, 0.2, rng)
    actual_next = rng.standard_normal()

    assert np.array_equal(transformed.u, velocity.u)
    assert np.array_equal(transformed.vx, velocity.vx)
    assert np.array_equal(transformed.vy, velocity.vy)
    assert actual_next == expected_next


def test_zero_temperature_ou_is_deterministic_exponential_damping():
    parameters = _control()
    velocity = LatticeVelocity(
        np.full((4, 4), 0.2),
        np.full((4, 4), -0.3),
        np.full((4, 4), 0.4),
    )
    bath = LangevinBath(temperature_K=0.0, gamma_u_per_fs=0.02, gamma_v_per_fs=0.03)
    dt = 0.5
    rng = np.random.default_rng(77)
    expected_next = np.random.default_rng(77).standard_normal()
    result = ou_velocity_step(velocity, parameters, bath, dt, rng)

    assert np.allclose(result.u, np.exp(-0.02 * dt) * velocity.u, rtol=0.0, atol=1.0e-15)
    assert np.allclose(result.vx, np.exp(-0.03 * dt) * velocity.vx, rtol=0.0, atol=1.0e-15)
    assert np.allclose(result.vy, np.exp(-0.03 * dt) * velocity.vy, rtol=0.0, atol=1.0e-15)
    # No random variates are consumed when the FDT amplitude is exactly zero.
    assert rng.standard_normal() == expected_next


def test_seeded_ou_stream_is_reproducible_and_channels_are_independent():
    parameters = _control()
    velocity = LatticeVelocity.zeros(4, 4)
    bath = LangevinBath(temperature_K=300.0, gamma_u_per_fs=0.01, gamma_v_per_fs=0.02)

    first = ou_velocity_step(
        velocity,
        parameters,
        bath,
        0.2,
        np.random.default_rng(2026),
    )
    second = ou_velocity_step(
        velocity,
        parameters,
        bath,
        0.2,
        np.random.default_rng(2026),
    )
    different = ou_velocity_step(
        velocity,
        parameters,
        bath,
        0.2,
        np.random.default_rng(2027),
    )

    assert np.array_equal(first.u, second.u)
    assert np.array_equal(first.vx, second.vx)
    assert np.array_equal(first.vy, second.vy)
    assert not np.array_equal(first.u, different.u)
    assert not np.array_equal(first.vx, first.vy)


def test_kinetic_temperature_uses_three_n_equipartition_dof():
    parameters = _control(nx=2, ny=2)
    mass_u, mass_v = lattice_masses_fs(parameters)
    target_temperature = 250.0
    # Give every degree of freedom exactly 1/2 kBT kinetic energy.
    u_speed = np.sqrt(BOLTZMANN_EV_PER_K * target_temperature / mass_u)
    v_speed = np.sqrt(BOLTZMANN_EV_PER_K * target_temperature / mass_v)
    velocity = LatticeVelocity(
        np.full((2, 2), u_speed),
        np.full((2, 2), v_speed),
        np.full((2, 2), -v_speed),
    )
    assert np.isclose(
        kinetic_temperature_K(velocity, parameters),
        target_temperature,
        rtol=2.0e-15,
        atol=0.0,
    )


def test_frictionless_baoab_reduces_to_velocity_verlet():
    parameters = _control()
    rng = np.random.default_rng(9001)
    state = LatticeState(
        0.01 * rng.normal(size=(4, 4)),
        np.zeros((4, 4)),
        np.zeros((4, 4)),
    )
    velocity = LatticeVelocity(
        1.0e-3 * rng.normal(size=(4, 4)),
        np.zeros((4, 4)),
        np.zeros((4, 4)),
    )
    force_at_state = _harmonic_force(parameters)
    force = force_at_state(state)
    dt = 0.2

    vv_state, vv_velocity, vv_force = velocity_verlet_step(
        state,
        velocity,
        force,
        parameters,
        dt,
        force_at_state,
    )
    baoab_state, baoab_velocity, baoab_force = baoab_step(
        state,
        velocity,
        force,
        parameters,
        LangevinBath(300.0, 0.0, 0.0),
        dt,
        force_at_state,
        np.random.default_rng(5),
    )

    for candidate, reference in (
        (baoab_state.u, vv_state.u),
        (baoab_state.vx, vv_state.vx),
        (baoab_state.vy, vv_state.vy),
        (baoab_velocity.u, vv_velocity.u),
        (baoab_velocity.vx, vv_velocity.vx),
        (baoab_velocity.vy, vv_velocity.vy),
        (baoab_force.u, vv_force.u),
    ):
        assert np.allclose(candidate, reference, rtol=0.0, atol=2.0e-16)


def test_ou_stationary_kinetic_variance_is_preserved_statistically():
    # This vectorized ensemble test validates both the FDT amplitude and the
    # effective-mass unit conversion without requiring a long trajectory.
    replicas = 50_000
    parameters = _control(nx=replicas, ny=1)
    mass_u, mass_v = lattice_masses_fs(parameters)
    temperature = 300.0
    bath = LangevinBath(temperature, 0.015, 0.022)
    initialization_rng = np.random.default_rng(123)
    velocity = LatticeVelocity(
        initialization_rng.normal(
            scale=np.sqrt(BOLTZMANN_EV_PER_K * temperature / mass_u),
            size=(1, replicas),
        ),
        initialization_rng.normal(
            scale=np.sqrt(BOLTZMANN_EV_PER_K * temperature / mass_v),
            size=(1, replicas),
        ),
        initialization_rng.normal(
            scale=np.sqrt(BOLTZMANN_EV_PER_K * temperature / mass_v),
            size=(1, replicas),
        ),
    )
    transformed = ou_velocity_step(
        velocity,
        parameters,
        bath,
        0.5,
        np.random.default_rng(456),
    )

    target = BOLTZMANN_EV_PER_K * temperature
    measured = (
        mass_u * np.mean(transformed.u**2),
        mass_v * np.mean(transformed.vx**2),
        mass_v * np.mean(transformed.vy**2),
    )
    for value in measured:
        assert abs(value / target - 1.0) < 0.02


def test_baoab_samples_harmonic_u_equipartition_control():
    # Many independent u oscillators are evolved in parallel.  This is a
    # deterministic fixed-seed statistical regression, not a material model.
    replicas = 2048
    parameters = _control(nx=replicas, ny=1)
    bath = LangevinBath(temperature_K=300.0, gamma_u_per_fs=0.01, gamma_v_per_fs=0.0)
    state = LatticeState.zeros(1, replicas)
    velocity = LatticeVelocity.zeros(1, replicas)
    force_at_state = _harmonic_force(parameters)
    force = force_at_state(state)
    rng = np.random.default_rng(24680)
    dt = 1.0

    for _ in range(3000):
        state, velocity, force = baoab_step(
            state,
            velocity,
            force,
            parameters,
            bath,
            dt,
            force_at_state,
            rng,
        )

    mass_u, _ = lattice_masses_fs(parameters)
    target_half_kbt = 0.5 * BOLTZMANN_EV_PER_K * bath.temperature_K
    mean_kinetic = 0.5 * mass_u * float(np.mean(velocity.u**2))
    mean_potential = 0.5 * parameters.k1 * float(np.mean(state.u**2))

    assert abs(mean_kinetic / target_half_kbt - 1.0) < 0.07
    assert abs(mean_potential / target_half_kbt - 1.0) < 0.07
