"""Classical zero-temperature integration gates for D2."""

from __future__ import annotations

import numpy as np

from holstein_peierls.dynamics.classical import (
    drift,
    half_kick,
    lattice_acceleration,
    velocity_verlet_step,
)
from holstein_peierls.dynamics.ehrenfest import LatticeVelocity, lattice_masses_fs
from holstein_peierls.gradients import LatticeGradient
from holstein_peierls.lattice import LatticeState
from holstein_peierls.parameters import StaticPolaronParameters


def _harmonic_force(
    parameters: StaticPolaronParameters,
):
    def force(state: LatticeState) -> LatticeGradient:
        zeros = np.zeros_like(state.u)
        return LatticeGradient(
            -parameters.k1 * state.u,
            zeros.copy(),
            zeros.copy(),
        )

    return force


def _integrate_harmonic(dt_fs: float, final_time_fs: float) -> tuple[float, float]:
    parameters = StaticPolaronParameters(nx=1, ny=1, polaron_position=1)
    state = LatticeState.zeros(1, 1)
    state.u[0, 0] = 0.1
    velocity = LatticeVelocity.zeros(1, 1)
    force_fn = _harmonic_force(parameters)
    force = force_fn(state)
    steps = int(round(final_time_fs / dt_fs))
    assert np.isclose(steps * dt_fs, final_time_fs)
    for _ in range(steps):
        state, velocity, force = velocity_verlet_step(
            state,
            velocity,
            force,
            parameters,
            dt_fs,
            force_fn,
        )
    return float(state.u[0, 0]), float(velocity.u[0, 0])


def test_half_kick_and_drift_match_direct_arithmetic() -> None:
    parameters = StaticPolaronParameters(nx=2, ny=2, polaron_position=1)
    state = LatticeState.zeros(2, 2)
    velocity = LatticeVelocity.zeros(2, 2)
    velocity.u[:] = 1.0e-3
    force = LatticeGradient(
        np.full((2, 2), 0.2),
        np.full((2, 2), -0.1),
        np.full((2, 2), 0.05),
    )
    dt = 0.2
    acceleration = lattice_acceleration(force, parameters)
    half = half_kick(velocity, force, parameters, dt)
    np.testing.assert_allclose(
        half.u,
        velocity.u + 0.5 * dt * acceleration.u,
        rtol=0.0,
        atol=1.0e-18,
    )
    moved = drift(state, half, dt)
    np.testing.assert_allclose(moved.u, dt * half.u, rtol=0.0, atol=1.0e-18)
    np.testing.assert_allclose(moved.vx, dt * half.vx, rtol=0.0, atol=1.0e-18)
    np.testing.assert_allclose(moved.vy, dt * half.vy, rtol=0.0, atol=1.0e-18)


def test_velocity_verlet_has_second_order_global_convergence() -> None:
    parameters = StaticPolaronParameters(nx=1, ny=1, polaron_position=1)
    mass_u, _ = lattice_masses_fs(parameters)
    omega = np.sqrt(parameters.k1 / mass_u)
    final_time = 20.0
    exact_q = 0.1 * np.cos(omega * final_time)
    exact_v = -0.1 * omega * np.sin(omega * final_time)

    errors = []
    for dt in (0.4, 0.2, 0.1):
        q, v = _integrate_harmonic(dt, final_time)
        errors.append(np.hypot(q - exact_q, (v - exact_v) / omega))

    order_coarse = np.log2(errors[0] / errors[1])
    order_fine = np.log2(errors[1] / errors[2])
    assert order_coarse > 1.9
    assert order_fine > 1.9


def test_velocity_verlet_harmonic_energy_error_is_bounded() -> None:
    parameters = StaticPolaronParameters(nx=1, ny=1, polaron_position=1)
    mass_u, _ = lattice_masses_fs(parameters)
    state = LatticeState.zeros(1, 1)
    state.u[0, 0] = 0.1
    velocity = LatticeVelocity.zeros(1, 1)
    force_fn = _harmonic_force(parameters)
    force = force_fn(state)
    dt = 0.2
    initial_energy = 0.5 * parameters.k1 * state.u[0, 0] ** 2
    deviations = []
    for _ in range(1000):
        state, velocity, force = velocity_verlet_step(
            state,
            velocity,
            force,
            parameters,
            dt,
            force_fn,
        )
        energy = (
            0.5 * parameters.k1 * state.u[0, 0] ** 2
            + 0.5 * mass_u * velocity.u[0, 0] ** 2
        )
        deviations.append(abs(energy - initial_energy))

    # Symplectic Verlet should show a small bounded oscillatory error rather
    # than secular energy drift on this exactly solvable harmonic control.
    assert max(deviations) / initial_energy < 2.0e-5
