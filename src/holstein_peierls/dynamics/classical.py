"""Deterministic classical lattice stepping for D2.

The routines in this module contain no thermostat, damping, random force or
electric field.  They implement the zero-temperature velocity-Verlet skeleton
independently of the electronic propagator so the classical integrator can be
validated in isolation before the Ehrenfest coupling is assembled.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..gradients import LatticeGradient
from ..lattice import LatticeState
from ..parameters import StaticPolaronParameters
from .ehrenfest import LatticeVelocity, lattice_masses_fs

ForceFunction = Callable[[LatticeState], LatticeGradient]


def _positive_dt(dt_fs: float) -> float:
    value = float(dt_fs)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("dt_fs must be finite and positive")
    return value


def lattice_acceleration(
    force: LatticeGradient,
    parameters: StaticPolaronParameters,
) -> LatticeGradient:
    """Convert forces in eV/A to accelerations in A/fs^2."""
    shape = (parameters.ny, parameters.nx)
    if force.u.shape != shape or force.vx.shape != shape or force.vy.shape != shape:
        raise ValueError("force shape does not match lattice parameters")
    mass_u, mass_v = lattice_masses_fs(parameters)
    return LatticeGradient(
        np.asarray(force.u / mass_u, dtype=np.float64),
        np.asarray(force.vx / mass_v, dtype=np.float64),
        np.asarray(force.vy / mass_v, dtype=np.float64),
    )


def half_kick(
    velocity: LatticeVelocity,
    force: LatticeGradient,
    parameters: StaticPolaronParameters,
    dt_fs: float,
) -> LatticeVelocity:
    """Apply one half velocity kick over a full-step interval ``dt_fs``."""
    dt = _positive_dt(dt_fs)
    velocity.validate((parameters.ny, parameters.nx))
    acceleration = lattice_acceleration(force, parameters)
    return LatticeVelocity(
        np.asarray(velocity.u + 0.5 * dt * acceleration.u, dtype=np.float64),
        np.asarray(velocity.vx + 0.5 * dt * acceleration.vx, dtype=np.float64),
        np.asarray(velocity.vy + 0.5 * dt * acceleration.vy, dtype=np.float64),
    )


def drift(
    state: LatticeState,
    half_step_velocity: LatticeVelocity,
    dt_fs: float,
) -> LatticeState:
    """Drift all lattice coordinates over one full interval."""
    dt = _positive_dt(dt_fs)
    state.validate()
    half_step_velocity.validate(state.shape)
    return LatticeState(
        np.asarray(state.u + dt * half_step_velocity.u, dtype=np.float64),
        np.asarray(state.vx + dt * half_step_velocity.vx, dtype=np.float64),
        np.asarray(state.vy + dt * half_step_velocity.vy, dtype=np.float64),
    )


def velocity_verlet_step(
    state: LatticeState,
    velocity: LatticeVelocity,
    force: LatticeGradient,
    parameters: StaticPolaronParameters,
    dt_fs: float,
    force_at_state: ForceFunction,
) -> tuple[LatticeState, LatticeVelocity, LatticeGradient]:
    """Advance an autonomous classical lattice by one velocity-Verlet step.

    This reference helper is used for classical validation problems where the
    force is a function of lattice coordinates only.  In coupled Ehrenfest D2,
    the second force also depends on the newly propagated electronic state and
    will therefore be supplied explicitly by the coupled orchestrator rather
    than hidden inside this function.
    """
    half_velocity = half_kick(velocity, force, parameters, dt_fs)
    new_state = drift(state, half_velocity, dt_fs)
    new_force = force_at_state(new_state)
    new_velocity = half_kick(half_velocity, new_force, parameters, dt_fs)
    return new_state, new_velocity, new_force
