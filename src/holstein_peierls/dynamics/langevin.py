"""Finite-temperature classical lattice bath for D4.

The archived dynamics already used a Markovian Langevin bath, but its discrete
implementation mixes several damping constants, recreates random streams at
half steps, and contains an incorrect active temperature diagnostic.  D4 keeps
the same classical fluctuation-dissipation physics while replacing the legacy
BBK-like update by a transparent BAOAB splitting.

For each velocity component,

    dv = [F(q)/m - gamma v] dt + sqrt(2 gamma k_B T / m) dW,

where ``gamma`` is expressed in 1/fs.  The Ornstein-Uhlenbeck (O) substep is
integrated exactly,

    v' = c v + sigma R,
    c = exp(-gamma dt),
    sigma^2 = (1-c^2) k_B T / m.

Masses are the already validated effective lattice masses in
``eV fs^2 / angstrom^2``.  Consequently ``sigma`` is directly in
``angstrom / fs``.

This module contains no electronic thermalization or decoherence.  Those are a
separate D5 physics choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.random import Generator

from ..gradients import LatticeGradient
from ..lattice import LatticeState
from ..parameters import StaticPolaronParameters
from .classical import drift, half_kick
from .ehrenfest import LatticeVelocity, lattice_kinetic_energy, lattice_masses_fs

# CODATA Boltzmann constant in eV/K.  Energies throughout this project are eV.
BOLTZMANN_EV_PER_K = 8.617333262145e-5

ForceFunction = Callable[[LatticeState], LatticeGradient]


@dataclass(frozen=True, slots=True)
class LangevinBath:
    """Markovian lattice-bath parameters used by the D4 BAOAB thermostat.

    ``gamma_u_per_fs`` damps the intramolecular ``u`` velocities.
    ``gamma_v_per_fs`` damps both intermolecular displacement fields ``vx`` and
    ``vy``.  The values are numerical bath parameters; this class does not
    assign material-specific damping constants.
    """

    temperature_K: float
    gamma_u_per_fs: float
    gamma_v_per_fs: float

    def __post_init__(self) -> None:
        values = (
            float(self.temperature_K),
            float(self.gamma_u_per_fs),
            float(self.gamma_v_per_fs),
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Langevin bath parameters must be finite")
        if self.temperature_K < 0.0:
            raise ValueError("temperature_K must be non-negative")
        if self.gamma_u_per_fs < 0.0 or self.gamma_v_per_fs < 0.0:
            raise ValueError("Langevin friction coefficients must be non-negative")

    @property
    def is_frictionless(self) -> bool:
        return self.gamma_u_per_fs == 0.0 and self.gamma_v_per_fs == 0.0


@dataclass(frozen=True, slots=True)
class OUParameters:
    """Exact Ornstein-Uhlenbeck coefficient and velocity-noise amplitude."""

    damping_factor: float
    velocity_sigma: float


def _positive_dt(dt_fs: float) -> float:
    dt = float(dt_fs)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_fs must be finite and positive")
    return dt


def ou_parameters(
    *,
    gamma_per_fs: float,
    temperature_K: float,
    mass_eV_fs2_per_A2: float,
    dt_fs: float,
) -> OUParameters:
    """Return exact O-step coefficients consistent with FDT.

    The stationary variance is

    ``<v^2> = k_B T / m``.
    """
    gamma = float(gamma_per_fs)
    temperature = float(temperature_K)
    mass = float(mass_eV_fs2_per_A2)
    dt = _positive_dt(dt_fs)
    if not np.isfinite(gamma) or gamma < 0.0:
        raise ValueError("gamma_per_fs must be finite and non-negative")
    if not np.isfinite(temperature) or temperature < 0.0:
        raise ValueError("temperature_K must be finite and non-negative")
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("mass must be finite and positive")

    damping = float(np.exp(-gamma * dt))
    variance = (1.0 - damping * damping) * BOLTZMANN_EV_PER_K * temperature / mass
    # Protect the exactly frictionless limit from tiny negative roundoff.
    sigma = float(np.sqrt(max(0.0, variance)))
    return OUParameters(damping, sigma)


def ou_velocity_step(
    velocity: LatticeVelocity,
    parameters: StaticPolaronParameters,
    bath: LangevinBath,
    dt_fs: float,
    rng: Generator,
) -> LatticeVelocity:
    """Apply the exact Ornstein-Uhlenbeck thermostat step to all velocities.

    A persistent caller-owned ``numpy.random.Generator`` is required.  The
    function deliberately performs no random draw for a channel whose noise
    amplitude is exactly zero.  Thus the frictionless limit is bitwise
    deterministic and a zero-temperature damped run does not consume RNG state.
    """
    if not isinstance(rng, Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    shape = (parameters.ny, parameters.nx)
    velocity.validate(shape)
    mass_u, mass_v = lattice_masses_fs(parameters)
    u_ou = ou_parameters(
        gamma_per_fs=bath.gamma_u_per_fs,
        temperature_K=bath.temperature_K,
        mass_eV_fs2_per_A2=mass_u,
        dt_fs=dt_fs,
    )
    v_ou = ou_parameters(
        gamma_per_fs=bath.gamma_v_per_fs,
        temperature_K=bath.temperature_K,
        mass_eV_fs2_per_A2=mass_v,
        dt_fs=dt_fs,
    )

    if u_ou.velocity_sigma == 0.0:
        u = np.asarray(u_ou.damping_factor * velocity.u, dtype=np.float64)
    else:
        u = np.asarray(
            u_ou.damping_factor * velocity.u
            + u_ou.velocity_sigma * rng.standard_normal(shape),
            dtype=np.float64,
        )

    if v_ou.velocity_sigma == 0.0:
        vx = np.asarray(v_ou.damping_factor * velocity.vx, dtype=np.float64)
        vy = np.asarray(v_ou.damping_factor * velocity.vy, dtype=np.float64)
    else:
        # Separate draws preserve independent baths for the two intermolecular
        # coordinate fields while sharing one reproducible persistent stream.
        vx = np.asarray(
            v_ou.damping_factor * velocity.vx
            + v_ou.velocity_sigma * rng.standard_normal(shape),
            dtype=np.float64,
        )
        vy = np.asarray(
            v_ou.damping_factor * velocity.vy
            + v_ou.velocity_sigma * rng.standard_normal(shape),
            dtype=np.float64,
        )

    return LatticeVelocity(u, vx, vy)


def kinetic_temperature_K(
    velocity: LatticeVelocity,
    parameters: StaticPolaronParameters,
    *,
    degrees_of_freedom: int | None = None,
) -> float:
    """Return the instantaneous kinetic temperature from equipartition.

    ``T = 2 K / (N_dof k_B)``.

    With no constrained collective modes, the lattice has ``3*N`` velocity
    degrees of freedom.  A caller that later removes translational/gauge modes
    must pass the corresponding reduced count explicitly.
    """
    velocity.validate((parameters.ny, parameters.nx))
    dof = 3 * parameters.n_sites if degrees_of_freedom is None else int(degrees_of_freedom)
    if dof <= 0:
        raise ValueError("degrees_of_freedom must be positive")
    kinetic = lattice_kinetic_energy(velocity, parameters)
    return float(2.0 * kinetic / (dof * BOLTZMANN_EV_PER_K))


def baoab_step(
    state: LatticeState,
    velocity: LatticeVelocity,
    force: LatticeGradient,
    parameters: StaticPolaronParameters,
    bath: LangevinBath,
    dt_fs: float,
    force_at_state: ForceFunction,
    rng: Generator,
) -> tuple[LatticeState, LatticeVelocity, LatticeGradient]:
    """Advance an autonomous classical lattice by one BAOAB step.

    Sequence: ``B(dt/2) A(dt/2) O(dt) A(dt/2) B(dt/2)``.

    When both friction coefficients vanish, ``O`` is the identity and the two
    adjacent half drifts combine exactly into the deterministic velocity-Verlet
    drift.  This provides a strict reduction gate to the validated D2 classical
    integrator.
    """
    dt = _positive_dt(dt_fs)
    state.validate()
    velocity.validate(state.shape)

    half_velocity = half_kick(velocity, force, parameters, dt)
    midpoint = drift(state, half_velocity, 0.5 * dt)
    thermostatted_velocity = ou_velocity_step(
        half_velocity,
        parameters,
        bath,
        dt,
        rng,
    )
    new_state = drift(midpoint, thermostatted_velocity, 0.5 * dt)
    new_force = force_at_state(new_state)
    new_velocity = half_kick(
        thermostatted_velocity,
        new_force,
        parameters,
        dt,
    )
    return new_state, new_velocity, new_force
