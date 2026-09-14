"""Classical lattice propagation on a fixed electronic Ehrenfest surface.

This control is used by IP1o.  The complete complex electronic wavefunction is
held fixed at the branch-point value while the lattice continues to evolve on

    V_lattice(q) + <psi_fixed|H(q, A_held)|psi_fixed>.

When the Peierls phase is held constant this surface is time independent.  A
velocity-Verlet step is therefore conservative up to the usual integration
error, while electronic redistribution and later carrier hops are excluded by
construction.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ..parameters import StaticPolaronParameters
from .classical import drift, half_kick
from .coupled import CoupledEhrenfestState
from .driven import field_dynamic_total_energy, field_ehrenfest_force


def _fixed_state(values: ArrayLike, dimension: int) -> np.ndarray:
    psi = np.asarray(values, dtype=np.complex128)
    if psi.ndim != 1 or psi.size != int(dimension):
        raise ValueError("fixed electronic state dimension does not match lattice")
    if not np.all(np.isfinite(psi)):
        raise ValueError("fixed electronic state must be finite")
    norm = float(np.vdot(psi, psi).real)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("fixed electronic state must have nonzero norm")
    return psi


def frozen_surface_total_energy(
    state: CoupledEhrenfestState,
    parameters: StaticPolaronParameters,
    fixed_electronic_state: ArrayLike,
    held_phase,
    time_fs: float,
) -> float:
    """Return the conservative total energy of the fixed-electronic surface."""
    psi = _fixed_state(fixed_electronic_state, parameters.n_sites)
    return float(
        field_dynamic_total_energy(
            state.lattice,
            state.velocity,
            parameters,
            psi,
            held_phase,
            float(time_fs),
        ).total
    )


def frozen_surface_verlet_step(
    state: CoupledEhrenfestState,
    parameters: StaticPolaronParameters,
    fixed_electronic_state: ArrayLike,
    held_phase,
    time_fs: float,
    dt_fs: float,
) -> CoupledEhrenfestState:
    """Advance one velocity-Verlet step with the electronic state held fixed."""
    dt = float(dt_fs)
    time = float(time_fs)
    if not np.isfinite(time) or not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("time_fs must be finite and dt_fs positive")
    psi = _fixed_state(fixed_electronic_state, parameters.n_sites)
    state.lattice.validate()
    state.velocity.validate((parameters.ny, parameters.nx))

    force_old = field_ehrenfest_force(
        state.lattice,
        parameters,
        psi,
        held_phase,
        time,
    )
    half_velocity = half_kick(state.velocity, force_old, parameters, dt)
    lattice_new = drift(state.lattice, half_velocity, dt)
    force_new = field_ehrenfest_force(
        lattice_new,
        parameters,
        psi,
        held_phase,
        time + dt,
    )
    velocity_new = half_kick(half_velocity, force_new, parameters, dt)
    return CoupledEhrenfestState(
        lattice_new,
        velocity_new,
        np.asarray(psi, dtype=np.complex128).copy(),
    )
