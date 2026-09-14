import numpy as np
import pytest

from holstein_peierls.dynamics.coupled import CoupledEhrenfestState
from holstein_peierls.dynamics.driven import field_ehrenfest_force
from holstein_peierls.dynamics.ehrenfest import LatticeVelocity
from holstein_peierls.dynamics.field import UniformElectricField2D
from holstein_peierls.dynamics.field_release import HeldPeierlsPhase2D
from holstein_peierls.dynamics.frozen_electronic_surface import (
    frozen_surface_total_energy,
    frozen_surface_verlet_step,
)
from holstein_peierls.lattice import LatticeState
from holstein_peierls.parameters import StaticPolaronParameters


def _normalized_state(n):
    values = np.arange(1, n + 1, dtype=np.float64) + 1j * np.arange(n, 0, -1)
    values = values.astype(np.complex128)
    return values / np.linalg.norm(values)


def test_frozen_surface_force_is_continuous_at_held_phase_switch():
    parameters = StaticPolaronParameters(nx=3, ny=3)
    lattice = LatticeState.zeros(3, 3)
    psi = _normalized_state(parameters.n_sites)
    driving = UniformElectricField2D.from_millivolt_per_angstrom(10.0)
    switch = 123.4
    held = HeldPeierlsPhase2D.from_driving_field(driving, switch)

    force_drive = field_ehrenfest_force(lattice, parameters, psi, driving, switch)
    force_held = field_ehrenfest_force(lattice, parameters, psi, held, switch)
    assert np.allclose(force_drive.u, force_held.u, rtol=0.0, atol=1.0e-14)
    assert np.allclose(force_drive.vx, force_held.vx, rtol=0.0, atol=1.0e-14)
    assert np.allclose(force_drive.vy, force_held.vy, rtol=0.0, atol=1.0e-14)


def test_frozen_surface_step_keeps_electronic_state_bitwise_fixed():
    parameters = StaticPolaronParameters(nx=3, ny=3)
    psi = _normalized_state(parameters.n_sites)
    state = CoupledEhrenfestState(
        LatticeState.zeros(3, 3),
        LatticeVelocity.zeros(3, 3),
        psi.copy(),
    )
    driving = UniformElectricField2D.from_millivolt_per_angstrom(10.0)
    held = HeldPeierlsPhase2D.from_driving_field(driving, 100.0)
    updated = frozen_surface_verlet_step(state, parameters, psi, held, 100.0, 0.01)
    assert np.array_equal(updated.electronic_state, psi)


def test_frozen_surface_is_conservative_for_held_phase():
    parameters = StaticPolaronParameters(nx=3, ny=3)
    psi = _normalized_state(parameters.n_sites)
    state = CoupledEhrenfestState(
        LatticeState.zeros(3, 3),
        LatticeVelocity.zeros(3, 3),
        psi.copy(),
    )
    driving = UniformElectricField2D.from_millivolt_per_angstrom(10.0)
    held = HeldPeierlsPhase2D.from_driving_field(driving, 100.0)
    initial = frozen_surface_total_energy(state, parameters, psi, held, 100.0)
    current = state
    dt = 0.01
    for step in range(200):
        current = frozen_surface_verlet_step(
            current,
            parameters,
            psi,
            held,
            100.0 + step * dt,
            dt,
        )
    final = frozen_surface_total_energy(current, parameters, psi, held, 102.0)
    assert abs(final - initial) < 1.0e-8
    assert np.array_equal(current.electronic_state, psi)
