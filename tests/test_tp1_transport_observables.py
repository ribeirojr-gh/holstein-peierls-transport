from __future__ import annotations

import numpy as np

from holstein_peierls.dynamics.driven import field_power
from holstein_peierls.dynamics.field import UniformElectricField2D
from holstein_peierls.dynamics.frozen import HBAR_EV_FS
from holstein_peierls.dynamics.transport_observables import (
    TransportKinematics2D,
    bond_probability_currents,
    continuity_population_derivative,
    field_power_from_particle_velocity,
    periodic_center_of_probability,
    schrodinger_population_derivative,
    transport_kinematics,
    trapezoidal_displacement_increment,
)
from holstein_peierls.lattice import LatticeState
from holstein_peierls.parameters import StaticPolaronParameters


def _parameters(nx: int = 4, ny: int = 4) -> StaticPolaronParameters:
    return StaticPolaronParameters(nx=nx, ny=ny, polaron_position=1)


def _deterministic_lattice(parameters: StaticPolaronParameters) -> LatticeState:
    y, x = np.indices((parameters.ny, parameters.nx), dtype=np.float64)
    return LatticeState(
        u=0.011 * np.sin(0.7 * x + 0.3 * y),
        vx=0.017 * np.cos(0.4 * x - 0.2 * y),
        vy=0.013 * np.sin(0.5 * y - 0.1 * x),
    )


def _complex_state(parameters: StaticPolaronParameters) -> np.ndarray:
    index = np.arange(parameters.n_sites, dtype=np.float64)
    psi = np.exp(0.17j * index) * (1.0 + 0.13 * np.cos(0.41 * index))
    return np.asarray(psi / np.linalg.norm(psi), dtype=np.complex128)


def test_localized_basis_state_has_zero_transport_current() -> None:
    parameters = _parameters()
    lattice = LatticeState.zeros(parameters.ny, parameters.nx)
    psi = np.zeros(parameters.n_sites, dtype=np.complex128)
    psi[5] = 1.0
    field = UniformElectricField2D.from_millivolt_per_angstrom(
        2.0, ax_angstrom=3.0, ay_angstrom=4.0
    )
    currents = bond_probability_currents(
        lattice, parameters, psi, field=field, time_fs=137.0
    )
    assert np.array_equal(currents.jx_per_fs, np.zeros_like(currents.jx_per_fs))
    assert np.array_equal(currents.jy_per_fs, np.zeros_like(currents.jy_per_fs))


def test_transport_observables_are_global_phase_invariant() -> None:
    parameters = _parameters()
    lattice = _deterministic_lattice(parameters)
    psi = _complex_state(parameters)
    field = UniformElectricField2D(
        ex_v_per_angstrom=0.0017,
        ey_v_per_angstrom=-0.0008,
        ax_angstrom=3.1,
        ay_angstrom=4.2,
    )
    first = bond_probability_currents(
        lattice, parameters, psi, field=field, time_fs=83.0
    )
    second = bond_probability_currents(
        lattice,
        parameters,
        psi * np.exp(0.713j),
        field=field,
        time_fs=83.0,
    )
    assert np.allclose(first.jx_per_fs, second.jx_per_fs, atol=2e-15, rtol=0.0)
    assert np.allclose(first.jy_per_fs, second.jy_per_fs, atol=2e-15, rtol=0.0)


def test_bond_currents_satisfy_discrete_continuity_equation() -> None:
    parameters = _parameters(5, 4)
    lattice = _deterministic_lattice(parameters)
    psi = _complex_state(parameters)
    field = UniformElectricField2D(
        ex_v_per_angstrom=0.0013,
        ey_v_per_angstrom=0.0007,
        ax_angstrom=3.0,
        ay_angstrom=3.7,
    )
    currents = bond_probability_currents(
        lattice, parameters, psi, field=field, time_fs=61.0
    )
    from_flux = continuity_population_derivative(currents)
    direct = schrodinger_population_derivative(
        lattice, parameters, psi, field=field, time_fs=61.0
    )
    assert np.max(np.abs(from_flux - direct)) < 1.0e-12
    assert abs(float(np.sum(from_flux))) < 1.0e-14


def test_plane_wave_matches_uniform_lattice_group_velocity() -> None:
    parameters = _parameters(6, 5)
    lattice = LatticeState.zeros(parameters.ny, parameters.nx)
    kx = 2.0 * np.pi / parameters.nx
    x = np.tile(np.arange(parameters.nx), parameters.ny)
    psi = np.exp(1.0j * kx * x) / np.sqrt(parameters.n_sites)
    ax = 3.25
    ay = 4.1
    kinematics = transport_kinematics(
        lattice,
        parameters,
        psi,
        ax_angstrom=ax,
        ay_angstrom=ay,
    )
    expected_vx = 2.0 * parameters.j0x * ax * np.sin(kx) / HBAR_EV_FS
    assert abs(kinematics.velocity_x_A_per_fs - expected_vx) < 2.0e-14
    assert abs(kinematics.velocity_y_A_per_fs) < 2.0e-14


def test_particle_velocity_reproduces_validated_field_power() -> None:
    parameters = _parameters(5, 4)
    lattice = _deterministic_lattice(parameters)
    psi = _complex_state(parameters)
    field = UniformElectricField2D(
        ex_v_per_angstrom=0.0018,
        ey_v_per_angstrom=-0.0006,
        ax_angstrom=3.2,
        ay_angstrom=4.0,
    )
    time_fs = 117.0
    kinematics = transport_kinematics(
        lattice, parameters, psi, field=field, time_fs=time_fs
    )
    from_velocity = field_power_from_particle_velocity(field, kinematics)
    direct = field_power(lattice, parameters, psi, field, time_fs)
    assert abs(from_velocity - direct) < 1.0e-12


def test_periodic_center_handles_packet_across_x_boundary() -> None:
    parameters = _parameters(4, 4)
    psi = np.zeros(parameters.n_sites, dtype=np.complex128)
    # Equal probability at x=3 and x=0 on the same row.  The circular center is
    # x=3.5 sites (equivalently -0.5), not the naive Cartesian midpoint x=1.5.
    psi[1 * parameters.nx + 3] = 1.0 / np.sqrt(2.0)
    psi[1 * parameters.nx + 0] = 1.0 / np.sqrt(2.0)
    center = periodic_center_of_probability(
        psi, parameters, ax_angstrom=3.0, ay_angstrom=4.0
    )
    assert center.x_A is not None
    assert center.y_A is not None
    assert abs(center.x_A - 10.5) < 1.0e-12
    assert abs(center.y_A - 4.0) < 1.0e-12
    assert center.x_resultant > 0.7


def test_uniform_state_has_undefined_circular_center() -> None:
    parameters = _parameters(4, 4)
    psi = np.ones(parameters.n_sites, dtype=np.complex128) / np.sqrt(parameters.n_sites)
    center = periodic_center_of_probability(
        psi,
        parameters,
        ax_angstrom=3.0,
        ay_angstrom=3.0,
        minimum_resultant=1.0e-12,
    )
    assert center.x_A is None
    assert center.y_A is None
    assert center.x_resultant < 1.0e-12
    assert center.y_resultant < 1.0e-12


def test_trapezoidal_displacement_is_exact_for_constant_velocity() -> None:
    old = TransportKinematics2D(0.0, 0.0, 0.037, -0.012)
    new = TransportKinematics2D(0.0, 0.0, 0.037, -0.012)
    dx, dy = trapezoidal_displacement_increment(old, new, 2.5)
    assert dx == 0.0925
    assert dy == -0.03
