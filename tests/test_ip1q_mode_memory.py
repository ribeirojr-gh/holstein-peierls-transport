import numpy as np
import pytest

from holstein_peierls.dynamics.driven import field_ehrenfest_force
from holstein_peierls.dynamics.ehrenfest import LatticeVelocity, lattice_masses_fs
from holstein_peierls.dynamics.field import UniformElectricField2D
from holstein_peierls.dynamics.field_release import HeldPeierlsPhase2D
from holstein_peierls.dynamics.mode_memory import (
    axis_resolved_velocity_spectrum,
    frozen_surface_equilibrium,
    harmonic_wave_numbers,
    modal_energy_arrays,
    q_participation_ratio,
    real_space_excitation_energies,
    ridge_diagnostics,
    traveling_vx_energy_split,
)
from holstein_peierls.lattice import LatticeState
from holstein_peierls.parameters import StaticPolaronParameters


def _max_force(force):
    return max(
        float(np.max(np.abs(force.u))),
        float(np.max(np.abs(force.vx))),
        float(np.max(np.abs(force.vy))),
    )


def test_frozen_surface_equilibrium_zeroes_force():
    parameters = StaticPolaronParameters(nx=6, ny=5)
    rng = np.random.default_rng(1234)
    psi = rng.normal(size=parameters.n_sites) + 1j * rng.normal(size=parameters.n_sites)
    psi = psi / np.linalg.norm(psi)
    field = UniformElectricField2D.from_millivolt_per_angstrom(10.0)
    held = HeldPeierlsPhase2D.from_driving_field(field, 2874.0)
    equilibrium = frozen_surface_equilibrium(parameters, psi, held)
    force = field_ehrenfest_force(equilibrium, parameters, psi, held, 9999.0)
    assert _max_force(force) < 1.0e-12
    assert np.max(np.abs(np.mean(equilibrium.vx, axis=1))) < 1.0e-13
    assert np.max(np.abs(np.mean(equilibrium.vy, axis=0))) < 1.0e-13


def test_modal_energy_matches_real_space_parseval():
    parameters = StaticPolaronParameters(nx=7, ny=6)
    rng = np.random.default_rng(7)
    equilibrium = LatticeState.zeros(parameters.ny, parameters.nx)
    lattice = LatticeState(
        rng.normal(scale=0.01, size=(parameters.ny, parameters.nx)),
        rng.normal(scale=0.01, size=(parameters.ny, parameters.nx)),
        rng.normal(scale=0.01, size=(parameters.ny, parameters.nx)),
    )
    velocity = LatticeVelocity(
        rng.normal(scale=1.0e-5, size=(parameters.ny, parameters.nx)),
        rng.normal(scale=1.0e-5, size=(parameters.ny, parameters.nx)),
        rng.normal(scale=1.0e-5, size=(parameters.ny, parameters.nx)),
    )
    real = real_space_excitation_energies(lattice, velocity, equilibrium, parameters)
    modal = modal_energy_arrays(lattice, velocity, equilibrium, parameters).sums
    for key in ("u_eV", "vx_eV", "vy_eV", "total_eV"):
        assert modal[key] == pytest.approx(real[key], rel=1.0e-12, abs=1.0e-12)


def test_traveling_vx_split_identifies_plus_x_wave_as_retrograde_for_minus_x_carrier():
    parameters = StaticPolaronParameters(nx=8, ny=4)
    equilibrium = LatticeState.zeros(parameters.ny, parameters.nx)
    mass_v = lattice_masses_fs(parameters)[1]
    q = 2.0 * np.pi / parameters.nx
    omega = 2.0 * np.sqrt(parameters.k2 / mass_v) * np.sin(0.5 * q)
    x = np.arange(parameters.nx, dtype=np.float64)[None, :]
    displacement = np.broadcast_to(np.cos(q * x), (parameters.ny, parameters.nx)).copy()
    velocity_x = np.broadcast_to(omega * np.sin(q * x), displacement.shape).copy()
    lattice = LatticeState(
        np.zeros_like(displacement),
        displacement,
        np.zeros_like(displacement),
    )
    velocity = LatticeVelocity(
        np.zeros_like(displacement),
        velocity_x,
        np.zeros_like(displacement),
    )
    split = traveling_vx_energy_split(
        lattice,
        velocity,
        equilibrium,
        parameters,
        carrier_dx_sites=-1,
    )
    assert split["retrograde_fraction_of_direction_resolved"] > 1.0 - 1.0e-12
    assert split["direction_resolved_fraction_of_vx"] > 1.0 - 1.0e-12


def test_q_participation_ratio_counts_equal_sectors():
    assert q_participation_ratio([1.0, 1.0, 1.0, 1.0]) == pytest.approx(4.0)
    assert q_participation_ratio([2.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_qomega_spectrum_and_ridge_find_exact_bin_tone():
    nt = 201
    dt = 10.0
    ny, nx = 3, 8
    tone_index = 5
    omega0 = 2.0 * np.pi * tone_index / (nt * dt)
    times = np.arange(nt, dtype=np.float64) * dt
    qx = harmonic_wave_numbers(nx)[1]
    x = np.arange(nx, dtype=np.float64)[None, :]
    spatial = np.broadcast_to(np.cos(qx * x), (ny, nx))
    frames = np.sin(omega0 * times)[:, None, None] * spatial[None, :, :]
    q_axis, omega, power = axis_resolved_velocity_spectrum(
        frames, dt, dispersive_axis="x"
    )
    assert power.shape == (nx, omega.size)
    energy = np.zeros(nx)
    energy[1] = 1.0
    analytic = np.zeros(nx)
    analytic[1] = omega0
    diag = ridge_diagnostics(power, omega, analytic, energy, minimum_energy_fraction=0.1)
    assert diag["eligible_sector_count"] == 1
    assert diag["within_one_frequency_bin"]
    assert diag["records"][0]["peak_omega_per_fs"] == pytest.approx(omega0)
    assert q_axis[1] == pytest.approx(qx)
