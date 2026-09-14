"""Tests for IP1f carrier-centered phonon-wake observables."""

from __future__ import annotations

import numpy as np
import pytest

from holstein_peierls.dynamics.ehrenfest import LatticeVelocity, lattice_kinetic_energy
from holstein_peierls.dynamics.phonon_wake import (
    backward_excess_centroid_sites,
    event_aligned_coordinates,
    fit_group_velocity_sites_per_fs,
    intermolecular_energy_flux,
    local_lattice_energy_density,
    longitudinal_flux_field,
    longitudinal_profile,
    wake_window_metrics,
)
from holstein_peierls.energy import lattice_energy
from holstein_peierls.lattice import LatticeState
from holstein_peierls.parameters import StaticPolaronParameters


def _random_state(seed: int = 7, size: int = 6):
    rng = np.random.default_rng(seed)
    shape = (size, size)
    lattice = LatticeState(
        rng.normal(scale=0.02, size=shape),
        rng.normal(scale=0.03, size=shape),
        rng.normal(scale=0.03, size=shape),
    )
    velocity = LatticeVelocity(
        rng.normal(scale=2.0e-4, size=shape),
        rng.normal(scale=2.0e-4, size=shape),
        rng.normal(scale=2.0e-4, size=shape),
    )
    parameters = StaticPolaronParameters(nx=size, ny=size)
    return lattice, velocity, parameters


def test_local_energy_density_sums_to_classical_lattice_energy() -> None:
    lattice, velocity, parameters = _random_state()
    density = local_lattice_energy_density(lattice, velocity, parameters)
    intra, inter = lattice_energy(lattice, parameters)
    kinetic = lattice_kinetic_energy(velocity, parameters)
    assert float(np.sum(density.total)) == pytest.approx(intra + inter + kinetic, abs=1.0e-12)


def test_event_coordinates_wrap_and_point_true_hop_to_positive_s() -> None:
    nx = ny = 6
    source = 2 * nx + 0
    target = 2 * nx + (nx - 1)  # -x through PBC
    coords = event_aligned_coordinates(source, target, nx, ny)
    tx = target % nx
    ty = target // nx
    assert coords.dx_sites == -1
    assert coords.dy_sites == 0
    assert int(coords.s_sites[ty, tx]) == 1
    assert int(coords.p_sites[ty, tx]) == 0


def test_longitudinal_profile_conserves_selected_scalar_sum() -> None:
    nx = ny = 8
    source = 3 * nx + 3
    target = source + 1
    coords = event_aligned_coordinates(source, target, nx, ny)
    values = np.arange(nx * ny, dtype=np.float64).reshape(ny, nx)
    _, profile = longitudinal_profile(values, coords)
    assert float(np.sum(profile)) == pytest.approx(float(np.sum(values)))


def test_harmonic_travelling_wave_energy_flux_has_expected_sign() -> None:
    size = 16
    parameters = StaticPolaronParameters(nx=size, ny=size)
    k = 2.0 * np.pi / size
    omega = 0.002
    x = np.arange(size, dtype=np.float64)
    q = np.cos(k * x)
    qdot_right = omega * np.sin(k * x)
    lattice = LatticeState(
        np.zeros((size, size)),
        np.tile(q, (size, 1)),
        np.zeros((size, size)),
    )
    velocity = LatticeVelocity(
        np.zeros((size, size)),
        np.tile(qdot_right, (size, 1)),
        np.zeros((size, size)),
    )
    flux = intermolecular_energy_flux(lattice, velocity, parameters)
    assert float(np.mean(flux.jx)) > 0.0
    assert float(np.max(np.abs(flux.jy))) == pytest.approx(0.0)


def test_longitudinal_flux_projection_changes_with_hop_direction() -> None:
    size = 8
    source = 3 * size + 3
    right = source + 1
    left = source - 1
    jx = np.ones((size, size), dtype=np.float64)
    jy = np.zeros((size, size), dtype=np.float64)
    from holstein_peierls.dynamics.phonon_wake import IntermolecularEnergyFlux

    flux = IntermolecularEnergyFlux(jx, jy)
    plus = longitudinal_flux_field(flux, event_aligned_coordinates(source, right, size, size))
    minus = longitudinal_flux_field(flux, event_aligned_coordinates(source, left, size, size))
    assert np.all(plus == 1.0)
    assert np.all(minus == -1.0)


def test_wake_metrics_detect_retrograde_excess_and_flux() -> None:
    s = np.arange(-5, 6, dtype=np.int64)
    excess = np.zeros(s.size, dtype=np.float64)
    flux = np.zeros(s.size, dtype=np.float64)
    excess[s == -3] = 2.0
    excess[s == 3] = 0.5
    flux[s == -3] = -0.4
    flux[s == 3] = 0.1
    metrics = wake_window_metrics(excess, flux, s, core_half_width_sites=1)
    assert metrics.signed_asymmetry > 0.0
    assert metrics.backward_positive_excess_eV > metrics.forward_positive_excess_eV
    assert metrics.backward_outward_flux_eV_per_fs > metrics.forward_outward_flux_eV_per_fs
    assert metrics.flux_bias > 0.0


def test_backward_centroid_and_group_velocity_fit_are_negative_for_retrograde_motion() -> None:
    s = np.arange(-8, 9, dtype=np.int64)
    times = np.array([200.0, 400.0, 600.0, 800.0, 1000.0])
    centers = []
    for center in (-2.5, -3.0, -3.5, -4.0, -4.5):
        profile = np.exp(-0.5 * ((s - center) / 0.7) ** 2)
        value = backward_excess_centroid_sites(profile, s, core_half_width_sites=1)
        assert value is not None
        centers.append(value)
    fit = fit_group_velocity_sites_per_fs(times, np.asarray(centers))
    assert fit["velocity_sites_per_fs"] is not None
    assert float(fit["velocity_sites_per_fs"]) < 0.0
    assert float(fit["velocity_sites_per_ps"]) < 0.0
    assert float(fit["r_squared"]) > 0.95
