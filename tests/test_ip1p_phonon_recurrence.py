"""Tests for the periodic-boundary phonon recurrence audit."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from holstein_peierls.dynamics.phonon_recurrence import (
    conservative_event_window_cutoff_fs,
    event_window_precedes_stationary_wrap,
    intermolecular_recurrence_scales,
)
from holstein_peierls.parameters import StaticPolaronParameters


def test_default_20x20_recurrence_scale_matches_harmonic_formula() -> None:
    parameters = StaticPolaronParameters(nx=20, ny=20)
    scales = intermolecular_recurrence_scales(
        parameters,
        lattice_spacing_A=3.0,
        gamma_v_per_fs=0.01,
    )
    mass_v = parameters.m2 / 1.0e6
    frequency_scale = np.sqrt(parameters.k2 / mass_v)
    assert scales.spring_frequency_scale_per_fs == pytest.approx(frequency_scale)
    assert scales.omega_max_per_fs == pytest.approx(2.0 * frequency_scale)
    assert scales.max_group_velocity_sites_per_fs == pytest.approx(frequency_scale)
    assert scales.max_group_velocity_A_per_fs == pytest.approx(3.0 * frequency_scale)
    assert scales.wrap_time_x_fs == pytest.approx(20.0 / frequency_scale)
    assert scales.wrap_time_y_fs == pytest.approx(20.0 / frequency_scale)


def test_wrap_time_scales_linearly_with_cell_length() -> None:
    p20 = StaticPolaronParameters(nx=20, ny=20)
    p40 = replace(p20, nx=40, ny=40, polaron_position=821)
    s20 = intermolecular_recurrence_scales(p20, lattice_spacing_A=3.0, gamma_v_per_fs=0.0)
    s40 = intermolecular_recurrence_scales(p40, lattice_spacing_A=3.0, gamma_v_per_fs=0.0)
    assert s40.wrap_time_x_fs == pytest.approx(2.0 * s20.wrap_time_x_fs)
    assert s40.wrap_time_y_fs == pytest.approx(2.0 * s20.wrap_time_y_fs)


def test_current_gamma_overdamps_all_harmonic_inter_modes() -> None:
    parameters = StaticPolaronParameters(nx=20, ny=20)
    scales = intermolecular_recurrence_scales(
        parameters,
        lattice_spacing_A=3.0,
        gamma_v_per_fs=0.01,
    )
    assert scales.all_nonzero_harmonic_modes_overdamped
    assert scales.damping_ratio_at_omega_max > 1.0


def test_frictionless_control_is_not_overdamped() -> None:
    parameters = StaticPolaronParameters(nx=20, ny=20)
    scales = intermolecular_recurrence_scales(
        parameters,
        lattice_spacing_A=3.0,
        gamma_v_per_fs=0.0,
    )
    assert not scales.all_nonzero_harmonic_modes_overdamped
    assert scales.damping_ratio_at_omega_max == 0.0


def test_event_cutoff_subtracts_full_post_event_window() -> None:
    parameters = StaticPolaronParameters(nx=20, ny=20)
    scales = intermolecular_recurrence_scales(
        parameters,
        lattice_spacing_A=3.0,
        gamma_v_per_fs=0.01,
    )
    cutoff = conservative_event_window_cutoff_fs(scales, event_half_window_fs=500.0)
    assert cutoff == pytest.approx(scales.earliest_stationary_carrier_wrap_fs - 500.0)
    assert event_window_precedes_stationary_wrap(
        cutoff,
        scales,
        event_half_window_fs=500.0,
    )
    assert not event_window_precedes_stationary_wrap(
        cutoff + 1.0,
        scales,
        event_half_window_fs=500.0,
    )


def test_invalid_inputs_are_rejected() -> None:
    parameters = StaticPolaronParameters(nx=20, ny=20)
    with pytest.raises(ValueError):
        intermolecular_recurrence_scales(parameters, lattice_spacing_A=0.0, gamma_v_per_fs=0.01)
    with pytest.raises(ValueError):
        intermolecular_recurrence_scales(parameters, lattice_spacing_A=3.0, gamma_v_per_fs=-1.0)
