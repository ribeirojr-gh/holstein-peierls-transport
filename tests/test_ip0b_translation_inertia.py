from __future__ import annotations

import numpy as np

from holstein_peierls.lattice import LatticeState
from holstein_peierls.parameters import StaticPolaronParameters
from holstein_peierls.translation_barrier import translate_lattice_state
from holstein_peierls.translation_inertia import (
    collective_angular_frequency_per_fs,
    mass_weighted_translation_metric,
    midpoint_low_state_spectrum,
    one_site_translation_mass_metric,
    quadratic_endpoint_curvature,
)


def _parameters(size: int = 4) -> StaticPolaronParameters:
    center = (size // 2) * size + (size // 2)
    return StaticPolaronParameters(
        nx=size,
        ny=size,
        polaron_position=center + 1,
        j0x=0.1,
        j0y=0.1,
    )


def test_mass_metric_uses_legacy_fs_mass_conversion_and_closes() -> None:
    parameters = _parameters()
    start = LatticeState.zeros(parameters.ny, parameters.nx)
    end = start.copy()
    end.u[0, 0] = 1.0
    end.vx[0, 1] = 2.0
    end.vy[1, 0] = 3.0

    metric = mass_weighted_translation_metric(start, end, parameters)

    assert np.isclose(metric.intramolecular_eV_fs2, 75_000.0)
    assert np.isclose(metric.vx_eV_fs2, 600_000.0)
    assert np.isclose(metric.vy_eV_fs2, 1_350_000.0)
    assert np.isclose(metric.total_eV_fs2, 2_025_000.0)
    assert np.isclose(
        metric.intermolecular_eV_fs2,
        metric.vx_eV_fs2 + metric.vy_eV_fs2,
    )


def test_identical_states_have_zero_mass_metric() -> None:
    parameters = _parameters()
    state = LatticeState.zeros(parameters.ny, parameters.nx)
    metric = mass_weighted_translation_metric(state, state.copy(), parameters)
    assert metric.total_eV_fs2 == 0.0


def test_quadratic_endpoint_curvature_recovers_exact_parabola() -> None:
    s = np.linspace(0.0, 1.0, 21)
    curvature = 0.84
    energy = 0.5 * curvature * s * s
    assert np.isclose(
        quadratic_endpoint_curvature(s, energy, side="start", fit_points=5),
        curvature,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_collective_frequency_matches_definition() -> None:
    assert np.isclose(
        collective_angular_frequency_per_fs(0.8, 200.0),
        np.sqrt(0.8 / 200.0),
    )


def test_isotropic_delta_distortion_has_equal_x_y_translation_metric() -> None:
    parameters = _parameters(6)
    state = LatticeState.zeros(parameters.ny, parameters.nx)
    center = parameters.ny // 2, parameters.nx // 2
    state.u[center] = -0.1
    state.vx[center] = 0.07
    state.vy[center] = 0.07

    mx = one_site_translation_mass_metric(state, parameters, "+x")
    my = one_site_translation_mass_metric(state, parameters, "+y")

    assert np.isclose(mx.total_eV_fs2, my.total_eV_fs2)


def test_midpoint_low_state_spectrum_is_sorted_and_nonnegative_gap() -> None:
    parameters = _parameters(4)
    state = LatticeState.zeros(parameters.ny, parameters.nx)
    state.u[parameters.ny // 2, parameters.nx // 2] = -0.05
    spectrum = midpoint_low_state_spectrum(
        state,
        parameters,
        "+x",
        source_site=parameters.polaron_index,
        level_count=4,
    )

    values = np.asarray(spectrum.eigenvalues_eV)
    assert values.shape == (4,)
    assert np.all(np.diff(values) >= -1.0e-12)
    assert spectrum.first_gap_eV >= -1.0e-12
    assert np.isclose(spectrum.gaps_from_ground_eV[0], 0.0)
