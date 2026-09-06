"""Tests for IP1c event-conditioned lattice/current diagnostics."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from holstein_peierls.dynamics.event_conditioned_hopping import (
    PeriodicTemplateCorrelationField,
    first_negative_to_nonnegative_crossing,
    project_displacement_on_hop,
)
from holstein_peierls.parameters import StaticPolaronParameters
from holstein_peierls.polaron import solve_static_polaron
from holstein_peierls.translation_barrier import (
    translate_lattice_state,
    translated_site_index,
)


def _relaxed_case(ratio: float = 0.5):
    base = StaticPolaronParameters(nx=4, ny=4, polaron_position=11)
    parameters = replace(base, j0y=base.j0x * ratio)
    result = solve_static_polaron(
        parameters,
        solver="dense_lowest",
        gradient_mode="optimized",
        legacy_convergence=False,
    )
    assert result.diagnostics.converged
    center = int(np.argmax(np.asarray(result.charge_density).reshape(-1, order="C")))
    return parameters, result.state, center


def test_template_correlation_components_sum_to_total() -> None:
    parameters, state, center = _relaxed_case()
    field = PeriodicTemplateCorrelationField(state, parameters, center)
    maps = field.maps(state)
    assert np.allclose(maps.total, maps.u + maps.bond_x + maps.bond_y, atol=1.0e-13)


def test_reference_template_has_unit_self_amplitude() -> None:
    parameters, state, center = _relaxed_case()
    field = PeriodicTemplateCorrelationField(state, parameters, center)
    maps = field.maps(state)
    total, u, bond_x, bond_y = field.amplitude_at_center(maps, center)
    assert np.isclose(total, 1.0, atol=1.0e-12)
    assert np.isclose(total, u + bond_x + bond_y, atol=1.0e-12)


def test_source_target_coordinate_changes_sign_under_exact_translation() -> None:
    parameters, state, center = _relaxed_case()
    target = translated_site_index(center, parameters, "+x")
    field = PeriodicTemplateCorrelationField(state, parameters, center)
    initial = field.source_target_coordinate(field.maps(state), center, target)
    translated = translate_lattice_state(state, "+x")
    final = field.source_target_coordinate(field.maps(translated), center, target)
    assert initial.coordinate < 0.0
    assert final.coordinate > 0.0
    assert np.isclose(initial.coordinate, -final.coordinate, atol=1.0e-10)


def test_source_target_coordinate_is_invariant_to_positive_template_scaling() -> None:
    parameters, state, center = _relaxed_case()
    target = translated_site_index(center, parameters, "+y")
    field = PeriodicTemplateCorrelationField(state, parameters, center)
    base_maps = field.maps(state)
    from holstein_peierls.lattice import LatticeState

    scaled = LatticeState(2.5 * state.u, 2.5 * state.vx, 2.5 * state.vy)
    scaled_maps = field.maps(scaled)
    q0 = field.source_target_coordinate(base_maps, center, target).coordinate
    q1 = field.source_target_coordinate(scaled_maps, center, target).coordinate
    assert np.isclose(q0, q1, atol=1.0e-12)


def test_shift_index_wraps_periodic_boundary() -> None:
    parameters, state, center = _relaxed_case()
    field = PeriodicTemplateCorrelationField(state, parameters, center)
    x = center % parameters.nx
    y = center // parameters.nx
    wrapped_center = ((x - 1) % parameters.nx) + parameters.nx * y
    iy, ix = field.shift_index_for_center(wrapped_center)
    assert iy == 0
    assert ix == parameters.nx - 1


def test_project_displacement_has_expected_signs() -> None:
    parallel, perpendicular = project_displacement_on_hop(3.0, 0.5, 1, 0)
    assert parallel == pytest.approx(3.0)
    assert perpendicular == pytest.approx(0.5)
    parallel, perpendicular = project_displacement_on_hop(3.0, 0.5, -1, 0)
    assert parallel == pytest.approx(-3.0)
    assert perpendicular == pytest.approx(-0.5)
    parallel, perpendicular = project_displacement_on_hop(0.5, -3.0, 0, -1)
    assert parallel == pytest.approx(3.0)
    assert perpendicular == pytest.approx(0.5)


def test_project_displacement_rejects_non_nearest_step() -> None:
    with pytest.raises(ValueError):
        project_displacement_on_hop(1.0, 1.0, 1, 1)


def test_first_negative_to_nonnegative_crossing_interpolates() -> None:
    times = np.asarray([-2.0, -1.0, 0.0, 1.0])
    values = np.asarray([-1.0, -0.5, 0.5, 1.0])
    crossing = first_negative_to_nonnegative_crossing(times, values)
    assert crossing == pytest.approx(-0.5)
