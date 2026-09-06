"""Tests for IP1b dressed-polaron lattice-distortion tracking."""

from __future__ import annotations

import numpy as np

from holstein_peierls.dynamics.dressed_hopping import (
    DistortionTemplateMatch,
    PeriodicDistortionTemplateMatcher,
    PersistentTemplateTracker,
    match_transition_events,
    prospective_bond_rank,
)
from holstein_peierls.dynamics.hopping_observables import PersistentHopEvent
from holstein_peierls.lattice import LatticeState
from holstein_peierls.parameters import StaticPolaronParameters
from holstein_peierls.translation_barrier import translate_lattice_state


def _template_case():
    parameters = StaticPolaronParameters(nx=4, ny=4, polaron_position=5)
    u = np.zeros((4, 4), dtype=np.float64)
    vx = np.zeros((4, 4), dtype=np.float64)
    vy = np.zeros((4, 4), dtype=np.float64)
    u[1, 1] = -0.20
    u[1, 2] = -0.05
    vx[1, 1] = 0.03
    vx[1, 2] = -0.02
    vy[1, 1] = -0.04
    vy[2, 1] = 0.01
    lattice = LatticeState(u, vx, vy)
    return parameters, lattice


def test_template_self_match_recovers_reference_center_and_unit_amplitude() -> None:
    parameters, lattice = _template_case()
    matcher = PeriodicDistortionTemplateMatcher(lattice, parameters, 5)
    result = matcher.match(lattice)
    assert result.center_site == 5
    assert result.shift_x_sites == 0
    assert result.shift_y_sites == 0
    assert np.isclose(result.amplitude, 1.0, atol=1.0e-12)
    assert np.isclose(
        result.u_amplitude + result.bond_x_amplitude + result.bond_y_amplitude,
        result.amplitude,
        atol=1.0e-12,
    )


def test_template_translation_recovers_periodic_center() -> None:
    parameters, lattice = _template_case()
    matcher = PeriodicDistortionTemplateMatcher(lattice, parameters, 5)
    moved = translate_lattice_state(lattice, "+x")
    result = matcher.match(moved)
    assert result.center_site == 6
    assert result.shift_x_sites == 1
    assert result.shift_y_sites == 0
    assert np.isclose(result.amplitude, 1.0, atol=1.0e-12)


def test_template_translation_wraps_periodically() -> None:
    parameters, lattice = _template_case()
    matcher = PeriodicDistortionTemplateMatcher(lattice, parameters, 5)
    moved = lattice.copy()
    for _ in range(3):
        moved = translate_lattice_state(moved, "-x")
    result = matcher.match(moved)
    assert result.center_site == 6
    assert result.shift_x_sites == 1


def test_uniform_inter_displacement_gauges_do_not_change_template_match() -> None:
    parameters, lattice = _template_case()
    matcher = PeriodicDistortionTemplateMatcher(lattice, parameters, 5)
    gauged = LatticeState(
        lattice.u.copy(),
        lattice.vx + 12.3,
        lattice.vy - 7.1,
    )
    result = matcher.match(gauged)
    assert result.center_site == 5
    assert np.isclose(result.amplitude, 1.0, atol=1.0e-12)


def _match(site: int, *, amplitude: float = 1.0, gap: float = 0.5) -> DistortionTemplateMatch:
    return DistortionTemplateMatch(
        center_site=site,
        shift_x_sites=0,
        shift_y_sites=0,
        amplitude=amplitude,
        second_amplitude=amplitude * (1.0 - gap),
        relative_gap=gap,
        u_amplitude=0.3,
        bond_x_amplitude=0.4,
        bond_y_amplitude=0.3,
    )


def test_persistent_template_tracker_requires_requested_persistence() -> None:
    tracker = PersistentTemplateTracker(4, 4, persistence_samples=3)
    assert tracker.update(_match(5), 0.0) is None
    assert tracker.update(_match(6), 2.0) is None
    assert tracker.update(_match(6), 4.0) is None
    event = tracker.update(_match(6), 6.0)
    assert event is not None
    assert event.source_site == 5
    assert event.target_site == 6
    assert event.is_nearest_neighbor
    assert event.direction == "+x"
    assert event.transition_start_time_fs == 2.0


def test_persistent_template_tracker_rejects_low_confidence_candidate() -> None:
    tracker = PersistentTemplateTracker(
        4,
        4,
        persistence_samples=2,
        minimum_amplitude=0.1,
        minimum_relative_gap=0.05,
    )
    assert tracker.update(_match(5), 0.0) is None
    assert tracker.update(_match(6, amplitude=0.01, gap=0.5), 2.0) is None
    assert tracker.update(_match(6, amplitude=1.0, gap=0.01), 4.0) is None
    assert tracker.update(_match(6), 6.0) is None
    event = tracker.update(_match(6), 8.0)
    assert event is not None


def _event(source: int, target: int, time_fs: float) -> PersistentHopEvent:
    dx = (target % 4) - (source % 4)
    dy = (target // 4) - (source // 4)
    return PersistentHopEvent(
        source_site=source,
        target_site=target,
        transition_start_time_fs=time_fs,
        accepted_time_fs=time_fs + 10.0,
        dx_sites=dx,
        dy_sites=dy,
        is_nearest_neighbor=abs(dx) + abs(dy) == 1,
        direction="+x" if (dx, dy) == (1, 0) else "nonlocal",
        persistence_samples=5,
    )


def test_transition_matching_uses_same_transition_and_closest_lag() -> None:
    electronic = (_event(5, 6, 100.0),)
    lattice = (
        _event(5, 6, 180.0),
        _event(5, 6, 130.0),
        _event(6, 7, 105.0),
    )
    matched = match_transition_events(electronic, lattice, maximum_abs_lag_fs=100.0)
    assert len(matched) == 1
    assert matched[0].lattice.transition_start_time_fs == 130.0
    assert matched[0].lag_fs == 30.0


def test_prospective_bond_rank_identifies_strongest_and_weakest() -> None:
    values = {"+x": 0.20, "-x": 0.10, "+y": 0.15, "-y": 0.05}
    assert prospective_bond_rank(values, "+x") == 1
    assert prospective_bond_rank(values, "-y") == 4
