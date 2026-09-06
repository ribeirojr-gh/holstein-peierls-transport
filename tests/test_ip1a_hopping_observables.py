"""Tests for IP1a persistent hopping and local-transfer diagnostics."""

from __future__ import annotations

import numpy as np

from holstein_peierls.dynamics.hopping_observables import (
    PersistentSiteTracker,
    directional_transfer_bias,
    local_bond_transfer_magnitudes,
    local_transfer_anisotropy,
    periodic_nearest_neighbor_step,
)
from holstein_peierls.lattice import LatticeState
from holstein_peierls.parameters import StaticPolaronParameters


def _population(site: int, *, n: int = 16, maximum: float = 0.8) -> np.ndarray:
    values = np.full(n, (1.0 - maximum) / (n - 1), dtype=np.float64)
    values[site] = maximum
    return values


def test_periodic_nearest_neighbor_step_wraps_x() -> None:
    # 4x4 C-order: site 3 is (3,0), site 0 is (0,0).
    assert periodic_nearest_neighbor_step(3, 0, 4, 4) == (1, 0)
    assert periodic_nearest_neighbor_step(0, 3, 4, 4) == (-1, 0)


def test_periodic_nearest_neighbor_step_wraps_y() -> None:
    # site 12=(0,3), site 0=(0,0).
    assert periodic_nearest_neighbor_step(12, 0, 4, 4) == (0, 1)
    assert periodic_nearest_neighbor_step(0, 12, 4, 4) == (0, -1)


def test_persistent_tracker_rejects_short_flicker() -> None:
    tracker = PersistentSiteTracker(4, 4, persistence_samples=3)
    assert tracker.update(_population(5), 0.0) is None
    assert tracker.current_site == 5
    assert tracker.update(_population(6), 2.0) is None
    assert tracker.update(_population(6), 4.0) is None
    assert tracker.update(_population(5), 6.0) is None
    assert tracker.current_site == 5


def test_persistent_tracker_accepts_neighbor_after_required_samples() -> None:
    tracker = PersistentSiteTracker(4, 4, persistence_samples=3)
    tracker.update(_population(5), 0.0)
    assert tracker.update(_population(6), 2.0) is None
    assert tracker.update(_population(6), 4.0) is None
    event = tracker.update(_population(6), 6.0)
    assert event is not None
    assert event.source_site == 5
    assert event.target_site == 6
    assert event.transition_start_time_fs == 2.0
    assert event.accepted_time_fs == 6.0
    assert event.is_nearest_neighbor
    assert event.direction == "+x"
    assert tracker.current_site == 6


def test_persistent_tracker_classifies_nonlocal_transition() -> None:
    tracker = PersistentSiteTracker(4, 4, persistence_samples=2)
    tracker.update(_population(0), 0.0)
    tracker.update(_population(5), 2.0)
    event = tracker.update(_population(5), 4.0)
    assert event is not None
    assert not event.is_nearest_neighbor
    assert event.direction == "nonlocal"


def test_persistent_tracker_requires_dominance_margin() -> None:
    tracker = PersistentSiteTracker(
        4,
        4,
        persistence_samples=2,
        minimum_dominance_margin=0.05,
    )
    tracker.update(_population(5), 0.0)
    ambiguous = np.zeros(16, dtype=np.float64)
    ambiguous[6] = 0.40
    ambiguous[5] = 0.38
    ambiguous[0] = 0.22
    assert tracker.update(ambiguous, 2.0) is None
    assert tracker.update(ambiguous, 4.0) is None
    assert tracker.current_site == 5


def test_local_bond_transfer_magnitudes_recover_bare_uniform_values() -> None:
    parameters = StaticPolaronParameters(nx=4, ny=4, polaron_position=6)
    lattice = LatticeState.zeros(4, 4)
    values = local_bond_transfer_magnitudes(lattice, parameters, 5)
    assert np.isclose(values["+x"], abs(parameters.j0x))
    assert np.isclose(values["-x"], abs(parameters.j0x))
    assert np.isclose(values["+y"], abs(parameters.j0y))
    assert np.isclose(values["-y"], abs(parameters.j0y))


def test_local_anisotropy_and_directional_bias_have_expected_sign() -> None:
    values = {"+x": 0.12, "-x": 0.08, "+y": 0.04, "-y": 0.04}
    expected_anisotropy = (0.12 - 0.04) / (0.12 + 0.04)
    assert np.isclose(local_transfer_anisotropy(values), expected_anisotropy)
    assert directional_transfer_bias(values, "+x") > 0.0
    assert directional_transfer_bias(values, "+y") < 0.0
