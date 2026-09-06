"""Tests for IP1d matched counterfactual precursor utilities."""

from __future__ import annotations

import numpy as np
import pytest

from holstein_peierls.dynamics.matched_precursors import (
    compare_true_direction,
    nearest_negative_to_nonnegative_crossing,
    nearest_neighbor_site,
    nearest_neighbor_targets,
    negative_to_nonnegative_crossings,
)


def test_nearest_neighbor_site_wraps_periodically() -> None:
    nx = ny = 4
    assert nearest_neighbor_site(3, "+x", nx, ny) == 0
    assert nearest_neighbor_site(0, "-x", nx, ny) == 3
    assert nearest_neighbor_site(12, "+y", nx, ny) == 0
    assert nearest_neighbor_site(0, "-y", nx, ny) == 12


def test_nearest_neighbor_targets_returns_four_distinct_sites() -> None:
    targets = nearest_neighbor_targets(5, 4, 4)
    assert set(targets) == {"+x", "-x", "+y", "-y"}
    assert len(set(targets.values())) == 4


def test_compare_true_direction_reports_advantage_and_rank() -> None:
    result = compare_true_direction(
        {"+x": 4.0, "-x": 1.0, "+y": 2.0, "-y": 3.0},
        "+x",
    )
    assert result.true_value == pytest.approx(4.0)
    assert result.counterfactual_mean == pytest.approx(2.0)
    assert result.advantage == pytest.approx(2.0)
    assert result.true_rank == 1
    assert result.true_is_maximum


def test_compare_true_direction_handles_nonmaximal_true_direction() -> None:
    result = compare_true_direction(
        {"+x": 1.0, "-x": 4.0, "+y": 2.0, "-y": 3.0},
        "+x",
    )
    assert result.true_rank == 4
    assert not result.true_is_maximum
    assert result.advantage < 0.0


def test_negative_to_nonnegative_crossings_returns_all_interpolated_crossings() -> None:
    times = np.asarray([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0])
    values = np.asarray([-1.0, 1.0, -1.0, -1.0, 1.0, 1.0])
    crossings = negative_to_nonnegative_crossings(times, values)
    assert np.allclose(crossings, np.asarray([-2.5, 0.5]))


def test_nearest_crossing_uses_reference_instead_of_first_crossing() -> None:
    times = np.asarray([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0])
    values = np.asarray([-1.0, 1.0, -1.0, -1.0, 1.0, 1.0])
    crossing = nearest_negative_to_nonnegative_crossing(times, values, 0.0)
    assert crossing == pytest.approx(0.5)


def test_nearest_crossing_returns_none_without_crossing() -> None:
    times = np.asarray([-1.0, 0.0, 1.0])
    values = np.asarray([-2.0, -1.0, -0.5])
    assert nearest_negative_to_nonnegative_crossing(times, values) is None


def test_invalid_direction_is_rejected() -> None:
    with pytest.raises(ValueError):
        nearest_neighbor_site(0, "+z", 4, 4)
