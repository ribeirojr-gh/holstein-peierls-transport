import pytest

from holstein_peierls.dynamics.recrossing_control import (
    classify_return_commitment,
    is_direct_return_event,
)


def _event(source=820, target=819, direction="-x", start=3344.0, dx=-1):
    return {
        "source_site": source,
        "target_site": target,
        "transition_start_time_fs": start,
        "accepted_time_fs": start + 48.0,
        "dx_sites": dx,
        "dy_sites": 0,
        "is_nearest_neighbor": True,
        "direction": direction,
    }


def test_direct_return_identification():
    assert is_direct_return_event(_event(), branch_site=820, previous_site=819)
    assert not is_direct_return_event(
        _event(source=820, target=821, direction="+x", dx=1),
        branch_site=820,
        previous_site=819,
    )
    assert not is_direct_return_event(None, branch_site=820, previous_site=819)


def test_presence_difference_changes_commitment():
    result = classify_return_commitment(
        _event(), None, branch_site=820, previous_site=819
    )
    assert result["event_presence_changed"]
    assert result["return_commitment_changed"]


def test_return_vs_other_direction_changes_commitment():
    result = classify_return_commitment(
        _event(),
        _event(source=820, target=821, direction="+x", start=3340.0, dx=1),
        branch_site=820,
        previous_site=819,
    )
    assert result["direct_return_status_changed"]
    assert result["first_event_direction_changed"]
    assert result["return_commitment_changed"]


def test_two_similar_returns_do_not_change_commitment():
    result = classify_return_commitment(
        _event(start=3344.0),
        _event(start=3400.0),
        branch_site=820,
        previous_site=819,
        time_difference_fs=100.0,
    )
    assert result["direct_return_start_time_difference_fs"] == pytest.approx(56.0)
    assert not result["direct_return_start_time_changed"]
    assert not result["return_commitment_changed"]


def test_two_returns_separated_by_threshold_change_commitment():
    result = classify_return_commitment(
        _event(start=3344.0),
        _event(start=3444.0),
        branch_site=820,
        previous_site=819,
        time_difference_fs=100.0,
    )
    assert result["direct_return_start_time_changed"]
    assert result["return_commitment_changed"]


def test_rejects_non_x_event():
    bad = _event()
    bad["dx_sites"] = 0
    bad["dy_sites"] = 1
    bad["direction"] = "+y"
    with pytest.raises(ValueError):
        classify_return_commitment(bad, None, branch_site=820, previous_site=819)
