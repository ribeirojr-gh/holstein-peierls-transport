import pytest

from holstein_peierls.dynamics.post_return_control import (
    classify_post_return_escape,
    is_direct_reescape_event,
)


def _event(source=819, target=820, start=3702.0, dx=1, direction="+x"):
    return {
        "source_site": source,
        "target_site": target,
        "transition_start_time_fs": float(start),
        "accepted_time_fs": float(start + 48.0),
        "dx_sites": dx,
        "dy_sites": 0,
        "is_nearest_neighbor": True,
        "direction": direction,
    }


def _classify(native, reversed_, threshold=100.0):
    return classify_post_return_escape(
        native, reversed_, returned_site=819, previous_site=820,
        time_difference_fs=threshold,
    )


def test_direct_reescape_identification():
    assert is_direct_reescape_event(
        _event(), returned_site=819, previous_site=820
    )
    assert not is_direct_reescape_event(
        _event(source=819, target=818, dx=-1, direction="-x"),
        returned_site=819, previous_site=820,
    )
    assert not is_direct_reescape_event(
        None, returned_site=819, previous_site=820
    )


def test_one_event_vs_none_changes_escape_status():
    result = _classify(None, _event())
    assert result["event_presence_changed"]
    assert result["reversed_direct_reescape"]
    assert result["reescape_status_changed"]


def test_reescape_vs_other_x_event_changes_escape_status():
    other = _event(source=819, target=818, dx=-1, direction="-x")
    result = _classify(_event(), other)
    assert result["direct_reescape_status_changed"]
    assert result["reescape_status_changed"]


def test_two_none_do_not_change_escape_status():
    result = _classify(None, None)
    assert not result["reescape_status_changed"]


def test_two_non_direct_events_do_not_promote_difference_post_hoc():
    left = _event(source=819, target=818, dx=-1, direction="-x")
    right = _event(source=818, target=819, dx=1, direction="+x")
    result = _classify(left, right)
    assert not result["direct_reescape_status_changed"]
    assert not result["reescape_status_changed"]


def test_two_direct_reescapes_below_threshold_do_not_change_status():
    result = _classify(_event(start=3702.0), _event(start=3798.0))
    assert result["direct_reescape_start_time_difference_fs"] == pytest.approx(96.0)
    assert not result["reescape_status_changed"]


def test_two_direct_reescapes_at_threshold_change_status():
    result = _classify(_event(start=3702.0), _event(start=3802.0))
    assert result["direct_reescape_start_time_changed"]
    assert result["reescape_status_changed"]


def test_reject_invalid_events_and_threshold():
    bad = _event()
    bad["dx_sites"] = 0
    bad["dy_sites"] = 1
    with pytest.raises(ValueError):
        _classify(bad, None)
    with pytest.raises(ValueError):
        _classify(None, None, threshold=0.0)
    with pytest.raises(ValueError):
        classify_post_return_escape(
            None, None, returned_site=819, previous_site=819
        )
