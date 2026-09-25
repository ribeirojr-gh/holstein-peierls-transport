import copy

import pytest

from holstein_peierls.dynamics.ip2b_ensemble import (
    first_x_event_within_window,
    is_direct_recross,
    summarize_ip2b_members,
    x_event_topology_history,
)


def _event(source, target, start, accepted, dx):
    return {
        "source_site": source,
        "target_site": target,
        "transition_start_time_fs": float(start),
        "accepted_time_fs": float(accepted),
        "dx_sites": int(dx),
        "dy_sites": 0,
        "is_nearest_neighbor": True,
        "direction": "+x" if dx > 0 else "-x",
    }


def _record(i, *, d=0.30, dctrl=0.0, valid=True):
    native_first = _event(819, 820, 3200, 3248, 1)
    reversed_first = _event(819, 820, 3210, 3258, 1)
    return {
        "member_id": i,
        "pair_id": i % 16,
        "pair_sign": 1 if i < 16 else -1,
        "valid_member": valid,
        "rejection_reasons": [] if valid else ["pre_intervention:synthetic"],
        "primary": None
        if not valid
        else {
            "D_i": float(d),
            "D_i_ctrl": float(dctrl),
            "first_l1_ge_0p10_offset_fs": 200.0,
            "first_l1_ge_0p25_offset_fs": 700.0 if d >= 0.25 else None,
        },
        "secondary_events": None
        if not valid
        else {
            "native_first_x_event": native_first,
            "reversed_first_x_event": reversed_first,
            "native_first_is_direct_recross": True,
            "reversed_first_is_direct_recross": True,
            "native_x_event_topology_history": [[819, 820]],
            "reversed_x_event_topology_history": [[819, 820]],
            "first_transition_start_abs_difference_fs": 10.0,
        },
    }


def test_event_helpers_require_start_and_acceptance_inside_window():
    events = [
        _event(819, 820, 1000, 1100, 1),
        _event(820, 819, 2500, 2600, -1),
        _event(819, 820, 2990, 3010, 1),
    ]
    first = first_x_event_within_window(events, branch_time_fs=1000, window_fs=2000)
    assert first["source_site"] == 819
    assert first["target_site"] == 820
    assert x_event_topology_history(
        events, branch_time_fs=1000, window_fs=2000
    ) == [[819, 820], [820, 819]]
    assert is_direct_recross(first, post_hop_site=819, pre_hop_site=820)


def test_primary_passes_for_complete_strong_synthetic_design():
    records = [_record(i, d=0.30 + 0.001 * i, dctrl=0.0) for i in range(32)]
    result = summarize_ip2b_members(records)
    assert result["valid_member_count"] == 32
    assert result["primary_ip2b_pass"]
    assert all(result["primary_gates"].values())
    assert result["primary_distribution"]["D_ge_0p25_fraction"] == 1.0
    assert result["primary_distribution"]["median_control_separation_ratio"] >= 100.0


def test_primary_fails_if_fewer_than_24_valid_members():
    records = [_record(i, valid=i < 23) for i in range(32)]
    result = summarize_ip2b_members(records)
    assert not result["primary_gates"]["at_least_24_valid_members"]
    assert not result["primary_ip2b_pass"]


def test_primary_fails_on_median_even_when_control_ratio_is_large():
    records = [_record(i, d=0.20, dctrl=0.0) for i in range(32)]
    result = summarize_ip2b_members(records)
    assert not result["primary_gates"]["median_D_ge_0p25"]
    assert result["primary_gates"]["median_control_ratio_ge_100"]
    assert not result["primary_ip2b_pass"]


def test_primary_fails_if_only_just_under_sixty_percent_reach_threshold():
    records = [
        _record(i, d=0.30 if i < 19 else 0.249, dctrl=0.0)
        for i in range(32)
    ]
    result = summarize_ip2b_members(records)
    assert result["primary_distribution"]["D_ge_0p25_count"] == 19
    assert result["primary_distribution"]["D_ge_0p25_fraction"] == pytest.approx(19 / 32)
    assert not result["primary_gates"]["fraction_D_ge_0p25_ge_0p60"]
    assert not result["primary_ip2b_pass"]


def test_primary_fails_when_control_floor_is_not_well_separated():
    records = [_record(i, d=0.30, dctrl=0.01) for i in range(32)]
    result = summarize_ip2b_members(records)
    assert result["primary_distribution"]["median_control_separation_ratio"] == pytest.approx(30.0)
    assert not result["primary_gates"]["median_control_ratio_ge_100"]
    assert not result["primary_ip2b_pass"]


def test_secondary_history_disagreement_is_descriptive_only():
    records = [_record(i) for i in range(32)]
    changed = copy.deepcopy(records[7])
    changed["secondary_events"]["reversed_x_event_topology_history"] = [
        [819, 818]
    ]
    records[7] = changed
    result = summarize_ip2b_members(records)
    assert result["secondary_events"]["history_disagreement_count"] == 1
    assert result["primary_ip2b_pass"]


def test_exact_member_ids_and_explicit_rejections_are_required():
    records = [_record(i) for i in range(32)]
    bad = copy.deepcopy(records)
    bad[-1]["member_id"] = 30
    with pytest.raises(ValueError):
        summarize_ip2b_members(bad)

    records[0]["valid_member"] = False
    records[0]["rejection_reasons"] = []
    result = summarize_ip2b_members(records)
    assert not result["primary_gates"]["all_32_members_reported_or_rejected"]
    assert not result["primary_ip2b_pass"]
