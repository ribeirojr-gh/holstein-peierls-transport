"""Finite-design aggregation helpers for IP2b.

These functions classify only preregistered deterministic-design endpoints.
They do not assign population probabilities, p-values, randomization tests,
transport coefficients, or material lifetimes.
"""

from __future__ import annotations

from math import isfinite

import numpy as np


PRIMARY_L1_THRESHOLD = 0.25
EARLY_L1_THRESHOLD = 0.10
MINIMUM_VALID_MEMBERS = 24
MINIMUM_THRESHOLD_FRACTION = 0.60
MINIMUM_CONTROL_RATIO_MEDIAN = 100.0
EXPECTED_MEMBER_IDS = tuple(range(32))


def is_nearest_neighbor_x_event(event: dict | None) -> bool:
    return bool(
        event is not None
        and event.get("is_nearest_neighbor", False)
        and abs(int(event.get("dx_sites", 0))) == 1
        and int(event.get("dy_sites", 0)) == 0
    )


def first_x_event_within_window(
    events: list[dict],
    *,
    branch_time_fs: float,
    window_fs: float = 2000.0,
) -> dict | None:
    """Return first persistent x event whose start and acceptance lie in window."""
    start = float(branch_time_fs)
    width = float(window_fs)
    if not isfinite(start) or not isfinite(width) or width <= 0.0:
        raise ValueError("branch_time_fs must be finite and window_fs positive")
    cutoff = start + width
    candidates = [
        event
        for event in events
        if is_nearest_neighbor_x_event(event)
        and start <= float(event["transition_start_time_fs"]) <= cutoff
        and float(event["accepted_time_fs"]) <= cutoff
    ]
    return None if not candidates else min(
        candidates, key=lambda item: float(item["transition_start_time_fs"])
    )


def x_event_topology_history(
    events: list[dict],
    *,
    branch_time_fs: float,
    window_fs: float = 2000.0,
) -> list[list[int]]:
    """Return ordered source/target topology for accepted persistent x events."""
    start = float(branch_time_fs)
    cutoff = start + float(window_fs)
    selected = [
        event
        for event in events
        if is_nearest_neighbor_x_event(event)
        and start <= float(event["transition_start_time_fs"]) <= cutoff
        and float(event["accepted_time_fs"]) <= cutoff
    ]
    selected.sort(key=lambda item: float(item["transition_start_time_fs"]))
    return [[int(event["source_site"]), int(event["target_site"])] for event in selected]


def is_direct_recross(
    event: dict | None,
    *,
    post_hop_site: int,
    pre_hop_site: int,
) -> bool:
    return bool(
        is_nearest_neighbor_x_event(event)
        and int(event["source_site"]) == int(post_hop_site)
        and int(event["target_site"]) == int(pre_hop_site)
    )


def summarize_ip2b_members(member_records: list[dict]) -> dict:
    """Apply the locked IP2b finite-design primary and secondary summaries."""
    if len(member_records) != 32:
        raise ValueError("IP2b aggregate requires exactly 32 member records")
    ids = [int(record["member_id"]) for record in member_records]
    if sorted(ids) != list(EXPECTED_MEMBER_IDS) or len(set(ids)) != 32:
        raise ValueError("member IDs must be exactly 0..31 without duplicates")

    valid = [record for record in member_records if bool(record.get("valid_member", False))]
    d = np.asarray([float(record["primary"]["D_i"]) for record in valid], dtype=np.float64)
    dctrl = np.asarray(
        [float(record["primary"]["D_i_ctrl"]) for record in valid], dtype=np.float64
    )
    if d.size and (
        not np.all(np.isfinite(d))
        or not np.all(np.isfinite(dctrl))
        or np.any(d < 0.0)
        or np.any(dctrl < 0.0)
    ):
        raise ValueError("valid member endpoints must be finite and nonnegative")

    if d.size:
        q1, median, q3 = np.percentile(d, [25.0, 50.0, 75.0], method="linear")
        threshold_count = int(np.count_nonzero(d >= PRIMARY_L1_THRESHOLD))
        threshold_fraction = float(threshold_count / d.size)
        ratios = d / np.maximum(dctrl, 1.0e-12)
        ratio_median = float(np.median(ratios))
        first010 = [
            record["primary"].get("first_l1_ge_0p10_offset_fs")
            for record in valid
            if record["primary"].get("first_l1_ge_0p10_offset_fs") is not None
        ]
        first025 = [
            record["primary"].get("first_l1_ge_0p25_offset_fs")
            for record in valid
            if record["primary"].get("first_l1_ge_0p25_offset_fs") is not None
        ]
    else:
        q1 = median = q3 = float("nan")
        threshold_count = 0
        threshold_fraction = 0.0
        ratios = np.asarray([], dtype=np.float64)
        ratio_median = float("nan")
        first010 = []
        first025 = []

    completeness = bool(
        len(member_records) == 32
        and all(
            bool(record.get("valid_member", False))
            or len(record.get("rejection_reasons", [])) > 0
            for record in member_records
        )
    )

    gates = {
        "at_least_24_valid_members": bool(len(valid) >= MINIMUM_VALID_MEMBERS),
        "median_D_ge_0p25": bool(d.size and median >= PRIMARY_L1_THRESHOLD),
        "fraction_D_ge_0p25_ge_0p60": bool(
            d.size and threshold_fraction >= MINIMUM_THRESHOLD_FRACTION
        ),
        "median_control_ratio_ge_100": bool(
            ratios.size and ratio_median >= MINIMUM_CONTROL_RATIO_MEDIAN
        ),
        "all_32_members_reported_or_rejected": completeness,
    }
    primary_pass = bool(all(gates.values()))

    secondary_records = []
    history_disagreement_count = 0
    first_presence_disagreement_count = 0
    first_direction_disagreement_count = 0
    native_direct_recross_count = 0
    reversed_direct_recross_count = 0
    transition_start_differences = []
    for record in valid:
        secondary = record["secondary_events"]
        native_first = secondary.get("native_first_x_event")
        reversed_first = secondary.get("reversed_first_x_event")
        native_history = secondary.get("native_x_event_topology_history", [])
        reversed_history = secondary.get("reversed_x_event_topology_history", [])
        history_disagreement = native_history != reversed_history
        history_disagreement_count += int(history_disagreement)
        presence_disagreement = (native_first is None) != (reversed_first is None)
        first_presence_disagreement_count += int(presence_disagreement)
        direction_disagreement = bool(
            native_first is not None
            and reversed_first is not None
            and str(native_first["direction"]) != str(reversed_first["direction"])
        )
        first_direction_disagreement_count += int(direction_disagreement)
        native_direct = bool(secondary.get("native_first_is_direct_recross", False))
        reversed_direct = bool(secondary.get("reversed_first_is_direct_recross", False))
        native_direct_recross_count += int(native_direct)
        reversed_direct_recross_count += int(reversed_direct)
        delta = secondary.get("first_transition_start_abs_difference_fs")
        if delta is not None:
            transition_start_differences.append(float(delta))
        secondary_records.append(
            {
                "member_id": int(record["member_id"]),
                "history_disagreement": history_disagreement,
                "first_presence_disagreement": presence_disagreement,
                "first_direction_disagreement": direction_disagreement,
                "native_first_is_direct_recross": native_direct,
                "reversed_first_is_direct_recross": reversed_direct,
                "first_transition_start_abs_difference_fs": delta,
            }
        )

    valid_count = len(valid)
    secondary_summary = {
        "history_disagreement_count": history_disagreement_count,
        "history_disagreement_fraction": (
            0.0 if valid_count == 0 else float(history_disagreement_count / valid_count)
        ),
        "first_presence_disagreement_count": first_presence_disagreement_count,
        "first_direction_disagreement_count": first_direction_disagreement_count,
        "native_direct_recross_count": native_direct_recross_count,
        "native_direct_recross_fraction": (
            0.0 if valid_count == 0 else float(native_direct_recross_count / valid_count)
        ),
        "reversed_direct_recross_count": reversed_direct_recross_count,
        "reversed_direct_recross_fraction": (
            0.0 if valid_count == 0 else float(reversed_direct_recross_count / valid_count)
        ),
        "first_transition_start_abs_difference_fs": transition_start_differences,
        "member_records": secondary_records,
    }

    pair_groups = []
    for pair_id in range(16):
        records = sorted(
            [r for r in member_records if int(r.get("pair_id", -1)) == pair_id],
            key=lambda item: int(item["member_id"]),
        )
        pair_groups.append(
            {
                "pair_id": pair_id,
                "member_ids": [int(r["member_id"]) for r in records],
                "valid_member_ids": [
                    int(r["member_id"]) for r in records if bool(r.get("valid_member", False))
                ],
                "D_i": [
                    float(r["primary"]["D_i"])
                    for r in records
                    if bool(r.get("valid_member", False))
                ],
            }
        )

    return {
        "valid_member_count": valid_count,
        "rejected_member_count": int(32 - valid_count),
        "rejected_members": [
            {
                "member_id": int(record["member_id"]),
                "reasons": list(record.get("rejection_reasons", [])),
            }
            for record in member_records
            if not bool(record.get("valid_member", False))
        ],
        "primary_distribution": {
            "D_i": d.tolist(),
            "D_i_ctrl": dctrl.tolist(),
            "control_separation_ratio": ratios.tolist(),
            "q1_D_i": None if not d.size else float(q1),
            "median_D_i": None if not d.size else float(median),
            "q3_D_i": None if not d.size else float(q3),
            "iqr_D_i": None if not d.size else float(q3 - q1),
            "D_ge_0p25_count": threshold_count,
            "D_ge_0p25_fraction": threshold_fraction,
            "median_control_separation_ratio": None if not ratios.size else ratio_median,
            "first_l1_ge_0p10_offsets_fs": [float(x) for x in first010],
            "first_l1_ge_0p25_offsets_fs": [float(x) for x in first025],
        },
        "primary_gates": gates,
        "primary_ip2b_pass": primary_pass,
        "secondary_events": secondary_summary,
        "antithetic_pair_descriptives": pair_groups,
        "interpretation_guard": {
            "finite_deterministic_design_only": True,
            "no_p_values_or_population_confidence_intervals": True,
            "no_hopping_probability_rate_mobility_or_activation_energy": True,
        },
    }
