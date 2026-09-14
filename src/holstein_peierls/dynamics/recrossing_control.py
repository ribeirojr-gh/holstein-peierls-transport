"""Prospective recrossing/commitment classification helpers for IP1t."""

from __future__ import annotations


def _is_x_event(event: dict | None) -> bool:
    if event is None:
        return False
    return bool(
        event.get("is_nearest_neighbor", False)
        and abs(int(event.get("dx_sites", 0))) == 1
        and int(event.get("dy_sites", 0)) == 0
    )


def is_direct_return_event(
    event: dict | None,
    *,
    branch_site: int,
    previous_site: int,
) -> bool:
    """Return whether an x event directly returns from branch_site to previous_site."""
    return bool(
        _is_x_event(event)
        and int(event["source_site"]) == int(branch_site)
        and int(event["target_site"]) == int(previous_site)
    )


def classify_return_commitment(
    native_first: dict | None,
    reversed_first: dict | None,
    *,
    branch_site: int,
    previous_site: int,
    time_difference_fs: float = 100.0,
) -> dict:
    """Evaluate the preregistered IP1t first-return/commitment criterion.

    Inputs are the first persistent nearest-neighbour x events already filtered
    to the prospective commitment window, or ``None`` when no event occurred.
    """
    threshold = float(time_difference_fs)
    if threshold <= 0.0:
        raise ValueError("time_difference_fs must be positive")
    if native_first is not None and not _is_x_event(native_first):
        raise ValueError("native_first must be a nearest-neighbour x event")
    if reversed_first is not None and not _is_x_event(reversed_first):
        raise ValueError("reversed_first must be a nearest-neighbour x event")

    native_return = is_direct_return_event(
        native_first, branch_site=branch_site, previous_site=previous_site
    )
    reversed_return = is_direct_return_event(
        reversed_first, branch_site=branch_site, previous_site=previous_site
    )

    presence_changed = (native_first is None) != (reversed_first is None)
    direct_return_changed = native_return != reversed_return
    direction_changed = bool(
        native_first is not None
        and reversed_first is not None
        and str(native_first["direction"]) != str(reversed_first["direction"])
    )
    return_time_difference = None
    return_time_changed = False
    if native_return and reversed_return:
        return_time_difference = abs(
            float(native_first["transition_start_time_fs"])
            - float(reversed_first["transition_start_time_fs"])
        )
        return_time_changed = bool(return_time_difference >= threshold)

    changed = bool(
        presence_changed
        or direct_return_changed
        or direction_changed
        or return_time_changed
    )
    return {
        "native_direct_return": native_return,
        "reversed_direct_return": reversed_return,
        "event_presence_changed": presence_changed,
        "direct_return_status_changed": direct_return_changed,
        "first_event_direction_changed": direction_changed,
        "direct_return_start_time_difference_fs": return_time_difference,
        "direct_return_start_time_changed": return_time_changed,
        "return_commitment_changed": changed,
    }
