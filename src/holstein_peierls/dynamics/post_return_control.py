"""Prospective post-return re-escape classification for IP1u."""

from __future__ import annotations


def _is_x_event(event: dict | None) -> bool:
    """A persistent nearest-neighbor x event, as reported by the event tracker."""
    return bool(
        event is not None
        and event.get("is_nearest_neighbor", False)
        and abs(int(event.get("dx_sites", 0))) == 1
        and int(event.get("dy_sites", 0)) == 0
    )



def first_x_event_in_window(
    events: list[dict], *, branch_time_fs: float, window_fs: float
) -> dict | None:
    """First confirmed x event whose start AND acceptance fall in the window."""
    if float(window_fs) <= 0.0:
        raise ValueError("window_fs must be positive")
    start = float(branch_time_fs)
    cutoff = start + float(window_fs)
    candidates = [
        event for event in events
        if _is_x_event(event)
        and start <= float(event["transition_start_time_fs"]) <= cutoff
        and float(event["accepted_time_fs"]) <= cutoff
    ]
    return None if not candidates else min(
        candidates, key=lambda item: float(item["transition_start_time_fs"])
    )


def is_direct_reescape_event(
    event: dict | None, *, returned_site: int, previous_site: int
) -> bool:
    """Identify a direct escape from the returned site back to the previous site."""
    return bool(
        _is_x_event(event)
        and int(event["source_site"]) == int(returned_site)
        and int(event["target_site"]) == int(previous_site)
    )


def classify_post_return_escape(
    native_first: dict | None,
    reversed_first: dict | None,
    *,
    returned_site: int,
    previous_site: int,
    time_difference_fs: float = 100.0,
) -> dict:
    """Apply the preregistered IP1u first-x-event escape-stability criterion.

    Inputs must already be filtered to the first persistent x event in the
    prospective 1.5 ps window (or None).  No later event may override this
    primary classification.
    """
    if not (float(time_difference_fs) > 0.0):
        raise ValueError("time_difference_fs must be positive")
    if returned_site == previous_site:
        raise ValueError("returned_site and previous_site must differ")
    if native_first is not None and not _is_x_event(native_first):
        raise ValueError("native_first must be a nearest-neighbor x event")
    if reversed_first is not None and not _is_x_event(reversed_first):
        raise ValueError("reversed_first must be a nearest-neighbor x event")

    native_direct = is_direct_reescape_event(
        native_first, returned_site=returned_site, previous_site=previous_site
    )
    reversed_direct = is_direct_reescape_event(
        reversed_first, returned_site=returned_site, previous_site=previous_site
    )
    presence_changed = (native_first is None) != (reversed_first is None)
    direct_status_changed = native_direct != reversed_direct

    start_difference = None
    time_changed = False
    if native_direct and reversed_direct:
        start_difference = abs(
            float(native_first["transition_start_time_fs"])
            - float(reversed_first["transition_start_time_fs"])
        )
        time_changed = bool(start_difference >= float(time_difference_fs))

    return {
        "native_direct_reescape": native_direct,
        "reversed_direct_reescape": reversed_direct,
        "event_presence_changed": presence_changed,
        "direct_reescape_status_changed": direct_status_changed,
        "direct_reescape_start_time_difference_fs": start_difference,
        "direct_reescape_start_time_changed": time_changed,
        "reescape_status_changed": bool(
            presence_changed or direct_status_changed or time_changed
        ),
    }
