"""Event-conditioned time-window helpers for field-driven phonon-wake analysis.

These utilities are analysis-only. They do not alter the Holstein-Peierls
dynamics. Their purpose is to define a reproducible pre-event baseline and a
post-event interval that ends before the next persistent carrier relocation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class EventWakeWindow:
    """Masks and resolved limits for one event-conditioned wake interval."""

    baseline_mask: BoolArray
    post_mask: BoolArray
    event_time_fs: float
    baseline_start_fs: float
    baseline_end_fs: float
    post_end_fs: float


def event_wake_window(
    times_fs: ArrayLike,
    event_time_fs: float,
    *,
    max_post_fs: float,
    baseline_start_offset_fs: float = -500.0,
    baseline_end_offset_fs: float = -100.0,
    next_event_time_fs: float | None = None,
    next_event_guard_fs: float = 0.0,
) -> EventWakeWindow:
    """Return baseline/post masks for a persistent carrier event.

    The post interval begins at the event start and is truncated either by
    ``max_post_fs`` or by the next event minus an optional guard.  The baseline
    interval is specified by offsets relative to the event start.
    """
    times = np.asarray(times_fs, dtype=np.float64).reshape(-1)
    if times.size < 3 or not np.all(np.isfinite(times)):
        raise ValueError("times_fs must contain at least three finite samples")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times_fs must be strictly increasing")

    event = float(event_time_fs)
    max_post = float(max_post_fs)
    b0 = float(baseline_start_offset_fs)
    b1 = float(baseline_end_offset_fs)
    guard = float(next_event_guard_fs)
    if not all(np.isfinite(value) for value in (event, max_post, b0, b1, guard)):
        raise ValueError("window parameters must be finite")
    if max_post <= 0.0:
        raise ValueError("max_post_fs must be positive")
    if not b0 < b1 < 0.0:
        raise ValueError("baseline offsets must satisfy start < end < 0")
    if guard < 0.0:
        raise ValueError("next_event_guard_fs must be non-negative")

    post_end = event + max_post
    if next_event_time_fs is not None:
        nxt = float(next_event_time_fs)
        if not np.isfinite(nxt) or nxt <= event:
            raise ValueError("next_event_time_fs must be finite and later than the event")
        post_end = min(post_end, nxt - guard)
    if post_end <= event:
        raise ValueError("resolved post-event window is empty")

    baseline_start = event + b0
    baseline_end = event + b1
    baseline = (times >= baseline_start) & (times <= baseline_end)
    post = (times >= event) & (times <= post_end)
    if np.count_nonzero(baseline) < 2:
        raise ValueError("baseline window contains fewer than two samples")
    if np.count_nonzero(post) < 2:
        raise ValueError("post-event window contains fewer than two samples")

    return EventWakeWindow(
        baseline_mask=np.asarray(baseline, dtype=bool),
        post_mask=np.asarray(post, dtype=bool),
        event_time_fs=event,
        baseline_start_fs=baseline_start,
        baseline_end_fs=baseline_end,
        post_end_fs=float(post_end),
    )


def baseline_subtracted_profiles(
    profiles: ArrayLike,
    baseline_mask: ArrayLike,
) -> FloatArray:
    """Subtract the pre-event mean from each spatial profile column."""
    values = np.asarray(profiles, dtype=np.float64)
    mask = np.asarray(baseline_mask, dtype=bool).reshape(-1)
    if values.ndim != 2 or values.shape[0] != mask.size:
        raise ValueError("profiles must have shape (time, space) matching baseline_mask")
    if not np.all(np.isfinite(values)):
        raise ValueError("profiles must contain only finite values")
    if np.count_nonzero(mask) < 2:
        raise ValueError("baseline_mask must select at least two samples")
    baseline = np.mean(values[mask], axis=0)
    return np.asarray(values - baseline[None, :], dtype=np.float64)


def positive_energy_directionality(backward_eV: float, forward_eV: float) -> float:
    """Return signed backward-vs-forward positive outward-energy asymmetry."""
    back = float(backward_eV)
    front = float(forward_eV)
    if not np.isfinite(back) or not np.isfinite(front) or back < 0.0 or front < 0.0:
        raise ValueError("energies must be finite and non-negative")
    denominator = back + front
    return 0.0 if denominator <= 1.0e-30 else float((back - front) / denominator)
