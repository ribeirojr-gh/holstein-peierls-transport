"""Matched event/background helpers for natural field-driven wake analysis.

These routines are passive analysis utilities. They do not modify the dynamics.
IP1k uses them to compare real persistent carrier relocations with event-free
pseudo-event windows drawn from the same deterministic field-driven trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class MatchedWindow:
    """Resolved baseline/post masks for one real or pseudo event center."""

    baseline_mask: BoolArray
    post_mask: BoolArray
    center_time_fs: float
    baseline_start_fs: float
    baseline_end_fs: float
    post_end_fs: float


def matched_window(
    times_fs: ArrayLike,
    center_time_fs: float,
    *,
    baseline_start_offset_fs: float = -700.0,
    baseline_end_offset_fs: float = -100.0,
    post_end_offset_fs: float = 800.0,
) -> MatchedWindow:
    """Return complete baseline/post masks around one analysis center."""
    times = np.asarray(times_fs, dtype=np.float64).reshape(-1)
    if times.size < 3 or not np.all(np.isfinite(times)):
        raise ValueError("times_fs must contain at least three finite samples")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times_fs must be strictly increasing")

    center = float(center_time_fs)
    b0 = float(baseline_start_offset_fs)
    b1 = float(baseline_end_offset_fs)
    post = float(post_end_offset_fs)
    if not all(np.isfinite(value) for value in (center, b0, b1, post)):
        raise ValueError("window parameters must be finite")
    if not b0 < b1 < 0.0:
        raise ValueError("baseline offsets must satisfy start < end < 0")
    if post <= 0.0:
        raise ValueError("post_end_offset_fs must be positive")

    baseline_start = center + b0
    baseline_end = center + b1
    post_end = center + post
    if baseline_start < times[0] - 1.0e-10 or post_end > times[-1] + 1.0e-10:
        raise ValueError("requested matched window is not fully contained in the trajectory")

    baseline_mask = (times >= baseline_start) & (times <= baseline_end)
    post_mask = (times >= center) & (times <= post_end)
    if np.count_nonzero(baseline_mask) < 2:
        raise ValueError("baseline window contains fewer than two samples")
    if np.count_nonzero(post_mask) < 2:
        raise ValueError("post-event window contains fewer than two samples")

    return MatchedWindow(
        baseline_mask=np.asarray(baseline_mask, dtype=bool),
        post_mask=np.asarray(post_mask, dtype=bool),
        center_time_fs=center,
        baseline_start_fs=baseline_start,
        baseline_end_fs=baseline_end,
        post_end_fs=post_end,
    )


def window_contains_event(
    center_time_fs: float,
    event_times_fs: ArrayLike,
    *,
    baseline_start_offset_fs: float = -700.0,
    post_end_offset_fs: float = 800.0,
    exclude_center_event: bool = False,
    atol_fs: float = 1.0e-8,
) -> bool:
    """Return whether another persistent-event start lies in the full window."""
    center = float(center_time_fs)
    events = np.asarray(event_times_fs, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(events)):
        raise ValueError("event_times_fs must be finite")
    lower = center + float(baseline_start_offset_fs)
    upper = center + float(post_end_offset_fs)
    inside = (events >= lower) & (events <= upper)
    if exclude_center_event:
        inside &= ~np.isclose(events, center, rtol=0.0, atol=float(atol_fs))
    return bool(np.any(inside))


def complete_real_event_times(
    times_fs: ArrayLike,
    event_times_fs: ArrayLike,
    *,
    baseline_start_offset_fs: float = -700.0,
    post_end_offset_fs: float = 800.0,
) -> FloatArray:
    """Return real event times with complete, non-overlapping local windows."""
    times = np.asarray(times_fs, dtype=np.float64).reshape(-1)
    events = np.asarray(event_times_fs, dtype=np.float64).reshape(-1)
    selected: list[float] = []
    for event in events:
        try:
            matched_window(
                times,
                float(event),
                baseline_start_offset_fs=baseline_start_offset_fs,
                post_end_offset_fs=post_end_offset_fs,
            )
        except ValueError:
            continue
        if window_contains_event(
            float(event),
            events,
            baseline_start_offset_fs=baseline_start_offset_fs,
            post_end_offset_fs=post_end_offset_fs,
            exclude_center_event=True,
        ):
            continue
        selected.append(float(event))
    return np.asarray(selected, dtype=np.float64)


def pseudo_event_times(
    times_fs: ArrayLike,
    event_times_fs: ArrayLike,
    *,
    cadence_fs: float = 100.0,
    baseline_start_offset_fs: float = -700.0,
    post_end_offset_fs: float = 800.0,
) -> FloatArray:
    """Return deterministic event-free pseudo centers on the sampled time grid."""
    times = np.asarray(times_fs, dtype=np.float64).reshape(-1)
    events = np.asarray(event_times_fs, dtype=np.float64).reshape(-1)
    if times.size < 3 or np.any(np.diff(times) <= 0.0):
        raise ValueError("times_fs must be a strictly increasing sampled trajectory")
    cadence = float(cadence_fs)
    if not np.isfinite(cadence) or cadence <= 0.0:
        raise ValueError("cadence_fs must be finite and positive")
    dt = float(np.mean(np.diff(times)))
    if not np.allclose(np.diff(times), dt, rtol=0.0, atol=max(1.0e-10, 1.0e-10 * abs(dt))):
        raise ValueError("times_fs must be uniformly sampled")
    stride = cadence / dt
    stride_int = int(round(stride))
    if stride_int <= 0 or not np.isclose(stride, stride_int, rtol=0.0, atol=1.0e-10):
        raise ValueError("cadence_fs must be an integer multiple of the sampling interval")

    selected: list[float] = []
    for index in range(0, times.size, stride_int):
        center = float(times[index])
        try:
            matched_window(
                times,
                center,
                baseline_start_offset_fs=baseline_start_offset_fs,
                post_end_offset_fs=post_end_offset_fs,
            )
        except ValueError:
            continue
        if window_contains_event(
            center,
            events,
            baseline_start_offset_fs=baseline_start_offset_fs,
            post_end_offset_fs=post_end_offset_fs,
            exclude_center_event=False,
        ):
            continue
        selected.append(center)
    return np.asarray(selected, dtype=np.float64)


def empirical_percentile(value: float, background: ArrayLike) -> float:
    """Return the inclusive empirical percentile of one value in a background set."""
    observed = float(value)
    samples = np.asarray(background, dtype=np.float64).reshape(-1)
    if not np.isfinite(observed) or samples.size == 0 or not np.all(np.isfinite(samples)):
        raise ValueError("value/background must be finite and background non-empty")
    return float(100.0 * np.count_nonzero(samples <= observed) / samples.size)


def packet_gate(
    *,
    lag_fs: float,
    correlation: float,
    speed_sites_per_ps: float,
    vmax_sites_per_ps: float,
    lag_min_fs: float = 300.0,
    lag_max_fs: float = 700.0,
    minimum_correlation: float = 0.80,
) -> bool:
    """Apply the preregistered IP1k packet-propagation gate."""
    lag = float(lag_fs)
    corr = float(correlation)
    speed = float(speed_sites_per_ps)
    vmax = float(vmax_sites_per_ps)
    lower = float(lag_min_fs)
    upper = float(lag_max_fs)
    if not all(np.isfinite(value) for value in (lag, corr, speed, vmax, lower, upper)):
        return False
    if not lower < upper or vmax <= 0.0:
        raise ValueError("invalid packet-gate bounds")
    return bool(
        lag > lower
        and lag < upper
        and corr >= float(minimum_correlation)
        and speed <= 1.05 * vmax
    )


def background_separated(
    *,
    packet_qualified: bool,
    energy_percentile: float,
    peak_percentile: float,
    threshold_percentile: float = 95.0,
) -> bool:
    """Return the IP1k amplitude/background separation decision."""
    threshold = float(threshold_percentile)
    energy = float(energy_percentile)
    peak = float(peak_percentile)
    if not all(np.isfinite(value) for value in (threshold, energy, peak)):
        return False
    return bool(packet_qualified and energy > threshold and peak > threshold)
