"""Utilities for inter-hop lattice-current memory diagnostics.

These helpers operate on already constructed outward boundary-current series.
They are deliberately agnostic to the electronic model and event detector so
that the numerical definitions used by IP1l can be unit tested independently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class FluxWindowMetrics:
    start_fs: float
    end_fs: float
    center_fs: float
    positive_energy_eV: float
    net_energy_eV: float
    peak_outward_flux_eV_per_fs: float
    mean_outward_flux_eV_per_fs: float


def _as_uniform_times(times_fs: ArrayLike) -> tuple[FloatArray, float]:
    times = np.asarray(times_fs, dtype=np.float64).reshape(-1)
    if times.size < 2 or not np.all(np.isfinite(times)):
        raise ValueError("times_fs must contain at least two finite samples")
    delta = np.diff(times)
    if np.any(delta <= 0.0):
        raise ValueError("times_fs must be strictly increasing")
    dt = float(np.mean(delta))
    if not np.allclose(delta, dt, rtol=0.0, atol=max(1.0e-10, 1.0e-10 * abs(dt))):
        raise ValueError("times_fs must be uniformly sampled")
    return times, dt


def interval_mask(times_fs: ArrayLike, start_fs: float, end_fs: float) -> NDArray[np.bool_]:
    """Inclusive mask for a finite ordered time interval."""
    times = np.asarray(times_fs, dtype=np.float64).reshape(-1)
    start = float(start_fs)
    end = float(end_fs)
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        raise ValueError("interval bounds must be finite and ordered")
    return (times >= start) & (times <= end)


def flux_window_metrics(
    times_fs: ArrayLike,
    outward_flux_eV_per_fs: ArrayLike,
    start_fs: float,
    end_fs: float,
) -> FluxWindowMetrics:
    """Integrate outward boundary flux inside one inclusive time window."""
    times, _ = _as_uniform_times(times_fs)
    flux = np.asarray(outward_flux_eV_per_fs, dtype=np.float64).reshape(-1)
    if flux.shape != times.shape or not np.all(np.isfinite(flux)):
        raise ValueError("outward flux must be finite and match times_fs")
    mask = interval_mask(times, start_fs, end_fs)
    if np.count_nonzero(mask) < 2:
        raise ValueError("window must contain at least two sampled times")
    local_t = times[mask]
    local_f = flux[mask]
    positive = float(np.trapezoid(np.clip(local_f, 0.0, None), local_t))
    net = float(np.trapezoid(local_f, local_t))
    return FluxWindowMetrics(
        start_fs=float(start_fs),
        end_fs=float(end_fs),
        center_fs=0.5 * (float(start_fs) + float(end_fs)),
        positive_energy_eV=positive,
        net_energy_eV=net,
        peak_outward_flux_eV_per_fs=float(np.max(local_f)),
        mean_outward_flux_eV_per_fs=float(np.mean(local_f)),
    )


def sliding_flux_windows(
    times_fs: ArrayLike,
    outward_flux_eV_per_fs: ArrayLike,
    start_fs: float,
    end_fs: float,
    *,
    width_fs: float,
    step_fs: float,
) -> list[FluxWindowMetrics]:
    """Return regularly stepped complete windows inside ``[start,end]``."""
    times, dt = _as_uniform_times(times_fs)
    width = float(width_fs)
    step = float(step_fs)
    if width <= 0.0 or step <= 0.0:
        raise ValueError("window width and step must be positive")
    for value, name in ((width, "width_fs"), (step, "step_fs")):
        ratio = value / dt
        if not np.isclose(ratio, round(ratio), rtol=0.0, atol=1.0e-10):
            raise ValueError(f"{name} must be an integer multiple of the sampling interval")
    first = float(start_fs)
    last_start = float(end_fs) - width
    if last_start < first - 1.0e-10:
        return []
    n = int(np.floor((last_start - first) / step + 1.0e-10)) + 1
    starts = first + step * np.arange(n, dtype=np.float64)
    return [
        flux_window_metrics(times, outward_flux_eV_per_fs, float(value), float(value + width))
        for value in starts
    ]


def nonoverlapping_flux_bins(
    times_fs: ArrayLike,
    outward_flux_eV_per_fs: ArrayLike,
    start_fs: float,
    end_fs: float,
    *,
    width_fs: float,
) -> list[FluxWindowMetrics]:
    """Return complete adjacent bins, dropping an incomplete trailing interval."""
    return sliding_flux_windows(
        times_fs,
        outward_flux_eV_per_fs,
        start_fs,
        end_fs,
        width_fs=width_fs,
        step_fs=width_fs,
    )


def empirical_percentile(value: float, background: ArrayLike) -> float:
    """Inclusive empirical percentile used as a robustness discriminator."""
    samples = np.asarray(background, dtype=np.float64).reshape(-1)
    if samples.size == 0 or not np.all(np.isfinite(samples)) or not np.isfinite(value):
        raise ValueError("value and background must be finite and background non-empty")
    return float(100.0 * np.count_nonzero(samples <= float(value)) / samples.size)


def memory_gate(
    records: list[dict],
    *,
    first_event_time_fs: float,
    minimum_late_offset_fs: float = 1000.0,
    percentile_threshold: float = 95.0,
) -> dict:
    """Evaluate late inter-hop memory from precomputed percentile records."""
    threshold_time = float(first_event_time_fs) + float(minimum_late_offset_fs)
    late = [record for record in records if float(record["center_fs"]) > threshold_time]
    passing = [
        record
        for record in late
        if float(record["positive_energy_percentile"]) >= float(percentile_threshold)
        and float(record["peak_flux_percentile"]) >= float(percentile_threshold)
    ]
    latest = None if not passing else max(float(record["center_fs"]) for record in passing)
    fraction = 0.0 if not late else float(len(passing) / len(late))
    return {
        "late_bin_count": int(len(late)),
        "late_passing_bin_count": int(len(passing)),
        "late_passing_fraction": fraction,
        "latest_passing_center_fs": latest,
        "late_memory_gate_pass": bool(passing),
    }


def incremental_pulse_gate(
    *,
    packet_qualified: bool,
    peak_percentile: float,
    positive_excess_energy_eV: float,
    percentile_threshold: float = 95.0,
) -> bool:
    """IP1l gate for renewed radiation on top of inherited wake memory."""
    return bool(
        packet_qualified
        and np.isfinite(peak_percentile)
        and float(peak_percentile) >= float(percentile_threshold)
        and np.isfinite(positive_excess_energy_eV)
        and float(positive_excess_energy_eV) > 0.0
    )
