"""Helpers for IP1m single-hop wake-memory diagnostics.

The functions here are intentionally model-agnostic. They operate on event
metadata and already-computed percentile records so that the isotropic
single-hop acceptance rules can be unit tested separately from the production
dynamics.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def continuation_neighbor(
    site: int,
    *,
    dx_sites: int,
    dy_sites: int,
    nx: int,
    ny: int,
) -> int:
    """Return the periodic nearest neighbor one step beyond ``site``.

    ``dx_sites`` and ``dy_sites`` are the carrier-event lattice displacements.
    Exactly one component must have magnitude one and the other must be zero.
    """
    nx_i = int(nx)
    ny_i = int(ny)
    site_i = int(site)
    dx = int(dx_sites)
    dy = int(dy_sites)
    if nx_i <= 0 or ny_i <= 0:
        raise ValueError("nx and ny must be positive")
    if not (0 <= site_i < nx_i * ny_i):
        raise ValueError("site is outside the periodic lattice")
    if abs(dx) + abs(dy) != 1:
        raise ValueError("event displacement must be one nearest-neighbor step")
    x = site_i % nx_i
    y = site_i // nx_i
    x2 = (x + dx) % nx_i
    y2 = (y + dy) % ny_i
    return int(y2 * nx_i + x2)


def jointly_background_separated(record: dict, *, threshold: float = 95.0) -> bool:
    """Return whether energy and peak percentiles both meet ``threshold``."""
    energy = float(record["positive_energy_percentile"])
    peak = float(record["peak_flux_percentile"])
    return bool(
        np.isfinite(energy)
        and np.isfinite(peak)
        and energy >= float(threshold)
        and peak >= float(threshold)
    )


def sustained_memory_gate(
    records: list[dict],
    *,
    first_event_time_fs: float,
    minimum_late_offset_fs: float = 1000.0,
    minimum_passing_bins: int = 3,
    minimum_latest_offset_fs: float = 1500.0,
    percentile_threshold: float = 95.0,
) -> dict:
    """Evaluate the preregistered IP1m late and sustained-memory gates."""
    t0 = float(first_event_time_fs)
    late_threshold = t0 + float(minimum_late_offset_fs)
    latest_threshold = t0 + float(minimum_latest_offset_fs)
    late = [record for record in records if float(record["center_fs"]) > late_threshold]
    passing = [
        record
        for record in late
        if jointly_background_separated(record, threshold=percentile_threshold)
    ]
    latest = None if not passing else max(float(record["center_fs"]) for record in passing)
    late_present = bool(passing)
    sustained = bool(
        len(passing) >= int(minimum_passing_bins)
        and latest is not None
        and latest >= latest_threshold
    )
    return {
        "late_bin_count": int(len(late)),
        "late_passing_bin_count": int(len(passing)),
        "late_passing_fraction": 0.0 if not late else float(len(passing) / len(late)),
        "latest_passing_center_fs": latest,
        "late_memory_present": late_present,
        "sustained_late_memory": sustained,
        "minimum_passing_bins": int(minimum_passing_bins),
        "minimum_latest_offset_fs": float(minimum_latest_offset_fs),
    }


def ordered_envelope_fit(
    distance_to_center_fs: dict[int, float | None],
    *,
    required_distances: Iterable[int] = (1, 2, 3),
) -> dict:
    """Fit first-arrival center versus distance when the primary arrivals order.

    The returned speed is only a coarse envelope scale. Missing or unordered
    primary arrivals make the fit unavailable rather than raising.
    """
    required = tuple(int(value) for value in required_distances)
    centers: list[float] = []
    for distance in required:
        value = distance_to_center_fs.get(distance)
        if value is None or not np.isfinite(float(value)):
            return {
                "available": False,
                "ordered_primary_arrivals": False,
                "slope_fs_per_site": None,
                "speed_sites_per_ps": None,
                "r_squared": None,
            }
        centers.append(float(value))
    ordered = bool(all(b > a for a, b in zip(centers[:-1], centers[1:])))
    if not ordered:
        return {
            "available": False,
            "ordered_primary_arrivals": False,
            "slope_fs_per_site": None,
            "speed_sites_per_ps": None,
            "r_squared": None,
        }
    x = np.asarray(required, dtype=np.float64)
    y = np.asarray(centers, dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    if not np.isfinite(slope) or slope <= 0.0:
        return {
            "available": False,
            "ordered_primary_arrivals": True,
            "slope_fs_per_site": None,
            "speed_sites_per_ps": None,
            "r_squared": None,
        }
    fitted = slope * x + intercept
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if ss_tot == 0.0 else float(1.0 - ss_res / ss_tot)
    return {
        "available": True,
        "ordered_primary_arrivals": True,
        "slope_fs_per_site": float(slope),
        "speed_sites_per_ps": float(1000.0 / slope),
        "r_squared": r2,
    }
