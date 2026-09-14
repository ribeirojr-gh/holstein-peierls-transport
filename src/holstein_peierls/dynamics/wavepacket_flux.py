"""Boundary-flux diagnostics for carrier-aligned lattice-radiation profiles.

These routines operate on already projected longitudinal harmonic energy-current
profiles.  They are intentionally separate from the older half-space summed
current diagnostic: summing current over an extended region is not a physical
flux through a boundary.  Here the outward flux is evaluated at a specific
longitudinal bond and propagation delays are estimated by correlation between
successive boundaries.

The resulting speed is a flux-packet propagation speed.  It is not, by itself,
a normal-mode-resolved material phonon group velocity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class FluxDelayEstimate:
    """Correlation delay between two neighboring outward-flux boundaries."""

    lag_fs: float
    speed_sites_per_ps: float
    correlation: float


def _uniform_dt(times_fs: ArrayLike) -> tuple[FloatArray, float]:
    times = np.asarray(times_fs, dtype=np.float64).reshape(-1)
    if times.size < 3 or not np.all(np.isfinite(times)):
        raise ValueError("times_fs must contain at least three finite samples")
    delta = np.diff(times)
    if np.any(delta <= 0.0):
        raise ValueError("times_fs must be strictly increasing")
    dt = float(np.mean(delta))
    if not np.allclose(delta, dt, rtol=0.0, atol=max(1.0e-10, 1.0e-10 * abs(dt))):
        raise ValueError("times_fs must be uniformly sampled")
    return times, dt


def outward_boundary_flux_series(
    longitudinal_flux_eV_per_fs: ArrayLike,
    s_axis: ArrayLike,
    distance_sites: int,
    *,
    side: str,
) -> FloatArray:
    """Return outward current through one carrier-centered longitudinal boundary.

    ``longitudinal_flux_eV_per_fs`` must have shape ``(time, s)`` and use the
    convention that positive current points toward +s.  For the forward side,
    outward current is therefore ``+j(s=+d)``.  For the backward side it is
    ``-j(s=-d)``.
    """
    flux = np.asarray(longitudinal_flux_eV_per_fs, dtype=np.float64)
    s = np.asarray(s_axis, dtype=np.int64).reshape(-1)
    if flux.ndim != 2 or flux.shape[1] != s.size:
        raise ValueError("flux must have shape (time, len(s_axis))")
    if not np.all(np.isfinite(flux)):
        raise ValueError("flux must contain only finite values")
    distance = int(distance_sites)
    if distance <= 0:
        raise ValueError("distance_sites must be positive")
    if side not in ("backward", "forward"):
        raise ValueError("side must be 'backward' or 'forward'")
    target = -distance if side == "backward" else distance
    matches = np.flatnonzero(s == target)
    if matches.size != 1:
        raise ValueError(f"s_axis does not contain unique boundary coordinate {target}")
    series = np.asarray(flux[:, int(matches[0])], dtype=np.float64)
    return -series if side == "backward" else series


def integrated_outward_energy_eV(
    times_fs: ArrayLike,
    outward_flux_eV_per_fs: ArrayLike,
    *,
    positive_only: bool = True,
) -> float:
    """Integrate outward energy crossing one boundary.

    With ``positive_only=True`` inward episodes are clipped to zero.  With it
    disabled the result is the signed net transported energy through the chosen
    boundary.
    """
    times, _ = _uniform_dt(times_fs)
    flux = np.asarray(outward_flux_eV_per_fs, dtype=np.float64).reshape(-1)
    if flux.shape != times.shape or not np.all(np.isfinite(flux)):
        raise ValueError("outward flux must be finite and match times_fs")
    values = np.clip(flux, 0.0, None) if positive_only else flux
    return float(np.trapezoid(values, times))


def normalized_flux_delay(
    times_fs: ArrayLike,
    upstream_outward_flux: ArrayLike,
    downstream_outward_flux: ArrayLike,
    *,
    lag_min_fs: float,
    lag_max_fs: float,
) -> FluxDelayEstimate:
    """Estimate packet delay from normalized overlap correlation.

    The downstream boundary is one lattice site farther from the source.  A
    positive fitted lag therefore maps to ``speed = 1 site / lag``.  Every lag
    uses only the overlapping samples and each overlap is demeaned before the
    normalized dot product is evaluated.
    """
    times, dt = _uniform_dt(times_fs)
    a = np.asarray(upstream_outward_flux, dtype=np.float64).reshape(-1)
    b = np.asarray(downstream_outward_flux, dtype=np.float64).reshape(-1)
    if a.shape != times.shape or b.shape != times.shape:
        raise ValueError("flux series must match times_fs")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("flux series must be finite")
    lag_min = float(lag_min_fs)
    lag_max = float(lag_max_fs)
    if not np.isfinite(lag_min) or not np.isfinite(lag_max) or lag_min <= 0.0 or lag_max < lag_min:
        raise ValueError("lag bounds must be finite, positive and ordered")
    k_min = max(1, int(np.ceil(lag_min / dt)))
    k_max = min(times.size - 2, int(np.floor(lag_max / dt)))
    if k_max < k_min:
        raise ValueError("lag interval contains no sampled lag")

    best_corr = -np.inf
    best_k = k_min
    for k in range(k_min, k_max + 1):
        x = np.asarray(a[:-k], dtype=np.float64)
        y = np.asarray(b[k:], dtype=np.float64)
        x = x - float(np.mean(x))
        y = y - float(np.mean(y))
        denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
        corr = -np.inf if denominator <= 1.0e-30 else float(np.dot(x, y) / denominator)
        if corr > best_corr:
            best_corr = corr
            best_k = k

    lag = float(best_k * dt)
    return FluxDelayEstimate(
        lag_fs=lag,
        speed_sites_per_ps=float(1000.0 / lag),
        correlation=float(best_corr),
    )


def boundary_chain_summary(
    times_fs: ArrayLike,
    longitudinal_flux_eV_per_fs: ArrayLike,
    s_axis: ArrayLike,
    *,
    side: str,
    distances_sites: tuple[int, ...] = (2, 3, 4, 5, 6),
    lag_min_fs: float = 300.0,
    lag_max_fs: float = 1000.0,
) -> dict:
    """Summarize boundary energies and one-site propagation delays."""
    distances = tuple(int(value) for value in distances_sites)
    if len(distances) < 2 or any(value <= 0 for value in distances):
        raise ValueError("at least two positive boundary distances are required")
    if any(b - a != 1 for a, b in zip(distances[:-1], distances[1:])):
        raise ValueError("boundary distances must be consecutive lattice sites")

    series = {
        distance: outward_boundary_flux_series(
            longitudinal_flux_eV_per_fs,
            s_axis,
            distance,
            side=side,
        )
        for distance in distances
    }
    boundaries = []
    for distance in distances:
        values = series[distance]
        boundaries.append(
            {
                "distance_sites": distance,
                "positive_outward_energy_eV": integrated_outward_energy_eV(
                    times_fs, values, positive_only=True
                ),
                "net_outward_energy_eV": integrated_outward_energy_eV(
                    times_fs, values, positive_only=False
                ),
                "peak_outward_flux_eV_per_fs": float(np.max(values)),
            }
        )

    delays = []
    for first, second in zip(distances[:-1], distances[1:]):
        estimate = normalized_flux_delay(
            times_fs,
            series[first],
            series[second],
            lag_min_fs=lag_min_fs,
            lag_max_fs=lag_max_fs,
        )
        delays.append(
            {
                "from_distance_sites": first,
                "to_distance_sites": second,
                "lag_fs": estimate.lag_fs,
                "speed_sites_per_ps": estimate.speed_sites_per_ps,
                "correlation": estimate.correlation,
            }
        )

    return {
        "side": side,
        "boundaries": boundaries,
        "delays": delays,
        "median_speed_sites_per_ps": float(
            np.median([record["speed_sites_per_ps"] for record in delays])
        ),
        "minimum_correlation": float(min(record["correlation"] for record in delays)),
    }
