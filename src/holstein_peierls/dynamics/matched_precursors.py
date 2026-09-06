"""Matched counterfactual controls for thermally assisted polaron relocation.

IP1d compares every accepted nearest-neighbour electronic relocation with the
three alternative nearest-neighbour directions available at the same source and
same time.  These helpers are deliberately agnostic about rates, diffusion and
mobility; they only support mechanistic event-conditioned diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .hopping_observables import site_xy


DIRECTIONS = ("+x", "-x", "+y", "-y")
_DIRECTION_STEPS = {
    "+x": (1, 0),
    "-x": (-1, 0),
    "+y": (0, 1),
    "-y": (0, -1),
}


@dataclass(frozen=True, slots=True)
class DirectionalComparison:
    """True-direction value compared with the other three directions."""

    true_value: float
    counterfactual_mean: float
    advantage: float
    true_rank: int
    true_is_maximum: bool


def nearest_neighbor_site(
    source_site: int,
    direction: str,
    nx: int,
    ny: int,
) -> int:
    """Return the periodic nearest-neighbour site in ``direction``."""
    if direction not in _DIRECTION_STEPS:
        raise ValueError(f"unsupported nearest-neighbour direction: {direction}")
    if nx <= 1 or ny <= 1:
        raise ValueError("nx and ny must both exceed one")
    source = int(source_site)
    if not 0 <= source < nx * ny:
        raise ValueError("source_site is outside the lattice")
    x, y = site_xy(source, nx, ny)
    dx, dy = _DIRECTION_STEPS[direction]
    xx = (x + dx) % nx
    yy = (y + dy) % ny
    return int(xx + nx * yy)


def nearest_neighbor_targets(source_site: int, nx: int, ny: int) -> dict[str, int]:
    """Return all four periodic nearest-neighbour targets."""
    return {
        direction: nearest_neighbor_site(source_site, direction, nx, ny)
        for direction in DIRECTIONS
    }


def compare_true_direction(
    values: dict[str, float],
    true_direction: str,
) -> DirectionalComparison:
    """Compare one direction against the matched alternatives.

    Rank is one for the largest value.  Ties receive the best compatible rank,
    which is appropriate for a symmetry-preserving diagnostic; ``true_is_maximum``
    explicitly reports whether the true value equals the finite maximum within a
    tight floating-point tolerance.
    """
    if set(values) != set(DIRECTIONS):
        raise ValueError("values must contain exactly +x, -x, +y and -y")
    if true_direction not in DIRECTIONS:
        raise ValueError("true_direction must be a nearest-neighbour direction")
    data = {key: float(values[key]) for key in DIRECTIONS}
    if not all(np.isfinite(value) for value in data.values()):
        raise ValueError("directional values must be finite")
    true_value = data[true_direction]
    others = [data[key] for key in DIRECTIONS if key != true_direction]
    counterfactual_mean = float(np.mean(others))
    rank = 1 + sum(value > true_value for value in others)
    maximum = max(data.values())
    tolerance = 1.0e-12 * max(1.0, abs(maximum), abs(true_value))
    return DirectionalComparison(
        true_value=true_value,
        counterfactual_mean=counterfactual_mean,
        advantage=float(true_value - counterfactual_mean),
        true_rank=int(rank),
        true_is_maximum=bool(abs(true_value - maximum) <= tolerance),
    )


def negative_to_nonnegative_crossings(
    times_fs: NDArray[np.float64],
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return all linearly interpolated negative-to-nonnegative crossings."""
    times = np.asarray(times_fs, dtype=np.float64)
    data = np.asarray(values, dtype=np.float64)
    if times.ndim != 1 or data.shape != times.shape or times.size < 2:
        raise ValueError("times and values must be equal one-dimensional arrays of length >=2")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(data)):
        raise ValueError("times and values must be finite")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be strictly increasing")
    crossings: list[float] = []
    for index in range(1, times.size):
        left = float(data[index - 1])
        right = float(data[index])
        if left < 0.0 <= right:
            fraction = 0.0 if right == left else -left / (right - left)
            crossings.append(float(times[index - 1] + fraction * (times[index] - times[index - 1])))
    return np.asarray(crossings, dtype=np.float64)


def nearest_negative_to_nonnegative_crossing(
    times_fs: NDArray[np.float64],
    values: NDArray[np.float64],
    reference_time_fs: float = 0.0,
) -> float | None:
    """Return the crossing closest in time to ``reference_time_fs``."""
    crossings = negative_to_nonnegative_crossings(times_fs, values)
    if crossings.size == 0:
        return None
    reference = float(reference_time_fs)
    if not np.isfinite(reference):
        raise ValueError("reference_time_fs must be finite")
    index = int(np.argmin(np.abs(crossings - reference)))
    return float(crossings[index])
