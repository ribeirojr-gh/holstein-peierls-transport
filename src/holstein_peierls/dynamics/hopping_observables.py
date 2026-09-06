"""Persistent nearest-neighbour hopping diagnostics on periodic lattices.

The transport observables in TP1 integrate probability current and are the
correct PBC-safe way to measure continuous displacement.  The present module
answers a different question: whether a localized polaron undergoes a
*persistent change of molecular residence site*.

A change of the instantaneous maximum-population site is not automatically a
hop.  The tracker below requires a candidate site to remain dominant for a
caller-selected number of samples and to exceed a minimum population/dominance
margin.  This suppresses short breathing/flicker events near a shared-charge
configuration.

No rate, diffusion coefficient, mobility, or activation energy is inferred by
this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike

from ..hamiltonian import bond_transfer_integrals
from ..lattice import LatticeState
from ..parameters import StaticPolaronParameters


DirectionLabel = str


def site_xy(site: int, nx: int, ny: int) -> tuple[int, int]:
    """Return ``(x, y)`` for a C-order flattened molecular-site index."""
    index = int(site)
    if nx <= 0 or ny <= 0 or not 0 <= index < nx * ny:
        raise ValueError("site is outside the periodic lattice")
    return index % nx, index // nx


def periodic_minimum_image_step(
    source_site: int,
    target_site: int,
    nx: int,
    ny: int,
) -> tuple[int, int]:
    """Return the integer minimum-image site displacement target-source."""
    sx, sy = site_xy(source_site, nx, ny)
    tx, ty = site_xy(target_site, nx, ny)

    def wrap(delta: int, length: int) -> int:
        value = int(delta)
        half = length // 2
        if value > half:
            value -= length
        elif value < -half:
            value += length
        return value

    return wrap(tx - sx, nx), wrap(ty - sy, ny)


def periodic_nearest_neighbor_step(
    source_site: int,
    target_site: int,
    nx: int,
    ny: int,
) -> tuple[int, int] | None:
    """Return an oriented nearest-neighbour step, or ``None`` otherwise."""
    dx, dy = periodic_minimum_image_step(source_site, target_site, nx, ny)
    if abs(dx) + abs(dy) != 1:
        return None
    return dx, dy


def hop_direction_label(dx_sites: int, dy_sites: int) -> DirectionLabel:
    """Return ``+x``, ``-x``, ``+y``, ``-y`` or ``nonlocal``."""
    step = (int(dx_sites), int(dy_sites))
    labels = {(1, 0): "+x", (-1, 0): "-x", (0, 1): "+y", (0, -1): "-y"}
    return labels.get(step, "nonlocal")


def local_bond_transfer_magnitudes(
    lattice: LatticeState,
    parameters: StaticPolaronParameters,
    site: int,
) -> dict[str, float]:
    """Return ``|J|`` on the four nearest-neighbour bonds around one site.

    The returned quantities are instantaneous Hamiltonian transfer magnitudes in
    eV and therefore include the Peierls modulation from the current lattice.
    """
    lattice.validate()
    if lattice.shape != (parameters.ny, parameters.nx):
        raise ValueError("lattice shape does not match parameters")
    x, y = site_xy(site, parameters.nx, parameters.ny)
    tx, ty = bond_transfer_integrals(lattice, parameters)
    return {
        "+x": abs(float(tx[y, x])),
        "-x": abs(float(tx[y, (x - 1) % parameters.nx])),
        "+y": abs(float(ty[y, x])),
        "-y": abs(float(ty[(y - 1) % parameters.ny, x])),
    }


def local_transfer_anisotropy(values: Mapping[str, float]) -> float:
    """Return ``(Jmax-Jmin)/(Jmax+Jmin)`` for four local bond magnitudes."""
    required = ("+x", "-x", "+y", "-y")
    array = np.asarray([float(values[key]) for key in required], dtype=np.float64)
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError("transfer magnitudes must be finite and non-negative")
    maximum = float(np.max(array))
    minimum = float(np.min(array))
    denominator = maximum + minimum
    return 0.0 if denominator == 0.0 else float((maximum - minimum) / denominator)


def directional_transfer_bias(
    values: Mapping[str, float],
    direction: DirectionLabel,
) -> float:
    """Compare the prospective hop bond with the mean of the other three.

    ``(J_hop-J_other)/(J_hop+J_other)`` lies in [-1, 1].  Positive values mean
    that the prospective hop direction is instantaneously stronger than the
    other local bonds on average.
    """
    if direction not in ("+x", "-x", "+y", "-y"):
        raise ValueError("direction must be one of +x, -x, +y, -y")
    required = ("+x", "-x", "+y", "-y")
    array = {key: float(values[key]) for key in required}
    if any(not np.isfinite(value) or value < 0.0 for value in array.values()):
        raise ValueError("transfer magnitudes must be finite and non-negative")
    selected = array[direction]
    other = float(np.mean([value for key, value in array.items() if key != direction]))
    denominator = selected + other
    return 0.0 if denominator == 0.0 else float((selected - other) / denominator)


@dataclass(frozen=True, slots=True)
class PersistentHopEvent:
    """A persistent residence-site transition accepted by the tracker."""

    source_site: int
    target_site: int
    transition_start_time_fs: float
    accepted_time_fs: float
    dx_sites: int
    dy_sites: int
    is_nearest_neighbor: bool
    direction: DirectionLabel
    persistence_samples: int


class PersistentSiteTracker:
    """Stateful persistent-dominant-site hop detector.

    A new site is accepted only after it has been the unique-enough dominant
    molecular population for ``persistence_samples`` consecutive observations.
    ``minimum_dominance_margin`` is the population difference between the first
    and second largest molecular populations.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        *,
        persistence_samples: int,
        minimum_max_population: float = 0.10,
        minimum_dominance_margin: float = 0.02,
    ) -> None:
        if nx < 3 or ny < 3:
            raise ValueError("persistent hopping diagnostics require nx, ny >= 3")
        if int(persistence_samples) <= 0:
            raise ValueError("persistence_samples must be positive")
        for name, value in (
            ("minimum_max_population", minimum_max_population),
            ("minimum_dominance_margin", minimum_dominance_margin),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        self.nx = int(nx)
        self.ny = int(ny)
        self.persistence_samples = int(persistence_samples)
        self.minimum_max_population = float(minimum_max_population)
        self.minimum_dominance_margin = float(minimum_dominance_margin)
        self.current_site: int | None = None
        self._pending_site: int | None = None
        self._pending_count = 0
        self._pending_start_time_fs: float | None = None

    def _clear_pending(self) -> None:
        self._pending_site = None
        self._pending_count = 0
        self._pending_start_time_fs = None

    def update(
        self,
        populations: ArrayLike,
        time_fs: float,
    ) -> PersistentHopEvent | None:
        """Process one normalized molecular population snapshot."""
        values = np.asarray(populations, dtype=np.float64).reshape(-1)
        if values.size != self.nx * self.ny:
            raise ValueError("population size does not match lattice")
        if not np.all(np.isfinite(values)) or np.any(values < -1.0e-12):
            raise ValueError("populations must be finite and non-negative")
        time = float(time_fs)
        if not np.isfinite(time):
            raise ValueError("time_fs must be finite")

        order = np.argsort(values)
        dominant = int(order[-1])
        maximum = float(values[dominant])
        second = float(values[order[-2]]) if values.size > 1 else 0.0
        confident = (
            maximum >= self.minimum_max_population
            and maximum - second >= self.minimum_dominance_margin
        )

        if self.current_site is None:
            if confident:
                self.current_site = dominant
            return None

        if dominant == self.current_site:
            self._clear_pending()
            return None

        if not confident:
            self._clear_pending()
            return None

        if self._pending_site != dominant:
            self._pending_site = dominant
            self._pending_count = 1
            self._pending_start_time_fs = time
        else:
            self._pending_count += 1

        if self._pending_count < self.persistence_samples:
            return None

        source = int(self.current_site)
        target = int(dominant)
        start_time = float(self._pending_start_time_fs)
        minimum_step = periodic_minimum_image_step(
            source, target, self.nx, self.ny
        )
        nearest = periodic_nearest_neighbor_step(source, target, self.nx, self.ny)
        dx, dy = minimum_step
        event = PersistentHopEvent(
            source_site=source,
            target_site=target,
            transition_start_time_fs=start_time,
            accepted_time_fs=time,
            dx_sites=int(dx),
            dy_sites=int(dy),
            is_nearest_neighbor=nearest is not None,
            direction=hop_direction_label(dx, dy) if nearest is not None else "nonlocal",
            persistence_samples=self.persistence_samples,
        )
        self.current_site = target
        self._clear_pending()
        return event
