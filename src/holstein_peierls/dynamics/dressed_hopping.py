"""Lattice-distortion tracking for dressed-polaron hopping diagnostics.

IP1a detected persistent changes of the dominant electronic population site.  A
true polaron hop should additionally translate the lattice deformation dressing
the charge.  This module provides a periodic matched filter for that deformation
and a persistence tracker for its inferred molecular center.

The template uses elastic-energy coordinates

    sqrt(K1) * u,
    sqrt(K2) * Delta_x vx,
    sqrt(K2) * Delta_y vy,

so uniform intermolecular displacement gauges drop out exactly and the three
features have a direct energetic interpretation.  Template matching is a
mechanistic diagnostic; it is not a reaction coordinate, free-energy surface,
or effective-mass calculation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from ..lattice import LatticeState
from ..parameters import StaticPolaronParameters
from .hopping_observables import (
    PersistentHopEvent,
    hop_direction_label,
    periodic_minimum_image_step,
    periodic_nearest_neighbor_step,
    site_xy,
)

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True, slots=True)
class ElasticDistortionFeatures:
    """Energy-weighted fields used by the periodic distortion matched filter."""

    u: FloatArray
    bond_x: FloatArray
    bond_y: FloatArray


@dataclass(frozen=True, slots=True)
class DistortionTemplateMatch:
    """Best periodic translation of a relaxed lattice-distortion template."""

    center_site: int
    shift_x_sites: int
    shift_y_sites: int
    amplitude: float
    second_amplitude: float
    relative_gap: float
    u_amplitude: float
    bond_x_amplitude: float
    bond_y_amplitude: float


@dataclass(frozen=True, slots=True)
class MatchedTransition:
    """Electronic/lattice transition pair with a signed lattice-minus-electronic lag."""

    electronic: PersistentHopEvent
    lattice: PersistentHopEvent
    lag_fs: float


def elastic_distortion_features(
    lattice: LatticeState,
    parameters: StaticPolaronParameters,
) -> ElasticDistortionFeatures:
    """Return gauge-free elastic-energy coordinates for one lattice state."""
    lattice.validate()
    if lattice.shape != (parameters.ny, parameters.nx):
        raise ValueError("lattice shape does not match parameters")
    du = np.asarray(lattice.u, dtype=np.float64)
    dx = np.asarray(np.roll(lattice.vx, -1, axis=1) - lattice.vx, dtype=np.float64)
    dy = np.asarray(np.roll(lattice.vy, -1, axis=0) - lattice.vy, dtype=np.float64)
    return ElasticDistortionFeatures(
        u=np.asarray(np.sqrt(parameters.k1) * du, dtype=np.float64),
        bond_x=np.asarray(np.sqrt(parameters.k2) * dx, dtype=np.float64),
        bond_y=np.asarray(np.sqrt(parameters.k2) * dy, dtype=np.float64),
    )


def _minimum_image_shift(index: int, length: int) -> int:
    value = int(index)
    if value > length // 2:
        value -= length
    return value


class PeriodicDistortionTemplateMatcher:
    """Locate the periodic translation that best matches a relaxed distortion.

    The cross-correlation is evaluated by FFT.  ``amplitude=1`` for an exact
    copy of the reference template (including any periodic translation), up to
    floating-point roundoff.  Thermal background can reduce or increase the raw
    amplitude, so the value is a diagnostic rather than a probability.
    """

    def __init__(
        self,
        reference_lattice: LatticeState,
        parameters: StaticPolaronParameters,
        reference_center_site: int,
    ) -> None:
        reference_lattice.validate()
        if reference_lattice.shape != (parameters.ny, parameters.nx):
            raise ValueError("reference lattice shape does not match parameters")
        center = int(reference_center_site)
        if not 0 <= center < parameters.n_sites:
            raise ValueError("reference_center_site is outside the lattice")
        self.parameters = parameters
        self.reference_center_site = center
        self.reference = elastic_distortion_features(reference_lattice, parameters)
        self._fft_u = np.fft.fft2(self.reference.u)
        self._fft_x = np.fft.fft2(self.reference.bond_x)
        self._fft_y = np.fft.fft2(self.reference.bond_y)
        self._norm_u2 = float(np.sum(self.reference.u * self.reference.u))
        self._norm_x2 = float(np.sum(self.reference.bond_x * self.reference.bond_x))
        self._norm_y2 = float(np.sum(self.reference.bond_y * self.reference.bond_y))
        self._norm2 = self._norm_u2 + self._norm_x2 + self._norm_y2
        if not np.isfinite(self._norm2) or self._norm2 <= 0.0:
            raise ValueError("reference lattice distortion has zero/invalid template norm")

    def match(self, lattice: LatticeState) -> DistortionTemplateMatch:
        """Return the best periodic template translation for ``lattice``."""
        features = elastic_distortion_features(lattice, self.parameters)
        spectrum = (
            np.fft.fft2(features.u) * np.conj(self._fft_u)
            + np.fft.fft2(features.bond_x) * np.conj(self._fft_x)
            + np.fft.fft2(features.bond_y) * np.conj(self._fft_y)
        )
        correlation = np.asarray(np.fft.ifft2(spectrum).real, dtype=np.float64)
        flat = correlation.reshape(-1)
        best_flat = int(np.argmax(flat))
        best_y, best_x = np.unravel_index(best_flat, correlation.shape)
        if flat.size > 1:
            second = float(np.partition(flat, -2)[-2])
        else:
            second = float(flat[best_flat])
        best = float(flat[best_flat])
        amplitude = best / self._norm2
        second_amplitude = second / self._norm2
        relative_gap = float((best - second) / max(abs(best), 1.0e-15))

        shift_x = _minimum_image_shift(int(best_x), self.parameters.nx)
        shift_y = _minimum_image_shift(int(best_y), self.parameters.ny)
        ref_x, ref_y = site_xy(
            self.reference_center_site,
            self.parameters.nx,
            self.parameters.ny,
        )
        center_x = (ref_x + int(best_x)) % self.parameters.nx
        center_y = (ref_y + int(best_y)) % self.parameters.ny
        center_site = int(center_x + self.parameters.nx * center_y)

        # Component amplitudes at the winning shift are inexpensive explicit
        # dot products and make the dressing contributions auditable.
        rolled_u = np.roll(self.reference.u, shift=(int(best_y), int(best_x)), axis=(0, 1))
        rolled_x = np.roll(self.reference.bond_x, shift=(int(best_y), int(best_x)), axis=(0, 1))
        rolled_y = np.roll(self.reference.bond_y, shift=(int(best_y), int(best_x)), axis=(0, 1))
        u_amp = float(np.sum(features.u * rolled_u) / self._norm2)
        x_amp = float(np.sum(features.bond_x * rolled_x) / self._norm2)
        y_amp = float(np.sum(features.bond_y * rolled_y) / self._norm2)
        return DistortionTemplateMatch(
            center_site=center_site,
            shift_x_sites=shift_x,
            shift_y_sites=shift_y,
            amplitude=float(amplitude),
            second_amplitude=float(second_amplitude),
            relative_gap=float(relative_gap),
            u_amplitude=u_amp,
            bond_x_amplitude=x_amp,
            bond_y_amplitude=y_amp,
        )


class PersistentTemplateTracker:
    """Persistence-filtered tracker for matched lattice-distortion centers."""

    def __init__(
        self,
        nx: int,
        ny: int,
        *,
        persistence_samples: int,
        minimum_amplitude: float = 0.05,
        minimum_relative_gap: float = 0.01,
    ) -> None:
        if nx < 3 or ny < 3:
            raise ValueError("template hopping diagnostics require nx, ny >= 3")
        if int(persistence_samples) <= 0:
            raise ValueError("persistence_samples must be positive")
        if not np.isfinite(minimum_amplitude) or minimum_amplitude < 0.0:
            raise ValueError("minimum_amplitude must be finite and non-negative")
        if not np.isfinite(minimum_relative_gap) or minimum_relative_gap < 0.0:
            raise ValueError("minimum_relative_gap must be finite and non-negative")
        self.nx = int(nx)
        self.ny = int(ny)
        self.persistence_samples = int(persistence_samples)
        self.minimum_amplitude = float(minimum_amplitude)
        self.minimum_relative_gap = float(minimum_relative_gap)
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
        match: DistortionTemplateMatch,
        time_fs: float,
    ) -> PersistentHopEvent | None:
        """Process one template-match snapshot and return an accepted transition."""
        time = float(time_fs)
        if not np.isfinite(time):
            raise ValueError("time_fs must be finite")
        confident = (
            np.isfinite(match.amplitude)
            and np.isfinite(match.relative_gap)
            and match.amplitude >= self.minimum_amplitude
            and match.relative_gap >= self.minimum_relative_gap
        )
        if not confident:
            self._clear_pending()
            return None
        dominant = int(match.center_site)
        if not 0 <= dominant < self.nx * self.ny:
            raise ValueError("matched center site is outside the lattice")
        if self.current_site is None:
            self.current_site = dominant
            return None
        if dominant == self.current_site:
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
        target = dominant
        dx, dy = periodic_minimum_image_step(source, target, self.nx, self.ny)
        nearest = periodic_nearest_neighbor_step(source, target, self.nx, self.ny)
        event = PersistentHopEvent(
            source_site=source,
            target_site=target,
            transition_start_time_fs=float(self._pending_start_time_fs),
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


def match_transition_events(
    electronic_events: Iterable[PersistentHopEvent],
    lattice_events: Iterable[PersistentHopEvent],
    *,
    maximum_abs_lag_fs: float,
) -> tuple[MatchedTransition, ...]:
    """Greedily pair same source->target transitions within a symmetric lag window.

    Each lattice event is used at most once.  The closest admissible lattice
    event in transition-start time is selected for each electronic event.
    """
    window = float(maximum_abs_lag_fs)
    if not np.isfinite(window) or window < 0.0:
        raise ValueError("maximum_abs_lag_fs must be finite and non-negative")
    lattice = list(lattice_events)
    used: set[int] = set()
    matched: list[MatchedTransition] = []
    for electronic in electronic_events:
        candidates: list[tuple[float, int]] = []
        for index, event in enumerate(lattice):
            if index in used:
                continue
            if (
                event.source_site != electronic.source_site
                or event.target_site != electronic.target_site
            ):
                continue
            lag = float(event.transition_start_time_fs - electronic.transition_start_time_fs)
            if abs(lag) <= window:
                candidates.append((abs(lag), index))
        if not candidates:
            continue
        _, chosen = min(candidates)
        used.add(chosen)
        lattice_event = lattice[chosen]
        matched.append(
            MatchedTransition(
                electronic=electronic,
                lattice=lattice_event,
                lag_fs=float(
                    lattice_event.transition_start_time_fs
                    - electronic.transition_start_time_fs
                ),
            )
        )
    return tuple(matched)


def prospective_bond_rank(values: dict[str, float], direction: str) -> int:
    """Return rank 1..4 of the prospective bond by transfer magnitude.

    Rank 1 means no competing local bond has a strictly larger magnitude.
    """
    required = ("+x", "-x", "+y", "-y")
    if direction not in required:
        raise ValueError("direction must be one of +x, -x, +y, -y")
    data = {key: float(values[key]) for key in required}
    if any(not np.isfinite(value) or value < 0.0 for value in data.values()):
        raise ValueError("transfer magnitudes must be finite and non-negative")
    selected = data[direction]
    return 1 + sum(value > selected + 1.0e-15 for key, value in data.items() if key != direction)
