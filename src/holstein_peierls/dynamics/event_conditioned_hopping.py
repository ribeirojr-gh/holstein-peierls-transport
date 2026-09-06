"""Event-conditioned charge/lattice translation diagnostics for IP1c.

IP1b compared two independently discretized residence-site trackers.  That is a
useful conservative test, but it can miss a continuous lattice reorganization
that does not generate the exact same discrete source->target template event.

This module instead evaluates, for any electronic nearest-neighbour event, how
strongly the instantaneous lattice distortion resembles the static polaron
translated to the event source and target sites.  The reference features are
those already validated in IP1b:

    sqrt(K1) * u,
    sqrt(K2) * Delta_x vx,
    sqrt(K2) * Delta_y vy.

No hopping rate, activation energy, diffusion coefficient, or mobility is
inferred here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..lattice import LatticeState
from ..parameters import StaticPolaronParameters
from .dressed_hopping import elastic_distortion_features
from .hopping_observables import site_xy

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class TemplateCorrelationMaps:
    """Normalized periodic template correlations for every lattice translation."""

    total: FloatArray
    u: FloatArray
    bond_x: FloatArray
    bond_y: FloatArray


@dataclass(frozen=True, slots=True)
class SourceTargetDressingCoordinate:
    """Source-vs-target template preference for one electronic hop pair."""

    source_amplitude: float
    target_amplitude: float
    coordinate: float
    u_difference: float
    bond_x_difference: float
    bond_y_difference: float


class PeriodicTemplateCorrelationField:
    """Evaluate the full periodic correlation field of a relaxed polaron template."""

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
        self._norm2 = float(
            np.sum(self.reference.u * self.reference.u)
            + np.sum(self.reference.bond_x * self.reference.bond_x)
            + np.sum(self.reference.bond_y * self.reference.bond_y)
        )
        if not np.isfinite(self._norm2) or self._norm2 <= 0.0:
            raise ValueError("reference lattice distortion has zero/invalid norm")

    def maps(self, lattice: LatticeState) -> TemplateCorrelationMaps:
        """Return component and total normalized periodic cross-correlations."""
        features = elastic_distortion_features(lattice, self.parameters)
        u = np.asarray(
            np.fft.ifft2(np.fft.fft2(features.u) * np.conj(self._fft_u)).real
            / self._norm2,
            dtype=np.float64,
        )
        bond_x = np.asarray(
            np.fft.ifft2(np.fft.fft2(features.bond_x) * np.conj(self._fft_x)).real
            / self._norm2,
            dtype=np.float64,
        )
        bond_y = np.asarray(
            np.fft.ifft2(np.fft.fft2(features.bond_y) * np.conj(self._fft_y)).real
            / self._norm2,
            dtype=np.float64,
        )
        return TemplateCorrelationMaps(
            total=np.asarray(u + bond_x + bond_y, dtype=np.float64),
            u=u,
            bond_x=bond_x,
            bond_y=bond_y,
        )

    def shift_index_for_center(self, center_site: int) -> tuple[int, int]:
        """Return correlation-map ``(y,x)`` index for a requested template center."""
        center = int(center_site)
        if not 0 <= center < self.parameters.n_sites:
            raise ValueError("center_site is outside the lattice")
        ref_x, ref_y = site_xy(
            self.reference_center_site,
            self.parameters.nx,
            self.parameters.ny,
        )
        x, y = site_xy(center, self.parameters.nx, self.parameters.ny)
        return (y - ref_y) % self.parameters.ny, (x - ref_x) % self.parameters.nx

    def amplitude_at_center(
        self,
        maps: TemplateCorrelationMaps,
        center_site: int,
    ) -> tuple[float, float, float, float]:
        """Return total, u, x-bond and y-bond amplitudes at one template center."""
        iy, ix = self.shift_index_for_center(center_site)
        return (
            float(maps.total[iy, ix]),
            float(maps.u[iy, ix]),
            float(maps.bond_x[iy, ix]),
            float(maps.bond_y[iy, ix]),
        )

    def source_target_coordinate(
        self,
        maps: TemplateCorrelationMaps,
        source_site: int,
        target_site: int,
    ) -> SourceTargetDressingCoordinate:
        """Return continuous lattice preference from source (-) toward target (+).

        ``coordinate = (A_target-A_source)/(|A_target|+|A_source|)`` and therefore
        lies in [-1,1] except for harmless floating-point roundoff.  Component
        differences retain the common full-template normalization and sum to the
        unnormalized total target-minus-source amplitude difference.
        """
        source = self.amplitude_at_center(maps, source_site)
        target = self.amplitude_at_center(maps, target_site)
        denominator = abs(source[0]) + abs(target[0])
        coordinate = 0.0 if denominator <= 1.0e-15 else (target[0] - source[0]) / denominator
        return SourceTargetDressingCoordinate(
            source_amplitude=float(source[0]),
            target_amplitude=float(target[0]),
            coordinate=float(np.clip(coordinate, -1.0, 1.0)),
            u_difference=float(target[1] - source[1]),
            bond_x_difference=float(target[2] - source[2]),
            bond_y_difference=float(target[3] - source[3]),
        )


def project_displacement_on_hop(
    displacement_x_A: float,
    displacement_y_A: float,
    dx_sites: int,
    dy_sites: int,
) -> tuple[float, float]:
    """Return displacement parallel and perpendicular to a nearest-neighbour hop.

    For a unit lattice step ``(dx,dy)``, positive parallel displacement is in the
    accepted electronic-hop direction.  The perpendicular convention is the
    right-handed in-plane rotation ``(-dy, dx)``.
    """
    dx = int(dx_sites)
    dy = int(dy_sites)
    if abs(dx) + abs(dy) != 1:
        raise ValueError("hop projection requires a nearest-neighbour unit step")
    x = float(displacement_x_A)
    y = float(displacement_y_A)
    if not np.isfinite(x) or not np.isfinite(y):
        raise ValueError("displacements must be finite")
    parallel = dx * x + dy * y
    perpendicular = -dy * x + dx * y
    return float(parallel), float(perpendicular)


def first_negative_to_nonnegative_crossing(
    times_fs: NDArray[np.float64],
    values: NDArray[np.float64],
) -> float | None:
    """Return linearly interpolated first negative-to-nonnegative crossing time."""
    times = np.asarray(times_fs, dtype=np.float64)
    data = np.asarray(values, dtype=np.float64)
    if times.ndim != 1 or data.shape != times.shape or times.size < 2:
        raise ValueError("times and values must be equal one-dimensional arrays of length >=2")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(data)):
        raise ValueError("times and values must be finite")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be strictly increasing")
    for index in range(1, times.size):
        left = float(data[index - 1])
        right = float(data[index])
        if left < 0.0 <= right:
            if right == left:
                return float(times[index])
            fraction = -left / (right - left)
            return float(times[index - 1] + fraction * (times[index] - times[index - 1]))
    return None
