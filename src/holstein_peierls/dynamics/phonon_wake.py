"""Carrier-centered lattice-energy and phonon-wake diagnostics for IP1f.

The routines in this module are passive observables. They do not modify the
Holstein-Peierls dynamics and they do not define a hopping rate, mobility, or a
material phonon lifetime.

Two ideas are kept separate:

1. a local lattice-energy density whose sum exactly reproduces the lattice
   potential plus kinetic energy used by the dynamics; and
2. the harmonic intermolecular energy current carried by the Peierls fields.

For a nearest-neighbour carrier relocation, periodic coordinates are projected
onto a signed longitudinal coordinate ``s`` along the electronic hop and a
transverse coordinate ``p``. Positive ``s`` is the carrier direction. Thus a
negative longitudinal energy-current component is retrograde with respect to
that event.

At finite temperature the absolute local energy contains a large thermal
background and the bound polaron deformation. IP1f therefore uses
pre-event-subtracted, event-conditioned profiles and excludes the immediate
polaron core when forming wake observables. Such quantities are diagnostics of
an emitted lattice wake; they are not a unique normal-mode decomposition into
free phonons.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..lattice import LatticeState
from ..parameters import StaticPolaronParameters
from .ehrenfest import LatticeVelocity, lattice_masses_fs
from .hopping_observables import periodic_nearest_neighbor_step, site_xy

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class LocalLatticeEnergyDensity:
    """Site/bond-assigned lattice energy densities in eV.

    ``u`` contains the local intramolecular potential plus kinetic energy.
    ``vx`` and ``vy`` contain the corresponding intermolecular kinetic energy
    plus the +x and +y strain-bond energies whose lower/left endpoint is the
    indexed site. Their sum reproduces the complete classical lattice energy.
    """

    u: FloatArray
    vx: FloatArray
    vy: FloatArray

    @property
    def total(self) -> FloatArray:
        return np.asarray(self.u + self.vx + self.vy, dtype=np.float64)

    @property
    def intermolecular(self) -> FloatArray:
        return np.asarray(self.vx + self.vy, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class IntermolecularEnergyFlux:
    """Harmonic Peierls-field energy current in eV/fs.

    ``jx[y,x]`` is the vx-chain energy current through the bond from ``(x,y)``
    to ``(x+1,y)``. ``jy[y,x]`` is the vy-chain current through the bond from
    ``(x,y)`` to ``(x,y+1)``. Positive values point in +x/+y respectively.
    """

    jx: FloatArray
    jy: FloatArray


@dataclass(frozen=True, slots=True)
class EventAlignedCoordinates:
    """Minimum-image coordinates relative to a nearest-neighbour hop source."""

    s_sites: NDArray[np.int64]
    p_sites: NDArray[np.int64]
    dx_sites: int
    dy_sites: int
    longitudinal_length: int


@dataclass(frozen=True, slots=True)
class WakeWindowMetrics:
    """Pre-subtracted front/back wake diagnostics for one time window."""

    backward_excess_eV: float
    forward_excess_eV: float
    signed_asymmetry: float
    backward_positive_excess_eV: float
    forward_positive_excess_eV: float
    backward_outward_flux_eV_per_fs: float
    forward_outward_flux_eV_per_fs: float
    flux_bias: float


def _validate_shape(
    lattice: LatticeState,
    velocity: LatticeVelocity,
    parameters: StaticPolaronParameters,
) -> None:
    lattice.validate()
    velocity.validate(lattice.shape)
    if lattice.shape != (parameters.ny, parameters.nx):
        raise ValueError("lattice shape does not match parameters")


def local_lattice_energy_density(
    lattice: LatticeState,
    velocity: LatticeVelocity,
    parameters: StaticPolaronParameters,
) -> LocalLatticeEnergyDensity:
    """Return a local decomposition of classical lattice energy.

    The convention assigns every +x/+y spring to its starting site. This makes
    the decomposition simple and exactly summable while retaining periodicity.
    """
    _validate_shape(lattice, velocity, parameters)
    mass_u, mass_v = lattice_masses_fs(parameters)
    dx = np.roll(lattice.vx, -1, axis=1) - lattice.vx
    dy = np.roll(lattice.vy, -1, axis=0) - lattice.vy
    e_u = 0.5 * mass_u * velocity.u**2 + 0.5 * parameters.k1 * lattice.u**2
    e_vx = 0.5 * mass_v * velocity.vx**2 + 0.5 * parameters.k2 * dx**2
    e_vy = 0.5 * mass_v * velocity.vy**2 + 0.5 * parameters.k2 * dy**2
    return LocalLatticeEnergyDensity(
        np.asarray(e_u, dtype=np.float64),
        np.asarray(e_vx, dtype=np.float64),
        np.asarray(e_vy, dtype=np.float64),
    )


def intermolecular_energy_flux(
    lattice: LatticeState,
    velocity: LatticeVelocity,
    parameters: StaticPolaronParameters,
) -> IntermolecularEnergyFlux:
    """Return the harmonic intermolecular energy-current field.

    For a nearest-neighbour harmonic chain with local symmetric bond partition,
    the bond current is

        j_{i->i+1} = -(K/2) (v_i + v_{i+1}) (q_{i+1} - q_i).

    The HP intermolecular Hamiltonian is the direct sum of vx chains along x
    and vy chains along y, so the formula applies component-wise.
    """
    _validate_shape(lattice, velocity, parameters)
    dx = np.roll(lattice.vx, -1, axis=1) - lattice.vx
    dy = np.roll(lattice.vy, -1, axis=0) - lattice.vy
    jx = -0.5 * parameters.k2 * (
        velocity.vx + np.roll(velocity.vx, -1, axis=1)
    ) * dx
    jy = -0.5 * parameters.k2 * (
        velocity.vy + np.roll(velocity.vy, -1, axis=0)
    ) * dy
    return IntermolecularEnergyFlux(
        np.asarray(jx, dtype=np.float64),
        np.asarray(jy, dtype=np.float64),
    )


def _canonical_periodic_coordinate(
    values: NDArray[np.integer] | int,
    length: int,
) -> NDArray[np.int64]:
    """Map periodic integer coordinates to one canonical minimum-image range.

    For even lengths the antipodal point has two equivalent signed
    representations.  We choose ``-L/2`` rather than ``+L/2`` so that a
    direction reversal cannot create an extra longitudinal profile bin.
    """
    if int(length) < 3:
        raise ValueError("periodic axis length must be at least three")
    array = np.asarray(values, dtype=np.int64)
    half = int(length) // 2
    return np.asarray(((array + half) % int(length)) - half, dtype=np.int64)


def _minimum_image_axis(length: int, origin: int) -> NDArray[np.int64]:
    values = np.arange(length, dtype=np.int64) - int(origin)
    return _canonical_periodic_coordinate(values, length)


def event_aligned_coordinates(
    source_site: int,
    target_site: int,
    nx: int,
    ny: int,
) -> EventAlignedCoordinates:
    """Return signed minimum-image coordinates with the hop pointing to +s.

    The post-rotation canonicalization is essential for even cells.  Without
    it, a -x or -y hop maps the unique antipodal ``-L/2`` site to ``+L/2``;
    ``np.bincount`` then creates ``L+1`` bins while the declared axis has only
    ``L`` entries.  Re-wrapping after the event rotation keeps all four hop
    directions on the same PBC-safe coordinate convention.
    """
    step = periodic_nearest_neighbor_step(source_site, target_site, nx, ny)
    if step is None:
        raise ValueError("event-aligned coordinates require a nearest-neighbour hop")
    dx_hop, dy_hop = step
    sx, sy = site_xy(source_site, nx, ny)
    dx_axis = _minimum_image_axis(nx, sx)
    dy_axis = _minimum_image_axis(ny, sy)
    dx_grid = np.broadcast_to(dx_axis[None, :], (ny, nx))
    dy_grid = np.broadcast_to(dy_axis[:, None], (ny, nx))

    raw_s = dx_grid * int(dx_hop) + dy_grid * int(dy_hop)
    raw_p = -dx_grid * int(dy_hop) + dy_grid * int(dx_hop)
    longitudinal_length = nx if dx_hop != 0 else ny
    transverse_length = ny if dx_hop != 0 else nx
    s = _canonical_periodic_coordinate(raw_s, longitudinal_length)
    p = _canonical_periodic_coordinate(raw_p, transverse_length)

    return EventAlignedCoordinates(
        np.asarray(s, dtype=np.int64),
        np.asarray(p, dtype=np.int64),
        int(dx_hop),
        int(dy_hop),
        int(longitudinal_length),
    )


def longitudinal_profile(
    values: NDArray[np.floating],
    coordinates: EventAlignedCoordinates,
    *,
    transverse_half_width_sites: int | None = None,
) -> tuple[NDArray[np.int64], FloatArray]:
    """Sum a scalar lattice field into signed longitudinal ``s`` bins."""
    array = np.asarray(values, dtype=np.float64)
    if array.shape != coordinates.s_sites.shape:
        raise ValueError("values shape does not match event coordinates")
    mask = np.ones(array.shape, dtype=bool)
    if transverse_half_width_sites is not None:
        width = int(transverse_half_width_sites)
        if width < 0:
            raise ValueError("transverse_half_width_sites must be non-negative")
        mask &= np.abs(coordinates.p_sites) <= width
    s_values = coordinates.s_sites[mask].reshape(-1)
    weights = array[mask].reshape(-1)
    s_min = -(coordinates.longitudinal_length // 2)
    s_axis = np.arange(
        s_min,
        s_min + coordinates.longitudinal_length,
        dtype=np.int64,
    )
    bins = s_values - s_min
    if np.any(bins < 0) or np.any(bins >= coordinates.longitudinal_length):
        raise ValueError("event-aligned longitudinal coordinate escaped canonical PBC range")
    profile = np.bincount(
        bins,
        weights=weights,
        minlength=coordinates.longitudinal_length,
    )
    if profile.size != coordinates.longitudinal_length:
        raise RuntimeError("longitudinal profile length does not match periodic axis")
    return s_axis, np.asarray(profile, dtype=np.float64)


def longitudinal_flux_field(
    flux: IntermolecularEnergyFlux,
    coordinates: EventAlignedCoordinates,
) -> FloatArray:
    """Project the lattice energy-current vector onto the carrier direction."""
    if flux.jx.shape != coordinates.s_sites.shape or flux.jy.shape != coordinates.s_sites.shape:
        raise ValueError("flux shape does not match event coordinates")
    return np.asarray(
        coordinates.dx_sites * flux.jx + coordinates.dy_sites * flux.jy,
        dtype=np.float64,
    )


def wake_window_metrics(
    excess_profile_eV: NDArray[np.floating],
    flux_profile_eV_per_fs: NDArray[np.floating],
    s_axis: NDArray[np.integer],
    *,
    core_half_width_sites: int = 1,
) -> WakeWindowMetrics:
    """Summarize one event-conditioned wake time window.

    ``excess_profile_eV`` and ``flux_profile_eV_per_fs`` are already averaged
    over the requested time window and pre-event-subtracted. The immediate
    polaron core ``|s| <= core_half_width_sites`` is excluded.
    """
    excess = np.asarray(excess_profile_eV, dtype=np.float64)
    flux = np.asarray(flux_profile_eV_per_fs, dtype=np.float64)
    s = np.asarray(s_axis, dtype=np.int64)
    if excess.shape != flux.shape or excess.shape != s.shape:
        raise ValueError("wake profiles and s_axis must have identical shapes")
    core = int(core_half_width_sites)
    if core < 0:
        raise ValueError("core_half_width_sites must be non-negative")
    back = s < -core
    front = s > core
    back_excess = float(np.sum(excess[back]))
    front_excess = float(np.sum(excess[front]))
    denominator = float(np.sum(np.abs(excess[back | front])))
    asymmetry = 0.0 if denominator <= 1.0e-20 else float(
        (back_excess - front_excess) / denominator
    )
    back_positive = float(np.sum(np.clip(excess[back], 0.0, None)))
    front_positive = float(np.sum(np.clip(excess[front], 0.0, None)))

    # Positive outward numbers mean energy is flowing away from the event core.
    backward_outward = float(-np.sum(flux[back]))
    forward_outward = float(np.sum(flux[front]))
    flux_denominator = abs(backward_outward) + abs(forward_outward)
    flux_bias = 0.0 if flux_denominator <= 1.0e-20 else float(
        (backward_outward - forward_outward) / flux_denominator
    )
    return WakeWindowMetrics(
        backward_excess_eV=back_excess,
        forward_excess_eV=front_excess,
        signed_asymmetry=asymmetry,
        backward_positive_excess_eV=back_positive,
        forward_positive_excess_eV=front_positive,
        backward_outward_flux_eV_per_fs=backward_outward,
        forward_outward_flux_eV_per_fs=forward_outward,
        flux_bias=flux_bias,
    )


def backward_excess_centroid_sites(
    excess_profile_eV: NDArray[np.floating],
    s_axis: NDArray[np.integer],
    *,
    core_half_width_sites: int = 1,
) -> float | None:
    """Return the centroid of positive excess energy behind the carrier."""
    excess = np.asarray(excess_profile_eV, dtype=np.float64)
    s = np.asarray(s_axis, dtype=np.float64)
    if excess.shape != s.shape:
        raise ValueError("excess_profile_eV and s_axis must have identical shapes")
    mask = s < -int(core_half_width_sites)
    weights = np.clip(excess[mask], 0.0, None)
    total = float(np.sum(weights))
    if total <= 1.0e-20:
        return None
    return float(np.sum(s[mask] * weights) / total)


def fit_group_velocity_sites_per_fs(
    times_fs: NDArray[np.floating],
    centroids_sites: NDArray[np.floating],
    *,
    minimum_points: int = 4,
) -> dict[str, float | int | None]:
    """Fit a linear backward-wake centroid velocity.

    Negative slope means propagation opposite to the event-aligned carrier
    direction. The fit is descriptive: a damped/noisy wake need not possess a
    well-defined quasiparticle group velocity.
    """
    times = np.asarray(times_fs, dtype=np.float64)
    centers = np.asarray(centroids_sites, dtype=np.float64)
    if times.shape != centers.shape:
        raise ValueError("times_fs and centroids_sites must have identical shapes")
    mask = np.isfinite(times) & np.isfinite(centers)
    times = times[mask]
    centers = centers[mask]
    required = int(minimum_points)
    if required < 2:
        raise ValueError("minimum_points must be at least two")
    if times.size < required:
        return {
            "n": int(times.size),
            "velocity_sites_per_fs": None,
            "velocity_sites_per_ps": None,
            "r_squared": None,
        }
    slope, intercept = np.polyfit(times, centers, 1)
    predicted = slope * times + intercept
    ss_res = float(np.sum((centers - predicted) ** 2))
    ss_tot = float(np.sum((centers - np.mean(centers)) ** 2))
    r2 = 1.0 if ss_tot <= 1.0e-30 else float(1.0 - ss_res / ss_tot)
    return {
        "n": int(times.size),
        "velocity_sites_per_fs": float(slope),
        "velocity_sites_per_ps": float(1000.0 * slope),
        "r_squared": r2,
    }
