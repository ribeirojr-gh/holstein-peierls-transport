"""Periodic-boundary phonon recurrence diagnostics.

The intermolecular Holstein-Peierls lattice coordinates ``vx`` and ``vy`` obey,
in the harmonic limit and away from the carrier,

    M_v q_ddot_n = -K_2 (2 q_n - q_{n-1} - q_{n+1}).

For a periodic nearest-neighbour chain the undamped dispersion is

    omega(k) = 2 sqrt(K_2/M_v) |sin(k/2)|,

with dimensionless lattice wave number ``k`` in radians per site.  Therefore the
maximum undamped group velocity is ``sqrt(K_2/M_v)`` sites/fs.  These utilities
provide a conservative ballistic recurrence scale for a finite periodic cell
and classify the Langevin damping regime used in the thermal dynamics.

This module does not claim that the carrier is stationary, that every emitted
wave packet travels at the maximum group velocity, or that the Langevin bath is
material-calibrated.  The recurrence time is a finite-size diagnostic, not a
physical phonon lifetime.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..parameters import StaticPolaronParameters
from .ehrenfest import lattice_masses_fs


@dataclass(frozen=True, slots=True)
class IntermolecularRecurrenceScales:
    """Harmonic finite-cell scales for the intermolecular lattice modes."""

    mass_v_eV_fs2_per_A2: float
    spring_frequency_scale_per_fs: float
    omega_max_per_fs: float
    max_group_velocity_sites_per_fs: float
    max_group_velocity_A_per_fs: float
    wrap_time_x_fs: float
    wrap_time_y_fs: float
    earliest_stationary_carrier_wrap_fs: float
    gamma_v_per_fs: float
    damping_ratio_at_omega_max: float
    all_nonzero_harmonic_modes_overdamped: bool


def intermolecular_recurrence_scales(
    parameters: StaticPolaronParameters,
    *,
    lattice_spacing_A: float,
    gamma_v_per_fs: float,
) -> IntermolecularRecurrenceScales:
    """Return conservative harmonic PBC recurrence and damping scales.

    ``earliest_stationary_carrier_wrap_fs`` is the shortest full-cell traversal
    time of an undamped intermolecular packet evaluated at the maximum group
    velocity, assuming the carrier stays at the emission site.  A moving carrier
    can change an actual collision time, so this quantity must not be treated as
    a rigorous lower bound for carrier-wave recollision.
    """
    spacing = float(lattice_spacing_A)
    gamma = float(gamma_v_per_fs)
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("lattice_spacing_A must be finite and positive")
    if not math.isfinite(gamma) or gamma < 0.0:
        raise ValueError("gamma_v_per_fs must be finite and non-negative")
    if parameters.nx <= 0 or parameters.ny <= 0:
        raise ValueError("lattice dimensions must be positive")
    if not math.isfinite(parameters.k2) or parameters.k2 <= 0.0:
        raise ValueError("k2 must be finite and positive")

    _, mass_v = lattice_masses_fs(parameters)
    scale = math.sqrt(float(parameters.k2) / float(mass_v))
    omega_max = 2.0 * scale
    velocity_sites = scale
    velocity_A = spacing * velocity_sites
    wrap_x = float(parameters.nx) / velocity_sites
    wrap_y = float(parameters.ny) / velocity_sites
    earliest = min(wrap_x, wrap_y)
    damping_ratio = 0.0 if omega_max == 0.0 else gamma / (2.0 * omega_max)
    # For q_ddot + gamma q_dot + omega_0^2 q = 0, overdamping requires
    # gamma >= 2 omega_0.  Since omega_0 <= omega_max for every harmonic mode,
    # gamma >= 2 omega_max overdamps all non-zero modes.
    all_overdamped = bool(gamma >= 2.0 * omega_max)
    return IntermolecularRecurrenceScales(
        mass_v_eV_fs2_per_A2=float(mass_v),
        spring_frequency_scale_per_fs=float(scale),
        omega_max_per_fs=float(omega_max),
        max_group_velocity_sites_per_fs=float(velocity_sites),
        max_group_velocity_A_per_fs=float(velocity_A),
        wrap_time_x_fs=float(wrap_x),
        wrap_time_y_fs=float(wrap_y),
        earliest_stationary_carrier_wrap_fs=float(earliest),
        gamma_v_per_fs=gamma,
        damping_ratio_at_omega_max=float(damping_ratio),
        all_nonzero_harmonic_modes_overdamped=all_overdamped,
    )


def conservative_event_window_cutoff_fs(
    scales: IntermolecularRecurrenceScales,
    *,
    event_half_window_fs: float,
) -> float:
    """Return the latest event start whose analysis window ends before wrap.

    This is a stationary-carrier ballistic guard only.  It is useful for
    separating an early-time subset that is independent of the first full-cell
    traversal of an undamped packet emitted at t=0.  It does not prove absence
    of finite-size effects for a moving carrier.
    """
    half = float(event_half_window_fs)
    if not math.isfinite(half) or half < 0.0:
        raise ValueError("event_half_window_fs must be finite and non-negative")
    return float(scales.earliest_stationary_carrier_wrap_fs - half)


def event_window_precedes_stationary_wrap(
    transition_start_time_fs: float,
    scales: IntermolecularRecurrenceScales,
    *,
    event_half_window_fs: float,
) -> bool:
    """Return whether a full event window ends before the stationary wrap scale."""
    start = float(transition_start_time_fs)
    if not math.isfinite(start):
        raise ValueError("transition_start_time_fs must be finite")
    cutoff = conservative_event_window_cutoff_fs(
        scales,
        event_half_window_fs=event_half_window_fs,
    )
    return bool(start <= cutoff)
