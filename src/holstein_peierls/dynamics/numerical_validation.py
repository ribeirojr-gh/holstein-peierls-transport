"""Numerical validation helpers shared by finite-size dynamics screens.

The generalized energy-balance residual used throughout the dynamics code is an
extensive quantity.  A fixed absolute tolerance calibrated on a 20x20 lattice
must therefore not be reused unchanged for substantially larger cells.

This module preserves the validated 20x20 tolerance exactly and scales it
linearly with the number of lattice sites for finite-size comparisons.  The
scaling is a numerical acceptance rule only; it is not a physical error model.
"""

from __future__ import annotations

import math

REFERENCE_ENERGY_BALANCE_SITES = 20 * 20
REFERENCE_ENERGY_BALANCE_TOLERANCE_EV = 5.0e-5


def size_scaled_energy_balance_tolerance_eV(
    n_sites: int,
    *,
    reference_sites: int = REFERENCE_ENERGY_BALANCE_SITES,
    reference_tolerance_eV: float = REFERENCE_ENERGY_BALANCE_TOLERANCE_EV,
) -> float:
    """Return an extensive energy-balance tolerance scaled by lattice size.

    The default reproduces the established 20x20 gate exactly::

        400 sites  -> 5e-5 eV
        1600 sites -> 2e-4 eV

    No time scaling is applied.  The helper is intended for comparisons in
    which the underlying integrator, timestep and balance definition are
    otherwise unchanged.
    """
    sites = int(n_sites)
    ref_sites = int(reference_sites)
    ref_tol = float(reference_tolerance_eV)
    if sites <= 0:
        raise ValueError("n_sites must be positive")
    if ref_sites <= 0:
        raise ValueError("reference_sites must be positive")
    if not math.isfinite(ref_tol) or ref_tol <= 0.0:
        raise ValueError("reference_tolerance_eV must be finite and positive")
    return float(ref_tol * sites / ref_sites)


def extensive_energy_balance_passes(
    maximum_residual_eV: float,
    n_sites: int,
    *,
    reference_sites: int = REFERENCE_ENERGY_BALANCE_SITES,
    reference_tolerance_eV: float = REFERENCE_ENERGY_BALANCE_TOLERANCE_EV,
) -> bool:
    """Return whether an extensive residual satisfies the size-aware gate."""
    residual = float(maximum_residual_eV)
    if not math.isfinite(residual) or residual < 0.0:
        return False
    tolerance = size_scaled_energy_balance_tolerance_eV(
        n_sites,
        reference_sites=reference_sites,
        reference_tolerance_eV=reference_tolerance_eV,
    )
    return bool(residual < tolerance)
