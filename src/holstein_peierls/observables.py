"""Static-polaron observables and legacy output conventions."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .parameters import StaticPolaronParameters

FloatArray = NDArray[np.float64]


def polaron_formation_energy(
    total_energy: float, parameters: StaticPolaronParameters
) -> float:
    """Return the legacy positive energy gain relative to the rigid band state."""
    return 2.0 * (parameters.j0x + parameters.j0y) - total_energy


def inverse_participation_ratio(charge_density: FloatArray) -> float:
    """Conventional IPR for a normalized one-particle charge density."""
    density = np.asarray(charge_density, dtype=np.float64)
    norm = float(np.sum(density))
    if norm == 0.0:
        raise ValueError("charge density has zero norm")
    normalized = density / norm
    return float(np.sum(normalized * normalized))


def legacy_ipr(charge_density: FloatArray) -> float:
    """Reproduce the non-standard IPR formula printed by ``rprop.f90``."""
    q = np.asarray(charge_density, dtype=np.float64)
    denominator = float(np.sum(q * q)) ** 2
    if denominator == 0.0:
        raise ValueError("charge density has zero norm")
    return float(np.sum(q**4) / denominator)
