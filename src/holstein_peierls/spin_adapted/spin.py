"""Spin algebra for the minimal open-shell singlet/triplet ansatz.

Miranda et al. (J. Chem. Phys. 134, 244101, 2011) write the lowest open-shell
singlet as an equal-weight sum of two determinants.  In the same determinant
ordering convention, the corresponding M_S=0 triplet uses the relative minus
sign.  This module keeps that algebra explicit instead of encoding spin labels
as metadata on a spin-blind electron-hole wavefunction.
"""

from __future__ import annotations

from enum import Enum
import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class SpinMultiplicity(str, Enum):
    """Spin sectors required by the first excited-state implementation."""

    SINGLET = "singlet"
    TRIPLET = "triplet"


def minimal_open_shell_coefficients(multiplicity: SpinMultiplicity) -> FloatArray:
    """Return normalized coefficients for the two-determinant M_S=0 ansatz.

    The determinant ordering follows the convention used in the supplied
    Miranda reference.  The singlet has equal signs; the M_S=0 triplet has
    opposite signs.  A high-spin M_S=1 triplet is represented by one determinant
    in actual open-shell HF calculations, but the two-component form is useful
    for testing spin adaptation and configuration-space projections.
    """
    scale = 1.0 / math.sqrt(2.0)
    if multiplicity is SpinMultiplicity.SINGLET:
        return np.asarray((scale, scale), dtype=np.float64)
    if multiplicity is SpinMultiplicity.TRIPLET:
        return np.asarray((scale, -scale), dtype=np.float64)
    raise ValueError(f"unsupported spin multiplicity: {multiplicity}")


def s2_eigenvalue(multiplicity: SpinMultiplicity) -> float:
    """Return S(S+1) in units of hbar^2 for the supported pure-spin states."""
    if multiplicity is SpinMultiplicity.SINGLET:
        return 0.0
    if multiplicity is SpinMultiplicity.TRIPLET:
        return 2.0
    raise ValueError(f"unsupported spin multiplicity: {multiplicity}")


def spin_exchange_sign(multiplicity: SpinMultiplicity) -> float:
    """Return the two-open-shell Coulomb-exchange energy sign.

    For two singly occupied spatial orbitals, the standard spin-adapted energy
    contains ``+K`` for the singlet and ``-K`` for the triplet, giving
    ``E_S - E_T = 2 K`` for positive exchange.  The production MCTDHF layer
    will obtain this through the shell-dependent exchange functional; this
    helper is also useful for reduced electron-hole control tests.
    """
    if multiplicity is SpinMultiplicity.SINGLET:
        return 1.0
    if multiplicity is SpinMultiplicity.TRIPLET:
        return -1.0
    raise ValueError(f"unsupported spin multiplicity: {multiplicity}")
