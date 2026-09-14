"""Controlled single-relocation helpers for IP1g lattice-radiation diagnostics.

These helpers construct an intentionally non-equilibrium initial condition by
periodically translating the electronic wavefunction while leaving the relaxed
polaron lattice untouched.  The resulting state is a *charge-relocation
quench*.  It is not a natural hopping event and must not be used to infer a
hopping rate, mobility, activation energy, or calibrated material dissipation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

ComplexArray = NDArray[np.complex128]


def translate_electronic_state(
    electronic_state: ArrayLike,
    nx: int,
    ny: int,
    *,
    dx_sites: int = 0,
    dy_sites: int = 0,
) -> ComplexArray:
    """Periodically translate a flattened C-order electronic wavefunction.

    Positive ``dx_sites`` moves the charge pattern toward +x and positive
    ``dy_sites`` toward +y.  The operation is unitary/permutational and therefore
    preserves the wavefunction norm exactly up to floating-point roundoff.
    """
    if int(nx) <= 0 or int(ny) <= 0:
        raise ValueError("nx and ny must be positive")
    psi = np.asarray(electronic_state, dtype=np.complex128)
    if psi.ndim != 1 or psi.size != int(nx) * int(ny):
        raise ValueError("electronic_state size does not match nx*ny")
    if not np.all(np.isfinite(psi)):
        raise ValueError("electronic_state must contain only finite values")
    grid = psi.reshape((int(ny), int(nx)), order="C")
    shifted = np.roll(grid, shift=(int(dy_sites), int(dx_sites)), axis=(0, 1))
    return np.asarray(shifted.reshape(-1, order="C"), dtype=np.complex128).copy()
