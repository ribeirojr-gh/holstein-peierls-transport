"""Lattice geometry helpers for the two-dimensional periodic model."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class LatticeState:
    """Classical lattice degrees of freedom.

    Arrays have shape ``(ny, nx)``. This is equivalent to the legacy Fortran
    mapping ``k = j + nx * (i - 1)`` when flattened in C order.
    """

    u: FloatArray
    vx: FloatArray
    vy: FloatArray

    @classmethod
    def zeros(cls, ny: int, nx: int) -> "LatticeState":
        shape = (ny, nx)
        return cls(
            u=np.zeros(shape, dtype=np.float64),
            vx=np.zeros(shape, dtype=np.float64),
            vy=np.zeros(shape, dtype=np.float64),
        )

    def copy(self) -> "LatticeState":
        return LatticeState(self.u.copy(), self.vx.copy(), self.vy.copy())

    @property
    def shape(self) -> tuple[int, int]:
        return self.u.shape

    def validate(self) -> None:
        if self.u.shape != self.vx.shape or self.u.shape != self.vy.shape:
            raise ValueError("u, vx, and vy must have identical shapes")
        if self.u.ndim != 2:
            raise ValueError("lattice arrays must be two-dimensional")


def flatten_site_array(array: FloatArray) -> FloatArray:
    """Flatten a lattice array using the legacy site ordering."""
    return np.asarray(array, dtype=np.float64).reshape(-1, order="C")


def reshape_site_array(array: FloatArray, ny: int, nx: int) -> FloatArray:
    """Convert a legacy flat site vector into ``(ny, nx)`` representation."""
    return np.asarray(array, dtype=np.float64).reshape((ny, nx), order="C")
