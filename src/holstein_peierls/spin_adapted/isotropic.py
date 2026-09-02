"""Canonical isotropic parameter set for the S0 implementation work.

The legacy/reference code uses the names ``j0x``, ``j0y``, ``alpha_intra`` and
``alpha_inter{x,y}``.  The research notation used for the new S0 work is
``J1``, ``J2``, ``alpha1`` and ``alpha2``.  The mapping adopted here is
explicit and intentionally narrow:

- ``J1 -> j0x`` and ``J2 -> j0y``;
- ``alpha1 -> alpha_intra`` (Holstein coupling);
- ``alpha2 -> alpha_interx = alpha_intery`` (Peierls coupling).

For the coding/validation phase the requested fully symmetric control is
``J1 = J2 = 0.100 eV`` and ``alpha1 = alpha2 = 3.0 eV/A``.  The elastic
constants are *not* changed by this simplification: ``K1`` and ``K2`` retain
the validated legacy values until a separate physical decision is made.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..exciton.parameters import ExcitonParameters
from ..parameters import StaticPolaronParameters


@dataclass(frozen=True, slots=True)
class IsotropicControlParameters:
    """Smallest parameter space used while implementing spin-adapted physics."""

    j1: float = 0.100
    j2: float = 0.100
    alpha1: float = 3.0
    alpha2: float = 3.0
    k1: float = 16.51
    k2: float = 0.51

    def __post_init__(self) -> None:
        if self.j1 < 0.0 or self.j2 < 0.0:
            raise ValueError("transfer-integral magnitudes must be non-negative")
        if self.alpha1 < 0.0 or self.alpha2 < 0.0:
            raise ValueError("electron-phonon couplings must be non-negative")
        if self.k1 <= 0.0 or self.k2 <= 0.0:
            raise ValueError("elastic constants must be positive")

    @property
    def is_isotropic(self) -> bool:
        """Return whether the deliberately reduced S0 control is symmetric."""
        return self.j1 == self.j2 and self.alpha1 == self.alpha2

    def to_polaron_parameters(
        self,
        *,
        nx: int = 20,
        ny: int = 20,
        polaron_position: int | None = None,
        **overrides: object,
    ) -> StaticPolaronParameters:
        """Map the S0 notation onto the validated one-carrier parameter class."""
        if polaron_position is None:
            polaron_position = (ny // 2) * nx + (nx // 2) + 1
        values: dict[str, object] = {
            "nx": nx,
            "ny": ny,
            "k1": self.k1,
            "k2": self.k2,
            "j0x": self.j1,
            "j0y": self.j2,
            "alpha_intra": self.alpha1,
            "alpha_interx": self.alpha2,
            "alpha_intery": self.alpha2,
            "polaron_position": polaron_position,
        }
        values.update(overrides)
        return StaticPolaronParameters(**values)

    def to_exciton_parameters(
        self,
        *,
        nx: int = 20,
        ny: int = 20,
        exciton_position: int | None = None,
        **overrides: object,
    ) -> ExcitonParameters:
        """Return an equal-electron/hole exciton control with isotropic lattice data."""
        if exciton_position is None:
            exciton_position = (ny // 2) * nx + (nx // 2) + 1
        values: dict[str, object] = {
            "nx": nx,
            "ny": ny,
            "k1": self.k1,
            "k2": self.k2,
            "electron_j0x": self.j1,
            "electron_j0y": self.j2,
            "hole_j0x": self.j1,
            "hole_j0y": self.j2,
            "electron_alpha_intra": self.alpha1,
            "electron_alpha_interx": self.alpha2,
            "electron_alpha_intery": self.alpha2,
            "hole_alpha_intra": self.alpha1,
            "hole_alpha_interx": self.alpha2,
            "hole_alpha_intery": self.alpha2,
            "exciton_position": exciton_position,
        }
        values.update(overrides)
        return ExcitonParameters(**values)
