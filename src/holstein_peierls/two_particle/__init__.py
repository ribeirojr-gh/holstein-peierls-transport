"""Experimental static two-particle solvers.

This subpackage contains research code that is intentionally isolated from the
validated single-polaron production path. The first implementation targets a
singlet Holstein-Hubbard bipolaron in the adiabatic limit.
"""

from .bipolaron import BipolaronGroundState, solve_bipolaron_ground_state
from .parameters import BipolaronParameters

__all__ = [
    "BipolaronGroundState",
    "BipolaronParameters",
    "solve_bipolaron_ground_state",
]
