"""Experimental static two-particle solvers.

This subpackage contains research code that is intentionally isolated from the
validated single-polaron production path. It includes the singlet
Holstein-Hubbard reference solver and the subsequent static Holstein-Peierls
extension.
"""

from .bipolaron import BipolaronGroundState, solve_bipolaron_ground_state
from .observables import PairObservables, pair_observables
from .parameters import BipolaronParameters
from .peierls import (
    BipolaronLatticeGradient,
    HolsteinPeierlsBipolaronResult,
    relax_static_holstein_peierls_bipolaron,
    solve_holstein_peierls_ground_state,
)

__all__ = [
    "BipolaronGroundState",
    "BipolaronLatticeGradient",
    "BipolaronParameters",
    "HolsteinPeierlsBipolaronResult",
    "PairObservables",
    "pair_observables",
    "relax_static_holstein_peierls_bipolaron",
    "solve_bipolaron_ground_state",
    "solve_holstein_peierls_ground_state",
]
