"""Holstein-Peierls charge-transport simulation tools.

Version 0.2 provides the validated static two-dimensional polaron calculation
with both a strict legacy-reference path and an optimized sparse CPU path. The
relaxed lattice/electronic state is intended to prepare future dynamical
simulations.
"""

from .parameters import StaticPolaronParameters
from .polaron import PolaronResult, solve_static_polaron

__all__ = ["StaticPolaronParameters", "PolaronResult", "solve_static_polaron"]
__version__ = "0.2.0a1"
