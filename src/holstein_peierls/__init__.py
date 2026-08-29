"""Holstein-Peierls charge-transport simulation tools.

Version 0.1 implements the static two-dimensional polaron calculation used to
prepare the relaxed lattice/electronic state before a dynamical simulation.
"""

from .parameters import StaticPolaronParameters
from .polaron import PolaronResult, solve_static_polaron

__all__ = ["StaticPolaronParameters", "PolaronResult", "solve_static_polaron"]
__version__ = "0.1.0a1"
