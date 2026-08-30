"""Holstein-Peierls charge-transport simulation tools.

Version 0.3 retains the validated static two-dimensional one-polaron reference
and optimized sparse CPU paths while adding an experimental correlated
static two-particle subpackage for singlet Holstein-Hubbard and
Holstein-Peierls-Hubbard bipolarons. The one-polaron public API remains the
stable top-level interface; two-particle research functionality is available
under ``holstein_peierls.two_particle``.
"""

from .parameters import StaticPolaronParameters
from .polaron import PolaronResult, solve_static_polaron

__all__ = ["StaticPolaronParameters", "PolaronResult", "solve_static_polaron"]
__version__ = "0.3.0a1"
