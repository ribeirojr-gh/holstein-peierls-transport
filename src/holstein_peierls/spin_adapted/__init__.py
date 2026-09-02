"""Spin-adapted static excited-state foundations.

This package is the S0 layer of the modernization roadmap.  It keeps the
validated distinguishable electron-hole solver intact while adding the spin
algebra and open-shell energy/Fock functionals required before a production
MCTDHF dynamics is introduced.
"""

from .isotropic import IsotropicControlParameters
from .open_shell import (
    CLOSED_SHELL_SINGLET,
    HIGH_SPIN_TRIPLET,
    OPEN_SHELL_SINGLET,
    OpenShellStateDefinition,
    general_open_shell_energy,
    shell_fock_matrices,
)
from .spin import SpinMultiplicity, minimal_open_shell_coefficients, s2_eigenvalue

__all__ = [
    "CLOSED_SHELL_SINGLET",
    "HIGH_SPIN_TRIPLET",
    "OPEN_SHELL_SINGLET",
    "IsotropicControlParameters",
    "OpenShellStateDefinition",
    "SpinMultiplicity",
    "general_open_shell_energy",
    "minimal_open_shell_coefficients",
    "s2_eigenvalue",
    "shell_fock_matrices",
]
