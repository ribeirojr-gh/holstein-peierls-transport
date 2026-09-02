"""Spin-adapted static excited-state foundations.

This package is the S0 layer of the modernization roadmap.  It keeps the
validated distinguishable electron-hole solver intact while adding the spin
algebra and open-shell energy/Fock functionals required before a production
MCTDHF dynamics is introduced.
"""

from .excitation_reference import (
    ReferencedExcitationEnergy,
    ReferencedExcitationState,
    StaticReferencedExcitationResult,
    density_density_control_interaction,
    reference_shell_sizes,
    referenced_excitation_energy,
    referenced_excitation_gradient,
    relax_referenced_excitation,
    solve_referenced_excitation,
    spin_summed_rdm,
)
from .isotropic import IsotropicControlParameters
from .open_shell import (
    CLOSED_SHELL_SINGLET,
    HIGH_SPIN_TRIPLET,
    OPEN_SHELL_SINGLET,
    OpenShellStateDefinition,
    general_open_shell_energy,
    shell_fock_matrices,
)
from .orbital_optimization import (
    OpenShellOrbitalResult,
    OrbitalOptimizationDiagnostics,
    open_shell_orbital_energy,
    optimize_open_shell_orbitals,
    orbital_rotation_gradient,
    shell_projectors_from_complete_orbitals,
)
from .pair_control import (
    ExchangeControl,
    SpinAdaptedPairState,
    StaticSpinAdaptedPairResult,
    exchange_control_matrix,
    relax_spin_adapted_pair,
    singlet_triplet_gap,
    solve_spin_adapted_pair,
)
from .spin import SpinMultiplicity, minimal_open_shell_coefficients, s2_eigenvalue

__all__ = [
    "CLOSED_SHELL_SINGLET",
    "ExchangeControl",
    "HIGH_SPIN_TRIPLET",
    "OPEN_SHELL_SINGLET",
    "IsotropicControlParameters",
    "OpenShellOrbitalResult",
    "OpenShellStateDefinition",
    "OrbitalOptimizationDiagnostics",
    "ReferencedExcitationEnergy",
    "ReferencedExcitationState",
    "SpinAdaptedPairState",
    "SpinMultiplicity",
    "StaticReferencedExcitationResult",
    "StaticSpinAdaptedPairResult",
    "density_density_control_interaction",
    "exchange_control_matrix",
    "general_open_shell_energy",
    "minimal_open_shell_coefficients",
    "open_shell_orbital_energy",
    "optimize_open_shell_orbitals",
    "orbital_rotation_gradient",
    "reference_shell_sizes",
    "referenced_excitation_energy",
    "referenced_excitation_gradient",
    "relax_referenced_excitation",
    "relax_spin_adapted_pair",
    "s2_eigenvalue",
    "shell_fock_matrices",
    "shell_projectors_from_complete_orbitals",
    "singlet_triplet_gap",
    "solve_referenced_excitation",
    "solve_spin_adapted_pair",
    "spin_summed_rdm",
]
