"""Spin-adapted static excited-state foundations and shared observables.

The package contains the stationary S0 layer and the integrator-independent O0
observables required before production MCTDHF dynamics is introduced.  The
validated distinguishable electron-hole solver remains intact while the
spin-adapted and multiconfigurational layers are developed alongside it.
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
from .observables import (
    channel_projector,
    channel_yield_from_density,
    channel_yield_from_state,
    configuration_channel_yield,
    instantaneous_occupation_numbers,
    occupation_numbers_from_propagated_orbitals,
    one_rdm_from_orbitals,
    slater_determinant_overlap,
)
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
from .relaxation_control import (
    IsotropicRelaxationSeed,
    SpinRelaxationBranchResult,
    half_filled_n_closed,
    harmonic_lattice_newton_direction,
    isotropic_relaxation_seed,
    isotropic_staggered_site_energies,
    relax_isotropic_spin_branch,
)
from .spin import SpinMultiplicity, minimal_open_shell_coefficients, s2_eigenvalue

__all__ = [
    "CLOSED_SHELL_SINGLET",
    "ExchangeControl",
    "HIGH_SPIN_TRIPLET",
    "OPEN_SHELL_SINGLET",
    "IsotropicControlParameters",
    "IsotropicRelaxationSeed",
    "OpenShellOrbitalResult",
    "OpenShellStateDefinition",
    "OrbitalOptimizationDiagnostics",
    "ReferencedExcitationEnergy",
    "ReferencedExcitationState",
    "SpinAdaptedPairState",
    "SpinMultiplicity",
    "SpinRelaxationBranchResult",
    "StaticReferencedExcitationResult",
    "StaticSpinAdaptedPairResult",
    "channel_projector",
    "channel_yield_from_density",
    "channel_yield_from_state",
    "configuration_channel_yield",
    "density_density_control_interaction",
    "exchange_control_matrix",
    "general_open_shell_energy",
    "half_filled_n_closed",
    "harmonic_lattice_newton_direction",
    "instantaneous_occupation_numbers",
    "isotropic_relaxation_seed",
    "isotropic_staggered_site_energies",
    "minimal_open_shell_coefficients",
    "occupation_numbers_from_propagated_orbitals",
    "one_rdm_from_orbitals",
    "open_shell_orbital_energy",
    "optimize_open_shell_orbitals",
    "orbital_rotation_gradient",
    "reference_shell_sizes",
    "referenced_excitation_energy",
    "referenced_excitation_gradient",
    "relax_isotropic_spin_branch",
    "relax_referenced_excitation",
    "relax_spin_adapted_pair",
    "s2_eigenvalue",
    "shell_fock_matrices",
    "shell_projectors_from_complete_orbitals",
    "singlet_triplet_gap",
    "slater_determinant_overlap",
    "solve_referenced_excitation",
    "solve_spin_adapted_pair",
    "spin_summed_rdm",
]
