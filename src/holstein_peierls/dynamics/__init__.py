"""Electronic dynamics building blocks.

D0a provides deterministic linear propagation under a frozen Hermitian
Hamiltonian. D0b adds the separate frozen-geometry nonlinear spin-adapted
projector benchmark. Classical lattice integration, thermostats, fields, and
coupled production dynamics remain outside this package until their dedicated
validation stages.
"""

from .benchmark import (
    FrozenBenchmarkRecord,
    FrozenBenchmarkSuite,
    benchmark_frozen_hamiltonian,
)
from .frozen import (
    HBAR_EV_FS,
    CountingMatrixHamiltonian,
    PropagationMetrics,
    PropagationStep,
    cfm4_frozen_limit_step,
    compare_to_reference,
    crank_nicolson_step,
    energy_expectation,
    exact_spectral_step,
    lanczos_exponential_step,
    rk4_step,
    rkf78_step,
)
from .spin_adapted import (
    ProjectorComparisonMetrics,
    ProjectorConstraintMetrics,
    ProjectorPropagationResult,
    compare_projector_states,
    integrate_dop853_projectors,
    integrate_predictor_exponential_midpoint,
    integrate_rk4_projectors,
    predictor_exponential_midpoint_step,
    projector_constraints,
    projector_distance,
    projector_energy,
    projector_rhs,
    rk4_projector_step,
    spin_summed_rdm,
    validate_projector_manifold,
    variational_generator,
)
from .spin_adapted_rkmk import (
    integrate_rkmk4_projectors,
    rkmk4_projector_step,
)

__all__ = [
    "HBAR_EV_FS",
    "CountingMatrixHamiltonian",
    "FrozenBenchmarkRecord",
    "FrozenBenchmarkSuite",
    "ProjectorComparisonMetrics",
    "ProjectorConstraintMetrics",
    "ProjectorPropagationResult",
    "PropagationMetrics",
    "PropagationStep",
    "benchmark_frozen_hamiltonian",
    "cfm4_frozen_limit_step",
    "compare_projector_states",
    "compare_to_reference",
    "crank_nicolson_step",
    "energy_expectation",
    "exact_spectral_step",
    "integrate_dop853_projectors",
    "integrate_predictor_exponential_midpoint",
    "integrate_rk4_projectors",
    "integrate_rkmk4_projectors",
    "lanczos_exponential_step",
    "predictor_exponential_midpoint_step",
    "projector_constraints",
    "projector_distance",
    "projector_energy",
    "projector_rhs",
    "rk4_projector_step",
    "rk4_step",
    "rkf78_step",
    "rkmk4_projector_step",
    "spin_summed_rdm",
    "validate_projector_manifold",
    "variational_generator",
]
