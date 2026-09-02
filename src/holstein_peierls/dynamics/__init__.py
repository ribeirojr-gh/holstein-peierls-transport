"""Electronic dynamics building blocks.

The first production-independent checkpoint is D0a: deterministic propagation
under a frozen Hermitian Hamiltonian.  Classical lattice integration,
thermostats, fields, and nonlinear spin-adapted orbital dynamics remain outside
this package until their dedicated validation stages.
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

__all__ = [
    "HBAR_EV_FS",
    "CountingMatrixHamiltonian",
    "FrozenBenchmarkRecord",
    "FrozenBenchmarkSuite",
    "PropagationMetrics",
    "PropagationStep",
    "benchmark_frozen_hamiltonian",
    "cfm4_frozen_limit_step",
    "compare_to_reference",
    "crank_nicolson_step",
    "energy_expectation",
    "exact_spectral_step",
    "lanczos_exponential_step",
    "rk4_step",
    "rkf78_step",
]
