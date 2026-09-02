"""Common benchmark harness for the deterministic D0a propagator comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import csr_matrix, issparse

from .frozen import (
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


@dataclass(frozen=True, slots=True)
class FrozenBenchmarkRecord:
    """Accuracy, work, and timing for one repeated frozen-H method."""

    method: str
    dt_fs: float
    steps: int
    total_time_fs: float
    elapsed_seconds: float
    seconds_per_step: float
    hamiltonian_applications: int
    eigendecompositions: int
    linear_solves: int
    approximate_workspace_bytes: int
    maximum_embedded_error_estimate: float | None
    maximum_krylov_dimension_used: int | None
    metrics: PropagationMetrics

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        return result


@dataclass(frozen=True, slots=True)
class FrozenBenchmarkSuite:
    """Complete D0a benchmark including untimed-reference metadata."""

    dimension: int
    dt_fs: float
    steps: int
    total_time_fs: float
    initial_energy_ev: float
    reference_elapsed_seconds: float
    records: tuple[FrozenBenchmarkRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "dt_fs": self.dt_fs,
            "steps": self.steps,
            "total_time_fs": self.total_time_fs,
            "initial_energy_ev": self.initial_energy_ev,
            "reference_elapsed_seconds": self.reference_elapsed_seconds,
            "records": [record.to_dict() for record in self.records],
        }


def _validate_benchmark_inputs(
    dense_hamiltonian: ArrayLike,
    action_hamiltonian: Any,
    initial_state: ArrayLike,
    dt_fs: float,
    steps: int,
) -> tuple[np.ndarray, csr_matrix, np.ndarray]:
    dense = np.asarray(dense_hamiltonian, dtype=np.complex128)
    if dense.ndim != 2 or dense.shape[0] != dense.shape[1]:
        raise ValueError("dense_hamiltonian must be square")
    if not np.allclose(dense, dense.conj().T, rtol=0.0, atol=1.0e-12):
        raise ValueError("dense_hamiltonian must be Hermitian")
    sparse = csr_matrix(action_hamiltonian, dtype=np.complex128)
    if sparse.shape != dense.shape:
        raise ValueError("action_hamiltonian shape must match dense_hamiltonian")
    if not np.allclose(sparse.toarray(), dense, rtol=0.0, atol=1.0e-13):
        raise ValueError("dense and action Hamiltonians must represent the same operator")
    state = np.asarray(initial_state, dtype=np.complex128)
    if state.ndim != 1 or state.size != dense.shape[0]:
        raise ValueError("initial_state dimension must match Hamiltonian")
    norm = float(np.linalg.norm(state))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("initial_state must have a finite non-zero norm")
    if not np.isfinite(dt_fs) or dt_fs <= 0.0:
        raise ValueError("dt_fs must be finite and positive")
    if steps <= 0:
        raise ValueError("steps must be positive")
    return dense, sparse, state


def _run_method(
    name: str,
    initial_state: np.ndarray,
    reference_state: np.ndarray,
    dense_hamiltonian: np.ndarray,
    dt_fs: float,
    steps: int,
    stepper: Callable[[np.ndarray], PropagationStep],
) -> FrozenBenchmarkRecord:
    state = np.asarray(initial_state, dtype=np.complex128).copy()
    hamiltonian_applications = 0
    eigendecompositions = 0
    linear_solves = 0
    workspace = 0
    embedded_errors: list[float] = []
    krylov_dimensions: list[int] = []

    start = perf_counter()
    for _ in range(steps):
        result = stepper(state)
        state = result.state
        hamiltonian_applications += result.hamiltonian_applications
        eigendecompositions += result.eigendecompositions
        linear_solves += result.linear_solves
        workspace = max(workspace, result.approximate_workspace_bytes)
        if result.embedded_error_estimate is not None:
            embedded_errors.append(result.embedded_error_estimate)
        if result.krylov_dimension_used is not None:
            krylov_dimensions.append(result.krylov_dimension_used)
    elapsed = perf_counter() - start

    metrics = compare_to_reference(reference_state, state, dense_hamiltonian)
    return FrozenBenchmarkRecord(
        method=name,
        dt_fs=float(dt_fs),
        steps=int(steps),
        total_time_fs=float(dt_fs * steps),
        elapsed_seconds=float(elapsed),
        seconds_per_step=float(elapsed / steps),
        hamiltonian_applications=hamiltonian_applications,
        eigendecompositions=eigendecompositions,
        linear_solves=linear_solves,
        approximate_workspace_bytes=int(workspace),
        maximum_embedded_error_estimate=(
            max(embedded_errors) if embedded_errors else None
        ),
        maximum_krylov_dimension_used=(
            max(krylov_dimensions) if krylov_dimensions else None
        ),
        metrics=metrics,
    )


def benchmark_frozen_hamiltonian(
    dense_hamiltonian: ArrayLike,
    action_hamiltonian: Any,
    initial_state: ArrayLike,
    *,
    dt_fs: float = 0.1,
    steps: int = 100,
    krylov_dimension: int = 12,
) -> FrozenBenchmarkSuite:
    """Run the common D0a comparison without renormalizing any candidate state.

    ``dense_hamiltonian`` is used for the legacy spectral reference and final
    diagnostics.  ``action_hamiltonian`` is the numerically identical sparse
    matrix used by RK/Krylov and Crank-Nicolson candidates, preventing the
    benchmark from hiding dense ``H @ psi`` work inside a nominally matrix-free
    propagator.
    """
    if krylov_dimension <= 0:
        raise ValueError("krylov_dimension must be positive")
    dense, sparse, state = _validate_benchmark_inputs(
        dense_hamiltonian,
        action_hamiltonian,
        initial_state,
        dt_fs,
        steps,
    )
    total_time = float(dt_fs * steps)

    reference_start = perf_counter()
    reference_state = exact_spectral_step(dense, state, total_time).state
    reference_elapsed = perf_counter() - reference_start
    initial_energy = energy_expectation(dense, state)

    spectral_record = _run_method(
        "legacy_spectral",
        state,
        reference_state,
        dense,
        dt_fs,
        steps,
        lambda psi: exact_spectral_step(dense, psi, dt_fs),
    )

    rk4_action = CountingMatrixHamiltonian(sparse)
    rk4_record = _run_method(
        "rk4",
        state,
        reference_state,
        dense,
        dt_fs,
        steps,
        lambda psi: rk4_step(rk4_action, psi, dt_fs),
    )

    rkf_action = CountingMatrixHamiltonian(sparse)
    rkf_record = _run_method(
        "rkf78_order8",
        state,
        reference_state,
        dense,
        dt_fs,
        steps,
        lambda psi: rkf78_step(rkf_action, psi, dt_fs),
    )

    lanczos_action = CountingMatrixHamiltonian(sparse)
    lanczos_record = _run_method(
        f"lanczos_m{krylov_dimension}",
        state,
        reference_state,
        dense,
        dt_fs,
        steps,
        lambda psi: lanczos_exponential_step(
            lanczos_action,
            psi,
            dt_fs,
            krylov_dimension=krylov_dimension,
        ),
    )

    cfm_action = CountingMatrixHamiltonian(sparse)
    cfm_record = _run_method(
        f"cfm4_frozen_m{krylov_dimension}",
        state,
        reference_state,
        dense,
        dt_fs,
        steps,
        lambda psi: cfm4_frozen_limit_step(
            cfm_action,
            psi,
            dt_fs,
            krylov_dimension=krylov_dimension,
        ),
    )

    crank_record = _run_method(
        "crank_nicolson",
        state,
        reference_state,
        dense,
        dt_fs,
        steps,
        lambda psi: crank_nicolson_step(sparse, psi, dt_fs),
    )

    return FrozenBenchmarkSuite(
        dimension=dense.shape[0],
        dt_fs=float(dt_fs),
        steps=int(steps),
        total_time_fs=total_time,
        initial_energy_ev=initial_energy,
        reference_elapsed_seconds=float(reference_elapsed),
        records=(
            spectral_record,
            rk4_record,
            rkf_record,
            lanczos_record,
            cfm_record,
            crank_record,
        ),
    )
