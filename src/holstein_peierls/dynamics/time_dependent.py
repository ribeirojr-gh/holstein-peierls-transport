"""Linear explicitly time-dependent electronic propagators for D1.

D1 keeps the classical lattice frozen while the Peierls electric-field phase
makes the one-particle Hamiltonian explicitly time dependent,

    d psi / dt = -(i / hbar) H(t) psi.

The high-accuracy reference is adaptive DOP853.  Fixed-step RK4 remains a
transparent baseline.  The main exponential candidate is the standard
symmetric two-exponential fourth-order commutator-free Magnus method evaluated
at the two Gauss-Legendre nodes, with each exponential action evaluated by the
validated D0a Lanczos kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

from .frozen import (
    HBAR_EV_FS,
    CountingMatrixHamiltonian,
    lanczos_exponential_step,
)

ComplexArray = NDArray[np.complex128]
HamiltonianFunction = Callable[[float], Any]


@dataclass(frozen=True, slots=True)
class TimeDependentPropagationResult:
    """Final state and work counters for a D1 propagation."""

    state: ComplexArray
    hamiltonian_evaluations: int
    hamiltonian_applications: int
    elapsed_seconds: float
    accepted_steps: int | None = None
    rhs_evaluations: int | None = None
    success: bool = True
    message: str = ""


@dataclass(frozen=True, slots=True)
class TimeDependentComparisonMetrics:
    """State-accuracy and norm metrics against a tightened D1 reference."""

    norm_error: float
    raw_state_error: float
    phase_aligned_state_error: float
    fidelity: float


def _state_vector(values: ArrayLike) -> ComplexArray:
    state = np.asarray(values, dtype=np.complex128)
    if state.ndim != 1 or state.size == 0:
        raise ValueError("state must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(state)):
        raise ValueError("state must contain only finite values")
    if float(np.linalg.norm(state)) == 0.0:
        raise ValueError("state norm must be non-zero")
    return state.copy()


def _matrix_at(
    hamiltonian_at: HamiltonianFunction,
    time_fs: float,
    *,
    dimension: int,
) -> Any:
    matrix = hamiltonian_at(float(time_fs))
    if not hasattr(matrix, "shape") or matrix.shape != (dimension, dimension):
        raise ValueError("time-dependent Hamiltonian has the wrong shape")
    # Dense conversion is used only for the inexpensive Hermiticity gate.  D1
    # benchmark cells are deliberately small; production sparse actions can
    # later replace this validation with a cheaper structural guarantee.
    dense = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    dense = np.asarray(dense, dtype=np.complex128)
    if not np.all(np.isfinite(dense)):
        raise ValueError("Hamiltonian must contain only finite values")
    if not np.allclose(dense, dense.conj().T, rtol=0.0, atol=1.0e-12):
        raise ValueError("time-dependent Hamiltonian must be Hermitian")
    return matrix


def compare_time_dependent_states(
    reference_state: ArrayLike,
    candidate_state: ArrayLike,
) -> TimeDependentComparisonMetrics:
    """Compare two propagated states modulo an arbitrary global phase."""
    reference = _state_vector(reference_state)
    candidate = _state_vector(candidate_state)
    if candidate.shape != reference.shape:
        raise ValueError("reference and candidate states must have equal shape")
    reference_norm = float(np.linalg.norm(reference))
    candidate_norm = float(np.linalg.norm(candidate))
    overlap = np.vdot(reference, candidate)
    normalized_overlap = overlap / (reference_norm * candidate_norm)
    fidelity = float(np.clip(abs(normalized_overlap) ** 2, 0.0, 1.0))
    if abs(overlap) > 0.0:
        phase = np.conj(overlap) / abs(overlap)
        aligned = candidate * phase
    else:
        aligned = candidate
    return TimeDependentComparisonMetrics(
        norm_error=abs(candidate_norm - reference_norm) / reference_norm,
        raw_state_error=float(np.linalg.norm(candidate - reference) / reference_norm),
        phase_aligned_state_error=float(
            np.linalg.norm(aligned - reference) / reference_norm
        ),
        fidelity=fidelity,
    )


def rk4_time_dependent_step(
    hamiltonian_at: HamiltonianFunction,
    state: ArrayLike,
    time_fs: float,
    dt_fs: float,
) -> TimeDependentPropagationResult:
    """One classical RK4 step with stage-consistent Hamiltonian times."""
    psi = _state_vector(state)
    time = float(time_fs)
    dt = float(dt_fs)
    if not np.isfinite(time) or not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("time must be finite and dt_fs must be finite and positive")
    evaluations = 0

    def rhs(stage_time: float, stage_state: ComplexArray) -> ComplexArray:
        nonlocal evaluations
        matrix = _matrix_at(hamiltonian_at, stage_time, dimension=psi.size)
        evaluations += 1
        return np.asarray((-1.0j / HBAR_EV_FS) * (matrix @ stage_state), dtype=np.complex128)

    start = perf_counter()
    k1 = rhs(time, psi)
    k2 = rhs(time + 0.5 * dt, psi + 0.5 * dt * k1)
    k3 = rhs(time + 0.5 * dt, psi + 0.5 * dt * k2)
    k4 = rhs(time + dt, psi + dt * k3)
    propagated = psi + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    elapsed = perf_counter() - start
    return TimeDependentPropagationResult(
        state=np.asarray(propagated, dtype=np.complex128),
        hamiltonian_evaluations=evaluations,
        hamiltonian_applications=4,
        elapsed_seconds=float(elapsed),
        accepted_steps=1,
        rhs_evaluations=4,
    )


def cfm4_lanczos_step(
    hamiltonian_at: HamiltonianFunction,
    state: ArrayLike,
    time_fs: float,
    dt_fs: float,
    *,
    krylov_dimension: int = 8,
    breakdown_tolerance: float = 1.0e-13,
) -> TimeDependentPropagationResult:
    """Fourth-order two-exponential commutator-free Magnus/Lanczos step.

    The Hamiltonian is sampled at the two Gauss nodes

        c1,2 = 1/2 -/+ sqrt(3)/6.

    With

        a1 = (3 - 2 sqrt(3))/12,
        a2 = (3 + 2 sqrt(3))/12,

    the propagator is

        exp[-i dt (a1 H1 + a2 H2)/hbar]
        exp[-i dt (a2 H1 + a1 H2)/hbar].

    The rightmost exponential acts first.  For constant H both weighted
    Hamiltonians reduce to H/2, exactly recovering the D0a frozen CF4 limit.
    """
    psi = _state_vector(state)
    time = float(time_fs)
    dt = float(dt_fs)
    if not np.isfinite(time) or not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("time must be finite and dt_fs must be finite and positive")
    if krylov_dimension <= 0:
        raise ValueError("krylov_dimension must be positive")

    sqrt3 = float(np.sqrt(3.0))
    c1 = 0.5 - sqrt3 / 6.0
    c2 = 0.5 + sqrt3 / 6.0
    a1 = (3.0 - 2.0 * sqrt3) / 12.0
    a2 = (3.0 + 2.0 * sqrt3) / 12.0

    start = perf_counter()
    h1 = _matrix_at(hamiltonian_at, time + c1 * dt, dimension=psi.size)
    h2 = _matrix_at(hamiltonian_at, time + c2 * dt, dimension=psi.size)
    right_matrix = a2 * h1 + a1 * h2
    left_matrix = a1 * h1 + a2 * h2

    right_action = CountingMatrixHamiltonian(right_matrix)
    first = lanczos_exponential_step(
        right_action,
        psi,
        dt,
        krylov_dimension=krylov_dimension,
        breakdown_tolerance=breakdown_tolerance,
    )
    left_action = CountingMatrixHamiltonian(left_matrix)
    second = lanczos_exponential_step(
        left_action,
        first.state,
        dt,
        krylov_dimension=krylov_dimension,
        breakdown_tolerance=breakdown_tolerance,
    )
    elapsed = perf_counter() - start
    return TimeDependentPropagationResult(
        state=second.state,
        hamiltonian_evaluations=2,
        hamiltonian_applications=(
            first.hamiltonian_applications + second.hamiltonian_applications
        ),
        elapsed_seconds=float(elapsed),
        accepted_steps=1,
    )


def integrate_rk4_time_dependent(
    hamiltonian_at: HamiltonianFunction,
    state: ArrayLike,
    *,
    dt_fs: float,
    steps: int,
    initial_time_fs: float = 0.0,
) -> TimeDependentPropagationResult:
    """Integrate an explicitly time-dependent Hamiltonian with fixed-step RK4."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    psi = _state_vector(state)
    time = float(initial_time_fs)
    total_evaluations = 0
    total_applications = 0
    start = perf_counter()
    for _ in range(steps):
        result = rk4_time_dependent_step(hamiltonian_at, psi, time, dt_fs)
        psi = result.state
        total_evaluations += result.hamiltonian_evaluations
        total_applications += result.hamiltonian_applications
        time += float(dt_fs)
    elapsed = perf_counter() - start
    return TimeDependentPropagationResult(
        state=psi,
        hamiltonian_evaluations=total_evaluations,
        hamiltonian_applications=total_applications,
        elapsed_seconds=float(elapsed),
        accepted_steps=steps,
        rhs_evaluations=4 * steps,
    )


def integrate_cfm4_lanczos(
    hamiltonian_at: HamiltonianFunction,
    state: ArrayLike,
    *,
    dt_fs: float,
    steps: int,
    initial_time_fs: float = 0.0,
    krylov_dimension: int = 8,
    breakdown_tolerance: float = 1.0e-13,
) -> TimeDependentPropagationResult:
    """Integrate D1 with fixed-step fourth-order CF Magnus + Lanczos."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    psi = _state_vector(state)
    time = float(initial_time_fs)
    total_evaluations = 0
    total_applications = 0
    start = perf_counter()
    for _ in range(steps):
        result = cfm4_lanczos_step(
            hamiltonian_at,
            psi,
            time,
            dt_fs,
            krylov_dimension=krylov_dimension,
            breakdown_tolerance=breakdown_tolerance,
        )
        psi = result.state
        total_evaluations += result.hamiltonian_evaluations
        total_applications += result.hamiltonian_applications
        time += float(dt_fs)
    elapsed = perf_counter() - start
    return TimeDependentPropagationResult(
        state=psi,
        hamiltonian_evaluations=total_evaluations,
        hamiltonian_applications=total_applications,
        elapsed_seconds=float(elapsed),
        accepted_steps=steps,
    )


def integrate_dop853_time_dependent(
    hamiltonian_at: HamiltonianFunction,
    state: ArrayLike,
    *,
    final_time_fs: float,
    initial_time_fs: float = 0.0,
    rtol: float = 1.0e-11,
    atol: float = 1.0e-13,
    max_step_fs: float = np.inf,
) -> TimeDependentPropagationResult:
    """Adaptive DOP853 reference for the linear explicitly time-dependent TDSE."""
    psi = _state_vector(state)
    initial_time = float(initial_time_fs)
    final_time = float(final_time_fs)
    if not np.isfinite(initial_time) or not np.isfinite(final_time):
        raise ValueError("integration times must be finite")
    if final_time <= initial_time:
        raise ValueError("final_time_fs must exceed initial_time_fs")
    if rtol <= 0.0 or atol <= 0.0 or max_step_fs <= 0.0:
        raise ValueError("DOP853 tolerances and max_step_fs must be positive")
    evaluations = 0

    def rhs(time: float, values: ComplexArray) -> ComplexArray:
        nonlocal evaluations
        matrix = _matrix_at(hamiltonian_at, time, dimension=psi.size)
        evaluations += 1
        return np.asarray((-1.0j / HBAR_EV_FS) * (matrix @ values), dtype=np.complex128)

    start = perf_counter()
    solution = solve_ivp(
        rhs,
        (initial_time, final_time),
        psi,
        method="DOP853",
        rtol=float(rtol),
        atol=float(atol),
        max_step=float(max_step_fs),
    )
    elapsed = perf_counter() - start
    return TimeDependentPropagationResult(
        state=np.asarray(solution.y[:, -1], dtype=np.complex128),
        hamiltonian_evaluations=evaluations,
        hamiltonian_applications=evaluations,
        elapsed_seconds=float(elapsed),
        accepted_steps=max(0, int(solution.t.size - 1)),
        rhs_evaluations=int(solution.nfev),
        success=bool(solution.success),
        message=str(solution.message),
    )
