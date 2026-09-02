"""Deterministic frozen-H electronic propagators for the D0a benchmark.

This module deliberately contains no lattice integrator, thermostat, electric
field, or material-specific parameterization.  It benchmarks only

    d|psi>/dt = -(i/hbar) H |psi>

for a fixed Hermitian Hamiltonian expressed in eV and a time step expressed in
femtoseconds.

The full spectral exponential reproduces the mathematical operation used by the
archived dynamics and is retained as an accuracy reference.  Matrix-vector
methods operate through :class:`CountingMatrixHamiltonian`, making the number of
``H @ psi`` evaluations an explicit benchmark metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csc_matrix, csr_matrix, eye as sparse_eye, issparse
from scipy.sparse.linalg import spsolve

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]

# CODATA/SI value converted from eV s to eV fs.  The project Hamiltonians are
# expressed in eV, so this is the natural constant for electronic propagation.
HBAR_EV_FS = 0.6582119569


@dataclass(frozen=True, slots=True)
class PropagationStep:
    """One electronic propagation result and its algorithmic work counters."""

    state: ComplexArray
    hamiltonian_applications: int = 0
    eigendecompositions: int = 0
    linear_solves: int = 0
    approximate_workspace_bytes: int = 0
    embedded_error_estimate: float | None = None
    krylov_dimension_used: int | None = None


@dataclass(frozen=True, slots=True)
class PropagationMetrics:
    """Accuracy metrics against an exact/reference electronic state."""

    norm_error: float
    raw_state_error: float
    phase_aligned_state_error: float
    fidelity: float
    energy_expectation: float
    reference_energy_expectation: float
    energy_error: float


class CountingMatrixHamiltonian:
    """Count applications of a fixed dense or sparse Hamiltonian matrix."""

    def __init__(self, matrix: Any) -> None:
        if not hasattr(matrix, "shape") or len(matrix.shape) != 2:
            raise ValueError("Hamiltonian must be a two-dimensional matrix")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Hamiltonian must be square")
        self.matrix = matrix
        self.dimension = int(matrix.shape[0])
        self.applications = 0

    def reset(self) -> None:
        """Reset the application counter without changing the Hamiltonian."""
        self.applications = 0

    def __call__(self, state: ArrayLike) -> ComplexArray:
        vector = _state_vector(state, dimension=self.dimension)
        self.applications += 1
        return np.asarray(self.matrix @ vector, dtype=np.complex128)


def _state_vector(values: ArrayLike, *, dimension: int | None = None) -> ComplexArray:
    vector = np.asarray(values, dtype=np.complex128)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("state must be a non-empty one-dimensional vector")
    if dimension is not None and vector.size != dimension:
        raise ValueError("state dimension does not match Hamiltonian")
    if not np.all(np.isfinite(vector)):
        raise ValueError("state must contain only finite values")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("state must have a finite non-zero norm")
    return vector


def _positive_time_step(dt_fs: float) -> float:
    value = float(dt_fs)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("dt_fs must be finite and positive")
    return value


def _dense_hermitian(matrix: Any, *, atol: float = 1.0e-12) -> ComplexArray:
    dense = matrix.toarray() if issparse(matrix) else np.asarray(matrix)
    dense = np.asarray(dense, dtype=np.complex128)
    if dense.ndim != 2 or dense.shape[0] != dense.shape[1]:
        raise ValueError("Hamiltonian must be a square matrix")
    if not np.all(np.isfinite(dense)):
        raise ValueError("Hamiltonian must contain only finite values")
    if not np.allclose(dense, dense.conj().T, rtol=0.0, atol=atol):
        raise ValueError("Hamiltonian must be Hermitian")
    return dense


def _schrodinger_rhs(
    action: CountingMatrixHamiltonian,
    state: ComplexArray,
) -> ComplexArray:
    return (-1.0j / HBAR_EV_FS) * action(state)


def exact_spectral_step(
    hamiltonian: Any,
    state: ArrayLike,
    dt_fs: float,
) -> PropagationStep:
    """Propagate with a complete Hermitian eigendecomposition.

    A fresh eigendecomposition is intentionally performed on every call.  This
    mirrors the archived algorithmic cost when the function is used repeatedly,
    even though a frozen Hamiltonian could trivially cache its eigensystem.
    """
    dt = _positive_time_step(dt_fs)
    dense = _dense_hermitian(hamiltonian)
    psi = _state_vector(state, dimension=dense.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(dense)
    coefficients = eigenvectors.conj().T @ psi
    phases = np.exp((-1.0j * dt / HBAR_EV_FS) * eigenvalues)
    propagated = eigenvectors @ (phases * coefficients)
    n = dense.shape[0]
    # Approximate propagation workspace beyond the caller-owned Hamiltonian:
    # eigenvectors/eigenvalues plus complex coefficient/state vectors.
    workspace = 16 * n * n + 40 * n
    return PropagationStep(
        state=np.asarray(propagated, dtype=np.complex128),
        eigendecompositions=1,
        approximate_workspace_bytes=workspace,
    )


def rk4_step(
    action: CountingMatrixHamiltonian,
    state: ArrayLike,
    dt_fs: float,
) -> PropagationStep:
    """Classical fixed-step fourth-order Runge-Kutta propagation."""
    dt = _positive_time_step(dt_fs)
    psi = _state_vector(state, dimension=action.dimension)
    before = action.applications
    k1 = _schrodinger_rhs(action, psi)
    k2 = _schrodinger_rhs(action, psi + 0.5 * dt * k1)
    k3 = _schrodinger_rhs(action, psi + 0.5 * dt * k2)
    k4 = _schrodinger_rhs(action, psi + dt * k3)
    propagated = psi + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    n = psi.size
    return PropagationStep(
        state=np.asarray(propagated, dtype=np.complex128),
        hamiltonian_applications=action.applications - before,
        approximate_workspace_bytes=6 * 16 * n,
    )


# Fehlberg's classical embedded 7(8) tableau.  The first set of output weights
# is seventh order; the second set below is the eighth-order solution.  All 13
# stages are evaluated so the embedded difference remains available as a local
# error diagnostic even though D0a uses a fixed time step.
_RKF78_A: tuple[tuple[float, ...], ...] = (
    (),
    (2.0 / 27.0,),
    (1.0 / 36.0, 1.0 / 12.0),
    (1.0 / 24.0, 0.0, 1.0 / 8.0),
    (5.0 / 12.0, 0.0, -25.0 / 16.0, 25.0 / 16.0),
    (1.0 / 20.0, 0.0, 0.0, 1.0 / 4.0, 1.0 / 5.0),
    (-25.0 / 108.0, 0.0, 0.0, 125.0 / 108.0, -65.0 / 27.0, 125.0 / 54.0),
    (31.0 / 300.0, 0.0, 0.0, 0.0, 61.0 / 225.0, -2.0 / 9.0, 13.0 / 900.0),
    (2.0, 0.0, 0.0, -53.0 / 6.0, 704.0 / 45.0, -107.0 / 9.0, 67.0 / 90.0, 3.0),
    (-91.0 / 108.0, 0.0, 0.0, 23.0 / 108.0, -976.0 / 135.0, 311.0 / 54.0, -19.0 / 60.0, 17.0 / 6.0, -1.0 / 12.0),
    (2383.0 / 4100.0, 0.0, 0.0, -341.0 / 164.0, 4496.0 / 1025.0, -301.0 / 82.0, 2133.0 / 4100.0, 45.0 / 82.0, 45.0 / 164.0, 18.0 / 41.0),
    (3.0 / 205.0, 0.0, 0.0, 0.0, 0.0, -6.0 / 41.0, -3.0 / 205.0, -3.0 / 41.0, 3.0 / 41.0, 6.0 / 41.0, 0.0),
    (-1777.0 / 4100.0, 0.0, 0.0, -341.0 / 164.0, 4496.0 / 1025.0, -289.0 / 82.0, 2193.0 / 4100.0, 51.0 / 82.0, 33.0 / 164.0, 12.0 / 41.0, 0.0, 1.0),
)
_RKF78_B7 = np.asarray(
    (41.0 / 840.0, 0.0, 0.0, 0.0, 0.0, 34.0 / 105.0, 9.0 / 35.0, 9.0 / 35.0, 9.0 / 280.0, 9.0 / 280.0, 41.0 / 840.0, 0.0, 0.0),
    dtype=np.float64,
)
_RKF78_B8 = np.asarray(
    (0.0, 0.0, 0.0, 0.0, 0.0, 34.0 / 105.0, 9.0 / 35.0, 9.0 / 35.0, 9.0 / 280.0, 9.0 / 280.0, 0.0, 41.0 / 840.0, 41.0 / 840.0),
    dtype=np.float64,
)


def rkf78_step(
    action: CountingMatrixHamiltonian,
    state: ArrayLike,
    dt_fs: float,
) -> PropagationStep:
    """Fixed-step Fehlberg 7(8) propagation, returning the order-8 solution."""
    dt = _positive_time_step(dt_fs)
    psi = _state_vector(state, dimension=action.dimension)
    before = action.applications
    stages: list[ComplexArray] = []
    for row in _RKF78_A:
        if row:
            increment = np.zeros_like(psi)
            for coefficient, stage in zip(row, stages, strict=True):
                if coefficient != 0.0:
                    increment += coefficient * stage
            stage_state = psi + dt * increment
        else:
            stage_state = psi
        stages.append(_schrodinger_rhs(action, stage_state))

    seventh = psi.copy()
    eighth = psi.copy()
    for b7, b8, stage in zip(_RKF78_B7, _RKF78_B8, stages, strict=True):
        if b7 != 0.0:
            seventh += dt * b7 * stage
        if b8 != 0.0:
            eighth += dt * b8 * stage
    embedded_error = float(np.linalg.norm(eighth - seventh))
    n = psi.size
    return PropagationStep(
        state=np.asarray(eighth, dtype=np.complex128),
        hamiltonian_applications=action.applications - before,
        approximate_workspace_bytes=15 * 16 * n,
        embedded_error_estimate=embedded_error,
    )


def lanczos_exponential_step(
    action: CountingMatrixHamiltonian,
    state: ArrayLike,
    dt_fs: float,
    *,
    krylov_dimension: int = 12,
    breakdown_tolerance: float = 1.0e-13,
) -> PropagationStep:
    """Apply the frozen-H exponential in a short Hermitian Lanczos space.

    The residual is fully reorthogonalized against the accumulated basis to
    suppress loss of orthogonality in the small Krylov spaces used by D0a.  A
    true Krylov breakdown terminates early and is an exact invariant-subspace
    result rather than an error.
    """
    dt = _positive_time_step(dt_fs)
    if krylov_dimension <= 0:
        raise ValueError("krylov_dimension must be positive")
    if breakdown_tolerance <= 0.0:
        raise ValueError("breakdown_tolerance must be positive")
    psi = _state_vector(state, dimension=action.dimension)
    maximum_dimension = min(int(krylov_dimension), action.dimension)
    initial_norm = float(np.linalg.norm(psi))
    q = psi / initial_norm
    q_previous: ComplexArray | None = None
    beta_previous = 0.0
    basis: list[ComplexArray] = []
    diagonal: list[float] = []
    off_diagonal: list[float] = []
    before = action.applications

    for index in range(maximum_dimension):
        basis.append(q.copy())
        residual = action(q)
        alpha_complex = np.vdot(q, residual)
        if abs(float(np.imag(alpha_complex))) > 1.0e-10:
            raise FloatingPointError("Lanczos alpha is not real; Hamiltonian may be non-Hermitian")
        alpha = float(np.real(alpha_complex))
        residual = residual - alpha * q
        if q_previous is not None:
            residual = residual - beta_previous * q_previous

        # Full reorthogonalization is inexpensive for the deliberately short
        # D0a spaces and makes the numerical gate reproducible.
        for vector in basis:
            residual = residual - vector * np.vdot(vector, residual)
        beta = float(np.linalg.norm(residual))
        diagonal.append(alpha)

        if index == maximum_dimension - 1 or beta <= breakdown_tolerance:
            break
        off_diagonal.append(beta)
        q_previous = q
        q = residual / beta
        beta_previous = beta

    used = len(diagonal)
    tridiagonal = np.diag(np.asarray(diagonal, dtype=np.float64))
    if used > 1:
        off = np.asarray(off_diagonal[: used - 1], dtype=np.float64)
        tridiagonal += np.diag(off, k=1) + np.diag(off, k=-1)
    eigenvalues, eigenvectors = np.linalg.eigh(tridiagonal)
    projected_initial = eigenvectors[0, :].conj()
    projected = eigenvectors @ (
        np.exp((-1.0j * dt / HBAR_EV_FS) * eigenvalues) * projected_initial
    )
    q_matrix = np.column_stack(basis[:used])
    propagated = initial_norm * (q_matrix @ projected)
    n = psi.size
    workspace = 16 * n * used + 16 * n + 8 * used * used
    return PropagationStep(
        state=np.asarray(propagated, dtype=np.complex128),
        hamiltonian_applications=action.applications - before,
        approximate_workspace_bytes=workspace,
        krylov_dimension_used=used,
    )


def cfm4_frozen_limit_step(
    action: CountingMatrixHamiltonian,
    state: ArrayLike,
    dt_fs: float,
    *,
    krylov_dimension: int = 12,
    breakdown_tolerance: float = 1.0e-13,
) -> PropagationStep:
    """Frozen-H reduction of the standard two-exponential fourth-order CF scheme.

    A fourth-order two-exponential commutator-free Magnus propagator evaluates
    two weighted Hamiltonians at Gauss nodes.  When ``H`` is constant, each
    weighted Hamiltonian reduces to ``H/2`` and the two exponentials commute.
    D0a therefore evaluates two half-step Lanczos exponentials.  The genuinely
    time-dependent Gauss-node combination belongs to D1 and is intentionally not
    implemented here.
    """
    dt = _positive_time_step(dt_fs)
    first = lanczos_exponential_step(
        action,
        state,
        0.5 * dt,
        krylov_dimension=krylov_dimension,
        breakdown_tolerance=breakdown_tolerance,
    )
    second = lanczos_exponential_step(
        action,
        first.state,
        0.5 * dt,
        krylov_dimension=krylov_dimension,
        breakdown_tolerance=breakdown_tolerance,
    )
    used = max(
        first.krylov_dimension_used or 0,
        second.krylov_dimension_used or 0,
    )
    return PropagationStep(
        state=second.state,
        hamiltonian_applications=(
            first.hamiltonian_applications + second.hamiltonian_applications
        ),
        approximate_workspace_bytes=max(
            first.approximate_workspace_bytes,
            second.approximate_workspace_bytes,
        ),
        krylov_dimension_used=used,
    )


def crank_nicolson_step(
    hamiltonian: Any,
    state: ArrayLike,
    dt_fs: float,
) -> PropagationStep:
    """Apply the unitary Cayley/Crank-Nicolson step for a frozen Hermitian H."""
    dt = _positive_time_step(dt_fs)
    if not hasattr(hamiltonian, "shape") or hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ValueError("Hamiltonian must be square")
    n = int(hamiltonian.shape[0])
    psi = _state_vector(state, dimension=n)
    factor = 0.5j * dt / HBAR_EV_FS

    if issparse(hamiltonian):
        matrix = csr_matrix(hamiltonian, dtype=np.complex128)
        identity = sparse_eye(n, dtype=np.complex128, format="csr")
        rhs = psi - factor * (matrix @ psi)
        lhs = csc_matrix(identity + factor * matrix)
        propagated = spsolve(lhs, rhs)
        # Sparse factorization storage is implementation/order dependent.  The
        # reported lower-bound workspace accounts only for explicit vectors and
        # sparse matrix data owned by this step.
        workspace = 5 * 16 * n + lhs.data.nbytes + lhs.indices.nbytes + lhs.indptr.nbytes
    else:
        matrix = _dense_hermitian(hamiltonian)
        identity = np.eye(n, dtype=np.complex128)
        rhs = psi - factor * (matrix @ psi)
        lhs = identity + factor * matrix
        propagated = np.linalg.solve(lhs, rhs)
        workspace = 2 * 16 * n * n + 4 * 16 * n

    return PropagationStep(
        state=np.asarray(propagated, dtype=np.complex128),
        hamiltonian_applications=1,
        linear_solves=1,
        approximate_workspace_bytes=int(workspace),
    )


def energy_expectation(hamiltonian: Any, state: ArrayLike) -> float:
    """Return ``<psi|H|psi>/<psi|psi>`` in eV for a non-zero state."""
    if not hasattr(hamiltonian, "shape") or hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ValueError("Hamiltonian must be square")
    psi = _state_vector(state, dimension=int(hamiltonian.shape[0]))
    norm_squared = float(np.vdot(psi, psi).real)
    value = np.vdot(psi, hamiltonian @ psi) / norm_squared
    if abs(float(np.imag(value))) > 1.0e-10:
        raise FloatingPointError("energy expectation has a non-negligible imaginary part")
    return float(np.real(value))


def compare_to_reference(
    reference_state: ArrayLike,
    candidate_state: ArrayLike,
    hamiltonian: Any,
) -> PropagationMetrics:
    """Compare a candidate state with a reference modulo global phase."""
    if not hasattr(hamiltonian, "shape") or hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ValueError("Hamiltonian must be square")
    dimension = int(hamiltonian.shape[0])
    reference = _state_vector(reference_state, dimension=dimension)
    candidate = _state_vector(candidate_state, dimension=dimension)
    reference_norm = float(np.linalg.norm(reference))
    candidate_norm = float(np.linalg.norm(candidate))
    normalized_overlap = np.vdot(reference, candidate) / (reference_norm * candidate_norm)
    fidelity = float(np.clip(abs(normalized_overlap) ** 2, 0.0, 1.0))
    overlap = np.vdot(reference, candidate)
    if abs(overlap) > 0.0:
        phase = np.conj(overlap) / abs(overlap)
        aligned = candidate * phase
    else:
        aligned = candidate
    raw_error = float(np.linalg.norm(candidate - reference) / reference_norm)
    aligned_error = float(np.linalg.norm(aligned - reference) / reference_norm)
    candidate_energy = energy_expectation(hamiltonian, candidate)
    reference_energy = energy_expectation(hamiltonian, reference)
    return PropagationMetrics(
        norm_error=abs(candidate_norm - reference_norm),
        raw_state_error=raw_error,
        phase_aligned_state_error=aligned_error,
        fidelity=fidelity,
        energy_expectation=candidate_energy,
        reference_energy_expectation=reference_energy,
        energy_error=abs(candidate_energy - reference_energy),
    )
