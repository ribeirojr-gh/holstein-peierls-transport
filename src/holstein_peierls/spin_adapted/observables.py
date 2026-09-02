"""Occupation numbers and channel yields for static and future time-dependent states.

The O0 layer is intentionally independent of any particular propagator.  It
implements observables directly from the propagated quantum state or its reduced
density matrix so that RK, Krylov, Magnus, MCTDHF, and static solvers share the
same definitions.

For an orthonormal set of instantaneous one-particle orbitals ``phi_l`` and a
one-particle reduced density matrix ``gamma``, the occupation is

    n_l = <phi_l | gamma | phi_l>.

Equivalently, for propagated orbitals ``psi_k`` with occupations ``f_k``,

    n_l = sum_k f_k |<phi_l | psi_k>|^2.

Channel yields are expectation values of projectors.  If the channel is spanned
by possibly non-orthogonal states collected as columns of ``S``, the orthogonal
projector onto their span is

    P = S (S^dagger S)^+ S^dagger,

where ``+`` denotes the Moore-Penrose pseudoinverse.  This construction avoids
double counting and retains interference inside multiconfigurational channels.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


def _as_complex_matrix(values: ArrayLike, *, name: str) -> ComplexArray:
    array = np.asarray(values, dtype=np.complex128)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    return array


def _validate_orthonormal_columns(
    orbitals: ComplexArray,
    *,
    atol: float,
    name: str,
) -> None:
    gram = orbitals.conj().T @ orbitals
    identity = np.eye(orbitals.shape[1], dtype=np.complex128)
    if not np.allclose(gram, identity, atol=atol, rtol=0.0):
        raise ValueError(f"{name} columns must be orthonormal")


def one_rdm_from_orbitals(
    propagated_orbitals: ArrayLike,
    occupations: ArrayLike,
    *,
    orthonormality_tolerance: float = 1.0e-10,
) -> ComplexArray:
    """Return ``gamma = Psi diag(f) Psi^dagger``.

    ``propagated_orbitals`` stores one orbital per column.  ``occupations`` may
    contain integer or fractional occupations, but negative occupations are not
    allowed.  The propagated orbitals are required to be orthonormal so that the
    trace of the returned RDM is exactly the total supplied occupation up to
    numerical roundoff.
    """
    psi = _as_complex_matrix(propagated_orbitals, name="propagated_orbitals")
    _validate_orthonormal_columns(
        psi,
        atol=orthonormality_tolerance,
        name="propagated_orbitals",
    )
    f = np.asarray(occupations, dtype=np.float64)
    if f.ndim != 1 or f.size != psi.shape[1]:
        raise ValueError("occupations must contain one value per propagated orbital")
    if np.any(~np.isfinite(f)) or np.any(f < 0.0):
        raise ValueError("occupations must be finite and non-negative")
    gamma = (psi * f[np.newaxis, :]) @ psi.conj().T
    return np.asarray(0.5 * (gamma + gamma.conj().T), dtype=np.complex128)


def instantaneous_occupation_numbers(
    one_rdm: ArrayLike,
    instantaneous_orbitals: ArrayLike,
    *,
    orthonormality_tolerance: float = 1.0e-10,
    hermiticity_tolerance: float = 1.0e-10,
) -> FloatArray:
    """Project a one-particle RDM onto instantaneous orthonormal orbitals."""
    gamma = _as_complex_matrix(one_rdm, name="one_rdm")
    if gamma.shape[0] != gamma.shape[1]:
        raise ValueError("one_rdm must be square")
    if not np.allclose(
        gamma,
        gamma.conj().T,
        atol=hermiticity_tolerance,
        rtol=0.0,
    ):
        raise ValueError("one_rdm must be Hermitian")

    phi = _as_complex_matrix(instantaneous_orbitals, name="instantaneous_orbitals")
    if phi.shape[0] != gamma.shape[0]:
        raise ValueError("instantaneous orbitals and one_rdm dimensions do not match")
    _validate_orthonormal_columns(
        phi,
        atol=orthonormality_tolerance,
        name="instantaneous_orbitals",
    )

    projected = phi.conj().T @ gamma @ phi
    diagonal = np.diag(projected)
    if np.max(np.abs(np.imag(diagonal)), initial=0.0) > hermiticity_tolerance:
        raise RuntimeError("projected occupations acquired a non-negligible imaginary part")
    result = np.asarray(np.real(diagonal), dtype=np.float64)
    tiny = 64.0 * np.finfo(np.float64).eps
    result[np.abs(result) < tiny] = 0.0
    return result


def occupation_numbers_from_propagated_orbitals(
    propagated_orbitals: ArrayLike,
    occupations: ArrayLike,
    instantaneous_orbitals: ArrayLike,
    *,
    orthonormality_tolerance: float = 1.0e-10,
) -> FloatArray:
    """Evaluate ``sum_k f_k |<phi_l|psi_k>|^2`` through the RDM definition."""
    gamma = one_rdm_from_orbitals(
        propagated_orbitals,
        occupations,
        orthonormality_tolerance=orthonormality_tolerance,
    )
    return instantaneous_occupation_numbers(
        gamma,
        instantaneous_orbitals,
        orthonormality_tolerance=orthonormality_tolerance,
    )


def channel_projector(
    channel_states: ArrayLike,
    *,
    rank_tolerance: float = 1.0e-12,
) -> ComplexArray:
    """Return the orthogonal projector onto a possibly non-orthogonal span."""
    states = _as_complex_matrix(channel_states, name="channel_states")
    if states.shape[1] == 0:
        raise ValueError("channel_states must contain at least one state")
    if rank_tolerance <= 0.0:
        raise ValueError("rank_tolerance must be positive")
    gram = states.conj().T @ states
    eigenvalues = np.linalg.eigvalsh(0.5 * (gram + gram.conj().T))
    scale = max(1.0, float(np.max(np.abs(eigenvalues), initial=0.0)))
    if float(np.max(eigenvalues, initial=0.0)) <= rank_tolerance * scale:
        raise ValueError("channel_states span is numerically zero")
    gram_pinv = np.linalg.pinv(gram, rcond=rank_tolerance, hermitian=True)
    projector = states @ gram_pinv @ states.conj().T
    projector = 0.5 * (projector + projector.conj().T)
    return np.asarray(projector, dtype=np.complex128)


def channel_yield_from_state(
    state: ArrayLike,
    projector: ArrayLike,
    *,
    normalization_tolerance: float = 1.0e-10,
) -> float:
    """Return ``<Psi|P|Psi>`` for a normalized pure state."""
    psi = np.asarray(state, dtype=np.complex128)
    if psi.ndim != 1:
        raise ValueError("state must be a one-dimensional vector")
    norm = float(np.vdot(psi, psi).real)
    if not np.isclose(norm, 1.0, atol=normalization_tolerance, rtol=0.0):
        raise ValueError("state must be normalized")
    p = _as_complex_matrix(projector, name="projector")
    if p.shape != (psi.size, psi.size):
        raise ValueError("projector dimension does not match state")
    value = np.vdot(psi, p @ psi)
    if abs(float(np.imag(value))) > normalization_tolerance:
        raise RuntimeError("channel yield acquired a non-negligible imaginary part")
    result = float(np.real(value))
    if result < 0.0 and result > -normalization_tolerance:
        result = 0.0
    if result > 1.0 and result < 1.0 + normalization_tolerance:
        result = 1.0
    return result


def channel_yield_from_density(
    density_matrix: ArrayLike,
    projector: ArrayLike,
    *,
    trace_tolerance: float = 1.0e-10,
) -> float:
    """Return ``Tr(rho P)`` for a normalized density operator."""
    rho = _as_complex_matrix(density_matrix, name="density_matrix")
    p = _as_complex_matrix(projector, name="projector")
    if rho.shape[0] != rho.shape[1] or p.shape != rho.shape:
        raise ValueError("density_matrix and projector must be square and same size")
    if not np.allclose(rho, rho.conj().T, atol=trace_tolerance, rtol=0.0):
        raise ValueError("density_matrix must be Hermitian")
    trace = np.trace(rho)
    if not np.isclose(trace, 1.0, atol=trace_tolerance, rtol=0.0):
        raise ValueError("density_matrix must have unit trace")
    value = np.trace(rho @ p)
    if abs(float(np.imag(value))) > trace_tolerance:
        raise RuntimeError("channel yield acquired a non-negligible imaginary part")
    result = float(np.real(value))
    if result < 0.0 and result > -trace_tolerance:
        result = 0.0
    if result > 1.0 and result < 1.0 + trace_tolerance:
        result = 1.0
    return result


def configuration_channel_yield(
    state_coefficients: ArrayLike,
    channel_states: ArrayLike,
    *,
    rank_tolerance: float = 1.0e-12,
) -> float:
    """Project a configuration-space state onto a multiconfigurational channel."""
    projector = channel_projector(channel_states, rank_tolerance=rank_tolerance)
    return channel_yield_from_state(state_coefficients, projector)


def slater_determinant_overlap(
    left_occupied_orbitals: ArrayLike,
    right_occupied_orbitals: ArrayLike,
) -> complex:
    """Return the overlap between two Slater determinants.

    For occupied spin-orbital coefficient matrices ``L`` and ``R`` with one
    occupied orbital per column, ``<Phi_L|Phi_R> = det(L^dagger R)``.
    """
    left = _as_complex_matrix(left_occupied_orbitals, name="left_occupied_orbitals")
    right = _as_complex_matrix(right_occupied_orbitals, name="right_occupied_orbitals")
    if left.shape != right.shape:
        raise ValueError("left and right occupied-orbital matrices must have same shape")
    overlap_matrix = left.conj().T @ right
    return complex(np.linalg.det(overlap_matrix))
