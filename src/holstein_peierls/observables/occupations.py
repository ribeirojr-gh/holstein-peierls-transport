"""Instantaneous orbital occupation numbers.

The legacy literature defines the occupation of instantaneous level ``l`` as

    n_l(t) = sum_k f_k |<phi_l(t)|psi_k(t)>|^2,

where ``psi_k`` are propagated occupied orbitals, ``f_k`` their occupations,
and ``phi_l`` instantaneous one-particle eigenstates.  In density-matrix form,

    n_l(t) = <phi_l(t)| gamma^(1)(t) |phi_l(t)>.

The RDM form is the canonical implementation because it also applies to the
spin-adapted/multiconfigurational states introduced in S0.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


def _as_square_matrix(values: ArrayLike, *, name: str) -> ComplexArray:
    matrix = np.asarray(values, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    return matrix


def _as_orbital_matrix(values: ArrayLike, *, n_basis: int, name: str) -> ComplexArray:
    orbitals = np.asarray(values, dtype=np.complex128)
    if orbitals.ndim != 2 or orbitals.shape[0] != n_basis:
        raise ValueError(f"{name} must have shape (n_basis, n_orbitals)")
    if orbitals.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one orbital")
    return orbitals


def _check_orthonormal(orbitals: ComplexArray, *, atol: float, name: str) -> None:
    overlap = orbitals.conj().T @ orbitals
    identity = np.eye(orbitals.shape[1], dtype=np.complex128)
    if not np.allclose(overlap, identity, rtol=0.0, atol=atol):
        raise ValueError(f"{name} must have orthonormal columns")


def instantaneous_occupations(
    one_particle_rdm: ArrayLike,
    instantaneous_orbitals: ArrayLike,
    *,
    check_orthonormal: bool = True,
    atol: float = 1.0e-10,
) -> FloatArray:
    """Project a one-particle RDM onto an instantaneous orbital basis.

    Parameters
    ----------
    one_particle_rdm
        Hermitian one-particle reduced density matrix ``gamma`` in the site or
        atomic-orbital basis.
    instantaneous_orbitals
        Columns are the instantaneous orbitals ``phi_l`` in the same basis.
        The matrix may contain a complete basis or only a selected spectral
        window.
    check_orthonormal
        Require the supplied instantaneous orbitals to be orthonormal.
    atol
        Absolute tolerance used for Hermiticity, orthonormality, and the
        imaginary part of projected diagonal elements.

    Returns
    -------
    numpy.ndarray
        ``n_l = <phi_l|gamma|phi_l>`` for every supplied orbital.
    """
    if atol <= 0.0:
        raise ValueError("atol must be positive")
    gamma = _as_square_matrix(one_particle_rdm, name="one_particle_rdm")
    if not np.allclose(gamma, gamma.conj().T, rtol=0.0, atol=atol):
        raise ValueError("one_particle_rdm must be Hermitian")

    orbitals = _as_orbital_matrix(
        instantaneous_orbitals,
        n_basis=gamma.shape[0],
        name="instantaneous_orbitals",
    )
    if check_orthonormal:
        _check_orthonormal(orbitals, atol=atol, name="instantaneous_orbitals")

    projected = orbitals.conj().T @ gamma @ orbitals
    diagonal = np.diag(projected)
    if np.max(np.abs(np.imag(diagonal))) > atol:
        raise FloatingPointError("projected occupations have a non-negligible imaginary part")
    return np.asarray(np.real(diagonal), dtype=np.float64)


def occupations_from_orbitals(
    propagated_orbitals: ArrayLike,
    orbital_occupations: ArrayLike,
    instantaneous_orbitals: ArrayLike,
    *,
    check_orthonormal: bool = True,
    atol: float = 1.0e-10,
) -> FloatArray:
    """Evaluate the literature overlap formula for propagated orbitals.

    This is a convenience wrapper around :func:`instantaneous_occupations`.
    It constructs

        gamma = sum_k f_k |psi_k><psi_k|

    and therefore exactly reproduces

        n_l = sum_k f_k |<phi_l|psi_k>|^2.

    The direct-RDM interface should be preferred for correlated or
    multiconfigurational states.
    """
    psi = np.asarray(propagated_orbitals, dtype=np.complex128)
    if psi.ndim != 2 or psi.shape[1] == 0:
        raise ValueError("propagated_orbitals must have shape (n_basis, n_orbitals)")
    occupations = np.asarray(orbital_occupations, dtype=np.float64)
    if occupations.ndim != 1 or occupations.shape[0] != psi.shape[1]:
        raise ValueError("orbital_occupations must match the propagated orbital count")
    if not np.all(np.isfinite(occupations)) or np.any(occupations < 0.0):
        raise ValueError("orbital_occupations must be finite and non-negative")
    if check_orthonormal:
        _check_orthonormal(psi, atol=atol, name="propagated_orbitals")

    gamma = (psi * occupations[np.newaxis, :]) @ psi.conj().T
    return instantaneous_occupations(
        gamma,
        instantaneous_orbitals,
        check_orthonormal=check_orthonormal,
        atol=atol,
    )
