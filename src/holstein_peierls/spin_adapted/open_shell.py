"""General open-shell Hartree-Fock/MCTDHF static functional.

The formulas implemented here are the time-independent energy and shell-Fock
counterparts of the general open-shell formalism used by Miranda et al.,
J. Chem. Phys. 134, 244101 (2011) and 244102 (2011).  They are written for a
site-basis density-density interaction kernel ``V[i,j]``.  This is the natural
form for Hubbard/PPP-like lattice controls and provides a small, testable core
before orbital optimization and real-time propagation are introduced.

For shell ``mu`` with occupation number ``n_mu`` and projector ``P_mu``:

E = sum_mu n_mu Tr(P_mu T)
  + 1/4 sum_mu,nu n_mu n_nu [2 a_mu,nu J_mu,nu - b_mu,nu K_mu,nu]

where

J_mu,nu = sum_ij P_mu[ii] P_nu[jj] V[i,j]
K_mu,nu = sum_ij P_mu[i,j] P_nu[j,i] V[i,j].

The corresponding shell-dependent Fock matrix is

F_mu = T
     + sum_nu n_nu a_mu,nu diag(V @ diag(P_nu))
     - 1/2 sum_nu n_nu b_mu,nu (V * P_nu).

The interaction kernel is kept independent of the classical lattice in S0, in
line with the project's current frozen-distance Coulomb convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
Matrix = NDArray[np.float64] | NDArray[np.complex128]


@dataclass(frozen=True, slots=True)
class OpenShellStateDefinition:
    """Occupation and state-parameter matrices for occupied orbital shells."""

    name: str
    occupations: tuple[int, ...]
    a: tuple[tuple[float, ...], ...]
    b: tuple[tuple[float, ...], ...]
    spin_s: float

    def __post_init__(self) -> None:
        n_shells = len(self.occupations)
        if n_shells == 0:
            raise ValueError("at least one occupied shell is required")
        if any(value not in (1, 2) for value in self.occupations):
            raise ValueError("occupied-shell occupations must be 1 or 2")
        if len(self.a) != n_shells or any(len(row) != n_shells for row in self.a):
            raise ValueError("a must be square with one row per occupied shell")
        if len(self.b) != n_shells or any(len(row) != n_shells for row in self.b):
            raise ValueError("b must be square with one row per occupied shell")
        a = np.asarray(self.a, dtype=np.float64)
        b = np.asarray(self.b, dtype=np.float64)
        if not np.allclose(a, a.T):
            raise ValueError("a must be symmetric")
        if not np.allclose(b, b.T):
            raise ValueError("b must be symmetric")
        if self.spin_s < 0.0:
            raise ValueError("spin quantum number must be non-negative")

    @property
    def n_shells(self) -> int:
        return len(self.occupations)

    @property
    def s2(self) -> float:
        return self.spin_s * (self.spin_s + 1.0)

    @property
    def a_matrix(self) -> FloatArray:
        return np.asarray(self.a, dtype=np.float64)

    @property
    def b_matrix(self) -> FloatArray:
        return np.asarray(self.b, dtype=np.float64)


CLOSED_SHELL_SINGLET = OpenShellStateDefinition(
    name="closed_shell_singlet",
    occupations=(2,),
    a=((1.0,),),
    b=((1.0,),),
    spin_s=0.0,
)

OPEN_SHELL_SINGLET = OpenShellStateDefinition(
    name="open_shell_singlet",
    occupations=(2, 1, 1),
    a=(
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
    ),
    b=(
        (1.0, 1.0, 1.0),
        (1.0, 2.0, -2.0),
        (1.0, -2.0, 2.0),
    ),
    spin_s=0.0,
)

HIGH_SPIN_TRIPLET = OpenShellStateDefinition(
    name="high_spin_triplet",
    occupations=(2, 1),
    a=(
        (1.0, 1.0),
        (1.0, 1.0),
    ),
    b=(
        (1.0, 1.0),
        (1.0, 2.0),
    ),
    spin_s=1.0,
)


def _validate_inputs(
    one_body: Matrix,
    interaction: FloatArray,
    projectors: tuple[Matrix, ...],
    definition: OpenShellStateDefinition,
) -> tuple[Matrix, FloatArray, tuple[Matrix, ...]]:
    t = np.asarray(one_body)
    v = np.asarray(interaction, dtype=np.float64)
    if t.ndim != 2 or t.shape[0] != t.shape[1]:
        raise ValueError("one_body must be a square matrix")
    n = t.shape[0]
    if v.shape != (n, n):
        raise ValueError("interaction must match the one-body matrix shape")
    if not np.allclose(t, t.conj().T):
        raise ValueError("one_body must be Hermitian")
    if not np.allclose(v, v.T):
        raise ValueError("interaction must be symmetric")
    if len(projectors) != definition.n_shells:
        raise ValueError("one projector is required for each occupied shell")

    checked: list[Matrix] = []
    for projector in projectors:
        p = np.asarray(projector)
        if p.shape != (n, n):
            raise ValueError("every shell projector must match one_body")
        if not np.allclose(p, p.conj().T, atol=1.0e-11):
            raise ValueError("shell projectors must be Hermitian")
        if not np.allclose(p @ p, p, atol=1.0e-9):
            raise ValueError("shell projectors must be idempotent")
        checked.append(p)

    for mu in range(len(checked)):
        for nu in range(mu + 1, len(checked)):
            if not np.allclose(checked[mu] @ checked[nu], 0.0, atol=1.0e-9):
                raise ValueError("occupied shell projectors must be mutually orthogonal")
    return t, v, tuple(checked)


def projector_from_orbitals(orbitals: Matrix) -> Matrix:
    """Build an orthogonal projector from column-orthonormal orbitals."""
    c = np.asarray(orbitals)
    if c.ndim != 2:
        raise ValueError("orbitals must be a matrix with orbitals in columns")
    overlap = c.conj().T @ c
    if not np.allclose(overlap, np.eye(c.shape[1]), atol=1.0e-10):
        raise ValueError("orbital columns must be orthonormal")
    return c @ c.conj().T


def _coulomb_contraction(p_mu: Matrix, p_nu: Matrix, v: FloatArray) -> float:
    density_mu = np.real(np.diag(p_mu))
    density_nu = np.real(np.diag(p_nu))
    return float(density_mu @ v @ density_nu)


def _exchange_contraction(p_mu: Matrix, p_nu: Matrix, v: FloatArray) -> float:
    value = np.sum(p_mu * p_nu.T * v)
    return float(np.real_if_close(value))


def general_open_shell_energy(
    one_body: Matrix,
    interaction: FloatArray,
    projectors: tuple[Matrix, ...],
    definition: OpenShellStateDefinition,
) -> float:
    """Evaluate the general open-shell electronic energy functional."""
    t, v, ps = _validate_inputs(one_body, interaction, projectors, definition)
    occupations = np.asarray(definition.occupations, dtype=np.float64)
    a = definition.a_matrix
    b = definition.b_matrix

    one_electron = 0.0
    for occupation, projector in zip(occupations, ps, strict=True):
        one_electron += occupation * float(np.real(np.trace(projector @ t)))

    two_electron = 0.0
    for mu, p_mu in enumerate(ps):
        for nu, p_nu in enumerate(ps):
            coulomb = _coulomb_contraction(p_mu, p_nu, v)
            exchange = _exchange_contraction(p_mu, p_nu, v)
            two_electron += (
                0.25
                * occupations[mu]
                * occupations[nu]
                * (2.0 * a[mu, nu] * coulomb - b[mu, nu] * exchange)
            )
    return float(one_electron + two_electron)


def shell_fock_matrices(
    one_body: Matrix,
    interaction: FloatArray,
    projectors: tuple[Matrix, ...],
    definition: OpenShellStateDefinition,
) -> tuple[Matrix, ...]:
    """Construct the shell-dependent Fock matrices of the open-shell model."""
    t, v, ps = _validate_inputs(one_body, interaction, projectors, definition)
    occupations = np.asarray(definition.occupations, dtype=np.float64)
    a = definition.a_matrix
    b = definition.b_matrix

    fock_matrices: list[Matrix] = []
    for mu in range(definition.n_shells):
        fock = np.array(t, copy=True, dtype=np.result_type(t, np.complex128))
        for nu, p_nu in enumerate(ps):
            density_nu = np.real(np.diag(p_nu))
            coulomb_potential = v @ density_nu
            fock += occupations[nu] * a[mu, nu] * np.diag(coulomb_potential)
            fock -= (
                0.5
                * occupations[nu]
                * b[mu, nu]
                * (v * p_nu)
            )
        fock = 0.5 * (fock + fock.conj().T)
        if np.isrealobj(t) and all(np.isrealobj(p) for p in ps):
            fock = np.real(fock)
        fock_matrices.append(fock)
    return tuple(fock_matrices)
