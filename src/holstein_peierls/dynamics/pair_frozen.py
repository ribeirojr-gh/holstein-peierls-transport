"""Complex-safe matrix-free frozen-H actions for D6 pair dynamics.

The static bipolaron and distinguishable-exciton solvers use real-valued
``LinearOperator`` objects because their adiabatic ground states can be chosen
real. Time propagation necessarily generates complex amplitudes, so D6 uses the
separate actions defined here rather than silently casting dynamic states to
real numbers.

The pair basis is always the ordered C-order product basis ``psi[i, j]``. For
bipolarons the physical spin-singlet spatial sector is the symmetric subspace;
for distinguishable electron-hole excitons no exchange symmetrization is
applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix

from ..exciton.interaction import electron_hole_interaction_matrix
from ..exciton.parameters import ExcitonParameters
from ..exciton.solver import build_carrier_hamiltonian
from ..lattice import LatticeState
from ..two_particle.interaction import pair_interaction_matrix
from ..two_particle.parameters import BipolaronParameters
from ..two_particle.peierls import build_one_particle_hamiltonian

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
PairSector = Literal["bipolaron", "exciton"]


@dataclass(slots=True)
class FrozenPairHamiltonianAction:
    """Matrix-free ``N^2`` Hamiltonian action with an application counter.

    ``left_hamiltonian`` acts on the first product-basis index and
    ``right_hamiltonian`` on the second. The interaction is diagonal in the
    ordered site-product basis.
    """

    left_hamiltonian: csr_matrix
    right_hamiltonian: csr_matrix
    interaction_eV: FloatArray
    sector: PairSector
    applications: int = 0

    def __post_init__(self) -> None:
        if self.left_hamiltonian.shape != self.right_hamiltonian.shape:
            raise ValueError("left and right one-particle Hamiltonians must have equal shape")
        if self.left_hamiltonian.shape[0] != self.left_hamiltonian.shape[1]:
            raise ValueError("one-particle Hamiltonians must be square")
        n = int(self.left_hamiltonian.shape[0])
        interaction = np.asarray(self.interaction_eV, dtype=np.float64)
        if interaction.shape != (n, n):
            raise ValueError("interaction matrix must have shape (n_sites, n_sites)")
        if not np.all(np.isfinite(interaction)):
            raise ValueError("interaction matrix must be finite")
        self.interaction_eV = interaction
        if self.sector not in ("bipolaron", "exciton"):
            raise ValueError("sector must be 'bipolaron' or 'exciton'")

    @property
    def n_sites(self) -> int:
        return int(self.left_hamiltonian.shape[0])

    @property
    def dimension(self) -> int:
        return self.n_sites * self.n_sites

    @property
    def shape(self) -> tuple[int, int]:
        return (self.dimension, self.dimension)

    def reset(self) -> None:
        self.applications = 0

    def __call__(self, state: ArrayLike) -> ComplexArray:
        vector = np.asarray(state, dtype=np.complex128).reshape(-1)
        if vector.size != self.dimension:
            raise ValueError("pair state dimension does not match Hamiltonian")
        if not np.all(np.isfinite(vector)):
            raise ValueError("pair state must be finite")
        psi = vector.reshape((self.n_sites, self.n_sites), order="C")
        left = self.left_hamiltonian @ psi
        right = (self.right_hamiltonian @ psi.T).T
        result = np.asarray(left + right, dtype=np.complex128)
        result += self.interaction_eV * psi
        self.applications += 1
        return np.asarray(result.ravel(order="C"), dtype=np.complex128)


def bipolaron_frozen_action(
    lattice: LatticeState,
    parameters: BipolaronParameters,
) -> FrozenPairHamiltonianAction:
    """Return the full Holstein-Peierls-Hubbard singlet-pair action."""
    h1 = build_one_particle_hamiltonian(lattice, parameters)
    return FrozenPairHamiltonianAction(
        left_hamiltonian=h1,
        right_hamiltonian=h1,
        interaction_eV=pair_interaction_matrix(parameters),
        sector="bipolaron",
    )


def exciton_frozen_action(
    lattice: LatticeState,
    parameters: ExcitonParameters,
) -> FrozenPairHamiltonianAction:
    """Return the distinguishable electron-hole frozen-H action."""
    return FrozenPairHamiltonianAction(
        left_hamiltonian=build_carrier_hamiltonian(lattice, parameters, "electron"),
        right_hamiltonian=build_carrier_hamiltonian(lattice, parameters, "hole"),
        interaction_eV=electron_hole_interaction_matrix(parameters),
        sector="exciton",
    )


def dense_hamiltonian_from_action(
    action: FrozenPairHamiltonianAction,
    *,
    maximum_dimension: int = 1024,
) -> ComplexArray:
    """Materialize a small pair Hamiltonian for validation only.

    Production D6 code must remain matrix-free. The dimension guard prevents an
    accidental ``N^4`` dense allocation for realistic lattices.
    """
    if action.dimension > maximum_dimension:
        raise ValueError(
            f"dense validation matrix disabled above dimension {maximum_dimension}; "
            f"received {action.dimension}"
        )
    before = action.applications
    matrix = np.empty((action.dimension, action.dimension), dtype=np.complex128)
    basis = np.zeros(action.dimension, dtype=np.complex128)
    for column in range(action.dimension):
        basis.fill(0.0)
        basis[column] = 1.0
        matrix[:, column] = action(basis)
    action.applications = before
    return matrix


def normalized_pair_state(state: ArrayLike, n_sites: int) -> ComplexArray:
    """Return a normalized flattened complex pair state without symmetrization."""
    vector = np.asarray(state, dtype=np.complex128).reshape(-1)
    if vector.size != n_sites * n_sites or not np.all(np.isfinite(vector)):
        raise ValueError("invalid pair-state shape or values")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("pair state must have finite non-zero norm")
    return np.asarray(vector / norm, dtype=np.complex128)


def symmetrized_bipolaron_state(state: ArrayLike, n_sites: int) -> ComplexArray:
    """Normalize and project a pair state onto the symmetric spatial sector."""
    vector = normalized_pair_state(state, n_sites)
    psi = vector.reshape((n_sites, n_sites), order="C")
    symmetric = 0.5 * (psi + psi.T)
    norm = float(np.linalg.norm(symmetric))
    if not np.isfinite(norm) or norm <= 1.0e-14:
        raise ValueError("bipolaron singlet projection produced a zero state")
    return np.asarray((symmetric / norm).ravel(order="C"), dtype=np.complex128)


def bipolaron_exchange_symmetry_error(state: ArrayLike, n_sites: int) -> float:
    """Return ``max|psi_ij-psi_ji|`` for a complex pair state."""
    psi = normalized_pair_state(state, n_sites).reshape((n_sites, n_sites), order="C")
    return float(np.max(np.abs(psi - psi.T)))


def bipolaron_one_body_density_matrix(
    state: ArrayLike,
    n_sites: int,
) -> ComplexArray:
    """Spin-summed bipolaron one-body RDM with trace two."""
    psi = normalized_pair_state(state, n_sites).reshape((n_sites, n_sites), order="C")
    return np.asarray(2.0 * (psi @ psi.conj().T), dtype=np.complex128)


def exciton_one_body_density_matrices(
    state: ArrayLike,
    n_sites: int,
) -> tuple[ComplexArray, ComplexArray]:
    """Return electron and hole RDMs, each normalized to trace one."""
    psi = normalized_pair_state(state, n_sites).reshape((n_sites, n_sites), order="C")
    gamma_e = psi @ psi.conj().T
    gamma_h = psi.T @ psi.conj()
    return (
        np.asarray(gamma_e, dtype=np.complex128),
        np.asarray(gamma_h, dtype=np.complex128),
    )


def electronic_energy_expectation(
    action: FrozenPairHamiltonianAction,
    state: ArrayLike,
) -> float:
    """Return the propagated-state electronic expectation value in eV."""
    psi = normalized_pair_state(state, action.n_sites)
    value = np.vdot(psi, action(psi))
    if abs(float(value.imag)) > 1.0e-10:
        raise FloatingPointError("pair Hamiltonian energy expectation is not real")
    return float(value.real)
