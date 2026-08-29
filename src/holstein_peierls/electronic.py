"""Electronic ground-state solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

from .hamiltonian import build_dense_hamiltonian, build_sparse_hamiltonian
from .lattice import LatticeState
from .parameters import StaticPolaronParameters

FloatArray = NDArray[np.float64]
SolverName = Literal["dense_full", "dense_lowest", "sparse"]


@dataclass(frozen=True, slots=True)
class GroundState:
    """Lowest electronic eigenstate for a fixed lattice configuration."""
    energy: float
    wavefunction: FloatArray

    @property
    def charge_density(self) -> FloatArray:
        return np.square(self.wavefunction)


def solve_ground_state(
    state: LatticeState,
    parameters: StaticPolaronParameters,
    *,
    solver: SolverName = "dense_lowest",
) -> GroundState:
    """Solve the electronic ground state using a selected CPU eigensolver."""
    if solver == "dense_full":
        hamiltonian = build_dense_hamiltonian(state, parameters)
        eigenvalues, eigenvectors = eigh(
            hamiltonian, driver="evd", overwrite_a=False, check_finite=False
        )
        energy = float(eigenvalues[0])
        wavefunction = np.asarray(eigenvectors[:, 0], dtype=np.float64)
    elif solver == "dense_lowest":
        hamiltonian = build_dense_hamiltonian(state, parameters)
        eigenvalues, eigenvectors = eigh(
            hamiltonian,
            subset_by_index=(0, 0),
            driver="evr",
            overwrite_a=False,
            check_finite=False,
        )
        energy = float(eigenvalues[0])
        wavefunction = np.asarray(eigenvectors[:, 0], dtype=np.float64)
    elif solver == "sparse":
        hamiltonian = build_sparse_hamiltonian(state, parameters)
        initial_vector = np.full(parameters.n_sites, 1.0 / np.sqrt(parameters.n_sites))
        eigenvalues, eigenvectors = eigsh(
            hamiltonian,
            k=1,
            which="SA",
            tol=1.0e-12,
            maxiter=10_000,
            v0=initial_vector,
        )
        energy = float(eigenvalues[0])
        wavefunction = np.asarray(eigenvectors[:, 0], dtype=np.float64)
    else:
        raise ValueError(f"unknown electronic solver: {solver}")

    anchor = int(np.argmax(np.abs(wavefunction)))
    if wavefunction[anchor] < 0.0:
        wavefunction = -wavefunction
    return GroundState(energy=energy, wavefunction=wavefunction)
