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


def _normalized_initial_vector(
    initial_wavefunction: FloatArray | None,
    n_sites: int,
) -> FloatArray:
    """Return a deterministic normalized starting vector for iterative solvers."""
    if initial_wavefunction is None:
        return np.full(n_sites, 1.0 / np.sqrt(n_sites), dtype=np.float64)

    vector = np.asarray(initial_wavefunction, dtype=np.float64).reshape(-1)
    if vector.size != n_sites:
        raise ValueError("initial wavefunction size does not match lattice")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("initial wavefunction must have a finite non-zero norm")
    return vector / norm


def solve_ground_state(
    state: LatticeState,
    parameters: StaticPolaronParameters,
    *,
    solver: SolverName = "dense_lowest",
    initial_wavefunction: FloatArray | None = None,
) -> GroundState:
    """Solve the electronic ground state using a selected CPU eigensolver.

    ``initial_wavefunction`` is used only by the iterative sparse solver. During
    RPROP relaxation, passing the previous electronic ground state provides an
    adiabatic warm start. This both reduces eigensolver work and prevents an
    arbitrary switch between translationally equivalent localized solutions on
    a periodic lattice.
    """
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
        initial_vector = _normalized_initial_vector(
            initial_wavefunction, parameters.n_sites
        )
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
