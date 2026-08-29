"""Electronic Hamiltonian for the static two-dimensional Holstein-Peierls model."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from .lattice import LatticeState
from .parameters import StaticPolaronParameters

FloatArray = NDArray[np.float64]


def bond_transfer_integrals(
    state: LatticeState, parameters: StaticPolaronParameters
) -> tuple[FloatArray, FloatArray]:
    """Return transfer integrals for bonds to +x and +y neighbours."""
    state.validate()
    vx_right = np.roll(state.vx, shift=-1, axis=1)
    vy_down = np.roll(state.vy, shift=-1, axis=0)
    tx = -parameters.j0x + parameters.alpha_interx * (vx_right - state.vx)
    ty = -parameters.j0y + parameters.alpha_intery * (vy_down - state.vy)
    return tx, ty


def build_dense_hamiltonian(
    state: LatticeState, parameters: StaticPolaronParameters
) -> FloatArray:
    """Construct the real symmetric Hamiltonian used by ``rprop.f90``."""
    ny, nx = state.shape
    if (ny, nx) != (parameters.ny, parameters.nx):
        raise ValueError("lattice shape does not match parameters")

    n = parameters.n_sites
    hamiltonian = np.zeros((n, n), dtype=np.float64)
    diagonal = parameters.alpha_intra * state.u.reshape(-1, order="C")
    np.fill_diagonal(hamiltonian, diagonal)

    tx, ty = bond_transfer_integrals(state, parameters)
    for y in range(ny):
        for x in range(nx):
            i = x + nx * y
            right = ((x + 1) % nx) + nx * y
            down = x + nx * ((y + 1) % ny)
            hamiltonian[i, right] = tx[y, x]
            hamiltonian[right, i] = tx[y, x]
            hamiltonian[i, down] = ty[y, x]
            hamiltonian[down, i] = ty[y, x]

    return hamiltonian


def build_sparse_hamiltonian(
    state: LatticeState, parameters: StaticPolaronParameters
) -> csr_matrix:
    """Construct a CSR Hamiltonian with the same matrix elements as the legacy code."""
    return csr_matrix(build_dense_hamiltonian(state, parameters))
