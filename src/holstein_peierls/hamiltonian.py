"""Electronic Hamiltonian for the static two-dimensional Holstein-Peierls model."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix

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


def _site_neighbours(parameters: StaticPolaronParameters) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Return flattened site, +x-neighbour, and +y-neighbour index arrays."""
    sites = np.arange(parameters.n_sites, dtype=np.int64).reshape(
        parameters.ny, parameters.nx
    )
    right = np.roll(sites, shift=-1, axis=1)
    down = np.roll(sites, shift=-1, axis=0)
    return sites.ravel(), right.ravel(), down.ravel()


def build_dense_hamiltonian(
    state: LatticeState, parameters: StaticPolaronParameters
) -> FloatArray:
    """Construct the real symmetric Hamiltonian used by ``rprop.f90``.

    The implementation is vectorized but preserves the same matrix elements and
    C-order site mapping as the archived Fortran program.
    """
    ny, nx = state.shape
    if (ny, nx) != (parameters.ny, parameters.nx):
        raise ValueError("lattice shape does not match parameters")

    n = parameters.n_sites
    hamiltonian = np.zeros((n, n), dtype=np.float64)
    diagonal = parameters.alpha_intra * state.u.reshape(-1, order="C")
    np.fill_diagonal(hamiltonian, diagonal)

    tx, ty = bond_transfer_integrals(state, parameters)
    sites, right, down = _site_neighbours(parameters)
    tx_flat = tx.ravel(order="C")
    ty_flat = ty.ravel(order="C")

    hamiltonian[sites, right] = tx_flat
    hamiltonian[right, sites] = tx_flat
    hamiltonian[sites, down] = ty_flat
    hamiltonian[down, sites] = ty_flat
    return hamiltonian


def build_sparse_hamiltonian(
    state: LatticeState, parameters: StaticPolaronParameters
) -> csr_matrix:
    """Construct the Hamiltonian directly in sparse CSR form.

    For production lattice sizes this avoids the previous dense ``N x N``
    allocation followed by dense-to-CSR conversion. Very small two-site
    periodic dimensions fall back to the dense reference path because the
    archived assignment semantics overwrite duplicate periodic bonds rather
    than summing them.
    """
    ny, nx = state.shape
    if (ny, nx) != (parameters.ny, parameters.nx):
        raise ValueError("lattice shape does not match parameters")
    if nx < 3 or ny < 3:
        return csr_matrix(build_dense_hamiltonian(state, parameters))

    sites, right, down = _site_neighbours(parameters)
    tx, ty = bond_transfer_integrals(state, parameters)
    diagonal = parameters.alpha_intra * state.u.ravel(order="C")
    tx_flat = tx.ravel(order="C")
    ty_flat = ty.ravel(order="C")

    rows = np.concatenate((sites, sites, right, sites, down))
    cols = np.concatenate((sites, right, sites, down, sites))
    data = np.concatenate((diagonal, tx_flat, tx_flat, ty_flat, ty_flat))

    return coo_matrix(
        (data, (rows, cols)), shape=(parameters.n_sites, parameters.n_sites)
    ).tocsr()
