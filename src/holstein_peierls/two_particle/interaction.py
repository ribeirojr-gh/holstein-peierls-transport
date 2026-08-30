"""Two-particle repulsive interaction models on the periodic molecular lattice."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .parameters import BipolaronParameters

FloatArray = NDArray[np.float64]


def minimum_image_offsets(
    parameters: BipolaronParameters,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return absolute minimum-image x/y separations in lattice-site units."""
    indices = np.arange(parameters.n_sites, dtype=np.int64)
    y = indices // parameters.nx
    x = indices % parameters.nx
    dx_raw = np.abs(x[:, None] - x[None, :])
    dy_raw = np.abs(y[:, None] - y[None, :])
    dx = np.minimum(dx_raw, parameters.nx - dx_raw)
    dy = np.minimum(dy_raw, parameters.ny - dy_raw)
    return dx.astype(np.int64), dy.astype(np.int64)


def pair_interaction_matrix(parameters: BipolaronParameters) -> FloatArray:
    """Return V_ij for onsite U plus nearest-neighbour V1.

    The ordered-pair basis uses one interaction value per configuration
    ``|i,j>``. ``hubbard_u`` acts only for ``i == j``. A positive
    ``nearest_neighbor_v`` acts on the four minimum-image nearest-neighbour
    configurations with squared lattice distance one. No material-specific
    long-range Coulomb geometry is assumed at this stage.
    """
    interaction = np.zeros(
        (parameters.n_sites, parameters.n_sites), dtype=np.float64
    )
    if parameters.hubbard_u != 0.0:
        np.fill_diagonal(interaction, parameters.hubbard_u)
    if parameters.nearest_neighbor_v != 0.0:
        dx, dy = minimum_image_offsets(parameters)
        nearest = ((dx == 1) & (dy == 0)) | ((dx == 0) & (dy == 1))
        interaction[nearest] += parameters.nearest_neighbor_v
    return interaction


def interaction_expectation(
    wavefunction: FloatArray,
    parameters: BipolaronParameters,
) -> float:
    """Return <Psi|V|Psi> for a normalized ordered-pair wavefunction."""
    psi = np.asarray(wavefunction, dtype=np.float64)
    expected = (parameters.n_sites, parameters.n_sites)
    if psi.shape != expected:
        raise ValueError("wavefunction shape does not match bipolaron parameters")
    return float(np.sum(np.square(psi) * pair_interaction_matrix(parameters)))
