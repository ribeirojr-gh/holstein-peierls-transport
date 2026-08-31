"""Two-particle repulsive interaction models on the periodic molecular lattice."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from .parameters import BipolaronParameters

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

# e^2 / (4 pi epsilon_0), expressed in eV angstrom.
COULOMB_PREFACTOR_EV_ANGSTROM = 14.3996454784255


@lru_cache(maxsize=16)
def _minimum_image_offsets_for_shape(nx: int, ny: int) -> tuple[IntArray, IntArray]:
    indices = np.arange(nx * ny, dtype=np.int64)
    y = indices // nx
    x = indices % nx
    dx_raw = np.abs(x[:, None] - x[None, :])
    dy_raw = np.abs(y[:, None] - y[None, :])
    dx = np.minimum(dx_raw, nx - dx_raw).astype(np.int64)
    dy = np.minimum(dy_raw, ny - dy_raw).astype(np.int64)
    dx.setflags(write=False)
    dy.setflags(write=False)
    return dx, dy


def minimum_image_offsets(
    parameters: BipolaronParameters,
) -> tuple[IntArray, IntArray]:
    """Return cached absolute minimum-image x/y separations in site units."""
    return _minimum_image_offsets_for_shape(parameters.nx, parameters.ny)


@lru_cache(maxsize=16)
def _minimum_image_distances_for_geometry(
    nx: int,
    ny: int,
    spacing_x: float,
    spacing_y: float,
) -> FloatArray:
    dx, dy = _minimum_image_offsets_for_shape(nx, ny)
    rx = dx.astype(np.float64) * spacing_x
    ry = dy.astype(np.float64) * spacing_y
    distance = np.sqrt(np.square(rx) + np.square(ry))
    distance.setflags(write=False)
    return distance


def minimum_image_distances_angstrom(parameters: BipolaronParameters) -> FloatArray:
    """Return cached physical minimum-image pair distances in angstrom.

    This helper is defined only when the long-range Coulomb model is enabled,
    because otherwise the effective lattice has no material-specific physical
    spacing attached to it.
    """
    if not parameters.long_range_coulomb:
        raise ValueError("physical pair distances require long_range_coulomb=True")
    assert parameters.lattice_spacing_x_angstrom is not None
    assert parameters.lattice_spacing_y_angstrom is not None
    return _minimum_image_distances_for_geometry(
        parameters.nx,
        parameters.ny,
        parameters.lattice_spacing_x_angstrom,
        parameters.lattice_spacing_y_angstrom,
    )


@lru_cache(maxsize=4)
def pair_interaction_matrix(parameters: BipolaronParameters) -> FloatArray:
    """Return the cached static two-carrier interaction matrix ``V_ij``.

    The ordered-pair basis uses one interaction value per configuration
    ``|i,j>``.

    - ``hubbard_u`` acts only for ``i == j``.
    - Without long-range Coulomb, ``nearest_neighbor_v`` is the validated
      extended-Hubbard nearest-neighbour repulsion.
    - With long-range Coulomb enabled, all distinct-site pairs first receive
      ``e^2 / (4 pi epsilon_0 epsilon_r r_ij)`` using physical minimum-image
      distances on the finite periodic cell.
    - If a nonzero ``nearest_neighbor_v`` is also supplied in long-range mode,
      it replaces the continuum value on the four cardinal nearest neighbours.
      It is never added to the continuum tail, avoiding implicit double
      counting of short-range screening.

    The returned array is read-only so it can safely be reused across repeated
    eigensolver calls during lattice relaxation.

    The long-range implementation is deliberately a minimum-image finite-cell
    model, not an Ewald sum. A periodic Ewald treatment of two like charges
    requires an explicit neutralizing convention and is a separate physical
    model.
    """
    interaction = np.zeros(
        (parameters.n_sites, parameters.n_sites), dtype=np.float64
    )

    if parameters.long_range_coulomb:
        assert parameters.relative_permittivity is not None
        distance = minimum_image_distances_angstrom(parameters)
        offsite = distance > 0.0
        interaction[offsite] = (
            COULOMB_PREFACTOR_EV_ANGSTROM
            / parameters.relative_permittivity
            / distance[offsite]
        )

    if parameters.nearest_neighbor_v != 0.0:
        dx, dy = minimum_image_offsets(parameters)
        nearest = ((dx == 1) & (dy == 0)) | ((dx == 0) & (dy == 1))
        if parameters.long_range_coulomb:
            interaction[nearest] = parameters.nearest_neighbor_v
        else:
            interaction[nearest] += parameters.nearest_neighbor_v

    if parameters.hubbard_u != 0.0:
        np.fill_diagonal(interaction, parameters.hubbard_u)

    interaction.setflags(write=False)
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
