"""Attractive electron-hole interaction models on the periodic reference lattice."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from ..two_particle.interaction import COULOMB_PREFACTOR_EV_ANGSTROM
from .parameters import ExcitonParameters

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@lru_cache(maxsize=16)
def _minimum_image_offsets(nx: int, ny: int) -> tuple[IntArray, IntArray]:
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


def minimum_image_offsets(parameters: ExcitonParameters) -> tuple[IntArray, IntArray]:
    """Return absolute minimum-image x/y separations in lattice-site units."""
    return _minimum_image_offsets(parameters.nx, parameters.ny)


@lru_cache(maxsize=16)
def _minimum_image_distances(nx: int, ny: int, spacing_x: float, spacing_y: float) -> FloatArray:
    dx, dy = _minimum_image_offsets(nx, ny)
    rx = dx.astype(np.float64) * spacing_x
    ry = dy.astype(np.float64) * spacing_y
    distances = np.sqrt(np.square(rx) + np.square(ry))
    distances.setflags(write=False)
    return distances


def minimum_image_distances_angstrom(parameters: ExcitonParameters) -> FloatArray:
    """Return physical e-h distances when a physical lattice spacing is defined."""
    if not parameters.long_range_coulomb:
        raise ValueError("physical pair distances require long_range_coulomb=True")
    assert parameters.lattice_spacing_x_angstrom is not None
    assert parameters.lattice_spacing_y_angstrom is not None
    return _minimum_image_distances(
        parameters.nx,
        parameters.ny,
        parameters.lattice_spacing_x_angstrom,
        parameters.lattice_spacing_y_angstrom,
    )


@lru_cache(maxsize=8)
def electron_hole_interaction_matrix(parameters: ExcitonParameters) -> FloatArray:
    """Return the non-positive static electron-hole interaction matrix."""
    interaction = np.zeros((parameters.n_sites, parameters.n_sites), dtype=np.float64)
    dx: IntArray | None = None
    dy: IntArray | None = None

    if parameters.long_range_coulomb:
        assert parameters.relative_permittivity is not None
        distance = minimum_image_distances_angstrom(parameters)
        offsite = distance > 0.0
        interaction[offsite] = -(
            COULOMB_PREFACTOR_EV_ANGSTROM
            / parameters.relative_permittivity
            / distance[offsite]
        )

    if parameters.nearest_neighbor_attraction != 0.0:
        dx, dy = minimum_image_offsets(parameters)
        nearest = ((dx == 1) & (dy == 0)) | ((dx == 0) & (dy == 1))
        interaction[nearest] = -parameters.nearest_neighbor_attraction

    if parameters.short_range_shell_attractions:
        if dx is None or dy is None:
            dx, dy = minimum_image_offsets(parameters)
        for shell_dx, shell_dy, magnitude in parameters.short_range_shell_attractions:
            shell = (dx == shell_dx) & (dy == shell_dy)
            interaction[shell] = -magnitude

    if parameters.onsite_attraction != 0.0:
        np.fill_diagonal(interaction, -parameters.onsite_attraction)

    interaction.setflags(write=False)
    return interaction


def interaction_expectation(wavefunction: FloatArray, parameters: ExcitonParameters) -> float:
    """Return the attractive interaction expectation for a normalized e-h state."""
    psi = np.asarray(wavefunction, dtype=np.float64)
    if psi.shape != (parameters.n_sites, parameters.n_sites):
        raise ValueError("wavefunction shape does not match exciton parameters")
    return float(np.sum(np.square(psi) * electron_hole_interaction_matrix(parameters)))
