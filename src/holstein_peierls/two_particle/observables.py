"""Observables for correlated two-particle states on a periodic 2D lattice."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .bipolaron import BipolaronGroundState
from .parameters import BipolaronParameters

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PairObservables:
    """Geometry-sensitive diagnostics of a normalized two-particle state.

    Distances are measured in lattice-site units using the minimum-image
    convention on the periodic rectangular lattice. The nearest-neighbour
    probability is additionally resolved into x and y contributions so that
    anisotropic molecular-crystal pair states can be identified.
    """

    onsite_probability: float
    nearest_neighbour_probability: float
    nearest_neighbour_x_probability: float
    nearest_neighbour_y_probability: float
    mean_separation: float
    rms_separation: float
    mean_dx: float
    mean_dy: float
    one_body_ipr: float
    radial_probability_by_r2: dict[int, float]


def periodic_pair_distance_matrices(
    parameters: BipolaronParameters,
) -> tuple[FloatArray, FloatArray, NDArray[np.int64]]:
    """Return minimum-image dx, dy, and squared-distance matrices."""
    n = parameters.n_sites
    indices = np.arange(n, dtype=np.int64)
    y = indices // parameters.nx
    x = indices % parameters.nx

    dx_raw = np.abs(x[:, None] - x[None, :])
    dy_raw = np.abs(y[:, None] - y[None, :])
    dx = np.minimum(dx_raw, parameters.nx - dx_raw).astype(np.float64)
    dy = np.minimum(dy_raw, parameters.ny - dy_raw).astype(np.float64)
    r2 = (dx.astype(np.int64) ** 2 + dy.astype(np.int64) ** 2).astype(np.int64)
    return dx, dy, r2


def pair_observables(
    ground_state: BipolaronGroundState,
    parameters: BipolaronParameters,
) -> PairObservables:
    """Calculate periodic pair-separation and localization observables."""
    probability = np.asarray(ground_state.probability, dtype=np.float64)
    expected_shape = (parameters.n_sites, parameters.n_sites)
    if probability.shape != expected_shape:
        raise ValueError("two-particle wavefunction shape does not match parameters")

    normalization = float(np.sum(probability))
    if not np.isclose(normalization, 1.0, atol=2.0e-10):
        raise ValueError("two-particle probability is not normalized")

    dx, dy, r2 = periodic_pair_distance_matrices(parameters)
    distance = np.sqrt(dx * dx + dy * dy)

    onsite_mask = r2 == 0
    nearest_x_mask = (dx == 1.0) & (dy == 0.0)
    nearest_y_mask = (dx == 0.0) & (dy == 1.0)

    onsite = float(np.sum(probability[onsite_mask]))
    nearest_x = float(np.sum(probability[nearest_x_mask]))
    nearest_y = float(np.sum(probability[nearest_y_mask]))
    nearest = nearest_x + nearest_y
    mean_dx = float(np.sum(probability * dx))
    mean_dy = float(np.sum(probability * dy))
    mean = float(np.sum(probability * distance))
    rms = float(np.sqrt(np.sum(probability * r2)))

    # Normalize the spin-summed site density to one before computing an IPR, so
    # it is directly comparable to the conventional one-particle IPR.
    normalized_density = ground_state.site_density / 2.0
    one_body_ipr = float(np.sum(np.square(normalized_density)))

    radial: dict[int, float] = {}
    for shell in np.unique(r2):
        radial[int(shell)] = float(np.sum(probability[r2 == shell]))

    return PairObservables(
        onsite_probability=onsite,
        nearest_neighbour_probability=nearest,
        nearest_neighbour_x_probability=nearest_x,
        nearest_neighbour_y_probability=nearest_y,
        mean_separation=mean,
        rms_separation=rms,
        mean_dx=mean_dx,
        mean_dy=mean_dy,
        one_body_ipr=one_body_ipr,
        radial_probability_by_r2=radial,
    )
