"""Observables for a distinguishable electron-hole exciton on a periodic lattice."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .interaction import minimum_image_offsets, minimum_image_distances_angstrom
from .parameters import ExcitonParameters
from .solver import ExcitonGroundState


@dataclass(frozen=True, slots=True)
class ExcitonObservables:
    """Localization and relative-coordinate observables for one exciton state."""

    electron_ipr: float
    hole_ipr: float
    onsite_probability: float
    mean_separation_sites: float
    rms_separation_sites: float
    mean_separation_angstrom: float | None
    rms_separation_angstrom: float | None


def exciton_observables(
    state: ExcitonGroundState,
    parameters: ExcitonParameters,
) -> ExcitonObservables:
    """Evaluate localization and e-h separation without assigning a material."""
    probability = state.probability
    expected = (parameters.n_sites, parameters.n_sites)
    if probability.shape != expected:
        raise ValueError("ground-state shape does not match exciton parameters")

    dx, dy = minimum_image_offsets(parameters)
    distance_sites = np.sqrt(
        np.square(dx.astype(np.float64)) + np.square(dy.astype(np.float64))
    )
    mean_sites = float(np.sum(probability * distance_sites))
    rms_sites = float(np.sqrt(np.sum(probability * np.square(distance_sites))))

    mean_angstrom: float | None = None
    rms_angstrom: float | None = None
    if parameters.long_range_coulomb:
        distance_angstrom = minimum_image_distances_angstrom(parameters)
        mean_angstrom = float(np.sum(probability * distance_angstrom))
        rms_angstrom = float(
            np.sqrt(np.sum(probability * np.square(distance_angstrom)))
        )

    return ExcitonObservables(
        electron_ipr=state.electron_ipr,
        hole_ipr=state.hole_ipr,
        onsite_probability=state.onsite_probability,
        mean_separation_sites=mean_sites,
        rms_separation_sites=rms_sites,
        mean_separation_angstrom=mean_angstrom,
        rms_separation_angstrom=rms_angstrom,
    )
