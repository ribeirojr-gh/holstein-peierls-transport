"""Collective-inertia and electronic-gap diagnostics for polaron translation.

IP0a showed that the frozen adiabatic translation barrier does not grow toward
isotropy.  IP0b therefore quantifies two dynamical ingredients that can suppress
translation even on a shallow potential-energy surface:

1. the mass-weighted lattice displacement required to translate the complete
   distortion by one molecular site; and
2. the lowest electronic splitting at the symmetric midpoint of the translation
   path.

The mass-weighted metric is a reaction-coordinate inertia for the chosen path,
not a complete band-polaron effective mass.  The midpoint gap is an adiabatic
low-state splitting diagnostic, not a nonadiabatic hopping rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import eigh

from .dynamics.ehrenfest import lattice_masses_fs
from .hamiltonian import build_dense_hamiltonian
from .lattice import LatticeState
from .parameters import StaticPolaronParameters
from .translation_barrier import (
    Direction,
    interpolate_lattice_states,
    translate_lattice_state,
    translated_site_index,
)

FloatArray = NDArray[np.float64]
EndpointSide = Literal["start", "end"]


@dataclass(frozen=True, slots=True)
class TranslationMassMetric:
    """Mass-weighted squared displacement along a one-site translation path.

    Each component has units of eV fs^2 because a lattice mass in
    eV fs^2/A^2 is multiplied by a squared displacement in A^2.  If the
    dimensionless path coordinate is ``s`` with ``q(s)=q0+s*dq``, this total is
    the generalized mass associated with ``s``.
    """

    intramolecular_eV_fs2: float
    vx_eV_fs2: float
    vy_eV_fs2: float

    @property
    def intermolecular_eV_fs2(self) -> float:
        return float(self.vx_eV_fs2 + self.vy_eV_fs2)

    @property
    def total_eV_fs2(self) -> float:
        return float(
            self.intramolecular_eV_fs2 + self.vx_eV_fs2 + self.vy_eV_fs2
        )


@dataclass(frozen=True, slots=True)
class MidpointElectronicSpectrum:
    """Lowest electronic levels at the symmetric frozen-path midpoint."""

    eigenvalues_eV: tuple[float, ...]
    gaps_from_ground_eV: tuple[float, ...]
    first_gap_eV: float
    source_population_ground: float
    target_population_ground: float
    source_target_population_difference: float


def mass_weighted_translation_metric(
    start: LatticeState,
    end: LatticeState,
    parameters: StaticPolaronParameters,
) -> TranslationMassMetric:
    """Return the lattice reaction-coordinate inertia between two states."""
    start.validate()
    end.validate()
    expected = (parameters.ny, parameters.nx)
    if start.shape != expected or end.shape != expected:
        raise ValueError("lattice shape does not match parameters")

    mass_u, mass_v = lattice_masses_fs(parameters)
    du = np.asarray(end.u - start.u, dtype=np.float64)
    dvx = np.asarray(end.vx - start.vx, dtype=np.float64)
    dvy = np.asarray(end.vy - start.vy, dtype=np.float64)
    return TranslationMassMetric(
        intramolecular_eV_fs2=float(mass_u * np.sum(du * du)),
        vx_eV_fs2=float(mass_v * np.sum(dvx * dvx)),
        vy_eV_fs2=float(mass_v * np.sum(dvy * dvy)),
    )


def one_site_translation_mass_metric(
    state: LatticeState,
    parameters: StaticPolaronParameters,
    direction: Direction,
) -> TranslationMassMetric:
    """Return the mass metric from a state to its exact one-site translation."""
    return mass_weighted_translation_metric(
        state,
        translate_lattice_state(state, direction),
        parameters,
    )


def quadratic_endpoint_curvature(
    fractions: ArrayLike,
    energies_eV: ArrayLike,
    *,
    side: EndpointSide = "start",
    fit_points: int = 5,
) -> float:
    """Estimate ``d^2E/ds^2`` at a path endpoint by a local quadratic fit.

    The reaction coordinate ``s`` is dimensionless.  The returned curvature is
    therefore in eV.  This is a diagnostic of the chosen path, not a Hessian
    normal-mode eigenvalue.
    """
    s = np.asarray(fractions, dtype=np.float64).reshape(-1)
    e = np.asarray(energies_eV, dtype=np.float64).reshape(-1)
    if s.size != e.size or s.size < 3:
        raise ValueError("fractions and energies must have the same length >= 3")
    if not np.all(np.isfinite(s)) or not np.all(np.isfinite(e)):
        raise ValueError("fractions and energies must be finite")
    count = int(fit_points)
    if count < 3 or count > s.size:
        raise ValueError("fit_points must lie between 3 and the number of samples")
    if np.any(np.diff(s) <= 0.0):
        raise ValueError("fractions must be strictly increasing")

    if side == "start":
        x = s[:count] - s[0]
        y = e[:count] - e[0]
    elif side == "end":
        x = s[-1] - s[-count:]
        y = e[-count:] - e[-1]
        order = np.argsort(x)
        x = x[order]
        y = y[order]
    else:
        raise ValueError("side must be 'start' or 'end'")

    coefficient = np.polyfit(x, y, deg=2)
    return float(2.0 * coefficient[0])


def collective_angular_frequency_per_fs(
    curvature_eV: float,
    mass_metric_eV_fs2: float,
) -> float:
    """Return ``sqrt(curvature/mass_metric)`` in fs^-1 for a stable endpoint."""
    curvature = float(curvature_eV)
    metric = float(mass_metric_eV_fs2)
    if not np.isfinite(curvature) or curvature <= 0.0:
        raise ValueError("curvature must be finite and positive")
    if not np.isfinite(metric) or metric <= 0.0:
        raise ValueError("mass metric must be finite and positive")
    return float(np.sqrt(curvature / metric))


def midpoint_low_state_spectrum(
    relaxed_state: LatticeState,
    parameters: StaticPolaronParameters,
    direction: Direction,
    *,
    source_site: int | None = None,
    level_count: int = 4,
) -> MidpointElectronicSpectrum:
    """Diagonalize the lowest electronic levels at the symmetric path midpoint."""
    relaxed_state.validate()
    if relaxed_state.shape != (parameters.ny, parameters.nx):
        raise ValueError("lattice shape does not match parameters")
    count = int(level_count)
    if count < 2 or count > parameters.n_sites:
        raise ValueError("level_count must lie between 2 and n_sites")

    translated = translate_lattice_state(relaxed_state, direction)
    midpoint = interpolate_lattice_states(relaxed_state, translated, 0.5)
    hamiltonian = build_dense_hamiltonian(midpoint, parameters)
    values, vectors = eigh(
        hamiltonian,
        subset_by_index=(0, count - 1),
        driver="evr",
        overwrite_a=False,
        check_finite=False,
    )
    values = np.asarray(values, dtype=np.float64)
    ground = np.asarray(vectors[:, 0], dtype=np.float64)
    population = np.square(ground)

    source = parameters.polaron_index if source_site is None else int(source_site)
    if not 0 <= source < parameters.n_sites:
        raise ValueError("source_site is outside the lattice")
    target = translated_site_index(source, parameters, direction)
    gaps = values - values[0]
    return MidpointElectronicSpectrum(
        eigenvalues_eV=tuple(float(value) for value in values),
        gaps_from_ground_eV=tuple(float(value) for value in gaps),
        first_gap_eV=float(gaps[1]),
        source_population_ground=float(population[source]),
        target_population_ground=float(population[target]),
        source_target_population_difference=float(population[source] - population[target]),
    )
