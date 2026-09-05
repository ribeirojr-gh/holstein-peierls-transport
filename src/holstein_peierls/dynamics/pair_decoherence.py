"""Instantaneous-decoherence controls for D6 pair dynamics.

D6d demonstrated electronic overheating in coherent finite-temperature pair
Ehrenfest trajectories.  D6e therefore extends the auditable D5 IDC controls to
the pair sectors, while respecting their different physical Hilbert spaces:

* bipolaron: symmetric spatial singlet sector only;
* distinguishable electron-hole exciton: full ordered pair sector.

The collapse weights use the same DP/BM/MA definitions validated in D5.  This
module does not assume that the one-polaron D5 scheme or decoherence interval is
transferable to pair dynamics; D6e benchmarks them again in the present model.

At an IDC event the lattice coordinates and velocities are fixed.  The
resulting electronic energy jump is returned explicitly as energy exchanged
with the added electronic environment.  No hidden velocity rescaling is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray

from .electronic_decoherence import (
    IDCScheme,
    adiabatic_populations,
    idc_collapse_probabilities,
)
from .pair_coupled import MovingPairHamiltonianFactory, PairParameters
from .pair_frozen import PairSector, normalized_pair_state
from .pair_thermalization import instantaneous_pair_spectrum, symmetric_pair_basis
from ..lattice import LatticeState

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PairInstantaneousDecoherenceEvent:
    """One stochastic collapse in the instantaneous physical pair eigenbasis."""

    sector: PairSector
    electronic_state: ComplexArray
    selected_state_index: int
    physical_sector_dimension: int
    adiabatic_energies_eV: FloatArray
    adiabatic_populations_before: FloatArray
    collapse_probabilities: FloatArray
    electronic_energy_before_eV: float
    electronic_energy_after_eV: float
    electronic_environment_exchange_eV: float
    scheme: IDCScheme


def _reconstruct_ordered_state(
    sector: PairSector,
    sector_state: ComplexArray,
    n_sites: int,
) -> ComplexArray:
    """Return a normalized ordered-basis vector from one physical-sector state."""
    coordinates = np.asarray(sector_state, dtype=np.complex128).reshape(-1)
    if sector == "bipolaron":
        basis = symmetric_pair_basis(n_sites)
        if coordinates.size != basis.shape[1]:
            raise ValueError("bipolaron sector state has the wrong dimension")
        ordered = basis @ coordinates
    elif sector == "exciton":
        if coordinates.size != n_sites * n_sites:
            raise ValueError("exciton sector state has the wrong dimension")
        ordered = coordinates
    else:
        raise ValueError("sector must be 'bipolaron' or 'exciton'")
    return normalized_pair_state(ordered, n_sites)


def apply_pair_instantaneous_decoherence(
    lattice: LatticeState,
    sector: PairSector,
    parameters: PairParameters,
    electronic_state: ArrayLike,
    temperature_K: float,
    scheme: IDCScheme,
    rng: Generator,
    *,
    factory: MovingPairHamiltonianFactory | None = None,
) -> PairInstantaneousDecoherenceEvent:
    """Collapse a pair state to one instantaneous physical-sector eigenstate."""
    if not isinstance(rng, Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if factory is None:
        factory = MovingPairHamiltonianFactory(sector, parameters)

    spectrum = instantaneous_pair_spectrum(
        lattice,
        sector,
        parameters,
        electronic_state,
        factory=factory,
    )
    population = adiabatic_populations(
        spectrum.eigenvectors,
        spectrum.state_coordinates,
    )
    probabilities = idc_collapse_probabilities(
        spectrum.eigenvalues_eV,
        population,
        temperature_K,
        scheme,
    )
    selected = int(rng.choice(spectrum.physical_sector_dimension, p=probabilities))
    selected_sector_state = np.asarray(
        spectrum.eigenvectors[:, selected], dtype=np.complex128
    )
    collapsed = _reconstruct_ordered_state(
        sector,
        selected_sector_state,
        parameters.n_sites,
    )

    before = float(np.dot(population, spectrum.eigenvalues_eV))
    after = float(spectrum.eigenvalues_eV[selected])
    return PairInstantaneousDecoherenceEvent(
        sector=sector,
        electronic_state=collapsed,
        selected_state_index=selected,
        physical_sector_dimension=spectrum.physical_sector_dimension,
        adiabatic_energies_eV=np.asarray(spectrum.eigenvalues_eV, dtype=np.float64),
        adiabatic_populations_before=np.asarray(population, dtype=np.float64),
        collapse_probabilities=np.asarray(probabilities, dtype=np.float64),
        electronic_energy_before_eV=before,
        electronic_energy_after_eV=after,
        electronic_environment_exchange_eV=float(after - before),
        scheme=scheme,
    )
