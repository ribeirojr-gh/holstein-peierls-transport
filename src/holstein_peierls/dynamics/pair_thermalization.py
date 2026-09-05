"""D6d diagnostic-only electronic thermalization analysis for pair dynamics.

D6c validates the classical finite-temperature BAOAB lattice bath but makes no
claim that coherent pair Ehrenfest dynamics samples the correct electronic
thermal distribution.  This module measures that question without modifying the
propagated state.

The instantaneous adiabatic reference space is sector dependent:

* bipolaron: the symmetric spatial singlet sector only, dimension N(N+1)/2;
* distinguishable electron-hole exciton: the full ordered pair space, N**2.

Including antisymmetric spatial states in the bipolaron canonical reference
would mix a different spin sector and is therefore explicitly forbidden.

The reported adiabatic populations are state-projection probabilities in the
instantaneous many-body eigenbasis.  Their Shannon entropy is not the von
Neumann entropy of the pure propagated state and is not a transport observable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .electronic_thermalization import (
    canonical_occupations,
    distribution_energy_eV,
    effective_inverse_temperature_eV_inv,
    effective_temperature_K,
    jensen_shannon_distance,
    occupation_entropy,
    occupation_participation_number,
    total_variation_distance,
)
from .langevin import BOLTZMANN_EV_PER_K
from .pair_coupled import MovingPairHamiltonianFactory, PairParameters
from .pair_frozen import (
    PairSector,
    bipolaron_exchange_symmetry_error,
    dense_hamiltonian_from_action,
    normalized_pair_state,
)
from ..lattice import LatticeState

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True, slots=True)
class PairAdiabaticSpectrum:
    """Instantaneous pair spectrum and state coordinates in its physical sector."""

    sector: PairSector
    eigenvalues_eV: FloatArray
    eigenvectors: ComplexArray
    state_coordinates: ComplexArray
    full_ordered_dimension: int
    physical_sector_dimension: int


@dataclass(frozen=True, slots=True)
class PairThermalizationSnapshot:
    """Instantaneous D6d many-body electronic thermalization diagnostics."""

    sector: PairSector
    full_ordered_dimension: int
    physical_sector_dimension: int
    eigenvalues_eV: FloatArray
    adiabatic_occupations: FloatArray
    canonical_occupations: FloatArray
    uniform_occupations: FloatArray
    propagated_energy_eV: float
    canonical_energy_eV: float
    uniform_energy_eV: float
    heating_coordinate: float
    tv_to_canonical: float
    js_to_canonical: float
    tv_to_uniform: float
    normalized_occupation_entropy: float
    occupation_participation_number: float
    ground_manifold_size: int
    ground_manifold_population: float
    canonical_ground_manifold_population: float
    beta_eff_eV_inv: float
    beta_bath_eV_inv: float
    beta_eff_over_beta_bath: float
    effective_temperature_K: float


def symmetric_pair_basis(n_sites: int) -> ComplexArray:
    """Return an orthonormal ordered-basis representation of the symmetric sector.

    Columns are |ii> and (|ij>+|ji>)/sqrt(2) for i<j.  The ordered pair basis is
    flattened in C order, matching all D6 pair Hamiltonian actions.
    """
    n = int(n_sites)
    if n <= 0:
        raise ValueError("n_sites must be positive")
    dimension = n * (n + 1) // 2
    basis = np.zeros((n * n, dimension), dtype=np.complex128)
    column = 0
    for i in range(n):
        basis[i * n + i, column] = 1.0
        column += 1
        for j in range(i + 1, n):
            basis[i * n + j, column] = 1.0 / np.sqrt(2.0)
            basis[j * n + i, column] = 1.0 / np.sqrt(2.0)
            column += 1
    if column != dimension:
        raise RuntimeError("internal symmetric-basis dimension mismatch")
    return basis


def _restricted_symmetric_hamiltonian(action, basis: ComplexArray) -> ComplexArray:
    """Build B^dagger H B for validation-sized bipolaron diagnostics only."""
    if basis.shape[0] != action.dimension:
        raise ValueError("basis dimension does not match pair Hamiltonian")
    before = action.applications
    applied = np.column_stack([action(basis[:, index]) for index in range(basis.shape[1])])
    action.applications = before
    restricted = basis.conj().T @ applied
    if not np.allclose(restricted, restricted.conj().T, rtol=0.0, atol=2.0e-12):
        raise FloatingPointError("restricted bipolaron Hamiltonian is not Hermitian")
    return np.asarray(restricted, dtype=np.complex128)


def instantaneous_pair_spectrum(
    lattice: LatticeState,
    sector: PairSector,
    parameters: PairParameters,
    electronic_state: ArrayLike,
    *,
    factory: MovingPairHamiltonianFactory | None = None,
    bipolaron_symmetry_tolerance: float = 1.0e-9,
    maximum_exciton_dimension: int = 1024,
) -> PairAdiabaticSpectrum:
    """Diagonalize the instantaneous Hamiltonian in the correct physical sector."""
    lattice.validate()
    if lattice.shape != (parameters.ny, parameters.nx):
        raise ValueError("lattice shape does not match pair parameters")
    if factory is None:
        factory = MovingPairHamiltonianFactory(sector, parameters)
    state = normalized_pair_state(electronic_state, parameters.n_sites)
    action = factory.at(lattice)

    if sector == "bipolaron":
        symmetry_error = bipolaron_exchange_symmetry_error(state, parameters.n_sites)
        if symmetry_error > float(bipolaron_symmetry_tolerance):
            raise ValueError(
                "bipolaron thermalization diagnostics require the propagated "
                "state to remain in the symmetric spatial singlet sector"
            )
        basis = symmetric_pair_basis(parameters.n_sites)
        coordinates = basis.conj().T @ state
        sector_norm = float(np.linalg.norm(coordinates))
        if abs(sector_norm - 1.0) > 1.0e-8:
            raise ValueError("bipolaron state has weight outside the symmetric sector")
        coordinates = np.asarray(coordinates / sector_norm, dtype=np.complex128)
        matrix = _restricted_symmetric_hamiltonian(action, basis)
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        return PairAdiabaticSpectrum(
            sector="bipolaron",
            eigenvalues_eV=np.asarray(eigenvalues, dtype=np.float64),
            eigenvectors=np.asarray(eigenvectors, dtype=np.complex128),
            state_coordinates=coordinates,
            full_ordered_dimension=action.dimension,
            physical_sector_dimension=basis.shape[1],
        )

    if sector == "exciton":
        matrix = dense_hamiltonian_from_action(
            action, maximum_dimension=int(maximum_exciton_dimension)
        )
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        return PairAdiabaticSpectrum(
            sector="exciton",
            eigenvalues_eV=np.asarray(eigenvalues, dtype=np.float64),
            eigenvectors=np.asarray(eigenvectors, dtype=np.complex128),
            state_coordinates=state,
            full_ordered_dimension=action.dimension,
            physical_sector_dimension=action.dimension,
        )

    raise ValueError("sector must be 'bipolaron' or 'exciton'")


def diagnose_pair_thermalization(
    lattice: LatticeState,
    sector: PairSector,
    parameters: PairParameters,
    electronic_state: ArrayLike,
    temperature_K: float,
    *,
    factory: MovingPairHamiltonianFactory | None = None,
    ground_degeneracy_atol_eV: float = 1.0e-10,
) -> PairThermalizationSnapshot:
    """Compare one propagated pair state with canonical and uniform references."""
    temperature = float(temperature_K)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("D6d diagnostics require a positive finite temperature")

    spectrum = instantaneous_pair_spectrum(
        lattice,
        sector,
        parameters,
        electronic_state,
        factory=factory,
    )
    amplitudes = spectrum.eigenvectors.conj().T @ spectrum.state_coordinates
    adiabatic = np.asarray(np.abs(amplitudes) ** 2, dtype=np.float64)
    adiabatic /= float(np.sum(adiabatic))
    canonical = canonical_occupations(spectrum.eigenvalues_eV, temperature)
    uniform = np.full(
        spectrum.physical_sector_dimension,
        1.0 / spectrum.physical_sector_dimension,
        dtype=np.float64,
    )

    propagated_energy = distribution_energy_eV(spectrum.eigenvalues_eV, adiabatic)
    canonical_energy = distribution_energy_eV(spectrum.eigenvalues_eV, canonical)
    uniform_energy = distribution_energy_eV(spectrum.eigenvalues_eV, uniform)
    denominator = uniform_energy - canonical_energy
    heating = (
        float((propagated_energy - canonical_energy) / denominator)
        if abs(denominator) > 1.0e-15
        else float("nan")
    )

    ground_energy = float(spectrum.eigenvalues_eV[0])
    ground_mask = (
        np.abs(spectrum.eigenvalues_eV - ground_energy)
        <= float(ground_degeneracy_atol_eV)
    )
    ground_population = float(np.sum(adiabatic[ground_mask]))
    canonical_ground = float(np.sum(canonical[ground_mask]))

    beta_bath = float(1.0 / (BOLTZMANN_EV_PER_K * temperature))
    beta_eff = effective_inverse_temperature_eV_inv(
        spectrum.eigenvalues_eV, propagated_energy
    )
    beta_ratio = (
        float(beta_eff / beta_bath)
        if np.isfinite(beta_eff)
        else float(np.copysign(np.inf, beta_eff))
    )
    entropy = occupation_entropy(adiabatic)
    normalized_entropy = (
        float(entropy / np.log(spectrum.physical_sector_dimension))
        if spectrum.physical_sector_dimension > 1
        else 0.0
    )

    return PairThermalizationSnapshot(
        sector=sector,
        full_ordered_dimension=spectrum.full_ordered_dimension,
        physical_sector_dimension=spectrum.physical_sector_dimension,
        eigenvalues_eV=spectrum.eigenvalues_eV,
        adiabatic_occupations=adiabatic,
        canonical_occupations=canonical,
        uniform_occupations=uniform,
        propagated_energy_eV=propagated_energy,
        canonical_energy_eV=canonical_energy,
        uniform_energy_eV=uniform_energy,
        heating_coordinate=heating,
        tv_to_canonical=total_variation_distance(adiabatic, canonical),
        js_to_canonical=jensen_shannon_distance(adiabatic, canonical),
        tv_to_uniform=total_variation_distance(adiabatic, uniform),
        normalized_occupation_entropy=normalized_entropy,
        occupation_participation_number=occupation_participation_number(adiabatic),
        ground_manifold_size=int(np.count_nonzero(ground_mask)),
        ground_manifold_population=ground_population,
        canonical_ground_manifold_population=canonical_ground,
        beta_eff_eV_inv=beta_eff,
        beta_bath_eV_inv=beta_bath,
        beta_eff_over_beta_bath=beta_ratio,
        effective_temperature_K=effective_temperature_K(beta_eff),
    )
