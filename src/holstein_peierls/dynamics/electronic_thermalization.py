"""D5a diagnostics for electronic thermalization in finite-temperature dynamics.

This module does not modify the propagated electronic state.  It measures how
instantaneous adiabatic occupations produced by a coherent trajectory compare
with canonical and infinite-temperature references for the same instantaneous
Hamiltonian.

For a normalized one-carrier state ``psi`` and instantaneous eigenvectors
``phi_l`` of ``H(q)``, the diagnostic occupation is

    p_l = |<phi_l|psi>|^2.

These are adiabatic state populations.  They are not natural occupations and
their Shannon entropy is not the von Neumann entropy of the pure propagated
state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

from ..hamiltonian import build_dense_hamiltonian
from ..lattice import LatticeState
from ..parameters import StaticPolaronParameters
from .langevin import BOLTZMANN_EV_PER_K

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def _finite_vector(values: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def _normalized_probabilities(values: ArrayLike, *, name: str) -> FloatArray:
    array = _finite_vector(values, name=name)
    if np.any(array < -1.0e-14):
        raise ValueError(f"{name} must be non-negative")
    array = np.maximum(array, 0.0)
    total = float(np.sum(array))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"{name} must have positive total weight")
    return np.asarray(array / total, dtype=np.float64)


def generalized_boltzmann_occupations(
    eigenvalues_eV: ArrayLike,
    beta_eV_inv: float,
) -> FloatArray:
    """Return normalized ``exp(-beta E)`` weights for signed inverse temperature.

    Signed ``beta`` is useful only as an energy-matched diagnostic for the
    bounded finite-dimensional electronic spectrum.  Physical bath references
    in this project always use positive temperature.
    """
    energies = _finite_vector(eigenvalues_eV, name="eigenvalues_eV")
    beta = float(beta_eV_inv)
    if not np.isfinite(beta):
        raise ValueError("beta_eV_inv must be finite")
    exponent = -beta * energies
    exponent -= float(np.max(exponent))
    weights = np.exp(exponent)
    return np.asarray(weights / float(np.sum(weights)), dtype=np.float64)


def canonical_occupations(
    eigenvalues_eV: ArrayLike,
    temperature_K: float,
    *,
    degeneracy_atol_eV: float = 1.0e-12,
) -> FloatArray:
    """Return one-particle canonical occupations at positive temperature.

    At exactly zero temperature, equal probability is assigned to the
    numerically degenerate ground-state manifold.
    """
    energies = _finite_vector(eigenvalues_eV, name="eigenvalues_eV")
    temperature = float(temperature_K)
    if not np.isfinite(temperature) or temperature < 0.0:
        raise ValueError("temperature_K must be finite and non-negative")
    if temperature == 0.0:
        ground = float(np.min(energies))
        mask = np.abs(energies - ground) <= float(degeneracy_atol_eV)
        result = np.zeros_like(energies)
        result[mask] = 1.0 / int(np.count_nonzero(mask))
        return result
    beta = 1.0 / (BOLTZMANN_EV_PER_K * temperature)
    return generalized_boltzmann_occupations(energies, beta)


def distribution_energy_eV(
    eigenvalues_eV: ArrayLike,
    probabilities: ArrayLike,
) -> float:
    energies = _finite_vector(eigenvalues_eV, name="eigenvalues_eV")
    population = _normalized_probabilities(probabilities, name="probabilities")
    if energies.size != population.size:
        raise ValueError("eigenvalues and probabilities must have equal size")
    return float(np.dot(population, energies))


def total_variation_distance(first: ArrayLike, second: ArrayLike) -> float:
    p = _normalized_probabilities(first, name="first")
    q = _normalized_probabilities(second, name="second")
    if p.size != q.size:
        raise ValueError("probability vectors must have equal size")
    return float(0.5 * np.sum(np.abs(p - q)))


def jensen_shannon_distance(first: ArrayLike, second: ArrayLike) -> float:
    """Return the Jensen-Shannon distance using natural logarithms."""
    p = _normalized_probabilities(first, name="first")
    q = _normalized_probabilities(second, name="second")
    if p.size != q.size:
        raise ValueError("probability vectors must have equal size")
    midpoint = 0.5 * (p + q)

    def kl_term(values: FloatArray) -> float:
        mask = values > 0.0
        return float(np.sum(values[mask] * np.log(values[mask] / midpoint[mask])))

    divergence = 0.5 * (kl_term(p) + kl_term(q))
    return float(np.sqrt(max(0.0, divergence)))


def occupation_entropy(probabilities: ArrayLike) -> float:
    """Return Shannon entropy of a specified occupation vector in nats."""
    population = _normalized_probabilities(probabilities, name="probabilities")
    mask = population > 0.0
    return float(-np.sum(population[mask] * np.log(population[mask])))


def occupation_participation_number(probabilities: ArrayLike) -> float:
    population = _normalized_probabilities(probabilities, name="probabilities")
    return float(1.0 / np.sum(population * population))


def effective_inverse_temperature_eV_inv(
    eigenvalues_eV: ArrayLike,
    target_energy_eV: float,
    *,
    energy_atol_eV: float = 1.0e-12,
) -> float:
    """Return signed beta whose canonical mean energy matches ``target_energy_eV``.

    For a bounded finite spectrum, the canonical mean energy is monotonic in
    beta and spans the interval from the maximum eigenvalue at beta -> -inf to
    the minimum eigenvalue at beta -> +inf.  ``beta=0`` is the uniform,
    infinite-temperature distribution.
    """
    energies = np.sort(_finite_vector(eigenvalues_eV, name="eigenvalues_eV"))
    target = float(target_energy_eV)
    if not np.isfinite(target):
        raise ValueError("target_energy_eV must be finite")
    minimum = float(energies[0])
    maximum = float(energies[-1])
    tolerance = float(energy_atol_eV)
    if target < minimum - tolerance or target > maximum + tolerance:
        raise ValueError("target energy lies outside the electronic spectrum")
    target = min(max(target, minimum), maximum)
    uniform_energy = float(np.mean(energies))
    if abs(target - uniform_energy) <= tolerance:
        return 0.0
    if abs(target - minimum) <= tolerance:
        return float(np.inf)
    if abs(target - maximum) <= tolerance:
        return float(-np.inf)

    def residual(beta: float) -> float:
        population = generalized_boltzmann_occupations(energies, beta)
        return distribution_energy_eV(energies, population) - target

    # The residual decreases monotonically with beta. Grow a symmetric bracket
    # until it contains the requested finite target energy.
    bound = 1.0
    for _ in range(60):
        low_value = residual(-bound)
        high_value = residual(bound)
        if low_value >= 0.0 and high_value <= 0.0:
            return float(brentq(residual, -bound, bound, xtol=1.0e-13, rtol=1.0e-13))
        bound *= 2.0
    raise RuntimeError("failed to bracket effective inverse temperature")


def effective_temperature_K(beta_eV_inv: float) -> float:
    """Convert signed inverse temperature to Kelvin, preserving infinities."""
    beta = float(beta_eV_inv)
    if np.isnan(beta):
        return float("nan")
    if beta == 0.0:
        return float(np.inf)
    if np.isposinf(beta):
        return 0.0
    if np.isneginf(beta):
        return -0.0
    return float(1.0 / (BOLTZMANN_EV_PER_K * beta))


@dataclass(frozen=True, slots=True)
class ElectronicThermalizationSnapshot:
    """Instantaneous D5a electronic thermalization diagnostics."""

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
    occupation_entropy: float
    normalized_occupation_entropy: float
    occupation_participation_number: float
    ground_manifold_size: int
    ground_manifold_population: float
    canonical_ground_manifold_population: float
    beta_eff_eV_inv: float
    beta_bath_eV_inv: float
    beta_eff_over_beta_bath: float
    effective_temperature_K: float


def diagnose_electronic_thermalization(
    state: LatticeState,
    parameters: StaticPolaronParameters,
    electronic_state: ArrayLike,
    temperature_K: float,
    *,
    ground_degeneracy_atol_eV: float = 1.0e-10,
) -> ElectronicThermalizationSnapshot:
    """Diagnose one propagated state against instantaneous thermal references."""
    state.validate()
    if state.shape != (parameters.ny, parameters.nx):
        raise ValueError("lattice shape does not match parameters")
    temperature = float(temperature_K)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("D5a finite-temperature diagnostics require temperature_K > 0")

    psi = np.asarray(electronic_state, dtype=np.complex128).reshape(-1)
    if psi.size != parameters.n_sites or not np.all(np.isfinite(psi)):
        raise ValueError("electronic state does not match parameters")
    norm_squared = float(np.vdot(psi, psi).real)
    if not np.isfinite(norm_squared) or norm_squared <= 0.0:
        raise ValueError("electronic state must have finite positive norm")
    psi = np.asarray(psi / np.sqrt(norm_squared), dtype=np.complex128)

    hamiltonian = build_dense_hamiltonian(state, parameters)
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    amplitudes = eigenvectors.conj().T @ psi
    adiabatic = _normalized_probabilities(np.abs(amplitudes) ** 2, name="adiabatic occupations")
    canonical = canonical_occupations(eigenvalues, temperature)
    uniform = np.full(eigenvalues.size, 1.0 / eigenvalues.size, dtype=np.float64)

    propagated_energy = distribution_energy_eV(eigenvalues, adiabatic)
    canonical_energy = distribution_energy_eV(eigenvalues, canonical)
    uniform_energy = distribution_energy_eV(eigenvalues, uniform)
    denominator = uniform_energy - canonical_energy
    heating = (
        float((propagated_energy - canonical_energy) / denominator)
        if abs(denominator) > 1.0e-15
        else float("nan")
    )

    ground_energy = float(eigenvalues[0])
    ground_mask = np.abs(eigenvalues - ground_energy) <= float(ground_degeneracy_atol_eV)
    ground_population = float(np.sum(adiabatic[ground_mask]))
    canonical_ground = float(np.sum(canonical[ground_mask]))

    beta_bath = float(1.0 / (BOLTZMANN_EV_PER_K * temperature))
    beta_eff = effective_inverse_temperature_eV_inv(eigenvalues, propagated_energy)
    if np.isfinite(beta_eff):
        beta_ratio = float(beta_eff / beta_bath)
    else:
        beta_ratio = float(np.copysign(np.inf, beta_eff))

    entropy = occupation_entropy(adiabatic)
    normalized_entropy = (
        float(entropy / np.log(eigenvalues.size)) if eigenvalues.size > 1 else 0.0
    )

    return ElectronicThermalizationSnapshot(
        eigenvalues_eV=np.asarray(eigenvalues, dtype=np.float64),
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
        occupation_entropy=entropy,
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
