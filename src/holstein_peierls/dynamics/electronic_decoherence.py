"""Instantaneous electronic decoherence controls for D5b.

D5a demonstrated progressive electronic overheating in the coherent Ehrenfest
control while the D4 lattice bath remained correctly thermalized. D5b therefore
implements explicit, auditable instantaneous-decoherence controls in the
instantaneous adiabatic representation.

The three schemes follow Si and Wu, J. Chem. Phys. 143, 024103 (2015):

- ``dp``: destruction of phase coherence only, with collapse probability equal
  to the pre-collapse adiabatic population;
- ``bm``: the same population reweighted by a Boltzmann factor;
- ``ma``: a Miller-Abrahams-like reweighting that suppresses transitions above
  the pre-collapse mean electronic energy.

These are stochastic model extensions, not exact consequences of Ehrenfest
physics. In particular, the decoherence interval is a model parameter and must
be benchmarked rather than silently identified with a material property.

A collapse changes the propagated electronic energy discontinuously at fixed
lattice coordinates and velocities. The change is returned explicitly as
``electronic_environment_exchange_eV``. Positive values mean that the
additional electronic environment injected energy into the matter subsystem;
negative values mean energy was removed. No hidden velocity rescaling is
performed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray

from ..hamiltonian import build_dense_hamiltonian
from ..lattice import LatticeState
from ..parameters import StaticPolaronParameters
from .langevin import BOLTZMANN_EV_PER_K

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
IDCScheme = Literal["dp", "bm", "ma"]


@dataclass(frozen=True, slots=True)
class InstantaneousDecoherenceEvent:
    """One stochastic collapse in the instantaneous adiabatic basis."""

    electronic_state: ComplexArray
    selected_state_index: int
    adiabatic_energies_eV: FloatArray
    adiabatic_populations_before: FloatArray
    collapse_probabilities: FloatArray
    electronic_energy_before_eV: float
    electronic_energy_after_eV: float
    electronic_environment_exchange_eV: float
    scheme: IDCScheme


def _normalized_probabilities(values: ArrayLike) -> FloatArray:
    probabilities = np.asarray(values, dtype=np.float64).reshape(-1)
    if probabilities.size == 0 or not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities must be a finite non-empty vector")
    if np.any(probabilities < -1.0e-15):
        raise ValueError("probabilities must be non-negative")
    probabilities = np.maximum(probabilities, 0.0)
    total = float(np.sum(probabilities))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("probabilities must have positive total weight")
    return np.asarray(probabilities / total, dtype=np.float64)


def _validate_scheme(scheme: str) -> IDCScheme:
    if scheme not in ("dp", "bm", "ma"):
        raise ValueError("scheme must be 'dp', 'bm', or 'ma'")
    return scheme  # type: ignore[return-value]


def adiabatic_populations(
    eigenvectors: ArrayLike,
    electronic_state: ArrayLike,
) -> FloatArray:
    """Return normalized ``|<phi_mu|psi>|^2`` populations."""
    vectors = np.asarray(eigenvectors, dtype=np.complex128)
    psi = np.asarray(electronic_state, dtype=np.complex128).reshape(-1)
    if vectors.ndim != 2 or vectors.shape[0] != psi.size:
        raise ValueError("eigenvector matrix and electronic state are incompatible")
    if not np.all(np.isfinite(vectors)) or not np.all(np.isfinite(psi)):
        raise ValueError("electronic inputs must be finite")
    norm_squared = float(np.vdot(psi, psi).real)
    if not np.isfinite(norm_squared) or norm_squared <= 0.0:
        raise ValueError("electronic state must have finite non-zero norm")
    coefficients = vectors.conj().T @ psi
    return _normalized_probabilities(np.abs(coefficients) ** 2 / norm_squared)


def idc_collapse_probabilities(
    energies_eV: ArrayLike,
    populations: ArrayLike,
    temperature_K: float,
    scheme: IDCScheme,
) -> FloatArray:
    """Return normalized stochastic collapse probabilities for one IDC event.

    ``dp`` uses the adiabatic populations directly.

    ``bm`` uses ``p_mu exp[-beta E_mu]``. A constant energy shift is removed
    before exponentiation for numerical stability.

    ``ma`` uses ``p_mu`` for states at or below the pre-collapse mean energy and
    ``p_mu exp[-beta(E_mu-Ebar)]`` above it.
    """
    selected_scheme = _validate_scheme(scheme)
    energies = np.asarray(energies_eV, dtype=np.float64).reshape(-1)
    population = _normalized_probabilities(populations)
    if energies.size != population.size or not np.all(np.isfinite(energies)):
        raise ValueError("energies and populations must be finite vectors of equal size")

    if selected_scheme == "dp":
        return population.copy()

    temperature = float(temperature_K)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("BM and MA schemes require positive finite temperature_K")
    beta = 1.0 / (BOLTZMANN_EV_PER_K * temperature)

    if selected_scheme == "bm":
        shifted = energies - float(np.min(energies))
        weights = population * np.exp(-beta * shifted)
    else:
        mean_energy = float(np.dot(population, energies))
        uphill = np.maximum(energies - mean_energy, 0.0)
        weights = population * np.exp(-beta * uphill)
    return _normalized_probabilities(weights)


def apply_instantaneous_decoherence(
    lattice: LatticeState,
    parameters: StaticPolaronParameters,
    electronic_state: ArrayLike,
    temperature_K: float,
    scheme: IDCScheme,
    rng: Generator,
) -> InstantaneousDecoherenceEvent:
    """Collapse the propagated state to one instantaneous adiabatic eigenstate.

    The Hamiltonian is diagonalized only at the decoherence event. D5b initially
    validates zero-field dynamics, so this function intentionally uses the
    zero-field one-carrier Hamiltonian. Field-driven IDC is a downstream gate.
    """
    if not isinstance(rng, Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    lattice.validate()
    if lattice.shape != (parameters.ny, parameters.nx):
        raise ValueError("lattice shape does not match parameters")

    hamiltonian = build_dense_hamiltonian(lattice, parameters)
    energies, vectors = np.linalg.eigh(hamiltonian)
    population = adiabatic_populations(vectors, electronic_state)
    probabilities = idc_collapse_probabilities(
        energies,
        population,
        temperature_K,
        scheme,
    )
    selected = int(rng.choice(energies.size, p=probabilities))
    collapsed = np.asarray(vectors[:, selected], dtype=np.complex128).copy()

    before = float(np.dot(population, energies))
    after = float(energies[selected])
    exchange = after - before
    return InstantaneousDecoherenceEvent(
        electronic_state=collapsed,
        selected_state_index=selected,
        adiabatic_energies_eV=np.asarray(energies, dtype=np.float64),
        adiabatic_populations_before=population,
        collapse_probabilities=probabilities,
        electronic_energy_before_eV=before,
        electronic_energy_after_eV=after,
        electronic_environment_exchange_eV=float(exchange),
        scheme=_validate_scheme(scheme),
    )
