"""Adapters from validated static states to O0 projection observables.

The O0 kernels operate on reduced density matrices and explicit configuration
expansions.  This module connects those representation-independent definitions
to the stationary polaron, bipolaron, distinguishable electron-hole exciton,
and spin-adapted S0 states without changing any underlying solver physics.

A deliberate guard is kept for the one-polaron path: :class:`PolaronResult`
contains only the charge density, which is insufficient to reconstruct the
one-particle coherences.  The adapter therefore accepts the validated
:class:`~holstein_peierls.electronic.GroundState` itself rather than inventing a
diagonal density matrix from ``|psi_i|^2``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..electronic import GroundState
from ..exciton.solver import ExcitonGroundState, StaticExcitonResult
from ..spin_adapted.excitation_reference import (
    ReferencedExcitationState,
    StaticReferencedExcitationResult,
    spin_summed_rdm,
)
from ..spin_adapted.open_shell import OpenShellStateDefinition
from ..spin_adapted.orbital_optimization import OpenShellOrbitalResult
from ..spin_adapted.spin import minimal_open_shell_coefficients
from ..two_particle.bipolaron import BipolaronGroundState, BipolaronResult

ComplexArray = NDArray[np.complex128]
SpinAdaptedComponent = Literal["neutral", "excited"]


@dataclass(frozen=True, slots=True)
class ExcitonOneParticleRDMs:
    """Electron and hole one-particle RDMs of a distinguishable exciton."""

    electron: ComplexArray
    hole: ComplexArray


@dataclass(frozen=True, slots=True)
class SpinAdaptedRDMs:
    """Neutral, excited, and excitation RDMs for the S0 reference convention."""

    neutral: ComplexArray
    excited: ComplexArray
    excitation: ComplexArray


@dataclass(frozen=True, slots=True)
class ConfigurationExpansion:
    """Explicit Slater-determinant expansion usable by O0 yield kernels."""

    configurations: tuple[ComplexArray, ...]
    coefficients: ComplexArray


def _normalized_vector(values: NDArray[np.generic], *, name: str) -> ComplexArray:
    vector = np.asarray(values, dtype=np.complex128).reshape(-1)
    if vector.size == 0:
        raise ValueError(f"{name} must not be empty")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError(f"{name} must have a finite non-zero norm")
    if not np.isclose(norm, 1.0, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"{name} must be normalized")
    return vector


def polaron_one_particle_rdm(state: GroundState) -> ComplexArray:
    """Return the rank-one one-particle RDM of a one-polaron electronic state."""
    psi = _normalized_vector(state.wavefunction, name="polaron wavefunction")
    return np.asarray(np.outer(psi, psi.conj()), dtype=np.complex128)


def polaron_configuration(state: GroundState) -> ComplexArray:
    """Return the one-occupied-orbital Slater representation of a polaron."""
    psi = _normalized_vector(state.wavefunction, name="polaron wavefunction")
    return psi[:, np.newaxis]


def _bipolaron_ground_state(
    state: BipolaronGroundState | BipolaronResult,
) -> BipolaronGroundState:
    return state.ground_state if isinstance(state, BipolaronResult) else state


def bipolaron_one_particle_rdm(
    state: BipolaronGroundState | BipolaronResult,
) -> ComplexArray:
    """Return the spin-summed bipolaron one-particle RDM with trace two."""
    ground_state = _bipolaron_ground_state(state)
    psi = np.asarray(ground_state.wavefunction, dtype=np.complex128)
    if psi.ndim != 2 or psi.shape[0] != psi.shape[1]:
        raise ValueError("bipolaron wavefunction must be a square ordered-pair matrix")
    norm = float(np.linalg.norm(psi))
    if not np.isclose(norm, 1.0, rtol=0.0, atol=1.0e-10):
        raise ValueError("bipolaron wavefunction must be normalized")
    gamma = 2.0 * (psi @ psi.conj().T)
    return np.asarray(0.5 * (gamma + gamma.conj().T), dtype=np.complex128)


def bipolaron_pair_state_vector(
    state: BipolaronGroundState | BipolaronResult,
) -> ComplexArray:
    """Flatten the normalized ordered-pair bipolaron state in C order."""
    ground_state = _bipolaron_ground_state(state)
    return _normalized_vector(
        np.asarray(ground_state.wavefunction).ravel(order="C"),
        name="bipolaron wavefunction",
    )


def _exciton_ground_state(
    state: ExcitonGroundState | StaticExcitonResult,
) -> ExcitonGroundState:
    return state.ground_state if isinstance(state, StaticExcitonResult) else state


def exciton_one_particle_rdms(
    state: ExcitonGroundState | StaticExcitonResult,
) -> ExcitonOneParticleRDMs:
    """Return separate unit-trace electron and hole RDMs of an exciton."""
    ground_state = _exciton_ground_state(state)
    psi = np.asarray(ground_state.wavefunction, dtype=np.complex128)
    if psi.ndim != 2:
        raise ValueError("exciton wavefunction must be an ordered electron-hole matrix")
    norm = float(np.linalg.norm(psi))
    if not np.isclose(norm, 1.0, rtol=0.0, atol=1.0e-10):
        raise ValueError("exciton wavefunction must be normalized")
    electron = psi @ psi.conj().T
    hole = psi.conj().T @ psi
    electron = 0.5 * (electron + electron.conj().T)
    hole = 0.5 * (hole + hole.conj().T)
    return ExcitonOneParticleRDMs(
        electron=np.asarray(electron, dtype=np.complex128),
        hole=np.asarray(hole, dtype=np.complex128),
    )


def exciton_pair_state_vector(
    state: ExcitonGroundState | StaticExcitonResult,
) -> ComplexArray:
    """Flatten the normalized ordered electron-hole state in C order."""
    ground_state = _exciton_ground_state(state)
    return _normalized_vector(
        np.asarray(ground_state.wavefunction).ravel(order="C"),
        name="exciton wavefunction",
    )


def open_shell_one_particle_rdm(
    state: OpenShellOrbitalResult,
    definition: OpenShellStateDefinition,
) -> ComplexArray:
    """Return ``sum_mu n_mu P_mu`` for one optimized open-shell state."""
    gamma = spin_summed_rdm(state.projectors, definition)
    return np.asarray(gamma, dtype=np.complex128)


def _referenced_state(
    state: ReferencedExcitationState | StaticReferencedExcitationResult,
) -> ReferencedExcitationState:
    return state.state if isinstance(state, StaticReferencedExcitationResult) else state


def spin_adapted_rdms(
    state: ReferencedExcitationState | StaticReferencedExcitationResult,
) -> SpinAdaptedRDMs:
    """Return the full S0 neutral/excited RDMs and their zero-trace difference."""
    referenced = _referenced_state(state)
    neutral = np.asarray(referenced.neutral_rdm, dtype=np.complex128)
    excited = np.asarray(referenced.excited_rdm, dtype=np.complex128)
    excitation = np.asarray(referenced.excitation_rdm, dtype=np.complex128)
    return SpinAdaptedRDMs(
        neutral=neutral,
        excited=excited,
        excitation=excitation,
    )


def _spin_orbital(spatial: ComplexArray, *, spin: Literal["alpha", "beta"]) -> ComplexArray:
    orbital = np.asarray(spatial, dtype=np.complex128).reshape(-1)
    zeros = np.zeros_like(orbital)
    if spin == "alpha":
        return np.concatenate((orbital, zeros))
    return np.concatenate((zeros, orbital))


def _closed_shell_columns(orbitals: ComplexArray, count: int) -> list[ComplexArray]:
    columns: list[ComplexArray] = []
    for index in range(count):
        spatial = orbitals[:, index]
        columns.append(_spin_orbital(spatial, spin="alpha"))
        columns.append(_spin_orbital(spatial, spin="beta"))
    return columns


def _determinant(columns: list[ComplexArray]) -> ComplexArray:
    if len(columns) == 0:
        raise ValueError("a determinant must contain at least one occupied spin orbital")
    configuration = np.column_stack(columns).astype(np.complex128, copy=False)
    overlap = configuration.conj().T @ configuration
    identity = np.eye(configuration.shape[1], dtype=np.complex128)
    if not np.allclose(overlap, identity, rtol=0.0, atol=1.0e-10):
        raise ValueError("constructed spin-orbital determinant is not orthonormal")
    return configuration


def spin_adapted_configuration_expansion(
    state: ReferencedExcitationState | StaticReferencedExcitationResult,
    *,
    component: SpinAdaptedComponent = "excited",
) -> ConfigurationExpansion:
    """Construct the explicit determinant expansion associated with an S0 state.

    The neutral closed-shell reference is one determinant.  The excited singlet
    and triplet are returned in the two-determinant ``M_S=0`` representation
    used by :func:`minimal_open_shell_coefficients`, even though the triplet
    orbital optimization itself uses the equivalent high-spin ``M_S=1``
    open-shell functional.

    Determinant phases are fixed explicitly.  With frontier spatial orbitals
    ``h`` and ``l``, the two open-shell determinants use the ordered frontier
    columns ``(h_alpha, l_beta)`` and ``(l_alpha, h_beta)``.  The reversal in the
    second determinant fixes its phase so that the project's established
    coefficient convention has equal signs for the singlet and opposite signs
    for the triplet.
    """
    referenced = _referenced_state(state)
    if component == "neutral":
        orbitals = np.asarray(referenced.neutral.orbitals, dtype=np.complex128)
        n_closed = referenced.neutral_shell_sizes[0]
        determinant = _determinant(_closed_shell_columns(orbitals, n_closed))
        return ConfigurationExpansion(
            configurations=(determinant,),
            coefficients=np.asarray([1.0 + 0.0j], dtype=np.complex128),
        )
    if component != "excited":
        raise ValueError(f"unsupported spin-adapted component: {component}")

    orbitals = np.asarray(referenced.excited.orbitals, dtype=np.complex128)
    n_closed = referenced.excited_shell_sizes[0]
    if orbitals.ndim != 2 or orbitals.shape[0] != orbitals.shape[1]:
        raise ValueError("S0 requires a complete square spatial-orbital matrix")
    if n_closed + 2 > orbitals.shape[1]:
        raise ValueError("S0 excited state does not contain two frontier orbitals")

    core = _closed_shell_columns(orbitals, n_closed)
    h = orbitals[:, n_closed]
    l = orbitals[:, n_closed + 1]
    determinant_one = _determinant(
        core
        + [
            _spin_orbital(h, spin="alpha"),
            _spin_orbital(l, spin="beta"),
        ]
    )
    determinant_two = _determinant(
        core
        + [
            _spin_orbital(l, spin="alpha"),
            _spin_orbital(h, spin="beta"),
        ]
    )
    coefficients = np.asarray(
        minimal_open_shell_coefficients(referenced.multiplicity),
        dtype=np.complex128,
    )
    return ConfigurationExpansion(
        configurations=(determinant_one, determinant_two),
        coefficients=coefficients,
    )
