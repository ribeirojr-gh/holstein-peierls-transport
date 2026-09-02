"""State-specific orbital optimization for the S0 open-shell functional.

This is the stationary counterpart needed before the time-dependent Miranda
orbital equations are introduced.  The optimizer works with a complete real
orthonormal orbital matrix and varies only rotations between subspaces with
different occupation numbers.  Rotations within a shell are gauge degrees of
freedom, while rotations between distinct shells with the same occupation are
held fixed in the minimal fixed-coefficient formalism, following the variational
restriction discussed by Miranda et al.

The energy derivative can be written in terms of shell projectors and Fock
matrices.  If ``P_mu`` is a shell projector and ``F_mu`` its Fock matrix,

    M = sum_mu n_mu [P_mu, F_mu]

is anti-Hermitian.  In the current real implementation it is skew-symmetric.
For an infinitesimal orbital rotation ``A`` the first-order energy change is
``dE = Tr(M A)``.  Choosing the allowed part of ``A=M`` is therefore a descent
direction because the trace of the square of a real skew-symmetric matrix is
non-positive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

from .open_shell import (
    Matrix,
    OpenShellStateDefinition,
    general_open_shell_energy,
    projector_from_orbitals,
    shell_fock_matrices,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class OrbitalOptimizationDiagnostics:
    iterations: int
    converged: bool
    final_energy: float
    final_max_gradient: float
    accepted_steps: int
    rejected_steps: int


@dataclass(frozen=True, slots=True)
class OpenShellOrbitalResult:
    orbitals: FloatArray
    projectors: tuple[FloatArray, ...]
    energy: float
    diagnostics: OrbitalOptimizationDiagnostics


def _validate_shell_sizes(
    n_orbitals: int,
    shell_sizes: tuple[int, ...],
    definition: OpenShellStateDefinition,
) -> None:
    if len(shell_sizes) != definition.n_shells:
        raise ValueError("shell_sizes must have one entry per occupied shell")
    if any(size < 0 for size in shell_sizes):
        raise ValueError("shell sizes must be non-negative")
    if sum(shell_sizes) > n_orbitals:
        raise ValueError("occupied shell sizes exceed the orbital-space dimension")


def _validate_orbitals(orbitals: Matrix) -> FloatArray:
    c = np.asarray(orbitals, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("a complete square orbital matrix is required")
    if not np.allclose(c.T @ c, np.eye(c.shape[0]), atol=1.0e-10):
        raise ValueError("orbital matrix must be orthonormal")
    return c.copy()


def shell_column_slices(shell_sizes: tuple[int, ...]) -> tuple[slice, ...]:
    slices: list[slice] = []
    start = 0
    for size in shell_sizes:
        slices.append(slice(start, start + size))
        start += size
    return tuple(slices)


def shell_projectors_from_complete_orbitals(
    orbitals: Matrix,
    shell_sizes: tuple[int, ...],
    definition: OpenShellStateDefinition,
) -> tuple[FloatArray, ...]:
    """Construct occupied-shell projectors from contiguous orbital columns."""
    c = _validate_orbitals(orbitals)
    _validate_shell_sizes(c.shape[1], shell_sizes, definition)
    projectors: list[FloatArray] = []
    for shell_slice in shell_column_slices(shell_sizes):
        block = c[:, shell_slice]
        if block.shape[1] == 0:
            projectors.append(np.zeros((c.shape[0], c.shape[0]), dtype=np.float64))
        else:
            projectors.append(
                np.asarray(projector_from_orbitals(block), dtype=np.float64)
            )
    return tuple(projectors)


def orbital_occupations(
    n_orbitals: int,
    shell_sizes: tuple[int, ...],
    definition: OpenShellStateDefinition,
) -> FloatArray:
    """Return occupation number (2,1,0) assigned to each orbital column."""
    _validate_shell_sizes(n_orbitals, shell_sizes, definition)
    result = np.zeros(n_orbitals, dtype=np.float64)
    start = 0
    for size, occupation in zip(
        shell_sizes, definition.occupations, strict=True
    ):
        result[start : start + size] = occupation
        start += size
    return result


def open_shell_orbital_energy(
    one_body: Matrix,
    interaction: FloatArray,
    orbitals: Matrix,
    shell_sizes: tuple[int, ...],
    definition: OpenShellStateDefinition,
) -> float:
    projectors = shell_projectors_from_complete_orbitals(
        orbitals, shell_sizes, definition
    )
    return general_open_shell_energy(
        one_body, interaction, projectors, definition
    )


def orbital_rotation_gradient(
    one_body: Matrix,
    interaction: FloatArray,
    orbitals: Matrix,
    shell_sizes: tuple[int, ...],
    definition: OpenShellStateDefinition,
) -> FloatArray:
    """Return the allowed anti-symmetric orbital-rotation energy gradient.

    Matrix elements connecting orbitals with the same occupation are set to
    zero.  This removes within-shell gauge rotations and, for the open-shell
    singlet, also freezes rotations between the two distinct singly occupied
    shells as required by the minimal fixed-coefficient ansatz.
    """
    c = _validate_orbitals(orbitals)
    projectors = shell_projectors_from_complete_orbitals(c, shell_sizes, definition)
    focks = shell_fock_matrices(one_body, interaction, projectors, definition)

    generator_site = np.zeros_like(c)
    for occupation, projector, fock in zip(
        definition.occupations, projectors, focks, strict=True
    ):
        generator_site += occupation * (projector @ fock - fock @ projector)

    generator_mo = c.T @ generator_site @ c
    generator_mo = 0.5 * (generator_mo - generator_mo.T)
    occupations = orbital_occupations(c.shape[1], shell_sizes, definition)
    allowed = occupations[:, None] != occupations[None, :]
    generator_mo = np.where(allowed, generator_mo, 0.0)
    return np.asarray(0.5 * (generator_mo - generator_mo.T), dtype=np.float64)


def optimize_open_shell_orbitals(
    one_body: Matrix,
    interaction: FloatArray,
    initial_orbitals: Matrix,
    shell_sizes: tuple[int, ...],
    definition: OpenShellStateDefinition,
    *,
    gradient_tolerance: float = 1.0e-9,
    max_iterations: int = 500,
    initial_step: float = 0.2,
    maximum_step: float = 2.0,
    minimum_step: float = 1.0e-12,
    growth_factor: float = 1.25,
    shrink_factor: float = 0.5,
    energy_decrease_tolerance: float = 1.0e-14,
) -> OpenShellOrbitalResult:
    """Minimize the fixed-coefficient open-shell energy by orbital rotations."""
    if gradient_tolerance <= 0.0:
        raise ValueError("gradient_tolerance must be positive")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if not 0.0 < minimum_step <= initial_step <= maximum_step:
        raise ValueError("require minimum_step <= initial_step <= maximum_step")
    if growth_factor <= 1.0:
        raise ValueError("growth_factor must exceed one")
    if not 0.0 < shrink_factor < 1.0:
        raise ValueError("shrink_factor must lie between zero and one")

    c = _validate_orbitals(initial_orbitals)
    _validate_shell_sizes(c.shape[1], shell_sizes, definition)
    energy = open_shell_orbital_energy(
        one_body, interaction, c, shell_sizes, definition
    )
    step = initial_step
    accepted = 0
    rejected = 0
    converged = False
    final_max_gradient = np.inf

    for iteration in range(1, max_iterations + 1):
        gradient = orbital_rotation_gradient(
            one_body, interaction, c, shell_sizes, definition
        )
        final_max_gradient = float(np.max(np.abs(gradient)))
        if final_max_gradient < gradient_tolerance:
            converged = True
            break

        trial_step = step
        accepted_this_iteration = False
        while trial_step >= minimum_step:
            rotation = expm(trial_step * gradient)
            candidate = np.asarray(c @ rotation, dtype=np.float64)
            candidate_energy = open_shell_orbital_energy(
                one_body, interaction, candidate, shell_sizes, definition
            )
            if candidate_energy < energy - energy_decrease_tolerance:
                c = candidate
                energy = candidate_energy
                step = min(trial_step * growth_factor, maximum_step)
                accepted += 1
                accepted_this_iteration = True
                break
            trial_step *= shrink_factor
            rejected += 1

        if not accepted_this_iteration:
            step = trial_step
            break

    projectors = shell_projectors_from_complete_orbitals(c, shell_sizes, definition)
    final_gradient = orbital_rotation_gradient(
        one_body, interaction, c, shell_sizes, definition
    )
    final_max_gradient = float(np.max(np.abs(final_gradient)))
    if final_max_gradient < gradient_tolerance:
        converged = True

    return OpenShellOrbitalResult(
        orbitals=c,
        projectors=projectors,
        energy=energy,
        diagnostics=OrbitalOptimizationDiagnostics(
            iterations=iteration,
            converged=converged,
            final_energy=energy,
            final_max_gradient=final_max_gradient,
            accepted_steps=accepted,
            rejected_steps=rejected,
        ),
    )
