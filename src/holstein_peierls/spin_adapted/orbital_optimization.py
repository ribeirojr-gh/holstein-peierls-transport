"""State-specific orbital optimization for the S0 open-shell functional.

This is the stationary counterpart needed before the time-dependent Miranda
orbital equations are introduced. The optimizer works with a complete real
orthonormal orbital matrix and varies only rotations between subspaces with
different occupation numbers. Rotations within a shell are gauge degrees of
freedom, while rotations between distinct shells with the same occupation are
held fixed in the minimal fixed-coefficient formalism, following the variational
restriction discussed by Miranda et al.

The energy derivative can be written in terms of shell projectors and Fock
matrices. If ``P_mu`` is a shell projector and ``F_mu`` its Fock matrix,

    M = sum_mu n_mu [P_mu, F_mu]

is anti-Hermitian. In the current real implementation it is skew-symmetric.
For an infinitesimal orbital rotation ``A`` the first-order energy change is
``dE = Tr(M A)``. Choosing ``A=M`` is therefore a descent direction because
the trace of the square of a real skew-symmetric matrix is non-positive.

The production optimizer accelerates this variational descent with a small
L-BFGS history expressed in the fixed site basis. Storing the tangent vectors
in the site basis provides an inexpensive vector transport between successive
orbital frames. Every proposed quasi-Newton direction is transformed back to
the current MO frame and projected onto the Miranda-allowed rotation space.
A non-descent direction, bad curvature update, or failed line search causes a
safe restart to the analytic steepest-descent generator.
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
    """Return occupation number (2, 1, or 0) assigned to each orbital column."""
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


def _allowed_rotation_mask(
    n_orbitals: int,
    shell_sizes: tuple[int, ...],
    definition: OpenShellStateDefinition,
) -> NDArray[np.bool_]:
    occupations = orbital_occupations(n_orbitals, shell_sizes, definition)
    return occupations[:, None] != occupations[None, :]


def _project_allowed_rotation(
    generator_mo: Matrix,
    allowed: NDArray[np.bool_],
) -> FloatArray:
    generator = np.asarray(generator_mo, dtype=np.float64)
    generator = 0.5 * (generator - generator.T)
    generator = np.where(allowed, generator, 0.0)
    return np.asarray(0.5 * (generator - generator.T), dtype=np.float64)


def orbital_rotation_gradient(
    one_body: Matrix,
    interaction: FloatArray,
    orbitals: Matrix,
    shell_sizes: tuple[int, ...],
    definition: OpenShellStateDefinition,
) -> FloatArray:
    """Return the allowed skew orbital-rotation descent generator.

    Matrix elements connecting orbitals with the same occupation are set to
    zero. This removes within-shell gauge rotations and, for the open-shell
    singlet, also freezes rotations between the two distinct singly occupied
    shells as required by the minimal fixed-coefficient ansatz.

    With the convention used here the first-order change under an infinitesimal
    rotation ``A`` is ``dE = Tr(G A)``. Hence ``G`` itself is a descent
    generator because ``Tr(G G) <= 0`` for a real skew-symmetric matrix.
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
    allowed = _allowed_rotation_mask(c.shape[1], shell_sizes, definition)
    return _project_allowed_rotation(generator_mo, allowed)


def _frobenius_inner(left: Matrix, right: Matrix) -> float:
    return float(np.sum(np.asarray(left) * np.asarray(right)))


def _lbfgs_direction(
    gradient_site: FloatArray,
    history: list[tuple[FloatArray, FloatArray, float]],
) -> FloatArray:
    """Return ``-H g`` from a compact L-BFGS history in the site basis."""
    if not history:
        return -gradient_site.copy()

    q = gradient_site.copy()
    alphas: list[float] = []
    for s_vec, y_vec, rho in reversed(history):
        alpha = rho * _frobenius_inner(s_vec, q)
        alphas.append(alpha)
        q = q - alpha * y_vec

    last_s, last_y, _ = history[-1]
    yy = _frobenius_inner(last_y, last_y)
    sy = _frobenius_inner(last_s, last_y)
    scale = sy / yy if yy > 0.0 and sy > 0.0 else 1.0
    inverse_hessian_gradient = scale * q

    for (s_vec, y_vec, rho), alpha in zip(history, reversed(alphas), strict=True):
        beta = rho * _frobenius_inner(y_vec, inverse_hessian_gradient)
        inverse_hessian_gradient = (
            inverse_hessian_gradient + s_vec * (alpha - beta)
        )
    return -inverse_hessian_gradient


def _site_descent_generator(
    one_body: Matrix,
    interaction: FloatArray,
    orbitals: FloatArray,
    shell_sizes: tuple[int, ...],
    definition: OpenShellStateDefinition,
) -> tuple[FloatArray, FloatArray]:
    generator_mo = orbital_rotation_gradient(
        one_body, interaction, orbitals, shell_sizes, definition
    )
    generator_site = orbitals @ generator_mo @ orbitals.T
    generator_site = 0.5 * (generator_site - generator_site.T)
    return generator_mo, np.asarray(generator_site, dtype=np.float64)


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
    energy_decrease_tolerance: float = 0.0,
    lbfgs_history_size: int = 8,
    armijo_constant: float = 1.0e-4,
    curvature_tolerance: float = 1.0e-10,
) -> OpenShellOrbitalResult:
    """Minimize the fixed-coefficient open-shell energy by orbital rotations.

    The primary search direction is a limited-memory BFGS approximation in the
    fixed site basis. The resulting tangent is projected into the currently
    allowed Miranda rotation space before an exponential retraction preserves
    orbital orthonormality exactly. A failed quasi-Newton direction triggers a
    history reset and a steepest-descent retry, so the analytic variational
    descent remains the robustness baseline.
    """
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
    if energy_decrease_tolerance < 0.0:
        raise ValueError("energy_decrease_tolerance must be non-negative")
    if lbfgs_history_size < 0:
        raise ValueError("lbfgs_history_size must be non-negative")
    if not 0.0 < armijo_constant < 1.0:
        raise ValueError("armijo_constant must lie between zero and one")
    if curvature_tolerance < 0.0:
        raise ValueError("curvature_tolerance must be non-negative")

    c = _validate_orbitals(initial_orbitals)
    _validate_shell_sizes(c.shape[1], shell_sizes, definition)
    allowed = _allowed_rotation_mask(c.shape[1], shell_sizes, definition)
    energy = open_shell_orbital_energy(
        one_body, interaction, c, shell_sizes, definition
    )
    step = initial_step
    accepted = 0
    rejected = 0
    converged = False
    final_max_gradient = np.inf
    history: list[tuple[FloatArray, FloatArray, float]] = []
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        descent_mo, descent_site = _site_descent_generator(
            one_body, interaction, c, shell_sizes, definition
        )
        final_max_gradient = float(np.max(np.abs(descent_mo)))
        if final_max_gradient < gradient_tolerance:
            converged = True
            break

        # In the Frobenius metric the conventional gradient is -descent_site.
        gradient_site = -descent_site
        proposed_site = _lbfgs_direction(gradient_site, history)
        proposed_mo = _project_allowed_rotation(c.T @ proposed_site @ c, allowed)
        directional_derivative = float(np.trace(descent_mo @ proposed_mo))

        # A trustworthy search direction must have negative dE/d(step).
        using_steepest_fallback = False
        if (
            not np.isfinite(directional_derivative)
            or directional_derivative >= 0.0
            or float(np.max(np.abs(proposed_mo))) == 0.0
        ):
            history.clear()
            proposed_mo = descent_mo.copy()
            proposed_site = descent_site.copy()
            directional_derivative = float(np.trace(descent_mo @ proposed_mo))
            using_steepest_fallback = True
        else:
            proposed_site = c @ proposed_mo @ c.T
            proposed_site = 0.5 * (proposed_site - proposed_site.T)

        trial_step = min(step, maximum_step)
        accepted_this_iteration = False
        while trial_step >= minimum_step:
            rotation = expm(trial_step * proposed_mo)
            candidate = np.asarray(c @ rotation, dtype=np.float64)
            candidate_energy = open_shell_orbital_energy(
                one_body, interaction, candidate, shell_sizes, definition
            )
            armijo_bound = (
                energy + armijo_constant * trial_step * directional_derivative
            )
            enough_energy_drop = (
                candidate_energy <= energy - energy_decrease_tolerance
            )
            if enough_energy_drop and candidate_energy <= armijo_bound:
                old_gradient_site = gradient_site
                old_direction_site = proposed_site
                c = candidate
                energy = candidate_energy
                accepted += 1
                accepted_this_iteration = True

                new_descent_mo, new_descent_site = _site_descent_generator(
                    one_body, interaction, c, shell_sizes, definition
                )
                new_gradient_site = -new_descent_site
                s_vec = trial_step * old_direction_site
                y_vec = new_gradient_site - old_gradient_site
                sy = _frobenius_inner(s_vec, y_vec)
                scale = max(
                    np.sqrt(_frobenius_inner(s_vec, s_vec))
                    * np.sqrt(_frobenius_inner(y_vec, y_vec)),
                    np.finfo(float).tiny,
                )
                if (
                    lbfgs_history_size > 0
                    and np.isfinite(sy)
                    and sy > curvature_tolerance * scale
                ):
                    history.append((s_vec, y_vec, 1.0 / sy))
                    if len(history) > lbfgs_history_size:
                        history.pop(0)
                elif not using_steepest_fallback:
                    # Bad curvature means the local inverse-Hessian model is
                    # unreliable. Restart rather than accumulate noisy history.
                    history.clear()

                step = min(trial_step * growth_factor, maximum_step)
                break
            trial_step *= shrink_factor
            rejected += 1

        if not accepted_this_iteration and not using_steepest_fallback:
            # Retry once with the guaranteed variational descent direction.
            history.clear()
            proposed_mo = descent_mo.copy()
            directional_derivative = float(np.trace(descent_mo @ proposed_mo))
            trial_step = min(initial_step, maximum_step)
            while trial_step >= minimum_step:
                rotation = expm(trial_step * proposed_mo)
                candidate = np.asarray(c @ rotation, dtype=np.float64)
                candidate_energy = open_shell_orbital_energy(
                    one_body, interaction, candidate, shell_sizes, definition
                )
                armijo_bound = (
                    energy + armijo_constant * trial_step * directional_derivative
                )
                if (
                    candidate_energy <= energy - energy_decrease_tolerance
                    and candidate_energy <= armijo_bound
                ):
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
