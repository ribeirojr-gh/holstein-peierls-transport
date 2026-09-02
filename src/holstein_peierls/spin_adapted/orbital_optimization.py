"""State-specific orbital optimization for the S0 open-shell functional.

The optimizer uses complete real orthonormal orbital matrices and varies only
rotations between subspaces with different occupation numbers.  Rotations
within a shell are gauge degrees of freedom; in the minimal fixed-coefficient
Miranda ansatz, rotations between distinct shells with the same occupation are
also held fixed.

For shell projectors ``P_mu`` and shell Fock matrices ``F_mu``, the allowed
orbital-rotation descent generator is obtained from

    M = sum_mu n_mu [P_mu, F_mu].

In the real implementation ``M`` is skew-symmetric.  The primary optimizer is
L-BFGS in a fixed site-basis tangent representation, followed by projection to
the allowed MO rotation space and an exponential retraction.

Two numerical safeguards are important for the extensive S0 calculations:

1. repeated floating-point matrix products can slowly erode orthogonality even
   though ``exp(A)`` is orthogonal for skew ``A`` in exact arithmetic.  Each
   trial orbital matrix is therefore projected back to the nearest QR frame,
   with column signs aligned to the unprojected trial;
2. close to the requested orbital-gradient tolerance, the expected energy
   decrease can be below the resolution of an O(1 eV) double-precision energy.
   In that narrow regime only, a step may be accepted when it strictly lowers
   the orbital gradient and changes the energy by no more than a bound derived
   from machine epsilon.  The convergence criterion itself is never relaxed.
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
BoolArray = NDArray[np.bool_]


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


def _orthonormalize_nearby(orbitals: Matrix) -> FloatArray:
    """Return an orthonormal frame closest to a nearly orthogonal trial matrix.

    QR is used because the trial already differs from an orthogonal matrix only
    by roundoff.  Signs are chosen so each corrected column has positive overlap
    with the corresponding uncorrected column; this prevents irrelevant sign
    flips from polluting quasi-Newton secants.
    """
    trial = np.asarray(orbitals, dtype=np.float64)
    q, _ = np.linalg.qr(trial)
    overlaps = np.sum(q * trial, axis=0)
    signs = np.sign(overlaps)
    signs[signs == 0.0] = 1.0
    q = q * signs[np.newaxis, :]
    return np.asarray(q, dtype=np.float64)


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
            projectors.append(np.asarray(projector_from_orbitals(block), dtype=np.float64))
    return tuple(projectors)


def orbital_occupations(
    n_orbitals: int,
    shell_sizes: tuple[int, ...],
    definition: OpenShellStateDefinition,
) -> FloatArray:
    """Return occupation number (2, 1, or 0) assigned to each orbital column."""
    _validate_shell_sizes(n_orbitals, shell_sizes, definition)
    occupations = np.zeros(n_orbitals, dtype=np.float64)
    start = 0
    for size, occupation in zip(shell_sizes, definition.occupations, strict=True):
        occupations[start : start + size] = occupation
        start += size
    return occupations


def open_shell_orbital_energy(
    one_body: Matrix,
    interaction: FloatArray,
    orbitals: Matrix,
    shell_sizes: tuple[int, ...],
    definition: OpenShellStateDefinition,
) -> float:
    projectors = shell_projectors_from_complete_orbitals(orbitals, shell_sizes, definition)
    return general_open_shell_energy(one_body, interaction, projectors, definition)


def _allowed_rotation_mask(
    n_orbitals: int,
    shell_sizes: tuple[int, ...],
    definition: OpenShellStateDefinition,
) -> BoolArray:
    occupations = orbital_occupations(n_orbitals, shell_sizes, definition)
    return occupations[:, None] != occupations[None, :]


def _project_allowed_rotation(generator_mo: Matrix, allowed: BoolArray) -> FloatArray:
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

    With the convention used here, for an infinitesimal rotation ``A`` the
    first-order energy change is ``dE = Tr(G A)``.  Hence ``A=G`` is a descent
    direction because ``Tr(G G) <= 0`` for a real skew-symmetric matrix.
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
    r = scale * q

    for (s_vec, y_vec, rho), alpha in zip(history, reversed(alphas), strict=True):
        beta = rho * _frobenius_inner(y_vec, r)
        r = r + s_vec * (alpha - beta)
    return -r


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


def _roundoff_energy_slack(current_energy: float, candidate_energy: float) -> float:
    scale = max(1.0, abs(current_energy), abs(candidate_energy))
    return 128.0 * np.finfo(np.float64).eps * scale


def _roundoff_floor_acceptance(
    *,
    current_energy: float,
    candidate_energy: float,
    current_max_gradient: float,
    candidate_max_gradient: float,
    gradient_tolerance: float,
    gradient_window_factor: float,
) -> bool:
    """Permit progress at the energy-resolution floor without relaxing the gate."""
    if current_max_gradient > gradient_window_factor * gradient_tolerance:
        return False
    if candidate_max_gradient >= current_max_gradient:
        return False
    return candidate_energy <= current_energy + _roundoff_energy_slack(
        current_energy, candidate_energy
    )


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
    roundoff_gradient_window_factor: float = 32.0,
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
    if energy_decrease_tolerance < 0.0:
        raise ValueError("energy_decrease_tolerance must be non-negative")
    if lbfgs_history_size < 0:
        raise ValueError("lbfgs_history_size must be non-negative")
    if not 0.0 < armijo_constant < 1.0:
        raise ValueError("armijo_constant must lie between zero and one")
    if curvature_tolerance < 0.0:
        raise ValueError("curvature_tolerance must be non-negative")
    if roundoff_gradient_window_factor <= 1.0:
        raise ValueError("roundoff_gradient_window_factor must exceed one")

    c = _validate_orbitals(initial_orbitals)
    _validate_shell_sizes(c.shape[1], shell_sizes, definition)
    allowed = _allowed_rotation_mask(c.shape[1], shell_sizes, definition)
    energy = open_shell_orbital_energy(one_body, interaction, c, shell_sizes, definition)

    step = initial_step
    accepted = 0
    rejected = 0
    converged = False
    final_max_gradient = np.inf
    history: list[tuple[FloatArray, FloatArray, float]] = []
    iteration = 0

    def line_search(
        proposed_mo: FloatArray,
        directional_derivative: float,
        trial_step_start: float,
        current_max_gradient: float,
    ) -> tuple[
        bool,
        FloatArray,
        float,
        float,
        FloatArray | None,
        FloatArray | None,
        bool,
    ]:
        nonlocal rejected
        trial_step = min(trial_step_start, maximum_step)
        while trial_step >= minimum_step:
            rotation = expm(trial_step * proposed_mo)
            candidate = _orthonormalize_nearby(c @ rotation)
            candidate_energy = open_shell_orbital_energy(
                one_body, interaction, candidate, shell_sizes, definition
            )
            armijo_bound = energy + armijo_constant * trial_step * directional_derivative
            ordinary = (
                candidate_energy <= energy - energy_decrease_tolerance
                and candidate_energy <= armijo_bound
            )

            candidate_mo: FloatArray | None = None
            candidate_site: FloatArray | None = None
            roundoff = False
            if not ordinary and (
                current_max_gradient
                <= roundoff_gradient_window_factor * gradient_tolerance
            ):
                candidate_mo, candidate_site = _site_descent_generator(
                    one_body, interaction, candidate, shell_sizes, definition
                )
                candidate_max = float(np.max(np.abs(candidate_mo)))
                roundoff = _roundoff_floor_acceptance(
                    current_energy=energy,
                    candidate_energy=candidate_energy,
                    current_max_gradient=current_max_gradient,
                    candidate_max_gradient=candidate_max,
                    gradient_tolerance=gradient_tolerance,
                    gradient_window_factor=roundoff_gradient_window_factor,
                )

            if ordinary or roundoff:
                if candidate_mo is None or candidate_site is None:
                    candidate_mo, candidate_site = _site_descent_generator(
                        one_body, interaction, candidate, shell_sizes, definition
                    )
                return (
                    True,
                    candidate,
                    candidate_energy,
                    trial_step,
                    candidate_mo,
                    candidate_site,
                    roundoff,
                )

            trial_step *= shrink_factor
            rejected += 1

        return False, c, energy, trial_step, None, None, False

    for iteration in range(1, max_iterations + 1):
        descent_mo, descent_site = _site_descent_generator(
            one_body, interaction, c, shell_sizes, definition
        )
        final_max_gradient = float(np.max(np.abs(descent_mo)))
        if final_max_gradient < gradient_tolerance:
            converged = True
            break

        gradient_site = -descent_site
        proposed_site = _lbfgs_direction(gradient_site, history)
        proposed_mo = _project_allowed_rotation(c.T @ proposed_site @ c, allowed)
        directional_derivative = float(np.trace(descent_mo @ proposed_mo))
        using_steepest = False

        if (
            not np.isfinite(directional_derivative)
            or directional_derivative >= 0.0
            or float(np.max(np.abs(proposed_mo))) == 0.0
        ):
            history.clear()
            proposed_mo = descent_mo.copy()
            proposed_site = descent_site.copy()
            directional_derivative = float(np.trace(descent_mo @ proposed_mo))
            using_steepest = True
        else:
            proposed_site = c @ proposed_mo @ c.T
            proposed_site = 0.5 * (proposed_site - proposed_site.T)

        (
            ok,
            candidate,
            candidate_energy,
            trial_step,
            new_descent_mo,
            new_descent_site,
            roundoff_step,
        ) = line_search(
            proposed_mo,
            directional_derivative,
            step,
            final_max_gradient,
        )

        if not ok and not using_steepest:
            history.clear()
            proposed_mo = descent_mo.copy()
            proposed_site = descent_site.copy()
            directional_derivative = float(np.trace(descent_mo @ proposed_mo))
            (
                ok,
                candidate,
                candidate_energy,
                trial_step,
                new_descent_mo,
                new_descent_site,
                roundoff_step,
            ) = line_search(
                proposed_mo,
                directional_derivative,
                initial_step,
                final_max_gradient,
            )
            using_steepest = True

        if not ok:
            step = trial_step
            break

        assert new_descent_mo is not None
        assert new_descent_site is not None
        old_gradient_site = gradient_site
        old_direction_site = proposed_site
        c = candidate
        energy = candidate_energy
        accepted += 1

        if roundoff_step:
            # Curvature inferred from energy-floor steps is not reliable.
            history.clear()
            step = min(max(trial_step, minimum_step) * growth_factor, initial_step)
            continue

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
            not using_steepest
            and lbfgs_history_size > 0
            and np.isfinite(sy)
            and sy > curvature_tolerance * scale
        ):
            history.append((s_vec, y_vec, 1.0 / sy))
            if len(history) > lbfgs_history_size:
                history.pop(0)
        elif not using_steepest:
            history.clear()

        step = min(trial_step * growth_factor, maximum_step)

    c = _orthonormalize_nearby(c)
    projectors = shell_projectors_from_complete_orbitals(c, shell_sizes, definition)
    final_gradient = orbital_rotation_gradient(
        one_body, interaction, c, shell_sizes, definition
    )
    final_max_gradient = float(np.max(np.abs(final_gradient)))
    converged = converged or final_max_gradient < gradient_tolerance
    final_energy = open_shell_orbital_energy(
        one_body, interaction, c, shell_sizes, definition
    )

    return OpenShellOrbitalResult(
        orbitals=c,
        projectors=projectors,
        energy=final_energy,
        diagnostics=OrbitalOptimizationDiagnostics(
            iterations=iteration,
            converged=converged,
            final_energy=final_energy,
            final_max_gradient=final_max_gradient,
            accepted_steps=accepted,
            rejected_steps=rejected,
        ),
    )
