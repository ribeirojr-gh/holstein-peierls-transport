"""Canonical S0 controls for fully relaxed singlet/triplet calculations.

The spin-adapted many-electron model is developed on the deliberately simple
isotropic parameter set selected for the framework implementation. This module
keeps three conventions explicit:

1. the neutral pi system is half filled (one electron per molecular site);
2. a HOMO->LUMO excitation therefore leaves ``N/2-1`` closed orbitals plus two
   frontier electrons for an even ``N``-site lattice;
3. small deterministic lattice seeds are used only to break the translational
   symmetry of the perfectly isotropic control. They are not material data.

The x- and y-bond seeds are related by the square-lattice symmetry and are used
as an important regression: after relaxation their energies must agree within
numerical tolerance if both converge to symmetry-equivalent minima.

The S0 relaxation loop deliberately performs only one normal neutral+excited
electronic optimization per lattice macro-iteration. The current electronic
state supplies the Hellmann-Feynman gradient; after an RPROP update, the next
geometry is solved with warm-started orbitals. If a warm start fails its strict
orbital-gradient gate, deterministic cold/reseeded recovery attempts are made
at the same geometry before any nuclear coordinate is moved. This makes the
structural path robust without ever using unconverged electronic forces.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..lattice import LatticeState
from ..parameters import StaticPolaronParameters
from .excitation_reference import (
    ReferencedExcitationRelaxationDiagnostics,
    ReferencedExcitationState,
    StaticReferencedExcitationResult,
    referenced_excitation_energy,
    referenced_excitation_gradient,
    solve_referenced_excitation,
)
from .spin import SpinMultiplicity


class IsotropicRelaxationSeed(str, Enum):
    """Small symmetry-breaking structural seeds for the S0 control."""

    ONSITE = "onsite"
    BOND_X = "bond_x"
    BOND_Y = "bond_y"


@dataclass(frozen=True, slots=True)
class SpinRelaxationBranchResult:
    """One spin/seed branch of the fully relaxed S0 control."""

    multiplicity: SpinMultiplicity
    seed: IsotropicRelaxationSeed
    result: StaticReferencedExcitationResult


def half_filled_n_closed(n_sites: int) -> int:
    """Return ``n_closed`` for one neutral HOMO->LUMO excitation at half filling."""
    if n_sites < 2:
        raise ValueError("at least two sites are required")
    if n_sites % 2 != 0:
        raise ValueError("the half-filled closed-shell reference requires even N")
    return n_sites // 2 - 1


def isotropic_relaxation_seed(
    parameters: StaticPolaronParameters,
    seed: IsotropicRelaxationSeed | str,
    *,
    amplitude: float = 1.0e-3,
) -> LatticeState:
    """Return a deterministic zero-background symmetry-breaking lattice seed."""
    if amplitude <= 0.0:
        raise ValueError("seed amplitude must be positive")
    try:
        seed_kind = IsotropicRelaxationSeed(seed)
    except ValueError as exc:
        raise ValueError(f"unsupported isotropic relaxation seed: {seed}") from exc

    shape = (parameters.ny, parameters.nx)
    u = np.zeros(shape, dtype=np.float64)
    vx = np.zeros(shape, dtype=np.float64)
    vy = np.zeros(shape, dtype=np.float64)
    cy = parameters.ny // 2
    cx = parameters.nx // 2

    if seed_kind is IsotropicRelaxationSeed.ONSITE:
        u[cy, cx] = -amplitude
        neighbours = {
            (cy, (cx + 1) % parameters.nx),
            (cy, (cx - 1) % parameters.nx),
            ((cy + 1) % parameters.ny, cx),
            ((cy - 1) % parameters.ny, cx),
        }
        share = amplitude / len(neighbours)
        for y, x in neighbours:
            u[y, x] += share
    elif seed_kind is IsotropicRelaxationSeed.BOND_X:
        vx[cy, cx] = +0.5 * amplitude
        vx[cy, (cx + 1) % parameters.nx] = -0.5 * amplitude
    else:
        vy[cy, cx] = +0.5 * amplitude
        vy[(cy + 1) % parameters.ny, cx] = -0.5 * amplitude

    return LatticeState(u=u, vx=vx, vy=vy)


def _rprop_coordinate_update(
    coordinate: np.ndarray,
    gradient: np.ndarray,
    previous_gradient: np.ndarray,
    step_size: np.ndarray,
    parameters: StaticPolaronParameters,
) -> np.ndarray:
    """Apply one component-wise RPROP update using the legacy conventions."""
    product = gradient * previous_gradient
    positive = product > 0.0
    negative = product < 0.0
    step_size[positive] = np.minimum(
        step_size[positive] * parameters.acceleration_factor,
        parameters.update_max,
    )
    step_size[negative] = np.maximum(
        step_size[negative] * parameters.deceleration_factor,
        parameters.update_min,
    )
    effective_gradient = gradient.copy()
    effective_gradient[negative] = 0.0
    delta = -np.sign(effective_gradient) * step_size
    coordinate += delta
    previous_gradient[...] = effective_gradient
    return delta


def _combine_best_electronic_results(
    states: list[ReferencedExcitationState],
) -> ReferencedExcitationState:
    """Combine the best independently optimized neutral/excited states.

    Neutral and excited variational problems are independent at a fixed lattice.
    Recovery attempts can therefore contribute their best converged component
    separately. If no attempt converged for a component, the smallest residual
    orbital gradient is retained so failure remains visible in diagnostics.
    """
    first = states[0]

    neutral_converged = [s.neutral for s in states if s.neutral.diagnostics.converged]
    if neutral_converged:
        neutral = min(neutral_converged, key=lambda result: result.energy)
    else:
        neutral = min(
            (s.neutral for s in states),
            key=lambda result: result.diagnostics.final_max_gradient,
        )

    excited_converged = [s.excited for s in states if s.excited.diagnostics.converged]
    if excited_converged:
        excited = min(excited_converged, key=lambda result: result.energy)
    else:
        excited = min(
            (s.excited for s in states),
            key=lambda result: result.diagnostics.final_max_gradient,
        )

    return ReferencedExcitationState(
        neutral=neutral,
        excited=excited,
        multiplicity=first.multiplicity,
        neutral_shell_sizes=first.neutral_shell_sizes,
        excited_shell_sizes=first.excited_shell_sizes,
    )


def _solve_electronic_with_recovery(
    lattice: LatticeState,
    parameters: StaticPolaronParameters,
    interaction: np.ndarray,
    *,
    n_closed: int,
    multiplicity: SpinMultiplicity,
    neutral_orbitals: np.ndarray | None,
    excited_orbitals: np.ndarray | None,
    orbital_gradient_tolerance: float,
    orbital_max_iterations: int,
) -> ReferencedExcitationState:
    """Solve one geometry, recovering deterministically from a bad warm start."""
    attempts: list[ReferencedExcitationState] = []
    primary = solve_referenced_excitation(
        lattice,
        parameters,
        interaction,
        n_closed=n_closed,
        multiplicity=multiplicity,
        initial_neutral_orbitals=neutral_orbitals,
        initial_excited_orbitals=excited_orbitals,
        orbital_gradient_tolerance=orbital_gradient_tolerance,
        orbital_max_iterations=orbital_max_iterations,
    )
    attempts.append(primary)
    if primary.neutral.diagnostics.converged and primary.excited.diagnostics.converged:
        return primary

    # First recovery: rebuild the neutral state canonically and seed the excited
    # optimization from the best available neutral frame. The larger iteration
    # budget is paid only after a failed primary warm start.
    excited_seed = (
        primary.neutral.orbitals
        if primary.neutral.diagnostics.converged
        else None
    )
    recovery = solve_referenced_excitation(
        lattice,
        parameters,
        interaction,
        n_closed=n_closed,
        multiplicity=multiplicity,
        initial_neutral_orbitals=None,
        initial_excited_orbitals=excited_seed,
        orbital_gradient_tolerance=orbital_gradient_tolerance,
        orbital_max_iterations=max(2 * orbital_max_iterations, orbital_max_iterations + 200),
    )
    attempts.append(recovery)
    combined = _combine_best_electronic_results(attempts)
    if combined.neutral.diagnostics.converged and combined.excited.diagnostics.converged:
        return combined

    # Final recovery: both sectors start from the canonical one-body basis.
    # This is deliberately expensive but deterministic and only reached if both
    # previous representations failed the strict variational gate.
    cold = solve_referenced_excitation(
        lattice,
        parameters,
        interaction,
        n_closed=n_closed,
        multiplicity=multiplicity,
        initial_neutral_orbitals=None,
        initial_excited_orbitals=None,
        orbital_gradient_tolerance=orbital_gradient_tolerance,
        orbital_max_iterations=max(4 * orbital_max_iterations, orbital_max_iterations + 500),
    )
    attempts.append(cold)
    return _combine_best_electronic_results(attempts)


def _single_solve_relaxation(
    parameters: StaticPolaronParameters,
    interaction: np.ndarray,
    *,
    n_closed: int,
    multiplicity: SpinMultiplicity,
    initial_lattice: LatticeState,
    orbital_gradient_tolerance: float,
    orbital_max_iterations: int,
    gradient_convergence_criterion: float,
) -> StaticReferencedExcitationResult:
    """Relax lattice and orbitals with one normal electronic solve per macro-step."""
    lattice = initial_lattice.copy()
    previous_u = np.zeros_like(lattice.u)
    previous_vx = np.zeros_like(lattice.vx)
    previous_vy = np.zeros_like(lattice.vy)
    step_u = np.full_like(lattice.u, parameters.update_start)
    step_vx = np.full_like(lattice.vx, parameters.update_start)
    step_vy = np.full_like(lattice.vy, parameters.update_start)

    neutral_orbitals: np.ndarray | None = None
    excited_orbitals: np.ndarray | None = None
    current: ReferencedExcitationState | None = None
    final_update = np.inf
    converged = False
    state_matches_lattice = False
    iteration = 0

    for iteration in range(1, parameters.max_iterations + 1):
        current = _solve_electronic_with_recovery(
            lattice,
            parameters,
            interaction,
            n_closed=n_closed,
            multiplicity=multiplicity,
            neutral_orbitals=neutral_orbitals,
            excited_orbitals=excited_orbitals,
            orbital_gradient_tolerance=orbital_gradient_tolerance,
            orbital_max_iterations=orbital_max_iterations,
        )
        state_matches_lattice = True
        neutral_ok = current.neutral.diagnostics.converged
        excited_ok = current.excited.diagnostics.converged
        gradient = referenced_excitation_gradient(lattice, parameters, current)
        gradient_value = gradient.maximum_absolute_component

        if (
            neutral_ok
            and excited_ok
            and final_update < parameters.convergence_criterion
            and gradient_value < gradient_convergence_criterion
        ):
            converged = True
            break

        if not (neutral_ok and excited_ok):
            break

        delta_u = _rprop_coordinate_update(
            lattice.u, gradient.u, previous_u, step_u, parameters
        )
        delta_vx = _rprop_coordinate_update(
            lattice.vx, gradient.vx, previous_vx, step_vx, parameters
        )
        delta_vy = _rprop_coordinate_update(
            lattice.vy, gradient.vy, previous_vy, step_vy, parameters
        )
        final_update = float(
            max(
                np.max(np.abs(delta_u)),
                np.max(np.abs(delta_vx)),
                np.max(np.abs(delta_vy)),
            )
        )
        neutral_orbitals = current.neutral.orbitals
        excited_orbitals = current.excited.orbitals
        state_matches_lattice = False

    assert current is not None

    if not state_matches_lattice:
        current = _solve_electronic_with_recovery(
            lattice,
            parameters,
            interaction,
            n_closed=n_closed,
            multiplicity=multiplicity,
            neutral_orbitals=neutral_orbitals,
            excited_orbitals=excited_orbitals,
            orbital_gradient_tolerance=orbital_gradient_tolerance,
            orbital_max_iterations=orbital_max_iterations,
        )

    final_energy = referenced_excitation_energy(lattice, parameters, current)
    final_gradient = referenced_excitation_gradient(lattice, parameters, current)
    final_gradient_value = final_gradient.maximum_absolute_component
    neutral_ok = current.neutral.diagnostics.converged
    excited_ok = current.excited.diagnostics.converged
    converged = converged or (
        neutral_ok
        and excited_ok
        and final_update < parameters.convergence_criterion
        and final_gradient_value < gradient_convergence_criterion
    )

    return StaticReferencedExcitationResult(
        lattice=lattice,
        state=current,
        energy=final_energy,
        gradient=final_gradient,
        diagnostics=ReferencedExcitationRelaxationDiagnostics(
            iterations=iteration,
            converged=converged,
            final_max_update=float(final_update),
            final_max_gradient=float(final_gradient_value),
            neutral_orbitals_converged=neutral_ok,
            excited_orbitals_converged=excited_ok,
        ),
    )


def relax_isotropic_spin_branch(
    parameters: StaticPolaronParameters,
    interaction: np.ndarray,
    *,
    multiplicity: SpinMultiplicity,
    seed: IsotropicRelaxationSeed | str,
    seed_amplitude: float = 1.0e-3,
    orbital_gradient_tolerance: float = 1.0e-8,
    orbital_max_iterations: int = 800,
    gradient_convergence_criterion: float = 1.0e-6,
) -> SpinRelaxationBranchResult:
    """Relax one half-filled singlet/triplet branch from a canonical seed."""
    if parameters.nx != parameters.ny:
        raise ValueError("the canonical isotropic relaxation benchmark requires nx=ny")
    if not np.isclose(parameters.j0x, parameters.j0y):
        raise ValueError("the canonical relaxation benchmark requires Jx=Jy")
    if not np.isclose(parameters.alpha_interx, parameters.alpha_intery):
        raise ValueError("the canonical relaxation benchmark requires alpha_x=alpha_y")

    seed_kind = IsotropicRelaxationSeed(seed)
    initial_lattice = isotropic_relaxation_seed(
        parameters,
        seed_kind,
        amplitude=seed_amplitude,
    )
    result = _single_solve_relaxation(
        parameters,
        interaction,
        n_closed=half_filled_n_closed(parameters.n_sites),
        multiplicity=multiplicity,
        initial_lattice=initial_lattice,
        orbital_gradient_tolerance=orbital_gradient_tolerance,
        orbital_max_iterations=orbital_max_iterations,
        gradient_convergence_criterion=gradient_convergence_criterion,
    )
    return SpinRelaxationBranchResult(
        multiplicity=multiplicity,
        seed=seed_kind,
        result=result,
    )
