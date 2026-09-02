"""Canonical S0 controls for fully relaxed singlet/triplet calculations.

The spin-adapted many-electron model is developed on the deliberately simple
isotropic parameter set selected for the framework implementation. This module
keeps three conventions explicit:

1. the neutral pi system is half filled (one electron per molecular site);
2. a HOMO->LUMO excitation therefore leaves ``N/2-1`` closed orbitals plus two
   frontier electrons for an even ``N``-site lattice;
3. small deterministic lattice seeds are used only to break the translational
   symmetry of the isotropic control. They are not material data.

A nearest-neighbour isotropic square lattice is exactly gapless at half filling
for every periodic even-site rectangular cell. That degeneracy makes a
state-specific HOMO->LUMO Born-Oppenheimer surface non-smooth and is therefore
a poor validation target. The S0 *benchmark* uses a checkerboard site-energy
control, ``+gap/2`` and ``-gap/2``, which preserves the x/y square symmetry but
opens a defined one-particle gap. This staggered term is a numerical validation
control, not a material parameter and not part of the historical carrier model.

The structural optimizer does not reuse the legacy component-wise RPROP step.
For the neutral-referenced excited-state surface the harmonic lattice Hessian is
known analytically: ``K1 I`` for the intramolecular coordinate and a periodic
one-dimensional Laplacian with stiffness ``K2`` for each intermolecular row or
column. We solve the corresponding Newton/preconditioned-gradient direction at
fixed electronic state and perform an Armijo line search on the fully
reoptimized Born-Oppenheimer energy. Translational zero modes of the Peierls
coordinates are fixed to zero mean through the Moore-Penrose pseudoinverse of
the periodic Laplacian.

Each accepted geometry carries an electronically converged neutral and excited
state. If a warm start fails its strict orbital-gradient gate, deterministic
cold/reseeded recovery attempts are made at the same geometry before the
candidate can enter the structural line search. This prevents unconverged
electronic forces from moving the lattice. Structural convergence requires both
the true last accepted coordinate update and the final structural gradient to
pass their independent gates.
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


def isotropic_staggered_site_energies(
    parameters: StaticPolaronParameters,
    gap: float,
) -> np.ndarray:
    """Return checkerboard ``+gap/2,-gap/2`` site energies for the S0 control.

    The full noninteracting band gap is ``gap``. For nonzero gap both periodic
    dimensions must be even so the bipartite checkerboard is compatible with
    the boundary conditions and remains exactly symmetric under x/y exchange.
    """
    if gap < 0.0:
        raise ValueError("staggered control gap must be non-negative")
    if gap == 0.0:
        return np.zeros((parameters.ny, parameters.nx), dtype=np.float64)
    if parameters.nx % 2 != 0 or parameters.ny % 2 != 0:
        raise ValueError("nonzero checkerboard gap requires even nx and ny")
    y, x = np.indices((parameters.ny, parameters.nx), dtype=np.int64)
    parity = np.where((x + y) % 2 == 0, -1.0, 1.0)
    return np.asarray(0.5 * gap * parity, dtype=np.float64)


def _electronic_control_lattice(
    lattice: LatticeState,
    parameters: StaticPolaronParameters,
    staggered_gap: float,
) -> LatticeState:
    """Embed the static checkerboard potential only in the electronic Hamiltonian.

    ``build_dense_hamiltonian`` represents diagonal site energy as
    ``alpha_intra*u``. We therefore add the static potential divided by
    ``alpha_intra`` to an internal lattice copy used only by the electronic
    solver. The returned physical lattice, elastic energy, and structural
    derivative continue to use the unshifted coordinates, so the checkerboard
    term has no spurious elastic cost or force.
    """
    if staggered_gap == 0.0:
        return lattice
    if parameters.alpha_intra == 0.0:
        raise ValueError("staggered control gap requires nonzero alpha_intra")
    site_energy = isotropic_staggered_site_energies(parameters, staggered_gap)
    return LatticeState(
        u=np.asarray(lattice.u + site_energy / parameters.alpha_intra, dtype=np.float64),
        vx=np.asarray(lattice.vx, dtype=np.float64).copy(),
        vy=np.asarray(lattice.vy, dtype=np.float64).copy(),
    )


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
        for y_index, x_index in neighbours:
            u[y_index, x_index] += share
    elif seed_kind is IsotropicRelaxationSeed.BOND_X:
        vx[cy, cx] = +0.5 * amplitude
        vx[cy, (cx + 1) % parameters.nx] = -0.5 * amplitude
    else:
        vy[cy, cx] = +0.5 * amplitude
        vy[(cy + 1) % parameters.ny, cx] = -0.5 * amplitude

    return LatticeState(u=u, vx=vx, vy=vy)


def _periodic_laplacian_pseudoinverse(
    rhs: np.ndarray,
    *,
    stiffness: float,
    axis: int,
) -> np.ndarray:
    """Solve ``stiffness * L x = rhs`` with the periodic zero mode fixed to zero."""
    values = np.asarray(rhs, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("periodic lattice solve requires a two-dimensional array")
    if stiffness <= 0.0:
        raise ValueError("lattice stiffness must be positive")
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")

    n = values.shape[axis]
    projected = values - np.mean(values, axis=axis, keepdims=True)
    transformed = np.fft.fft(projected, axis=axis)
    wave = 2.0 * np.pi * np.arange(n, dtype=np.float64) / float(n)
    eigenvalues = stiffness * (2.0 - 2.0 * np.cos(wave))
    inverse = np.zeros_like(eigenvalues)
    nonzero = eigenvalues > 64.0 * np.finfo(np.float64).eps * stiffness
    inverse[nonzero] = 1.0 / eigenvalues[nonzero]

    shape = [1, 1]
    shape[axis] = n
    solution = np.fft.ifft(transformed * inverse.reshape(shape), axis=axis).real
    return np.asarray(solution, dtype=np.float64)


def harmonic_lattice_newton_direction(
    parameters: StaticPolaronParameters,
    gradient,
) -> LatticeState:
    """Return the exact fixed-electronic-state Newton direction for the lattice."""
    delta_u = -np.asarray(gradient.u, dtype=np.float64) / parameters.k1
    delta_vx = _periodic_laplacian_pseudoinverse(
        -np.asarray(gradient.vx, dtype=np.float64),
        stiffness=parameters.k2,
        axis=1,
    )
    delta_vy = _periodic_laplacian_pseudoinverse(
        -np.asarray(gradient.vy, dtype=np.float64),
        stiffness=parameters.k2,
        axis=0,
    )
    return LatticeState(u=delta_u, vx=delta_vx, vy=delta_vy)


def _maximum_lattice_component(lattice: LatticeState) -> float:
    return float(
        max(
            np.max(np.abs(lattice.u)),
            np.max(np.abs(lattice.vx)),
            np.max(np.abs(lattice.vy)),
        )
    )


def _lattice_inner_product(gradient, direction: LatticeState) -> float:
    return float(
        np.sum(np.asarray(gradient.u) * direction.u)
        + np.sum(np.asarray(gradient.vx) * direction.vx)
        + np.sum(np.asarray(gradient.vy) * direction.vy)
    )


def _displaced_lattice(
    lattice: LatticeState,
    direction: LatticeState,
    step: float,
) -> LatticeState:
    return LatticeState(
        u=np.asarray(lattice.u + step * direction.u, dtype=np.float64),
        vx=np.asarray(lattice.vx + step * direction.vx, dtype=np.float64),
        vy=np.asarray(lattice.vy + step * direction.vy, dtype=np.float64),
    )


def _combine_best_electronic_results(
    states: list[ReferencedExcitationState],
) -> ReferencedExcitationState:
    """Combine the best independently optimized neutral/excited states."""
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
    staggered_gap: float,
) -> ReferencedExcitationState:
    """Solve one geometry, recovering deterministically from a bad warm start."""
    electronic_lattice = _electronic_control_lattice(
        lattice, parameters, staggered_gap
    )
    attempts: list[ReferencedExcitationState] = []
    primary = solve_referenced_excitation(
        electronic_lattice,
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

    excited_seed = primary.neutral.orbitals if primary.neutral.diagnostics.converged else None
    recovery = solve_referenced_excitation(
        electronic_lattice,
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

    cold = solve_referenced_excitation(
        electronic_lattice,
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


def _harmonic_preconditioned_relaxation(
    parameters: StaticPolaronParameters,
    interaction: np.ndarray,
    *,
    n_closed: int,
    multiplicity: SpinMultiplicity,
    initial_lattice: LatticeState,
    orbital_gradient_tolerance: float,
    orbital_max_iterations: int,
    gradient_convergence_criterion: float,
    staggered_gap: float,
    armijo_constant: float = 1.0e-4,
    line_search_shrink: float = 0.5,
    minimum_line_search_step: float = 2.0**-24,
) -> StaticReferencedExcitationResult:
    """Relax the Born-Oppenheimer surface using the exact harmonic preconditioner."""
    if not 0.0 < armijo_constant < 1.0:
        raise ValueError("armijo_constant must lie between zero and one")
    if not 0.0 < line_search_shrink < 1.0:
        raise ValueError("line_search_shrink must lie between zero and one")
    if not 0.0 < minimum_line_search_step < 1.0:
        raise ValueError("minimum_line_search_step must lie between zero and one")

    lattice = initial_lattice.copy()
    current = _solve_electronic_with_recovery(
        lattice,
        parameters,
        interaction,
        n_closed=n_closed,
        multiplicity=multiplicity,
        neutral_orbitals=None,
        excited_orbitals=None,
        orbital_gradient_tolerance=orbital_gradient_tolerance,
        orbital_max_iterations=orbital_max_iterations,
        staggered_gap=staggered_gap,
    )
    # No coordinate move has been made yet. This also lets an already stationary
    # seed satisfy the update gate without manufacturing a fictitious update.
    final_update = 0.0
    converged = False
    iteration = 0

    for iteration in range(1, parameters.max_iterations + 1):
        neutral_ok = current.neutral.diagnostics.converged
        excited_ok = current.excited.diagnostics.converged
        gradient = referenced_excitation_gradient(lattice, parameters, current)
        gradient_value = gradient.maximum_absolute_component

        if not (neutral_ok and excited_ok):
            break
        if (
            final_update < parameters.convergence_criterion
            and gradient_value < gradient_convergence_criterion
        ):
            converged = True
            break

        direction = harmonic_lattice_newton_direction(parameters, gradient)
        direction_max = _maximum_lattice_component(direction)
        directional_derivative = _lattice_inner_product(gradient, direction)
        if (
            not np.isfinite(directional_derivative)
            or directional_derivative >= 0.0
            or not np.isfinite(direction_max)
            or direction_max == 0.0
        ):
            break

        current_energy = referenced_excitation_energy(lattice, parameters, current)
        trial_step = 1.0
        accepted = False
        while trial_step >= minimum_line_search_step:
            candidate_lattice = _displaced_lattice(lattice, direction, trial_step)
            candidate_state = _solve_electronic_with_recovery(
                candidate_lattice,
                parameters,
                interaction,
                n_closed=n_closed,
                multiplicity=multiplicity,
                neutral_orbitals=current.neutral.orbitals,
                excited_orbitals=current.excited.orbitals,
                orbital_gradient_tolerance=orbital_gradient_tolerance,
                orbital_max_iterations=orbital_max_iterations,
                staggered_gap=staggered_gap,
            )
            candidate_ok = (
                candidate_state.neutral.diagnostics.converged
                and candidate_state.excited.diagnostics.converged
            )
            if candidate_ok:
                candidate_energy = referenced_excitation_energy(
                    candidate_lattice, parameters, candidate_state
                )
                armijo_bound = (
                    current_energy.total
                    + armijo_constant * trial_step * directional_derivative
                )
                if candidate_energy.total <= armijo_bound:
                    lattice = candidate_lattice
                    current = candidate_state
                    final_update = trial_step * direction_max
                    accepted = True
                    break
            trial_step *= line_search_shrink

        if not accepted:
            break

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
    staggered_gap: float = 0.8,
) -> SpinRelaxationBranchResult:
    """Relax one half-filled singlet/triplet branch from a canonical gapped seed."""
    if parameters.nx != parameters.ny:
        raise ValueError("the canonical isotropic relaxation benchmark requires nx=ny")
    if not np.isclose(parameters.j0x, parameters.j0y):
        raise ValueError("the canonical relaxation benchmark requires Jx=Jy")
    if not np.isclose(parameters.alpha_interx, parameters.alpha_intery):
        raise ValueError("the canonical relaxation benchmark requires alpha_x=alpha_y")
    isotropic_staggered_site_energies(parameters, staggered_gap)

    seed_kind = IsotropicRelaxationSeed(seed)
    initial_lattice = isotropic_relaxation_seed(
        parameters,
        seed_kind,
        amplitude=seed_amplitude,
    )
    result = _harmonic_preconditioned_relaxation(
        parameters,
        interaction,
        n_closed=half_filled_n_closed(parameters.n_sites),
        multiplicity=multiplicity,
        initial_lattice=initial_lattice,
        orbital_gradient_tolerance=orbital_gradient_tolerance,
        orbital_max_iterations=orbital_max_iterations,
        gradient_convergence_criterion=gradient_convergence_criterion,
        staggered_gap=staggered_gap,
    )
    return SpinRelaxationBranchResult(
        multiplicity=multiplicity,
        seed=seed_kind,
        result=result,
    )
