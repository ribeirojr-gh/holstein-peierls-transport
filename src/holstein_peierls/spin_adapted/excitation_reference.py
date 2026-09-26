"""Neutral-reference excited-state energy and lattice forces for S0.

The original molecular-crystal Holstein-Peierls model is parameterized around
the neutral molecular lattice and couples explicitly to an *additional* charge.
A many-electron open-shell implementation must therefore not let the entire
neutral electronic background drive the classical coordinates.

For a neutral excitation we use the referenced electronic energy

    Delta E_el(q) = E_excited(q) - E_neutral(q)

and the effective adiabatic surface

    E_ref(q) = E_lattice(q) + Delta E_el(q).

Both electronic states are variationally optimized at the same lattice q.  With
the present frozen density-density interaction kernel, Hellmann-Feynman then
gives structural forces from the **excitation density matrix**

    Delta gamma = gamma_excited - gamma_neutral,

rather than from all occupied electrons.  The trace of Delta gamma is exactly
zero for the neutral HOMO->LUMO singlet/triplet constructions used here.

This is the effective-model analogue of coupling lattice distortions to the
charge-density redistribution caused by a neutral excitation.  It also makes
the zero-excitation limit force-free with respect to the neutral electronic
background, as required by the historical excess-carrier Hamiltonian.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..exciton.solver import ExcitonLatticeGradient
from ..hamiltonian import build_dense_hamiltonian
from ..lattice import LatticeState
from ..parameters import StaticPolaronParameters
from .open_shell import (
    CLOSED_SHELL_SINGLET,
    HIGH_SPIN_TRIPLET,
    OPEN_SHELL_SINGLET,
    OpenShellStateDefinition,
)
from .orbital_optimization import OpenShellOrbitalResult, optimize_open_shell_orbitals
from .spin import SpinMultiplicity

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ReferencedExcitationState:
    """Neutral reference and one spin-adapted excited state at fixed geometry."""

    neutral: OpenShellOrbitalResult
    excited: OpenShellOrbitalResult
    multiplicity: SpinMultiplicity
    neutral_shell_sizes: tuple[int, ...]
    excited_shell_sizes: tuple[int, ...]

    @property
    def electronic_excitation_energy(self) -> float:
        return self.excited.energy - self.neutral.energy

    @property
    def neutral_rdm(self) -> FloatArray:
        return spin_summed_rdm(
            self.neutral.projectors,
            CLOSED_SHELL_SINGLET,
        )

    @property
    def excited_rdm(self) -> FloatArray:
        definition = excited_state_definition(self.multiplicity)
        return spin_summed_rdm(self.excited.projectors, definition)

    @property
    def excitation_rdm(self) -> FloatArray:
        return self.excited_rdm - self.neutral_rdm

    @property
    def excitation_density(self) -> FloatArray:
        return np.real(np.diag(self.excitation_rdm))

    @property
    def particle_number_change(self) -> float:
        return float(np.real(np.trace(self.excitation_rdm)))


@dataclass(frozen=True, slots=True)
class ReferencedExcitationEnergy:
    electronic_excitation: float
    lattice: float
    total: float


@dataclass(frozen=True, slots=True)
class ReferencedExcitationRelaxationDiagnostics:
    iterations: int
    converged: bool
    final_max_update: float
    final_max_gradient: float
    neutral_orbitals_converged: bool
    excited_orbitals_converged: bool


@dataclass(frozen=True, slots=True)
class StaticReferencedExcitationResult:
    lattice: LatticeState
    state: ReferencedExcitationState
    energy: ReferencedExcitationEnergy
    gradient: ExcitonLatticeGradient
    diagnostics: ReferencedExcitationRelaxationDiagnostics


def excited_state_definition(
    multiplicity: SpinMultiplicity,
) -> OpenShellStateDefinition:
    if multiplicity is SpinMultiplicity.SINGLET:
        return OPEN_SHELL_SINGLET
    if multiplicity is SpinMultiplicity.TRIPLET:
        return HIGH_SPIN_TRIPLET
    raise ValueError(f"unsupported spin multiplicity: {multiplicity}")


def spin_summed_rdm(
    projectors: tuple[FloatArray, ...],
    definition: OpenShellStateDefinition,
) -> FloatArray:
    """Return ``gamma = sum_mu n_mu P_mu`` for an open-shell state."""
    if len(projectors) != definition.n_shells:
        raise ValueError("projector count does not match state definition")
    result = np.zeros_like(np.asarray(projectors[0]), dtype=np.float64)
    for occupation, projector in zip(
        definition.occupations, projectors, strict=True
    ):
        result += occupation * np.asarray(projector, dtype=np.float64)
    return 0.5 * (result + result.T)


def reference_shell_sizes(
    n_closed: int,
    multiplicity: SpinMultiplicity,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return neutral and excited shell sizes for one HOMO->LUMO excitation.

    ``n_closed`` counts the doubly occupied orbitals *below* the two frontier
    orbitals.  The neutral state therefore has ``n_closed+1`` doubly occupied
    orbitals.  The excited singlet has one singly occupied HOMO and one singly
    occupied LUMO in separate shells, while the high-spin triplet places both in
    one singly occupied shell.
    """
    if n_closed < 0:
        raise ValueError("n_closed must be non-negative")
    neutral = (n_closed + 1,)
    if multiplicity is SpinMultiplicity.SINGLET:
        excited = (n_closed, 1, 1)
    elif multiplicity is SpinMultiplicity.TRIPLET:
        excited = (n_closed, 2)
    else:
        raise ValueError(f"unsupported spin multiplicity: {multiplicity}")
    return neutral, excited


def density_density_control_interaction(
    parameters: StaticPolaronParameters,
    *,
    onsite_u: float = 0.0,
    nearest_neighbor_v: float = 0.0,
) -> FloatArray:
    """Build a simple periodic isotropic density-density validation kernel."""
    if onsite_u < 0.0 or nearest_neighbor_v < 0.0:
        raise ValueError("interaction magnitudes must be non-negative")
    n = parameters.n_sites
    v = np.zeros((n, n), dtype=np.float64)
    np.fill_diagonal(v, onsite_u)
    if nearest_neighbor_v == 0.0:
        return v
    sites = np.arange(n, dtype=np.int64).reshape(parameters.ny, parameters.nx)
    for y in range(parameters.ny):
        for x in range(parameters.nx):
            i = int(sites[y, x])
            neighbours = {
                int(sites[y, (x + 1) % parameters.nx]),
                int(sites[y, (x - 1) % parameters.nx]),
                int(sites[(y + 1) % parameters.ny, x]),
                int(sites[(y - 1) % parameters.ny, x]),
            }
            for j in neighbours:
                if j != i:
                    v[i, j] = nearest_neighbor_v
    return 0.5 * (v + v.T)


def _initial_orbitals(one_body: FloatArray) -> FloatArray:
    _, orbitals = np.linalg.eigh(np.asarray(one_body, dtype=np.float64))
    return np.asarray(orbitals, dtype=np.float64)


def solve_referenced_excitation(
    lattice: LatticeState,
    parameters: StaticPolaronParameters,
    interaction: FloatArray,
    *,
    n_closed: int,
    multiplicity: SpinMultiplicity,
    initial_neutral_orbitals: FloatArray | None = None,
    initial_excited_orbitals: FloatArray | None = None,
    orbital_gradient_tolerance: float = 1.0e-9,
    orbital_max_iterations: int = 500,
) -> ReferencedExcitationState:
    """Optimize neutral and excited electronic states at one fixed lattice."""
    lattice.validate()
    if lattice.shape != (parameters.ny, parameters.nx):
        raise ValueError("lattice shape does not match parameters")
    v = np.asarray(interaction, dtype=np.float64)
    if v.shape != (parameters.n_sites, parameters.n_sites):
        raise ValueError("interaction shape does not match lattice")

    one_body = build_dense_hamiltonian(lattice, parameters)
    canonical = (
        _initial_orbitals(one_body)
        if initial_neutral_orbitals is None or initial_excited_orbitals is None
        else None
    )
    neutral_initial = (
        canonical
        if initial_neutral_orbitals is None
        else np.asarray(initial_neutral_orbitals, dtype=np.float64)
    )
    excited_initial = (
        canonical
        if initial_excited_orbitals is None
        else np.asarray(initial_excited_orbitals, dtype=np.float64)
    )
    assert neutral_initial is not None and excited_initial is not None
    neutral_sizes, excited_sizes = reference_shell_sizes(n_closed, multiplicity)
    if sum(neutral_sizes) > parameters.n_sites:
        raise ValueError("neutral occupied space exceeds lattice orbital space")
    if sum(excited_sizes) > parameters.n_sites:
        raise ValueError("excited occupied space exceeds lattice orbital space")

    neutral = optimize_open_shell_orbitals(
        one_body,
        v,
        neutral_initial,
        neutral_sizes,
        CLOSED_SHELL_SINGLET,
        gradient_tolerance=orbital_gradient_tolerance,
        max_iterations=orbital_max_iterations,
    )
    excited = optimize_open_shell_orbitals(
        one_body,
        v,
        excited_initial,
        excited_sizes,
        excited_state_definition(multiplicity),
        gradient_tolerance=orbital_gradient_tolerance,
        max_iterations=orbital_max_iterations,
    )
    return ReferencedExcitationState(
        neutral=neutral,
        excited=excited,
        multiplicity=multiplicity,
        neutral_shell_sizes=neutral_sizes,
        excited_shell_sizes=excited_sizes,
    )


def elastic_lattice_energy(
    lattice: LatticeState,
    parameters: StaticPolaronParameters,
) -> float:
    """Return the neutral-reference harmonic lattice energy."""
    lattice.validate()
    intra = 0.5 * parameters.k1 * float(np.sum(np.square(lattice.u)))
    dx = np.roll(lattice.vx, shift=-1, axis=1) - lattice.vx
    dy = np.roll(lattice.vy, shift=-1, axis=0) - lattice.vy
    inter = 0.5 * parameters.k2 * float(
        np.sum(np.square(dx)) + np.sum(np.square(dy))
    )
    return intra + inter


def referenced_excitation_energy(
    lattice: LatticeState,
    parameters: StaticPolaronParameters,
    state: ReferencedExcitationState,
) -> ReferencedExcitationEnergy:
    lattice_part = elastic_lattice_energy(lattice, parameters)
    electronic = state.electronic_excitation_energy
    return ReferencedExcitationEnergy(
        electronic_excitation=electronic,
        lattice=lattice_part,
        total=electronic + lattice_part,
    )


def referenced_excitation_gradient(
    lattice: LatticeState,
    parameters: StaticPolaronParameters,
    state: ReferencedExcitationState,
) -> ExcitonLatticeGradient:
    """Hellmann-Feynman gradient of ``E_lattice + E_exc - E_neutral``."""
    delta_gamma = state.excitation_rdm
    n = parameters.n_sites
    sites = np.arange(n, dtype=np.int64).reshape(parameters.ny, parameters.nx)
    left = np.roll(sites, shift=1, axis=1)
    right = np.roll(sites, shift=-1, axis=1)
    up = np.roll(sites, shift=1, axis=0)
    down = np.roll(sites, shift=-1, axis=0)

    delta_density = np.real(np.diag(delta_gamma)).reshape(
        parameters.ny, parameters.nx
    )
    grad_u = parameters.k1 * lattice.u + parameters.alpha_intra * delta_density

    vx_left = np.roll(lattice.vx, shift=1, axis=1)
    vx_right = np.roll(lattice.vx, shift=-1, axis=1)
    grad_vx = parameters.k2 * (2.0 * lattice.vx - vx_left - vx_right)
    grad_vx += 2.0 * parameters.alpha_interx * (
        delta_gamma[sites, left] - delta_gamma[sites, right]
    )

    vy_up = np.roll(lattice.vy, shift=1, axis=0)
    vy_down = np.roll(lattice.vy, shift=-1, axis=0)
    grad_vy = parameters.k2 * (2.0 * lattice.vy - vy_up - vy_down)
    grad_vy += 2.0 * parameters.alpha_intery * (
        delta_gamma[sites, up] - delta_gamma[sites, down]
    )

    return ExcitonLatticeGradient(
        u=np.asarray(grad_u, dtype=np.float64),
        vx=np.asarray(grad_vx, dtype=np.float64),
        vy=np.asarray(grad_vy, dtype=np.float64),
    )


def _rprop_update(
    coordinate: FloatArray,
    gradient: FloatArray,
    previous_gradient: FloatArray,
    step_size: FloatArray,
    parameters: StaticPolaronParameters,
) -> FloatArray:
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


def relax_referenced_excitation(
    parameters: StaticPolaronParameters,
    interaction: FloatArray,
    *,
    n_closed: int,
    multiplicity: SpinMultiplicity,
    initial_lattice: LatticeState | None = None,
    orbital_gradient_tolerance: float = 1.0e-9,
    orbital_max_iterations: int = 500,
    gradient_convergence_criterion: float = 1.0e-6,
) -> StaticReferencedExcitationResult:
    """Relax the neutral-referenced spin-adapted excitation and classical lattice."""
    shape = (parameters.ny, parameters.nx)
    if initial_lattice is None:
        lattice = LatticeState(
            u=np.zeros(shape, dtype=np.float64),
            vx=np.zeros(shape, dtype=np.float64),
            vy=np.zeros(shape, dtype=np.float64),
        )
    else:
        if initial_lattice.shape != shape:
            raise ValueError("initial lattice shape does not match parameters")
        lattice = initial_lattice.copy()

    previous_u = np.zeros_like(lattice.u)
    previous_vx = np.zeros_like(lattice.vx)
    previous_vy = np.zeros_like(lattice.vy)
    step_u = np.full_like(lattice.u, parameters.update_start)
    step_vx = np.full_like(lattice.vx, parameters.update_start)
    step_vy = np.full_like(lattice.vy, parameters.update_start)
    neutral_orbitals: FloatArray | None = None
    excited_orbitals: FloatArray | None = None
    converged = False
    final_update = np.inf
    final_gradient_value = np.inf
    final_state: ReferencedExcitationState | None = None
    final_energy: ReferencedExcitationEnergy | None = None
    final_gradient: ExcitonLatticeGradient | None = None

    for iteration in range(1, parameters.max_iterations + 1):
        current = solve_referenced_excitation(
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
        gradient = referenced_excitation_gradient(lattice, parameters, current)
        delta_u = _rprop_update(
            lattice.u, gradient.u, previous_u, step_u, parameters
        )
        delta_vx = _rprop_update(
            lattice.vx, gradient.vx, previous_vx, step_vx, parameters
        )
        delta_vy = _rprop_update(
            lattice.vy, gradient.vy, previous_vy, step_vy, parameters
        )

        final_state = solve_referenced_excitation(
            lattice,
            parameters,
            interaction,
            n_closed=n_closed,
            multiplicity=multiplicity,
            initial_neutral_orbitals=current.neutral.orbitals,
            initial_excited_orbitals=current.excited.orbitals,
            orbital_gradient_tolerance=orbital_gradient_tolerance,
            orbital_max_iterations=orbital_max_iterations,
        )
        neutral_orbitals = final_state.neutral.orbitals
        excited_orbitals = final_state.excited.orbitals
        final_energy = referenced_excitation_energy(lattice, parameters, final_state)
        final_gradient = referenced_excitation_gradient(
            lattice, parameters, final_state
        )
        final_update = float(
            max(
                np.max(np.abs(delta_u)),
                np.max(np.abs(delta_vx)),
                np.max(np.abs(delta_vy)),
            )
        )
        final_gradient_value = final_gradient.maximum_absolute_component
        converged = (
            final_update < parameters.convergence_criterion
            and final_gradient_value < gradient_convergence_criterion
            and final_state.neutral.diagnostics.converged
            and final_state.excited.diagnostics.converged
        )
        if converged:
            break

    assert final_state is not None
    assert final_energy is not None
    assert final_gradient is not None
    return StaticReferencedExcitationResult(
        lattice=lattice,
        state=final_state,
        energy=final_energy,
        gradient=final_gradient,
        diagnostics=ReferencedExcitationRelaxationDiagnostics(
            iterations=iteration,
            converged=converged,
            final_max_update=final_update,
            final_max_gradient=final_gradient_value,
            neutral_orbitals_converged=final_state.neutral.diagnostics.converged,
            excited_orbitals_converged=final_state.excited.diagnostics.converged,
        ),
    )
