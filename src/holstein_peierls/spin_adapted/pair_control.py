"""Reduced spin-adapted electron-hole control used during S0 validation.

This module is deliberately *not* the production MCTDHF implementation.  It
adds a frozen short-range exchange kernel to the already validated
spin-blind/distinguishable electron-hole Hamiltonian so that the following
cross-checks can be performed before orbital-optimized open-shell SCF is added:

- exchange -> 0 recovers the spin-blind pair model;
- positive exchange produces the standard two-open-shell ordering
  E_singlet - E_triplet > 0 for the same spatial pair;
- the same one-body reduced density matrices drive the Holstein-Peierls
  Hellmann-Feynman force;
- singlet and triplet lattice relaxations can be exercised through one API.

The exchange kernel here is a numerical control parameter, not a fitted
material property.  Production singlet/triplet calculations will use the
open-shell shell-Fock functional and its self-consistent orbitals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator, eigsh

from ..exciton.interaction import electron_hole_interaction_matrix
from ..exciton.parameters import ExcitonParameters
from ..exciton.solver import (
    ExcitonEnergy,
    ExcitonLatticeGradient,
    ExcitonRelaxationDiagnostics,
    build_carrier_hamiltonian,
    initial_lattice_state,
    initial_wavefunction,
    lattice_energy,
)
from ..lattice import LatticeState
from .spin import SpinMultiplicity, spin_exchange_sign

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ExchangeControl:
    """Frozen short-range electron-hole exchange magnitudes in eV."""

    onsite: float = 0.0
    nearest_neighbor: float = 0.0

    def __post_init__(self) -> None:
        if self.onsite < 0.0 or self.nearest_neighbor < 0.0:
            raise ValueError("exchange magnitudes must be non-negative")


@dataclass(frozen=True, slots=True)
class SpinAdaptedPairState:
    """Lowest pair eigenstate in a selected spin sector for a fixed lattice."""

    energy: float
    wavefunction: FloatArray
    multiplicity: SpinMultiplicity

    @property
    def electron_density_matrix(self) -> FloatArray:
        return self.wavefunction @ self.wavefunction.T

    @property
    def hole_density_matrix(self) -> FloatArray:
        return self.wavefunction.T @ self.wavefunction

    @property
    def onsite_probability(self) -> float:
        return float(np.sum(np.square(np.diag(self.wavefunction))))


@dataclass(frozen=True, slots=True)
class StaticSpinAdaptedPairResult:
    lattice: LatticeState
    ground_state: SpinAdaptedPairState
    energy: ExcitonEnergy
    diagnostics: ExcitonRelaxationDiagnostics


def exchange_control_matrix(
    parameters: ExcitonParameters,
    exchange: ExchangeControl,
) -> FloatArray:
    """Build a periodic onsite/cardinal exchange kernel ``K[i_e,i_h]``."""
    n = parameters.n_sites
    result = np.zeros((n, n), dtype=np.float64)
    for electron in range(n):
        ey, ex = divmod(electron, parameters.nx)
        for hole in range(n):
            hy, hx = divmod(hole, parameters.nx)
            dx_raw = abs(ex - hx)
            dy_raw = abs(ey - hy)
            dx = min(dx_raw, parameters.nx - dx_raw) if parameters.nx > 1 else 0
            dy = min(dy_raw, parameters.ny - dy_raw) if parameters.ny > 1 else 0
            if dx == 0 and dy == 0:
                result[electron, hole] = exchange.onsite
            elif dx + dy == 1:
                result[electron, hole] = exchange.nearest_neighbor
    return result


def spin_adapted_pair_interaction(
    parameters: ExcitonParameters,
    exchange: ExchangeControl,
    multiplicity: SpinMultiplicity,
) -> FloatArray:
    """Return direct attraction plus the spin-dependent exchange contribution."""
    direct = electron_hole_interaction_matrix(parameters)
    exchange_matrix = exchange_control_matrix(parameters, exchange)
    return direct + spin_exchange_sign(multiplicity) * exchange_matrix


def spin_adapted_pair_operator(
    state: LatticeState,
    parameters: ExcitonParameters,
    exchange: ExchangeControl,
    multiplicity: SpinMultiplicity,
) -> LinearOperator:
    """Return the matrix-free reduced spin-adapted pair Hamiltonian."""
    if state.shape != (parameters.ny, parameters.nx):
        raise ValueError("lattice shape does not match exciton parameters")
    electron_h = build_carrier_hamiltonian(state, parameters, "electron")
    hole_h = build_carrier_hamiltonian(state, parameters, "hole")
    interaction = spin_adapted_pair_interaction(parameters, exchange, multiplicity)
    n = parameters.n_sites

    def matvec(vector: FloatArray) -> FloatArray:
        psi = np.asarray(vector, dtype=np.float64).reshape((n, n), order="C")
        result = electron_h @ psi + (hole_h @ psi.T).T
        result = np.asarray(result, dtype=np.float64)
        result += interaction * psi
        return result.ravel(order="C")

    return LinearOperator((n * n, n * n), matvec=matvec, dtype=np.float64)


def _normalize(wavefunction: FloatArray) -> FloatArray:
    psi = np.asarray(wavefunction, dtype=np.float64).copy()
    norm = float(np.linalg.norm(psi))
    if not np.isfinite(norm) or norm < 1.0e-14:
        raise RuntimeError("spin-adapted pair wavefunction has invalid norm")
    psi /= norm
    anchor = np.unravel_index(int(np.argmax(np.abs(psi))), psi.shape)
    if psi[anchor] < 0.0:
        psi = -psi
    return psi


def solve_spin_adapted_pair(
    state: LatticeState,
    parameters: ExcitonParameters,
    exchange: ExchangeControl,
    multiplicity: SpinMultiplicity,
    *,
    initial_state: FloatArray | None = None,
) -> SpinAdaptedPairState:
    """Solve the lowest reduced pair state for one spin multiplicity."""
    n = parameters.n_sites
    operator = spin_adapted_pair_operator(state, parameters, exchange, multiplicity)
    if initial_state is None:
        v0 = initial_wavefunction(parameters, "frenkel").ravel(order="C")
    else:
        v0 = _normalize(np.asarray(initial_state).reshape((n, n))).ravel(order="C")

    dimension = n * n
    if dimension <= 4:
        dense = np.column_stack(
            [operator @ np.eye(dimension, dtype=np.float64)[:, i] for i in range(dimension)]
        )
        eigenvalues, eigenvectors = np.linalg.eigh(dense)
        vector = eigenvectors[:, int(np.argmin(eigenvalues))]
    else:
        _, eigenvectors = eigsh(
            operator,
            k=1,
            which="SA",
            v0=v0,
            tol=parameters.eigensolver_tolerance,
            maxiter=parameters.eigensolver_max_iterations,
        )
        vector = eigenvectors[:, 0]

    psi = _normalize(vector.reshape((n, n), order="C"))
    applied = operator @ psi.ravel(order="C")
    energy = float(np.dot(psi.ravel(order="C"), applied))
    return SpinAdaptedPairState(
        energy=energy,
        wavefunction=psi,
        multiplicity=multiplicity,
    )


def spin_adapted_pair_gradient(
    state: LatticeState,
    parameters: ExcitonParameters,
    exchange: ExchangeControl,
    multiplicity: SpinMultiplicity,
    *,
    ground_state: SpinAdaptedPairState | None = None,
) -> tuple[ExcitonLatticeGradient, SpinAdaptedPairState]:
    """Return the shared-lattice Hellmann-Feynman gradient.

    Direct attraction and exchange are frozen with respect to the lattice in
    this S0 control, so they enter the force only through the optimized RDMs.
    """
    if ground_state is None:
        ground_state = solve_spin_adapted_pair(
            state, parameters, exchange, multiplicity
        )
    gamma_e = ground_state.electron_density_matrix
    gamma_h = ground_state.hole_density_matrix
    sites = np.arange(parameters.n_sites, dtype=np.int64).reshape(
        parameters.ny, parameters.nx
    )
    left = np.roll(sites, shift=1, axis=1)
    right = np.roll(sites, shift=-1, axis=1)
    up = np.roll(sites, shift=1, axis=0)
    down = np.roll(sites, shift=-1, axis=0)

    density_e = np.diag(gamma_e).reshape((parameters.ny, parameters.nx))
    density_h = np.diag(gamma_h).reshape((parameters.ny, parameters.nx))

    grad_u = (
        parameters.k1 * state.u
        + parameters.electron_alpha_intra * density_e
        + parameters.hole_alpha_intra * density_h
    )

    vx_left = np.roll(state.vx, shift=1, axis=1)
    vx_right = np.roll(state.vx, shift=-1, axis=1)
    grad_vx = parameters.k2 * (2.0 * state.vx - vx_left - vx_right)
    grad_vx += 2.0 * parameters.electron_alpha_interx * (
        gamma_e[sites, left] - gamma_e[sites, right]
    )
    grad_vx += 2.0 * parameters.hole_alpha_interx * (
        gamma_h[sites, left] - gamma_h[sites, right]
    )

    vy_up = np.roll(state.vy, shift=1, axis=0)
    vy_down = np.roll(state.vy, shift=-1, axis=0)
    grad_vy = parameters.k2 * (2.0 * state.vy - vy_up - vy_down)
    grad_vy += 2.0 * parameters.electron_alpha_intery * (
        gamma_e[sites, up] - gamma_e[sites, down]
    )
    grad_vy += 2.0 * parameters.hole_alpha_intery * (
        gamma_h[sites, up] - gamma_h[sites, down]
    )

    return (
        ExcitonLatticeGradient(
            u=np.asarray(grad_u, dtype=np.float64),
            vx=np.asarray(grad_vx, dtype=np.float64),
            vy=np.asarray(grad_vy, dtype=np.float64),
        ),
        ground_state,
    )


def _rprop_update(
    coordinate: FloatArray,
    gradient: FloatArray,
    previous_gradient: FloatArray,
    step_size: FloatArray,
    parameters: ExcitonParameters,
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


def relax_spin_adapted_pair(
    parameters: ExcitonParameters,
    exchange: ExchangeControl,
    multiplicity: SpinMultiplicity,
    *,
    initial_lattice: LatticeState | None = None,
) -> StaticSpinAdaptedPairResult:
    """Relax the reduced spin-adapted pair on the shared classical lattice."""
    if initial_lattice is None:
        lattice = initial_lattice_state(parameters, "frenkel")
    else:
        if initial_lattice.shape != (parameters.ny, parameters.nx):
            raise ValueError("initial lattice shape does not match parameters")
        lattice = initial_lattice.copy()

    previous_u = np.zeros_like(lattice.u)
    previous_vx = np.zeros_like(lattice.vx)
    previous_vy = np.zeros_like(lattice.vy)
    step_u = np.full_like(lattice.u, parameters.update_start)
    step_vx = np.full_like(lattice.vx, parameters.update_start)
    step_vy = np.full_like(lattice.vy, parameters.update_start)
    cached = initial_wavefunction(parameters, "frenkel")

    converged = False
    final_update = np.inf
    final_gradient = np.inf
    final_state: SpinAdaptedPairState | None = None
    final_energy: ExcitonEnergy | None = None

    for iteration in range(1, parameters.max_iterations + 1):
        current = solve_spin_adapted_pair(
            lattice,
            parameters,
            exchange,
            multiplicity,
            initial_state=cached,
        )
        gradient, _ = spin_adapted_pair_gradient(
            lattice,
            parameters,
            exchange,
            multiplicity,
            ground_state=current,
        )
        delta_u = _rprop_update(lattice.u, gradient.u, previous_u, step_u, parameters)
        delta_vx = _rprop_update(
            lattice.vx, gradient.vx, previous_vx, step_vx, parameters
        )
        delta_vy = _rprop_update(
            lattice.vy, gradient.vy, previous_vy, step_vy, parameters
        )

        final_state = solve_spin_adapted_pair(
            lattice,
            parameters,
            exchange,
            multiplicity,
            initial_state=current.wavefunction,
        )
        lattice_part = lattice_energy(lattice, parameters)
        final_energy = ExcitonEnergy(
            electronic=final_state.energy,
            lattice=lattice_part,
            total=final_state.energy + lattice_part,
        )
        checked_gradient, _ = spin_adapted_pair_gradient(
            lattice,
            parameters,
            exchange,
            multiplicity,
            ground_state=final_state,
        )
        cached = final_state.wavefunction
        final_update = float(
            max(
                np.max(np.abs(delta_u)),
                np.max(np.abs(delta_vx)),
                np.max(np.abs(delta_vy)),
            )
        )
        final_gradient = checked_gradient.maximum_absolute_component
        converged = (
            final_update < parameters.convergence_criterion
            and final_gradient < parameters.gradient_convergence_criterion
        )
        if converged:
            break

    assert final_state is not None and final_energy is not None
    return StaticSpinAdaptedPairResult(
        lattice=lattice,
        ground_state=final_state,
        energy=final_energy,
        diagnostics=ExcitonRelaxationDiagnostics(
            iterations=iteration,
            converged=converged,
            final_max_update=final_update,
            final_max_gradient=final_gradient,
        ),
    )


def singlet_triplet_gap(
    state: LatticeState,
    parameters: ExcitonParameters,
    exchange: ExchangeControl,
) -> float:
    """Return ``E_singlet - E_triplet`` for one fixed lattice."""
    singlet = solve_spin_adapted_pair(
        state, parameters, exchange, SpinMultiplicity.SINGLET
    )
    triplet = solve_spin_adapted_pair(
        state, parameters, exchange, SpinMultiplicity.TRIPLET
    )
    return singlet.energy - triplet.energy
