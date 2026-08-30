"""Static singlet extended Holstein-Peierls-Hubbard bipolaron solver.

The one-particle bond Hamiltonian follows the same sign and periodic-index
convention as the validated single-polaron solver. Electronic forces are
obtained by replacing the one-particle density matrix with the spin-summed
one-body reduced density matrix of the correlated two-particle singlet.
The pair interaction is currently lattice-geometry independent: onsite U and
nearest-neighbour V1 alter the electronic state but add no explicit classical
force beyond their effect on that state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import LinearOperator, eigsh

from ..lattice import LatticeState
from .bipolaron import (
    BipolaronEnergy,
    BipolaronGroundState,
    BipolaronRelaxationDiagnostics,
    InitializationMode,
    _apply_two_particle_hamiltonian,
    _initial_singlet_vector,
    _normalize_symmetric_wavefunction,
    initial_distortion,
)
from .interaction import pair_interaction_matrix
from .parameters import BipolaronParameters

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class BipolaronLatticeGradient:
    """Energy derivatives with respect to ``u``, ``vx``, and ``vy``."""

    u: FloatArray
    vx: FloatArray
    vy: FloatArray

    @property
    def maximum_absolute_component(self) -> float:
        return float(
            max(
                np.max(np.abs(self.u)),
                np.max(np.abs(self.vx)),
                np.max(np.abs(self.vy)),
            )
        )


@dataclass(frozen=True, slots=True)
class HolsteinPeierlsBipolaronResult:
    """Fully relaxed static two-particle Holstein-Peierls state."""

    lattice: LatticeState
    ground_state: BipolaronGroundState
    energy: BipolaronEnergy
    diagnostics: BipolaronRelaxationDiagnostics

    @property
    def u(self) -> FloatArray:
        return self.lattice.u

    @property
    def vx(self) -> FloatArray:
        return self.lattice.vx

    @property
    def vy(self) -> FloatArray:
        return self.lattice.vy

    @property
    def onsite_pair_probability(self) -> float:
        return self.ground_state.onsite_pair_probability


def _validate_lattice(state: LatticeState, parameters: BipolaronParameters) -> None:
    state.validate()
    if state.shape != (parameters.ny, parameters.nx):
        raise ValueError("lattice shape does not match bipolaron parameters")


def bond_transfer_integrals(
    state: LatticeState,
    parameters: BipolaronParameters,
) -> tuple[FloatArray, FloatArray]:
    """Return Peierls-modulated transfer integrals on +x and +y bonds."""
    _validate_lattice(state, parameters)
    vx_right = np.roll(state.vx, shift=-1, axis=1)
    vy_down = np.roll(state.vy, shift=-1, axis=0)
    tx = -parameters.j0x + parameters.alpha_interx * (vx_right - state.vx)
    ty = -parameters.j0y + parameters.alpha_intery * (vy_down - state.vy)
    return np.asarray(tx, dtype=np.float64), np.asarray(ty, dtype=np.float64)


def _site_neighbours(
    parameters: BipolaronParameters,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    sites = np.arange(parameters.n_sites, dtype=np.int64).reshape(
        parameters.ny, parameters.nx
    )
    return (
        sites.ravel(order="C"),
        np.roll(sites, shift=-1, axis=1).ravel(order="C"),
        np.roll(sites, shift=-1, axis=0).ravel(order="C"),
    )


def build_one_particle_hamiltonian(
    state: LatticeState,
    parameters: BipolaronParameters,
) -> csr_matrix:
    """Build the Peierls-modulated one-carrier Hamiltonian in CSR form."""
    _validate_lattice(state, parameters)
    n = parameters.n_sites
    diagonal = parameters.alpha_intra * state.u.ravel(order="C")
    tx, ty = bond_transfer_integrals(state, parameters)

    if parameters.nx < 3 or parameters.ny < 3:
        h = np.zeros((n, n), dtype=np.float64)
        np.fill_diagonal(h, diagonal)
        sites = np.arange(n, dtype=np.int64).reshape(parameters.ny, parameters.nx)
        for y in range(parameters.ny):
            for x in range(parameters.nx):
                i = int(sites[y, x])
                jx = int(sites[y, (x + 1) % parameters.nx])
                jy = int(sites[(y + 1) % parameters.ny, x])
                h[i, jx] = h[jx, i] = tx[y, x]
                h[i, jy] = h[jy, i] = ty[y, x]
        return csr_matrix(h)

    sites, right, down = _site_neighbours(parameters)
    tx_flat = tx.ravel(order="C")
    ty_flat = ty.ravel(order="C")
    rows = np.concatenate((sites, sites, right, sites, down))
    cols = np.concatenate((sites, right, sites, down, sites))
    data = np.concatenate((diagonal, tx_flat, tx_flat, ty_flat, ty_flat))
    return coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def two_particle_linear_operator(
    state: LatticeState,
    parameters: BipolaronParameters,
) -> LinearOperator:
    """Return ``H1(q) tensor I + I tensor H1(q) + V_ij`` matrix-free."""
    h1 = build_one_particle_hamiltonian(state, parameters)
    interaction = pair_interaction_matrix(parameters)
    n = parameters.n_sites

    def matvec(vector: FloatArray) -> FloatArray:
        psi = np.asarray(vector, dtype=np.float64).reshape((n, n), order="C")
        return _apply_two_particle_hamiltonian(psi, h1, interaction).ravel(order="C")

    return LinearOperator((n * n, n * n), matvec=matvec, dtype=np.float64)


def solve_holstein_peierls_ground_state(
    state: LatticeState,
    parameters: BipolaronParameters,
    *,
    initial_wavefunction: FloatArray | None = None,
) -> BipolaronGroundState:
    """Solve the lowest symmetric spatial state for a fixed full lattice."""
    _validate_lattice(state, parameters)
    n = parameters.n_sites
    operator = two_particle_linear_operator(state, parameters)

    if initial_wavefunction is None:
        v0 = _initial_singlet_vector(parameters)
    else:
        psi0 = np.asarray(initial_wavefunction, dtype=np.float64).reshape((n, n))
        v0 = _normalize_symmetric_wavefunction(psi0).ravel(order="C")

    _, eigenvectors = eigsh(
        operator,
        k=1,
        which="SA",
        v0=v0,
        tol=parameters.eigensolver_tolerance,
        maxiter=parameters.eigensolver_max_iterations,
    )
    psi = _normalize_symmetric_wavefunction(eigenvectors[:, 0].reshape((n, n)))
    applied = operator @ psi.ravel(order="C")
    energy = float(np.dot(psi.ravel(order="C"), applied))
    return BipolaronGroundState(energy=energy, wavefunction=psi)


def lattice_energy(
    state: LatticeState,
    parameters: BipolaronParameters,
) -> float:
    """Return the complete Holstein plus Peierls elastic energy."""
    _validate_lattice(state, parameters)
    intra = 0.5 * parameters.k1 * float(np.sum(np.square(state.u)))
    dx = np.roll(state.vx, shift=-1, axis=1) - state.vx
    dy = np.roll(state.vy, shift=-1, axis=0) - state.vy
    inter = 0.5 * parameters.k2 * float(
        np.sum(np.square(dx)) + np.sum(np.square(dy))
    )
    return intra + inter


def total_energy(
    state: LatticeState,
    parameters: BipolaronParameters,
    *,
    ground_state: BipolaronGroundState | None = None,
) -> tuple[BipolaronEnergy, BipolaronGroundState]:
    """Return electronic, elastic, and total adiabatic energy."""
    if ground_state is None:
        ground_state = solve_holstein_peierls_ground_state(state, parameters)
    lattice = lattice_energy(state, parameters)
    return (
        BipolaronEnergy(
            electronic=ground_state.energy,
            lattice=lattice,
            total=ground_state.energy + lattice,
        ),
        ground_state,
    )


def energy_gradient(
    state: LatticeState,
    parameters: BipolaronParameters,
    *,
    ground_state: BipolaronGroundState | None = None,
) -> tuple[BipolaronLatticeGradient, BipolaronGroundState]:
    """Hellmann-Feynman gradient for all three classical lattice fields."""
    _validate_lattice(state, parameters)
    if ground_state is None:
        ground_state = solve_holstein_peierls_ground_state(state, parameters)

    gamma = ground_state.one_body_density_matrix
    sites = np.arange(parameters.n_sites, dtype=np.int64).reshape(
        parameters.ny, parameters.nx
    )
    left = np.roll(sites, shift=1, axis=1)
    right = np.roll(sites, shift=-1, axis=1)
    up = np.roll(sites, shift=1, axis=0)
    down = np.roll(sites, shift=-1, axis=0)

    density = np.diag(gamma).reshape((parameters.ny, parameters.nx), order="C")
    gamma_left = gamma[sites, left]
    gamma_right = gamma[sites, right]
    gamma_up = gamma[sites, up]
    gamma_down = gamma[sites, down]

    vx_left = np.roll(state.vx, shift=1, axis=1)
    vx_right = np.roll(state.vx, shift=-1, axis=1)
    vy_up = np.roll(state.vy, shift=1, axis=0)
    vy_down = np.roll(state.vy, shift=-1, axis=0)

    grad_u = parameters.k1 * state.u + parameters.alpha_intra * density
    grad_vx = (
        parameters.k2 * (2.0 * state.vx - vx_left - vx_right)
        + 2.0 * parameters.alpha_interx * (gamma_left - gamma_right)
    )
    grad_vy = (
        parameters.k2 * (2.0 * state.vy - vy_up - vy_down)
        + 2.0 * parameters.alpha_intery * (gamma_up - gamma_down)
    )

    return (
        BipolaronLatticeGradient(
            u=np.asarray(grad_u, dtype=np.float64),
            vx=np.asarray(grad_vx, dtype=np.float64),
            vy=np.asarray(grad_vy, dtype=np.float64),
        ),
        ground_state,
    )


def initial_lattice_state(
    parameters: BipolaronParameters,
    mode: InitializationMode = "onsite",
) -> LatticeState:
    """Return a Holstein seed with initially unstrained Peierls coordinates."""
    return LatticeState(
        u=initial_distortion(parameters, mode),
        vx=np.zeros((parameters.ny, parameters.nx), dtype=np.float64),
        vy=np.zeros((parameters.ny, parameters.nx), dtype=np.float64),
    )


def _rprop_update(
    coordinate: FloatArray,
    gradient: FloatArray,
    previous_gradient: FloatArray,
    step_size: FloatArray,
    parameters: BipolaronParameters,
) -> tuple[FloatArray, FloatArray]:
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
    return delta, effective_gradient


def relax_static_holstein_peierls_bipolaron(
    parameters: BipolaronParameters,
    *,
    initialization: InitializationMode = "onsite",
    initial_state: LatticeState | None = None,
) -> HolsteinPeierlsBipolaronResult:
    """Relax ``u``, ``vx``, and ``vy`` while requiring stationary forces."""
    if initial_state is None:
        lattice = initial_lattice_state(parameters, initialization)
    else:
        _validate_lattice(initial_state, parameters)
        lattice = initial_state.copy()

    previous_u = np.zeros_like(lattice.u)
    previous_vx = np.zeros_like(lattice.vx)
    previous_vy = np.zeros_like(lattice.vy)
    step_u = np.full_like(lattice.u, parameters.update_start)
    step_vx = np.full_like(lattice.vx, parameters.update_start)
    step_vy = np.full_like(lattice.vy, parameters.update_start)

    cached_state: BipolaronGroundState | None = None
    converged = False
    final_max_update = np.inf
    final_max_gradient = np.inf
    final_energy: BipolaronEnergy | None = None
    final_state: BipolaronGroundState | None = None

    for iteration in range(1, parameters.max_iterations + 1):
        gradient, current_state = energy_gradient(
            lattice,
            parameters,
            ground_state=cached_state,
        )

        delta_u, _ = _rprop_update(
            lattice.u, gradient.u, previous_u, step_u, parameters
        )
        delta_vx, _ = _rprop_update(
            lattice.vx, gradient.vx, previous_vx, step_vx, parameters
        )
        delta_vy, _ = _rprop_update(
            lattice.vy, gradient.vy, previous_vy, step_vy, parameters
        )

        final_state = solve_holstein_peierls_ground_state(
            lattice,
            parameters,
            initial_wavefunction=current_state.wavefunction,
        )
        final_energy, _ = total_energy(
            lattice,
            parameters,
            ground_state=final_state,
        )
        final_gradient, _ = energy_gradient(
            lattice,
            parameters,
            ground_state=final_state,
        )
        cached_state = final_state

        final_max_update = float(
            max(
                np.max(np.abs(delta_u)),
                np.max(np.abs(delta_vx)),
                np.max(np.abs(delta_vy)),
            )
        )
        final_max_gradient = final_gradient.maximum_absolute_component
        converged = (
            final_max_update < parameters.convergence_criterion
            and final_max_gradient < parameters.gradient_convergence_criterion
        )
        if converged:
            break

    assert final_energy is not None and final_state is not None
    return HolsteinPeierlsBipolaronResult(
        lattice=lattice,
        ground_state=final_state,
        energy=final_energy,
        diagnostics=BipolaronRelaxationDiagnostics(
            iterations=iteration,
            converged=converged,
            final_max_update=final_max_update,
            final_max_gradient=final_max_gradient,
        ),
    )
