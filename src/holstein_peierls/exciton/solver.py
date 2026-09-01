"""Static adiabatic Holstein-Peierls solver for a distinguishable e-h pair.

The wavefunction ``psi[i, j]`` places the electron on site ``i`` and the hole
on site ``j``.  No exchange projection is applied because the two carriers are
distinguishable.  Electron and hole one-particle Hamiltonians may have
independent hopping and electron-phonon parameters while sharing the same
classical lattice fields ``u``, ``vx`` and ``vy``.

The e-h interaction is frozen with respect to the classical lattice coordinates
in this reference implementation.  It affects the electronic state and hence
the Hellmann-Feynman lattice force, but it contributes no explicit Coulomb
force from changing molecular separations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import LinearOperator, eigsh

from ..lattice import LatticeState
from .interaction import electron_hole_interaction_matrix
from .parameters import ExcitonParameters

FloatArray = NDArray[np.float64]
Carrier = Literal["electron", "hole"]
ExcitonInitialization = Literal[
    "frenkel", "ct_x", "ct_y", "diagonal", "separated", "zero"
]


@dataclass(frozen=True, slots=True)
class ExcitonGroundState:
    """Lowest distinguishable electron-hole eigenstate for a fixed lattice."""

    energy: float
    wavefunction: FloatArray

    @property
    def probability(self) -> FloatArray:
        return np.square(self.wavefunction)

    @property
    def electron_density_matrix(self) -> FloatArray:
        """Electron one-body reduced density matrix with trace one."""
        return self.wavefunction @ self.wavefunction.T

    @property
    def hole_density_matrix(self) -> FloatArray:
        """Hole one-body reduced density matrix with trace one."""
        return self.wavefunction.T @ self.wavefunction

    @property
    def electron_density(self) -> FloatArray:
        return np.diag(self.electron_density_matrix)

    @property
    def hole_density(self) -> FloatArray:
        return np.diag(self.hole_density_matrix)

    @property
    def onsite_probability(self) -> float:
        return float(np.sum(np.square(np.diag(self.wavefunction))))

    @property
    def electron_ipr(self) -> float:
        density = self.electron_density
        return float(np.sum(np.square(density)))

    @property
    def hole_ipr(self) -> float:
        density = self.hole_density
        return float(np.sum(np.square(density)))


@dataclass(frozen=True, slots=True)
class ExcitonEnergy:
    electronic: float
    lattice: float
    total: float


@dataclass(frozen=True, slots=True)
class ExcitonLatticeGradient:
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
class ExcitonRelaxationDiagnostics:
    iterations: int
    converged: bool
    final_max_update: float
    final_max_gradient: float


@dataclass(frozen=True, slots=True)
class StaticExcitonResult:
    lattice: LatticeState
    ground_state: ExcitonGroundState
    energy: ExcitonEnergy
    diagnostics: ExcitonRelaxationDiagnostics


def _validate_lattice(state: LatticeState, parameters: ExcitonParameters) -> None:
    state.validate()
    if state.shape != (parameters.ny, parameters.nx):
        raise ValueError("lattice shape does not match exciton parameters")


def _carrier_parameters(
    parameters: ExcitonParameters,
    carrier: Carrier,
) -> tuple[float, float, float, float, float]:
    if carrier == "electron":
        return (
            parameters.electron_j0x,
            parameters.electron_j0y,
            parameters.electron_alpha_intra,
            parameters.electron_alpha_interx,
            parameters.electron_alpha_intery,
        )
    if carrier == "hole":
        return (
            parameters.hole_j0x,
            parameters.hole_j0y,
            parameters.hole_alpha_intra,
            parameters.hole_alpha_interx,
            parameters.hole_alpha_intery,
        )
    raise ValueError(f"unknown carrier: {carrier}")


def bond_transfer_integrals(
    state: LatticeState,
    parameters: ExcitonParameters,
    carrier: Carrier,
) -> tuple[FloatArray, FloatArray]:
    """Return carrier-specific Peierls-modulated +x/+y hopping matrix elements."""
    _validate_lattice(state, parameters)
    j0x, j0y, _, alpha_x, alpha_y = _carrier_parameters(parameters, carrier)
    vx_right = np.roll(state.vx, shift=-1, axis=1)
    vy_down = np.roll(state.vy, shift=-1, axis=0)
    tx = -j0x + alpha_x * (vx_right - state.vx)
    ty = -j0y + alpha_y * (vy_down - state.vy)
    return np.asarray(tx, dtype=np.float64), np.asarray(ty, dtype=np.float64)


def _site_neighbours(
    parameters: ExcitonParameters,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    sites = np.arange(parameters.n_sites, dtype=np.int64).reshape(
        parameters.ny, parameters.nx
    )
    return (
        sites.ravel(order="C"),
        np.roll(sites, shift=-1, axis=1).ravel(order="C"),
        np.roll(sites, shift=-1, axis=0).ravel(order="C"),
    )


def build_carrier_hamiltonian(
    state: LatticeState,
    parameters: ExcitonParameters,
    carrier: Carrier,
) -> csr_matrix:
    """Build one carrier's Holstein-Peierls Hamiltonian in CSR form."""
    _validate_lattice(state, parameters)
    j0x, j0y, alpha_intra, _, _ = _carrier_parameters(parameters, carrier)
    del j0x, j0y
    n = parameters.n_sites
    diagonal = alpha_intra * state.u.ravel(order="C")
    tx, ty = bond_transfer_integrals(state, parameters, carrier)

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


def _apply_exciton_hamiltonian(
    psi: FloatArray,
    electron_hamiltonian: csr_matrix,
    hole_hamiltonian: csr_matrix,
    interaction: FloatArray,
) -> FloatArray:
    """Apply ``He tensor I + I tensor Hh + V_eh`` to ``psi[i_e,i_h]``."""
    electron_part = electron_hamiltonian @ psi
    hole_part = (hole_hamiltonian @ psi.T).T
    result = np.asarray(electron_part + hole_part, dtype=np.float64)
    result += interaction * psi
    return result


def exciton_linear_operator(
    state: LatticeState,
    parameters: ExcitonParameters,
) -> LinearOperator:
    """Return the matrix-free distinguishable electron-hole Hamiltonian."""
    electron_h = build_carrier_hamiltonian(state, parameters, "electron")
    hole_h = build_carrier_hamiltonian(state, parameters, "hole")
    interaction = electron_hole_interaction_matrix(parameters)
    n = parameters.n_sites

    def matvec(vector: FloatArray) -> FloatArray:
        psi = np.asarray(vector, dtype=np.float64).reshape((n, n), order="C")
        return _apply_exciton_hamiltonian(
            psi, electron_h, hole_h, interaction
        ).ravel(order="C")

    return LinearOperator((n * n, n * n), matvec=matvec, dtype=np.float64)


def _normalize_wavefunction(wavefunction: FloatArray) -> FloatArray:
    psi = np.asarray(wavefunction, dtype=np.float64).copy()
    norm = float(np.linalg.norm(psi))
    if not np.isfinite(norm) or norm < 1.0e-14:
        raise RuntimeError("exciton wavefunction has zero or invalid norm")
    psi /= norm
    anchor = np.unravel_index(int(np.argmax(np.abs(psi))), psi.shape)
    if psi[anchor] < 0.0:
        psi = -psi
    return psi


def _pair_sites(
    parameters: ExcitonParameters,
    mode: ExcitonInitialization,
) -> tuple[int, int]:
    center = parameters.exciton_index
    cy, cx = divmod(center, parameters.nx)
    if mode in {"frenkel", "zero"}:
        return center, center
    if mode == "ct_x":
        return center, cy * parameters.nx + (cx + 1) % parameters.nx
    if mode == "ct_y":
        return center, ((cy + 1) % parameters.ny) * parameters.nx + cx
    if mode == "diagonal":
        return (
            center,
            ((cy + 1) % parameters.ny) * parameters.nx
            + (cx + 1) % parameters.nx,
        )
    if mode == "separated":
        sy = (cy + parameters.ny // 2) % parameters.ny
        sx = (cx + parameters.nx // 2) % parameters.nx
        return center, sy * parameters.nx + sx
    raise ValueError(f"unknown exciton initialization mode: {mode}")


def initial_wavefunction(
    parameters: ExcitonParameters,
    mode: ExcitonInitialization = "frenkel",
) -> FloatArray:
    """Return a localized electronic seed for a Frenkel/CT/separated branch."""
    electron_site, hole_site = _pair_sites(parameters, mode)
    psi = np.zeros((parameters.n_sites, parameters.n_sites), dtype=np.float64)
    psi[electron_site, hole_site] = 1.0
    return psi


def solve_exciton_ground_state(
    state: LatticeState,
    parameters: ExcitonParameters,
    *,
    initial_state: FloatArray | None = None,
) -> ExcitonGroundState:
    """Solve the lowest distinguishable e-h state for a fixed lattice."""
    _validate_lattice(state, parameters)
    n = parameters.n_sites
    operator = exciton_linear_operator(state, parameters)
    if initial_state is None:
        v0 = initial_wavefunction(parameters, "frenkel").ravel(order="C")
    else:
        v0 = _normalize_wavefunction(
            np.asarray(initial_state, dtype=np.float64).reshape((n, n))
        ).ravel(order="C")

    _, eigenvectors = eigsh(
        operator,
        k=1,
        which="SA",
        v0=v0,
        tol=parameters.eigensolver_tolerance,
        maxiter=parameters.eigensolver_max_iterations,
    )
    psi = _normalize_wavefunction(eigenvectors[:, 0].reshape((n, n), order="C"))
    applied = operator @ psi.ravel(order="C")
    energy = float(np.dot(psi.ravel(order="C"), applied))
    return ExcitonGroundState(energy=energy, wavefunction=psi)


def lattice_energy(state: LatticeState, parameters: ExcitonParameters) -> float:
    """Return complete Holstein plus Peierls elastic energy."""
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
    parameters: ExcitonParameters,
    *,
    ground_state: ExcitonGroundState | None = None,
) -> tuple[ExcitonEnergy, ExcitonGroundState]:
    if ground_state is None:
        ground_state = solve_exciton_ground_state(state, parameters)
    lattice = lattice_energy(state, parameters)
    return (
        ExcitonEnergy(
            electronic=ground_state.energy,
            lattice=lattice,
            total=ground_state.energy + lattice,
        ),
        ground_state,
    )


def energy_gradient(
    state: LatticeState,
    parameters: ExcitonParameters,
    *,
    ground_state: ExcitonGroundState | None = None,
) -> tuple[ExcitonLatticeGradient, ExcitonGroundState]:
    """Hellmann-Feynman gradient for the shared ``u``, ``vx`` and ``vy`` fields."""
    _validate_lattice(state, parameters)
    if ground_state is None:
        ground_state = solve_exciton_ground_state(state, parameters)

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

    vx_left = np.roll(state.vx, shift=1, axis=1)
    vx_right = np.roll(state.vx, shift=-1, axis=1)
    vy_up = np.roll(state.vy, shift=1, axis=0)
    vy_down = np.roll(state.vy, shift=-1, axis=0)

    grad_u = (
        parameters.k1 * state.u
        + parameters.electron_alpha_intra * density_e
        + parameters.hole_alpha_intra * density_h
    )
    grad_vx = parameters.k2 * (2.0 * state.vx - vx_left - vx_right)
    grad_vx += 2.0 * parameters.electron_alpha_interx * (
        gamma_e[sites, left] - gamma_e[sites, right]
    )
    grad_vx += 2.0 * parameters.hole_alpha_interx * (
        gamma_h[sites, left] - gamma_h[sites, right]
    )
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


def initial_lattice_state(
    parameters: ExcitonParameters,
    mode: ExcitonInitialization = "frenkel",
) -> LatticeState:
    """Return a localized Holstein seed and initially unstrained Peierls fields."""
    shape = (parameters.ny, parameters.nx)
    u = np.zeros(shape, dtype=np.float64)
    if mode != "zero":
        electron_site, hole_site = _pair_sites(parameters, mode)
        ey, ex = divmod(electron_site, parameters.nx)
        hy, hx = divmod(hole_site, parameters.nx)
        u[ey, ex] -= parameters.electron_alpha_intra / parameters.k1
        u[hy, hx] -= parameters.hole_alpha_intra / parameters.k1
    return LatticeState(
        u=u,
        vx=np.zeros(shape, dtype=np.float64),
        vy=np.zeros(shape, dtype=np.float64),
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


def relax_static_exciton(
    parameters: ExcitonParameters,
    *,
    initialization: ExcitonInitialization = "frenkel",
    initial_lattice: LatticeState | None = None,
) -> StaticExcitonResult:
    """Relax the shared lattice fields for one distinguishable electron-hole pair."""
    if initial_lattice is None:
        lattice = initial_lattice_state(parameters, initialization)
    else:
        _validate_lattice(initial_lattice, parameters)
        lattice = initial_lattice.copy()

    previous_u = np.zeros_like(lattice.u)
    previous_vx = np.zeros_like(lattice.vx)
    previous_vy = np.zeros_like(lattice.vy)
    step_u = np.full_like(lattice.u, parameters.update_start)
    step_vx = np.full_like(lattice.vx, parameters.update_start)
    step_vy = np.full_like(lattice.vy, parameters.update_start)

    cached_state: ExcitonGroundState | None = initial_wavefunction(
        parameters, initialization
    )
    converged = False
    final_max_update = np.inf
    final_max_gradient = np.inf
    final_energy: ExcitonEnergy | None = None
    final_state: ExcitonGroundState | None = None

    for iteration in range(1, parameters.max_iterations + 1):
        current_state = solve_exciton_ground_state(
            lattice,
            parameters,
            initial_state=cached_state.wavefunction
            if isinstance(cached_state, ExcitonGroundState)
            else cached_state,
        )
        gradient, _ = energy_gradient(
            lattice, parameters, ground_state=current_state
        )

        delta_u = _rprop_update(
            lattice.u, gradient.u, previous_u, step_u, parameters
        )
        delta_vx = _rprop_update(
            lattice.vx, gradient.vx, previous_vx, step_vx, parameters
        )
        delta_vy = _rprop_update(
            lattice.vy, gradient.vy, previous_vy, step_vy, parameters
        )

        final_state = solve_exciton_ground_state(
            lattice,
            parameters,
            initial_state=current_state.wavefunction,
        )
        final_energy, _ = total_energy(
            lattice, parameters, ground_state=final_state
        )
        final_gradient, _ = energy_gradient(
            lattice, parameters, ground_state=final_state
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
    return StaticExcitonResult(
        lattice=lattice,
        ground_state=final_state,
        energy=final_energy,
        diagnostics=ExcitonRelaxationDiagnostics(
            iterations=iteration,
            converged=converged,
            final_max_update=final_max_update,
            final_max_gradient=final_max_gradient,
        ),
    )


def binding_energy(
    exciton_total_energy: float,
    electron_polaron_total_energy: float,
    hole_polaron_total_energy: float,
) -> float:
    """Return ``E_e-pol + E_h-pol - E_X``; positive values mean bound."""
    return (
        float(electron_polaron_total_energy)
        + float(hole_polaron_total_energy)
        - float(exciton_total_energy)
    )
