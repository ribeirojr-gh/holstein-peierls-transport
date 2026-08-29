"""Analytical gradients for lattice relaxation."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from .electronic import GroundState, SolverName, solve_ground_state
from .lattice import LatticeState
from .parameters import StaticPolaronParameters

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class LatticeGradient:
    u: FloatArray
    vx: FloatArray
    vy: FloatArray

    @property
    def maximum_absolute_component(self) -> float:
        return float(max(np.max(np.abs(self.u)), np.max(np.abs(self.vx)), np.max(np.abs(self.vy))))


def energy_gradient(
    state: LatticeState,
    parameters: StaticPolaronParameters,
    *,
    solver: SolverName = "dense_lowest",
    ground_state: GroundState | None = None,
) -> tuple[LatticeGradient, GroundState]:
    """Evaluate the lattice-energy gradient using legacy arithmetic ordering.

    RPROP depends only on derivative signs. At symmetric sites, algebraically
    equivalent simplifications can alter signs of round-off-level derivatives,
    so the reference path preserves the density matrix, loops, and expression
    ordering of ``gradientenergy`` in the archived Fortran source.
    """
    if ground_state is None:
        ground_state = solve_ground_state(state, parameters, solver=solver)

    ny, nx = state.shape
    nxy = parameters.n_sites
    psi = np.asarray(ground_state.wavefunction, dtype=np.float64)
    rho = np.outer(psi, psi)
    u = state.u.reshape(-1, order="C")
    vx = state.vx.reshape(-1, order="C")
    vy = state.vy.reshape(-1, order="C")

    grad_u = np.empty(nxy, dtype=np.float64)
    grad_vx = np.empty(nxy, dtype=np.float64)
    grad_vy = np.empty(nxy, dtype=np.float64)

    for i in range(nxy):
        grad_u[i] = -(-parameters.k1 * u[i] - parameters.alpha_intra * rho[i, i])

    for iy in range(ny):
        for jx in range(1, nx - 1):
            k = jx + nx * iy
            grad_vx[k] = -(
                -parameters.k2 * (2.0 * vx[k] - vx[k - 1] - vx[k + 1])
                - parameters.alpha_interx
                * (rho[k, k - 1] - rho[k + 1, k] + rho[k - 1, k] - rho[k, k + 1])
            )

    for iy in range(ny):
        k = nx * iy
        left = k - 1 + nx
        grad_vx[k] = -(
            -parameters.k2 * (2.0 * vx[k] - vx[left] - vx[k + 1])
            - parameters.alpha_interx
            * (rho[k, left] - rho[k + 1, k] + rho[left, k] - rho[k, k + 1])
        )

    for iy in range(ny):
        k = nx - 1 + nx * iy
        right = k - nx + 1
        grad_vx[k] = -(
            -parameters.k2 * (2.0 * vx[k] - vx[k - 1] - vx[right])
            - parameters.alpha_interx
            * (rho[k, k - 1] - rho[right, k] + rho[k - 1, k] - rho[k, right])
        )

    for iy in range(1, ny - 1):
        for jx in range(nx):
            k = jx + nx * iy
            grad_vy[k] = -(
                -parameters.k2 * (2.0 * vy[k] - vy[k - nx] - vy[k + nx])
                - parameters.alpha_intery
                * (rho[k, k - nx] - rho[k + nx, k] + rho[k - nx, k] - rho[k, k + nx])
            )

    for jx in range(nx):
        k = jx
        up = k + nx * (ny - 1)
        grad_vy[k] = -(
            -parameters.k2 * (2.0 * vy[k] - vy[up] - vy[k + nx])
            - parameters.alpha_intery
            * (rho[k, up] - rho[k + nx, k] + rho[up, k] - rho[k, k + nx])
        )

    for jx in range(nx):
        k = jx + nx * (ny - 1)
        down = jx
        grad_vy[k] = -(
            -parameters.k2 * (2.0 * vy[k] - vy[k - nx] - vy[down])
            - parameters.alpha_intery
            * (rho[k, k - nx] - rho[down, k] + rho[k - nx, k] - rho[k, down])
        )

    shape = state.shape
    return LatticeGradient(
        grad_u.reshape(shape, order="C"),
        grad_vx.reshape(shape, order="C"),
        grad_vy.reshape(shape, order="C"),
    ), ground_state
