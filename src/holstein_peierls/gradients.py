"""Analytical gradients for lattice relaxation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .electronic import GroundState, SolverName, solve_ground_state
from .lattice import LatticeState
from .parameters import StaticPolaronParameters

FloatArray = NDArray[np.float64]
GradientMode = Literal["reference", "optimized"]


@dataclass(frozen=True, slots=True)
class LatticeGradient:
    """Energy derivatives with respect to the three lattice coordinates."""

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


def _reference_gradient(
    state: LatticeState,
    parameters: StaticPolaronParameters,
    ground_state: GroundState,
) -> LatticeGradient:
    """Evaluate the legacy-ordered gradient for regression calculations.

    This path intentionally preserves the full density matrix, explicit loops,
    and arithmetic ordering of ``gradientenergy`` in the archived Fortran code.
    RPROP depends on derivative signs, so this implementation remains available
    whenever strict historical regression is more important than performance.
    """
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
    )


def _optimized_gradient(
    state: LatticeState,
    parameters: StaticPolaronParameters,
    ground_state: GroundState,
) -> LatticeGradient:
    """Evaluate the same analytical gradient with O(N) storage and vector math.

    The legacy implementation builds the full density matrix
    ``rho[i, j] = psi[i] * psi[j]`` even though only diagonal and nearest-neighbour
    elements are used. For a real one-particle ground state, the electronic
    parts reduce exactly to local products of the wavefunction:

    ``rho[i,left] + rho[left,i] - rho[right,i] - rho[i,right]``
    ``= 2 * psi[i] * (psi[left] - psi[right])``.

    This path therefore requires only O(N) temporary storage instead of O(N^2).
    """
    psi = np.asarray(ground_state.wavefunction, dtype=np.float64).reshape(
        state.shape, order="C"
    )

    psi_left = np.roll(psi, shift=1, axis=1)
    psi_right = np.roll(psi, shift=-1, axis=1)
    psi_up = np.roll(psi, shift=1, axis=0)
    psi_down = np.roll(psi, shift=-1, axis=0)

    vx_left = np.roll(state.vx, shift=1, axis=1)
    vx_right = np.roll(state.vx, shift=-1, axis=1)
    vy_up = np.roll(state.vy, shift=1, axis=0)
    vy_down = np.roll(state.vy, shift=-1, axis=0)

    grad_u = parameters.k1 * state.u + parameters.alpha_intra * np.square(psi)
    grad_vx = (
        parameters.k2 * (2.0 * state.vx - vx_left - vx_right)
        + 2.0
        * parameters.alpha_interx
        * psi
        * (psi_left - psi_right)
    )
    grad_vy = (
        parameters.k2 * (2.0 * state.vy - vy_up - vy_down)
        + 2.0
        * parameters.alpha_intery
        * psi
        * (psi_up - psi_down)
    )

    return LatticeGradient(
        np.asarray(grad_u, dtype=np.float64),
        np.asarray(grad_vx, dtype=np.float64),
        np.asarray(grad_vy, dtype=np.float64),
    )


def energy_gradient(
    state: LatticeState,
    parameters: StaticPolaronParameters,
    *,
    solver: SolverName = "dense_lowest",
    ground_state: GroundState | None = None,
    mode: GradientMode = "optimized",
) -> tuple[LatticeGradient, GroundState]:
    """Evaluate the lattice-energy gradient.

    Parameters
    ----------
    mode:
        ``"optimized"`` uses O(N) storage and vectorized nearest-neighbour
        products. ``"reference"`` preserves the historical density-matrix and
        loop ordering for strict regression against the archived Fortran code.
    """
    if ground_state is None:
        ground_state = solve_ground_state(state, parameters, solver=solver)

    if mode == "optimized":
        gradient = _optimized_gradient(state, parameters, ground_state)
    elif mode == "reference":
        gradient = _reference_gradient(state, parameters, ground_state)
    else:
        raise ValueError(f"unknown gradient mode: {mode}")

    return gradient, ground_state
