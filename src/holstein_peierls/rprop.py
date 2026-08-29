"""Resilient backpropagation minimizer with an explicit legacy-compatibility mode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .electronic import GroundState, SolverName
from .energy import EnergyBreakdown, total_energy
from .gradients import LatticeGradient, energy_gradient
from .lattice import LatticeState
from .parameters import StaticPolaronParameters

FloatArray = NDArray[np.float64]
ProgressCallback = Callable[[int, EnergyBreakdown, LatticeState, LatticeGradient], None]


@dataclass(frozen=True, slots=True)
class RPropDiagnostics:
    iterations: int
    converged_u: bool
    converged_vx: bool
    converged_vy: bool
    final_max_delta_u: float
    final_max_delta_vx: float
    final_max_delta_vy: float


@dataclass(frozen=True, slots=True)
class RPropOutput:
    state: LatticeState
    energy: EnergyBreakdown
    ground_state: GroundState
    diagnostics: RPropDiagnostics


@dataclass(slots=True)
class _AxisMemory:
    previous_gradient: FloatArray
    delta: FloatArray
    update: FloatArray
    previous_delta: FloatArray
    previous_update: FloatArray

    @classmethod
    def create(cls, shape: tuple[int, int], update_start: float) -> "_AxisMemory":
        zeros = np.zeros(shape, dtype=np.float64)
        return cls(
            previous_gradient=zeros.copy(),
            delta=zeros.copy(),
            update=np.full(shape, update_start, dtype=np.float64),
            previous_delta=zeros.copy(),
            previous_update=zeros.copy(),
        )


def _legacy_axis_step(
    values: FloatArray,
    gradient: FloatArray,
    memory: _AxisMemory,
    parameters: StaticPolaronParameters,
) -> None:
    """Apply one RPROP update following the archived branch logic."""
    product = gradient * memory.previous_gradient
    positive = product > 0.0
    negative = product < 0.0
    zero = ~(positive | negative)

    if np.any(positive):
        memory.update[positive] = np.minimum(
            memory.previous_update[positive] * parameters.acceleration_factor,
            parameters.update_max,
        )
        memory.delta[positive] = -np.sign(gradient[positive]) * memory.update[positive]
        values[positive] += memory.delta[positive]

    if np.any(negative):
        memory.update[negative] = np.maximum(
            memory.previous_update[negative] * parameters.deceleration_factor,
            parameters.update_min,
        )
        values[negative] -= memory.previous_delta[negative]
        gradient[negative] = 0.0

    if np.any(zero):
        memory.delta[zero] = -np.sign(gradient[zero]) * memory.update[zero]
        values[zero] += memory.delta[zero]


def minimize_legacy_rprop(
    initial_state: LatticeState,
    parameters: StaticPolaronParameters,
    *,
    solver: SolverName = "dense_lowest",
    stop_when_all_coordinates_converge: bool = False,
    progress: ProgressCallback | None = None,
) -> RPropOutput:
    """Relax a polaron with the historical RPROP update equations."""
    state = initial_state.copy()
    state.validate()
    shape = state.shape
    memories = {
        "u": _AxisMemory.create(shape, parameters.update_start),
        "vx": _AxisMemory.create(shape, parameters.update_start),
        "vy": _AxisMemory.create(shape, parameters.update_start),
    }

    total_energy(state, parameters, solver=solver)
    converged_u = converged_vx = converged_vy = False
    final_energy: EnergyBreakdown | None = None
    final_ground_state: GroundState | None = None
    iterations_done = 0

    for iteration in range(1, parameters.max_iterations + 1):
        gradient, _ = energy_gradient(state, parameters, solver=solver)
        current = {
            "u": np.array(gradient.u, copy=True),
            "vx": np.array(gradient.vx, copy=True),
            "vy": np.array(gradient.vy, copy=True),
        }
        values = {"u": state.u, "vx": state.vx, "vy": state.vy}

        for name in ("u", "vx", "vy"):
            _legacy_axis_step(values[name], current[name], memories[name], parameters)

        final_energy, final_ground_state = total_energy(state, parameters, solver=solver)
        iterations_done = iteration

        du = memories["u"].delta
        dvx = memories["vx"].delta
        dvy = memories["vy"].delta
        converged_u = bool(np.max(np.abs(du)) < parameters.convergence_criterion)
        converged_vx = bool(np.max(np.abs(dvx)) < parameters.convergence_criterion)
        converged_vy = bool(np.max(np.abs(dvy)) < parameters.convergence_criterion)

        for name in ("u", "vx", "vy"):
            memory = memories[name]
            memory.previous_update[...] = memory.update
            memory.previous_delta[...] = memory.delta
            memory.previous_gradient[...] = current[name]

        if progress is not None:
            progress(iteration, final_energy, state, LatticeGradient(current["u"], current["vx"], current["vy"]))

        done = (
            converged_u and converged_vx and converged_vy
            if stop_when_all_coordinates_converge
            else converged_u
        )
        if done:
            break

    assert final_energy is not None and final_ground_state is not None
    diagnostics = RPropDiagnostics(
        iterations=iterations_done,
        converged_u=converged_u,
        converged_vx=converged_vx,
        converged_vy=converged_vy,
        final_max_delta_u=float(np.max(np.abs(memories["u"].delta))),
        final_max_delta_vx=float(np.max(np.abs(memories["vx"].delta))),
        final_max_delta_vy=float(np.max(np.abs(memories["vy"].delta))),
    )
    return RPropOutput(state, final_energy, final_ground_state, diagnostics)
