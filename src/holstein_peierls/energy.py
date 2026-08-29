"""Energy terms of the static Holstein-Peierls problem."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .electronic import GroundState, SolverName, solve_ground_state
from .lattice import LatticeState
from .parameters import StaticPolaronParameters


@dataclass(frozen=True, slots=True)
class EnergyBreakdown:
    intramolecular_lattice: float
    intermolecular_lattice: float
    electronic: float

    @property
    def total(self) -> float:
        return self.intramolecular_lattice + self.intermolecular_lattice + self.electronic


def lattice_energy(
    state: LatticeState, parameters: StaticPolaronParameters
) -> tuple[float, float]:
    """Return intra- and intermolecular lattice potential energies."""
    intra = 0.5 * parameters.k1 * float(np.sum(state.u * state.u))
    dx = np.roll(state.vx, -1, axis=1) - state.vx
    dy = np.roll(state.vy, -1, axis=0) - state.vy
    inter = 0.5 * parameters.k2 * float(np.sum(dx * dx) + np.sum(dy * dy))
    return intra, inter


def total_energy(
    state: LatticeState,
    parameters: StaticPolaronParameters,
    *,
    solver: SolverName = "dense_lowest",
    ground_state: GroundState | None = None,
) -> tuple[EnergyBreakdown, GroundState]:
    """Return total energy and the electronic ground state."""
    if ground_state is None:
        ground_state = solve_ground_state(state, parameters, solver=solver)
    intra, inter = lattice_energy(state, parameters)
    return EnergyBreakdown(intra, inter, ground_state.energy), ground_state
