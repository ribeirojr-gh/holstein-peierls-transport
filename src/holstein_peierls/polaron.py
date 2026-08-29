"""High-level static-polaron workflow."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from .electronic import SolverName
from .lattice import LatticeState
from .observables import inverse_participation_ratio, legacy_ipr, polaron_formation_energy
from .parameters import StaticPolaronParameters
from .rprop import RPropDiagnostics, minimize_legacy_rprop

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PolaronResult:
    state: LatticeState
    total_energy: float
    electronic_energy: float
    formation_energy: float
    charge_density: FloatArray
    ipr: float
    legacy_ipr: float
    diagnostics: RPropDiagnostics


def prepare_initial_state(
    parameters: StaticPolaronParameters,
    initial_state: LatticeState | None = None,
    *,
    apply_legacy_seed: bool = True,
) -> LatticeState:
    """Prepare the initial lattice exactly as the historical program does."""
    state = (
        LatticeState.zeros(parameters.ny, parameters.nx)
        if initial_state is None
        else initial_state.copy()
    )
    state.validate()
    if apply_legacy_seed:
        flat = state.u.reshape(-1, order="C")
        flat[parameters.polaron_index] = -0.1
    return state


def solve_static_polaron(
    parameters: StaticPolaronParameters,
    *,
    initial_state: LatticeState | None = None,
    solver: SolverName = "dense_lowest",
    legacy_convergence: bool = False,
    apply_legacy_seed: bool = True,
) -> PolaronResult:
    """Relax the lattice around one excess charge.

    The modern default requires all three lattice fields to converge. Set
    ``legacy_convergence=True`` only to reproduce the historical u-only stop.
    """
    state = prepare_initial_state(
        parameters, initial_state, apply_legacy_seed=apply_legacy_seed
    )
    output = minimize_legacy_rprop(
        state,
        parameters,
        solver=solver,
        stop_when_all_coordinates_converge=not legacy_convergence,
    )
    density = output.ground_state.charge_density.reshape(
        (parameters.ny, parameters.nx), order="C"
    )
    return PolaronResult(
        state=output.state,
        total_energy=output.energy.total,
        electronic_energy=output.energy.electronic,
        formation_energy=polaron_formation_energy(output.energy.total, parameters),
        charge_density=density,
        ipr=inverse_participation_ratio(density),
        legacy_ipr=legacy_ipr(density),
        diagnostics=output.diagnostics,
    )
