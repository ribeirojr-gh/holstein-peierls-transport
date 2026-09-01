"""Competing relaxed branches for the static reference exciton."""

from __future__ import annotations

from dataclasses import dataclass

from .parameters import ExcitonParameters
from .solver import ExcitonInitialization, StaticExcitonResult, relax_static_exciton


DEFAULT_EXCITON_BRANCHES: tuple[ExcitonInitialization, ...] = (
    "frenkel",
    "ct_x",
    "ct_y",
    "diagonal",
    "separated",
)
DEFAULT_BRANCH_ENERGY_TIE_TOLERANCE_EV = 1.0e-10


@dataclass(frozen=True, slots=True)
class ExcitonBranchOutcome:
    """One relaxed result together with the seed used to reach it."""

    mode: ExcitonInitialization
    result: StaticExcitonResult


@dataclass(frozen=True, slots=True)
class ExcitonBranchScan:
    outcomes: tuple[ExcitonBranchOutcome, ...]
    best_mode: ExcitonInitialization
    best_result: StaticExcitonResult
    degenerate_best_modes: tuple[ExcitonInitialization, ...]

    @property
    def all_converged(self) -> bool:
        return all(item.result.diagnostics.converged for item in self.outcomes)


def relax_exciton_branches(
    parameters: ExcitonParameters,
    *,
    modes: tuple[ExcitonInitialization, ...] = DEFAULT_EXCITON_BRANCHES,
    energy_tie_tolerance_ev: float = DEFAULT_BRANCH_ENERGY_TIE_TOLERANCE_EV,
) -> ExcitonBranchScan:
    """Relax competing seeds and select the lowest converged energy basin.

    ``mode`` labels the *initialization*, not the topology of the final state.
    Different seeds can converge to the same minimum.  Energies within
    ``energy_tie_tolerance_ev`` of the minimum are therefore recorded as a
    degenerate group and the first requested mode is used as a deterministic
    canonical representative.  This prevents floating-point noise from
    relabelling one physical minimum as a different seed branch.

    Only converged branches are eligible for promotion.  If no branch reaches
    both the displacement and gradient criteria, a ``RuntimeError`` is raised
    rather than silently selecting a non-stationary state.
    """
    if not modes:
        raise ValueError("at least one exciton branch must be requested")
    if len(set(modes)) != len(modes):
        raise ValueError("exciton branch modes must be unique")
    if energy_tie_tolerance_ev < 0.0:
        raise ValueError("energy_tie_tolerance_ev must be non-negative")

    outcomes = tuple(
        ExcitonBranchOutcome(
            mode=mode,
            result=relax_static_exciton(parameters, initialization=mode),
        )
        for mode in modes
    )
    converged = [item for item in outcomes if item.result.diagnostics.converged]
    if not converged:
        raise RuntimeError("no static exciton branch converged")

    minimum_energy = min(item.result.energy.total for item in converged)
    degenerate = tuple(
        item
        for item in converged
        if item.result.energy.total - minimum_energy <= energy_tie_tolerance_ev
    )
    best = degenerate[0]
    return ExcitonBranchScan(
        outcomes=outcomes,
        best_mode=best.mode,
        best_result=best.result,
        degenerate_best_modes=tuple(item.mode for item in degenerate),
    )
