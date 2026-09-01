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


@dataclass(frozen=True, slots=True)
class ExcitonBranchOutcome:
    mode: ExcitonInitialization
    result: StaticExcitonResult


@dataclass(frozen=True, slots=True)
class ExcitonBranchScan:
    outcomes: tuple[ExcitonBranchOutcome, ...]
    best_mode: ExcitonInitialization
    best_result: StaticExcitonResult

    @property
    def all_converged(self) -> bool:
        return all(item.result.diagnostics.converged for item in self.outcomes)


def relax_exciton_branches(
    parameters: ExcitonParameters,
    *,
    modes: tuple[ExcitonInitialization, ...] = DEFAULT_EXCITON_BRANCHES,
) -> ExcitonBranchScan:
    """Relax competing Frenkel/CT/separated seeds and select the lowest energy.

    Only converged branches are eligible for promotion.  If no branch reaches
    both the displacement and gradient criteria, a ``RuntimeError`` is raised
    rather than silently selecting a non-stationary state.
    """
    if not modes:
        raise ValueError("at least one exciton branch must be requested")
    if len(set(modes)) != len(modes):
        raise ValueError("exciton branch modes must be unique")

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
    best = min(converged, key=lambda item: item.result.energy.total)
    return ExcitonBranchScan(
        outcomes=outcomes,
        best_mode=best.mode,
        best_result=best.result,
    )
