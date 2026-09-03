"""Explicitly time-dependent spin-adapted projector dynamics for D1.

D0b validated the nonlinear fixed-coefficient open-shell projector equations at
frozen geometry with a time-independent one-body Hamiltonian.  D1 changes only
that one ingredient: the Peierls electric-field phase makes ``T(t)`` explicitly
time dependent while the classical lattice and density-density interaction
remain frozen.

Two numerical paths are provided:

- tight adaptive DOP853 as the independent high-accuracy reference; and
- fourth-order RKMK with stage-consistent times as the structure-preserving
  fixed-step candidate carried forward from D0b.

No lattice motion, thermostat, spin-orbit coupling, or configuration-amplitude
dynamics is introduced here.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

from ..spin_adapted.open_shell import OpenShellStateDefinition
from .spin_adapted import (
    ProjectorConstraintMetrics,
    ProjectorPropagationResult,
    projector_constraints,
    projector_distance,
    projector_rhs,
    spin_summed_rdm,
    validate_projector_manifold,
    variational_generator,
)
from .spin_adapted_rkmk import (
    apply_algebra_coordinate,
    dexp_inverse_fourth_order,
)

ComplexArray = NDArray[np.complex128]
OneBodyFunction = Callable[[float], Any]


@dataclass(frozen=True, slots=True)
class TimeDependentProjectorComparison:
    """Gauge-invariant D1b state comparison at the same final time."""

    projector_distance: float
    rdm_distance: float
    constraints: ProjectorConstraintMetrics


def _one_body_at(
    one_body_at: OneBodyFunction,
    time_fs: float,
    *,
    dimension: int,
) -> ComplexArray:
    matrix = one_body_at(float(time_fs))
    dense = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    dense = np.asarray(dense, dtype=np.complex128)
    if dense.shape != (dimension, dimension):
        raise ValueError("time-dependent one-body matrix has the wrong shape")
    if not np.all(np.isfinite(dense)):
        raise ValueError("time-dependent one-body matrix must be finite")
    if not np.allclose(dense, dense.conj().T, rtol=0.0, atol=1.0e-12):
        raise ValueError("time-dependent one-body matrix must be Hermitian")
    return dense


def compare_time_dependent_projectors(
    reference: Sequence[ArrayLike],
    candidate: Sequence[ArrayLike],
    definition: OpenShellStateDefinition,
) -> TimeDependentProjectorComparison:
    """Compare final shell projectors and spin-summed RDMs without energy gates."""
    p_distance = projector_distance(reference, candidate)
    gamma_reference = spin_summed_rdm(reference, definition)
    gamma_candidate = spin_summed_rdm(candidate, definition)
    scale = float(np.linalg.norm(gamma_reference))
    if scale == 0.0:
        raise ValueError("reference RDM must have non-zero norm")
    rdm_distance = float(np.linalg.norm(gamma_candidate - gamma_reference) / scale)
    return TimeDependentProjectorComparison(
        projector_distance=p_distance,
        rdm_distance=rdm_distance,
        constraints=projector_constraints(candidate, definition),
    )


def _algebra_rhs_time_dependent(
    one_body_at: OneBodyFunction,
    interaction: ArrayLike,
    base_projectors: Sequence[ComplexArray],
    definition: OpenShellStateDefinition,
    time_fs: float,
    omega: ComplexArray,
) -> ComplexArray:
    current = apply_algebra_coordinate(base_projectors, omega)
    one_body = _one_body_at(one_body_at, time_fs, dimension=current[0].shape[0])
    generator = variational_generator(
        one_body,
        interaction,
        current,
        definition,
    )
    return dexp_inverse_fourth_order(omega, generator)


def rkmk4_time_dependent_projector_step(
    one_body_at: OneBodyFunction,
    interaction: ArrayLike,
    projectors: Sequence[ArrayLike],
    definition: OpenShellStateDefinition,
    time_fs: float,
    dt_fs: float,
) -> tuple[ComplexArray, ...]:
    """Advance one stage-time-consistent fourth-order RKMK D1b step."""
    time = float(time_fs)
    dt = float(dt_fs)
    if not np.isfinite(time) or not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("time must be finite and dt_fs must be finite and positive")
    ps = tuple(np.asarray(p, dtype=np.complex128) for p in projectors)
    validate_projector_manifold(ps, definition, tolerance=1.0e-7)
    dimension = ps[0].shape[0]
    zero = np.zeros((dimension, dimension), dtype=np.complex128)

    k1 = _algebra_rhs_time_dependent(
        one_body_at, interaction, ps, definition, time, zero
    )
    k2 = _algebra_rhs_time_dependent(
        one_body_at,
        interaction,
        ps,
        definition,
        time + 0.5 * dt,
        0.5 * dt * k1,
    )
    k3 = _algebra_rhs_time_dependent(
        one_body_at,
        interaction,
        ps,
        definition,
        time + 0.5 * dt,
        0.5 * dt * k2,
    )
    k4 = _algebra_rhs_time_dependent(
        one_body_at,
        interaction,
        ps,
        definition,
        time + dt,
        dt * k3,
    )
    omega = (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    omega = np.asarray(0.5 * (omega - omega.conj().T), dtype=np.complex128)
    return apply_algebra_coordinate(ps, omega)


def integrate_rkmk4_time_dependent_projectors(
    one_body_at: OneBodyFunction,
    interaction: ArrayLike,
    projectors: Sequence[ArrayLike],
    definition: OpenShellStateDefinition,
    *,
    dt_fs: float,
    steps: int,
    initial_time_fs: float = 0.0,
) -> ProjectorPropagationResult:
    """Integrate D1b with fixed-step fourth-order structure-preserving RKMK."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    validate_projector_manifold(projectors, definition)
    ps = tuple(np.asarray(p, dtype=np.complex128) for p in projectors)
    time = float(initial_time_fs)
    start = perf_counter()
    for _ in range(steps):
        ps = rkmk4_time_dependent_projector_step(
            one_body_at,
            interaction,
            ps,
            definition,
            time,
            dt_fs,
        )
        time += float(dt_fs)
    elapsed = perf_counter() - start
    return ProjectorPropagationResult(
        projectors=ps,
        rhs_evaluations=4 * steps,
        elapsed_seconds=float(elapsed),
        accepted_steps=steps,
    )


def _pack(projectors: Sequence[ComplexArray]) -> ComplexArray:
    return np.concatenate(
        [np.asarray(p, dtype=np.complex128).ravel(order="C") for p in projectors]
    )


def _unpack(
    values: ArrayLike,
    *,
    dimension: int,
    n_shells: int,
) -> tuple[ComplexArray, ...]:
    vector = np.asarray(values, dtype=np.complex128).reshape(-1)
    block = dimension * dimension
    if vector.size != n_shells * block:
        raise ValueError("packed projector vector has the wrong size")
    return tuple(
        np.asarray(
            vector[index * block : (index + 1) * block].reshape(
                (dimension, dimension), order="C"
            ),
            dtype=np.complex128,
        )
        for index in range(n_shells)
    )


def integrate_dop853_time_dependent_projectors(
    one_body_at: OneBodyFunction,
    interaction: ArrayLike,
    projectors: Sequence[ArrayLike],
    definition: OpenShellStateDefinition,
    *,
    final_time_fs: float,
    initial_time_fs: float = 0.0,
    rtol: float = 1.0e-11,
    atol: float = 1.0e-13,
    max_step_fs: float = np.inf,
) -> ProjectorPropagationResult:
    """Adaptive DOP853 reference for nonlinear spin-adapted D1b dynamics."""
    initial_time = float(initial_time_fs)
    final_time = float(final_time_fs)
    if not np.isfinite(initial_time) or not np.isfinite(final_time):
        raise ValueError("integration times must be finite")
    if final_time <= initial_time:
        raise ValueError("final_time_fs must exceed initial_time_fs")
    if rtol <= 0.0 or atol <= 0.0 or max_step_fs <= 0.0:
        raise ValueError("DOP853 tolerances and max_step_fs must be positive")
    validate_projector_manifold(projectors, definition)
    ps = tuple(np.asarray(p, dtype=np.complex128) for p in projectors)
    dimension = ps[0].shape[0]
    y0 = _pack(ps)

    def rhs(time: float, values: ComplexArray) -> ComplexArray:
        current = _unpack(
            values,
            dimension=dimension,
            n_shells=definition.n_shells,
        )
        one_body = _one_body_at(one_body_at, time, dimension=dimension)
        return _pack(projector_rhs(one_body, interaction, current, definition))

    start = perf_counter()
    solution = solve_ivp(
        rhs,
        (initial_time, final_time),
        y0,
        method="DOP853",
        rtol=float(rtol),
        atol=float(atol),
        max_step=float(max_step_fs),
    )
    elapsed = perf_counter() - start
    final_projectors = _unpack(
        solution.y[:, -1],
        dimension=dimension,
        n_shells=definition.n_shells,
    )
    return ProjectorPropagationResult(
        projectors=final_projectors,
        rhs_evaluations=int(solution.nfev),
        elapsed_seconds=float(elapsed),
        accepted_steps=max(0, int(solution.t.size - 1)),
        success=bool(solution.success),
        message=str(solution.message),
    )
