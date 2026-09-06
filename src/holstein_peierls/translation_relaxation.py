"""Transversely relaxed one-site polaron translation paths.

IP0a used a linear interpolation between two exactly translated relaxed lattice
states.  IP0b showed that this frozen path has a shallow barrier along the easy
axis but a strongly anisotropy-dependent lattice reaction-coordinate inertia.

IP0c lowers the frozen path by minimizing the adiabatic ground-state energy in
hyperplanes perpendicular to the endpoint translation vector.  The scalar
translation coordinate is held fixed while every orthogonal lattice degree of
freedom is allowed to relax.  Uniform ``vx`` and ``vy`` gauge modes are removed
explicitly.

This construction is a *constrained relaxed path*.  It is more physical than the
frozen linear interpolation, but it is not yet claimed to be a true NEB/string
minimum-energy path or a finite-temperature free-energy barrier.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from .electronic import GroundState, SolverName, solve_ground_state
from .energy import total_energy
from .gradients import energy_gradient
from .lattice import LatticeState
from .parameters import StaticPolaronParameters
from .translation_barrier import Direction, translate_lattice_state

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ConstrainedRelaxationDiagnostics:
    converged: bool
    optimizer_success: bool
    iterations: int
    function_evaluations: int
    projected_gradient_max_eV_per_A: float
    reaction_coordinate_error: float
    energy_lowering_eV: float
    message: str


@dataclass(frozen=True, slots=True)
class RelaxedTranslationImage:
    fraction: float
    state: LatticeState
    energy_eV: float
    frozen_energy_eV: float
    electronic_energy_eV: float
    charge_density: FloatArray
    ipr: float
    participation_number: float
    source_population: float
    target_population: float
    diagnostics: ConstrainedRelaxationDiagnostics


@dataclass(frozen=True, slots=True)
class RelaxedTranslationProfile:
    direction: Direction
    source_site: int
    target_site: int
    images: tuple[RelaxedTranslationImage, ...]
    barrier_eV: float
    barrier_image_index: int
    endpoint_energy_mismatch_eV: float
    maximum_reaction_coordinate_error: float
    maximum_projected_gradient_eV_per_A: float
    all_images_converged: bool


def flatten_lattice_state(state: LatticeState) -> FloatArray:
    """Flatten ``u``, ``vx`` and ``vy`` in the project's C-order convention."""
    state.validate()
    return np.concatenate(
        [
            np.asarray(state.u, dtype=np.float64).reshape(-1, order="C"),
            np.asarray(state.vx, dtype=np.float64).reshape(-1, order="C"),
            np.asarray(state.vy, dtype=np.float64).reshape(-1, order="C"),
        ]
    )


def unflatten_lattice_state(values: ArrayLike, shape: tuple[int, int]) -> LatticeState:
    """Inverse of :func:`flatten_lattice_state`."""
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    n = int(shape[0] * shape[1])
    if vector.size != 3 * n:
        raise ValueError("lattice vector has the wrong dimension")
    if not np.all(np.isfinite(vector)):
        raise ValueError("lattice vector must contain finite values")
    return LatticeState(
        vector[:n].reshape(shape, order="C").copy(),
        vector[n : 2 * n].reshape(shape, order="C").copy(),
        vector[2 * n :].reshape(shape, order="C").copy(),
    )


def _remove_inter_displacement_gauges(vector: FloatArray, shape: tuple[int, int]) -> FloatArray:
    """Remove uniform vx/vy components from a vector in lattice space."""
    output = np.asarray(vector, dtype=np.float64).reshape(-1).copy()
    n = int(shape[0] * shape[1])
    if output.size != 3 * n:
        raise ValueError("lattice vector has the wrong dimension")
    output[n : 2 * n] -= float(np.mean(output[n : 2 * n]))
    output[2 * n :] -= float(np.mean(output[2 * n :]))
    return output


def translation_direction_vector(start: LatticeState, end: LatticeState) -> FloatArray:
    """Return the gauge-free endpoint displacement vector."""
    start.validate()
    end.validate()
    if start.shape != end.shape:
        raise ValueError("endpoint shapes must match")
    direction = flatten_lattice_state(end) - flatten_lattice_state(start)
    direction = _remove_inter_displacement_gauges(direction, start.shape)
    norm2 = float(np.dot(direction, direction))
    if not np.isfinite(norm2) or norm2 <= 0.0:
        raise ValueError("translation direction must have finite non-zero length")
    return direction


def project_orthogonal_translation_subspace(
    vector: ArrayLike,
    direction: ArrayLike,
    shape: tuple[int, int],
) -> FloatArray:
    """Project a lattice-space vector off translation and uniform gauge modes."""
    value = _remove_inter_displacement_gauges(
        np.asarray(vector, dtype=np.float64), shape
    )
    d = _remove_inter_displacement_gauges(
        np.asarray(direction, dtype=np.float64), shape
    )
    d2 = float(np.dot(d, d))
    if not np.isfinite(d2) or d2 <= 0.0:
        raise ValueError("direction must have finite non-zero length")
    value -= d * (float(np.dot(value, d)) / d2)
    return _remove_inter_displacement_gauges(value, shape)


def reaction_coordinate_fraction(
    state: LatticeState,
    start: LatticeState,
    end: LatticeState,
) -> float:
    """Projection of a state onto the endpoint translation coordinate."""
    if state.shape != start.shape or state.shape != end.shape:
        raise ValueError("all lattice shapes must match")
    d = translation_direction_vector(start, end)
    displacement = flatten_lattice_state(state) - flatten_lattice_state(start)
    displacement = _remove_inter_displacement_gauges(displacement, state.shape)
    return float(np.dot(displacement, d) / np.dot(d, d))


def _apply_reference_gauge(
    vector: FloatArray,
    reference: FloatArray,
    shape: tuple[int, int],
) -> FloatArray:
    """Set uniform vx/vy means to the reference means without changing physics."""
    output = np.asarray(vector, dtype=np.float64).reshape(-1).copy()
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    n = int(shape[0] * shape[1])
    for slc in (slice(n, 2 * n), slice(2 * n, 3 * n)):
        output[slc] += float(np.mean(ref[slc]) - np.mean(output[slc]))
    return output


def relax_constrained_translation_image(
    start: LatticeState,
    end: LatticeState,
    parameters: StaticPolaronParameters,
    fraction: float,
    *,
    solver: SolverName = "sparse",
    max_iterations: int = 300,
    gradient_tolerance_eV_per_A: float = 2.0e-6,
) -> RelaxedTranslationImage:
    """Relax one fixed-translation hyperplane with analytic adiabatic gradients.

    ``fraction`` fixes the projection onto the straight endpoint displacement.
    L-BFGS optimizes a redundant full-dimensional variable whose translation and
    gauge components are projected out before every energy evaluation.  This
    avoids constructing an explicit 3N-by-(3N-3) null-space basis.
    """
    start.validate()
    end.validate()
    expected = (parameters.ny, parameters.nx)
    if start.shape != expected or end.shape != expected:
        raise ValueError("lattice shape does not match parameters")
    s = float(fraction)
    if not np.isfinite(s) or s < 0.0 or s > 1.0:
        raise ValueError("fraction must lie in [0, 1]")
    if int(max_iterations) <= 0:
        raise ValueError("max_iterations must be positive")
    gtol = float(gradient_tolerance_eV_per_A)
    if not np.isfinite(gtol) or gtol <= 0.0:
        raise ValueError("gradient tolerance must be finite and positive")

    shape = start.shape
    q0 = flatten_lattice_state(start)
    q1 = flatten_lattice_state(end)
    d = translation_direction_vector(start, end)
    q_reference = (1.0 - s) * q0 + s * q1
    q_reference = _apply_reference_gauge(q_reference, q0, shape)

    previous_wavefunction: FloatArray | None = None

    def evaluate(auxiliary: FloatArray) -> tuple[float, FloatArray, GroundState]:
        nonlocal previous_wavefunction
        orthogonal = project_orthogonal_translation_subspace(auxiliary, d, shape)
        q = _apply_reference_gauge(q_reference + orthogonal, q0, shape)
        lattice = unflatten_lattice_state(q, shape)
        ground = solve_ground_state(
            lattice,
            parameters,
            solver=solver,
            initial_wavefunction=previous_wavefunction,
        )
        previous_wavefunction = np.asarray(ground.wavefunction, dtype=np.float64)
        gradient, _ = energy_gradient(
            lattice,
            parameters,
            solver=solver,
            ground_state=ground,
            mode="optimized",
        )
        energy, _ = total_energy(
            lattice,
            parameters,
            solver=solver,
            ground_state=ground,
        )
        raw_gradient = flatten_lattice_state(
            LatticeState(gradient.u, gradient.vx, gradient.vy)
        )
        projected_gradient = project_orthogonal_translation_subspace(
            raw_gradient, d, shape
        )
        return float(energy.total), projected_gradient, ground

    zero = np.zeros_like(q_reference)
    frozen_energy, _, _ = evaluate(zero)

    if s == 0.0 or s == 1.0:
        lattice = start.copy() if s == 0.0 else end.copy()
        ground = solve_ground_state(lattice, parameters, solver=solver)
        energy, _ = total_energy(lattice, parameters, ground_state=ground, solver=solver)
        gradient, _ = energy_gradient(
            lattice, parameters, ground_state=ground, solver=solver, mode="optimized"
        )
        projected = project_orthogonal_translation_subspace(
            flatten_lattice_state(LatticeState(gradient.u, gradient.vx, gradient.vy)),
            d,
            shape,
        )
        optimized = np.asarray(ground.wavefunction, dtype=np.float64)
        population = np.square(optimized)
        source = int(np.argmax(np.square(solve_ground_state(start, parameters, solver=solver).wavefunction)))
        target = int(np.argmax(np.square(solve_ground_state(end, parameters, solver=solver).wavefunction)))
        ipr = float(np.sum(population * population))
        return RelaxedTranslationImage(
            fraction=s,
            state=lattice,
            energy_eV=float(energy.total),
            frozen_energy_eV=float(frozen_energy),
            electronic_energy_eV=float(ground.energy),
            charge_density=population.reshape(shape, order="C"),
            ipr=ipr,
            participation_number=float(1.0 / ipr),
            source_population=float(population[source]),
            target_population=float(population[target]),
            diagnostics=ConstrainedRelaxationDiagnostics(
                converged=bool(np.max(np.abs(projected)) <= 1.0e-5),
                optimizer_success=True,
                iterations=0,
                function_evaluations=1,
                projected_gradient_max_eV_per_A=float(np.max(np.abs(projected))),
                reaction_coordinate_error=float(abs(reaction_coordinate_fraction(lattice, start, end) - s)),
                energy_lowering_eV=float(frozen_energy - energy.total),
                message="fixed relaxed endpoint",
            ),
        )

    cache: dict[str, object] = {}

    def objective(auxiliary: FloatArray) -> tuple[float, FloatArray]:
        energy_value, gradient_value, ground_value = evaluate(auxiliary)
        cache["ground"] = ground_value
        return energy_value, gradient_value

    result = minimize(
        objective,
        zero,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": int(max_iterations),
            "gtol": gtol,
            "ftol": 1.0e-13,
            "maxls": 50,
            "maxcor": 12,
        },
    )

    auxiliary = np.asarray(result.x, dtype=np.float64)
    orthogonal = project_orthogonal_translation_subspace(auxiliary, d, shape)
    q_final = _apply_reference_gauge(q_reference + orthogonal, q0, shape)
    lattice = unflatten_lattice_state(q_final, shape)
    ground = solve_ground_state(
        lattice,
        parameters,
        solver=solver,
        initial_wavefunction=previous_wavefunction,
    )
    gradient, _ = energy_gradient(
        lattice, parameters, solver=solver, ground_state=ground, mode="optimized"
    )
    energy, _ = total_energy(
        lattice, parameters, solver=solver, ground_state=ground
    )
    projected = project_orthogonal_translation_subspace(
        flatten_lattice_state(LatticeState(gradient.u, gradient.vx, gradient.vy)),
        d,
        shape,
    )
    projected_max = float(np.max(np.abs(projected)))
    coordinate_error = float(abs(reaction_coordinate_fraction(lattice, start, end) - s))
    population = np.square(np.asarray(ground.wavefunction, dtype=np.float64))
    source_ground = solve_ground_state(start, parameters, solver=solver)
    end_ground = solve_ground_state(end, parameters, solver=solver)
    source = int(np.argmax(np.square(source_ground.wavefunction)))
    target = int(np.argmax(np.square(end_ground.wavefunction)))
    ipr = float(np.sum(population * population))
    converged = bool(projected_max <= 1.0e-5 and coordinate_error <= 1.0e-10)

    return RelaxedTranslationImage(
        fraction=s,
        state=lattice,
        energy_eV=float(energy.total),
        frozen_energy_eV=float(frozen_energy),
        electronic_energy_eV=float(ground.energy),
        charge_density=population.reshape(shape, order="C"),
        ipr=ipr,
        participation_number=float(1.0 / ipr),
        source_population=float(population[source]),
        target_population=float(population[target]),
        diagnostics=ConstrainedRelaxationDiagnostics(
            converged=converged,
            optimizer_success=bool(result.success),
            iterations=int(result.nit),
            function_evaluations=int(result.nfev),
            projected_gradient_max_eV_per_A=projected_max,
            reaction_coordinate_error=coordinate_error,
            energy_lowering_eV=float(frozen_energy - energy.total),
            message=str(result.message),
        ),
    )


def constrained_translation_profile(
    relaxed_state: LatticeState,
    parameters: StaticPolaronParameters,
    *,
    direction: Direction = "+x",
    image_count: int = 9,
    solver: SolverName = "sparse",
    max_iterations: int = 300,
    gradient_tolerance_eV_per_A: float = 2.0e-6,
) -> RelaxedTranslationProfile:
    """Build a transversely relaxed profile at fixed translation fractions."""
    relaxed_state.validate()
    if relaxed_state.shape != (parameters.ny, parameters.nx):
        raise ValueError("lattice shape does not match parameters")
    count = int(image_count)
    if count < 3 or count % 2 == 0:
        raise ValueError("image_count must be an odd integer >= 3")

    translated = translate_lattice_state(relaxed_state, direction)
    start_ground = solve_ground_state(relaxed_state, parameters, solver=solver)
    end_ground = solve_ground_state(translated, parameters, solver=solver)
    source = int(np.argmax(np.square(start_ground.wavefunction)))
    target = int(np.argmax(np.square(end_ground.wavefunction)))

    images = tuple(
        relax_constrained_translation_image(
            relaxed_state,
            translated,
            parameters,
            float(fraction),
            solver=solver,
            max_iterations=max_iterations,
            gradient_tolerance_eV_per_A=gradient_tolerance_eV_per_A,
        )
        for fraction in np.linspace(0.0, 1.0, count)
    )
    energies = np.asarray([image.energy_eV for image in images], dtype=np.float64)
    endpoint_reference = 0.5 * float(energies[0] + energies[-1])
    barrier_index = int(np.argmax(energies))
    return RelaxedTranslationProfile(
        direction=direction,
        source_site=source,
        target_site=target,
        images=images,
        barrier_eV=float(energies[barrier_index] - endpoint_reference),
        barrier_image_index=barrier_index,
        endpoint_energy_mismatch_eV=float(abs(energies[0] - energies[-1])),
        maximum_reaction_coordinate_error=float(
            max(image.diagnostics.reaction_coordinate_error for image in images)
        ),
        maximum_projected_gradient_eV_per_A=float(
            max(image.diagnostics.projected_gradient_max_eV_per_A for image in images)
        ),
        all_images_converged=all(image.diagnostics.converged for image in images),
    )
