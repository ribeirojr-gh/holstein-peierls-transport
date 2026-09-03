"""Frozen-geometry spin-adapted projector dynamics for D0b.

D0a benchmarks the linear TDSE under a fixed one-particle Hamiltonian.  D0b is
separate because the spin-adapted open-shell Fock operators depend on the
instantaneous electronic state.  The natural gauge-invariant variables are the
occupied-shell projectors ``P_mu`` rather than individual orbital phases.

For the restricted orbital manifold already validated in S0, define

    M = sum_mu n_mu [P_mu, F_mu]

with shell occupations ``n_mu`` and shell-dependent Fock matrices ``F_mu``.
The Dirac-Frenkel/unitary-orbital equation fixes the off-diagonal blocks of a
single anti-Hermitian site-basis generator ``K`` through

    K_ab = P_a M P_b / (i hbar (n_a - n_b)),   n_a != n_b,

where the virtual complement is included as an occupation-zero subspace.
Rotations inside equal-occupation subspaces are gauge-fixed to zero, matching
the occupation-stratified tangent convention used by the S0 optimizer.  All
shells then evolve under the same generator,

    dP_mu/dt = [K, P_mu].

Using one common generator is essential: propagating each shell independently
with its own Fock operator would generally destroy mutual orthogonality.

This module is a D0b numerical foundation, not yet coupled electron-lattice
dynamics.  Geometry, external fields, thermostats, and intersystem crossing are
excluded.  The open-shell singlet therefore retains the same fixed-coefficient,
equal-occupation gauge restriction as S0; a richer configuration-amplitude
backend can be added later without changing this projector-level benchmark API.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from ..spin_adapted.open_shell import OpenShellStateDefinition
from .frozen import HBAR_EV_FS

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ProjectorConstraintMetrics:
    """Deviation from the exact shell-projector manifold."""

    maximum_hermiticity_error: float
    maximum_idempotency_error: float
    maximum_mutual_orthogonality_error: float
    particle_number: float


@dataclass(frozen=True, slots=True)
class ProjectorPropagationResult:
    """Final projector state and numerical work counters."""

    projectors: tuple[ComplexArray, ...]
    rhs_evaluations: int
    elapsed_seconds: float
    accepted_steps: int | None = None
    rejected_or_internal_steps: int | None = None
    success: bool = True
    message: str = ""


@dataclass(frozen=True, slots=True)
class ProjectorComparisonMetrics:
    """Gauge-invariant accuracy and conservation metrics for D0b."""

    projector_distance: float
    rdm_distance: float
    initial_energy: float
    final_energy: float
    energy_drift: float
    constraints: ProjectorConstraintMetrics


def _model_arrays(
    one_body: ArrayLike,
    interaction: ArrayLike,
    definition: OpenShellStateDefinition,
) -> tuple[ComplexArray, FloatArray]:
    t = np.asarray(one_body, dtype=np.complex128)
    v = np.asarray(interaction, dtype=np.float64)
    if t.ndim != 2 or t.shape[0] != t.shape[1]:
        raise ValueError("one_body must be a square matrix")
    if not np.all(np.isfinite(t)):
        raise ValueError("one_body must contain only finite values")
    if not np.allclose(t, t.conj().T, rtol=0.0, atol=1.0e-12):
        raise ValueError("one_body must be Hermitian")
    if v.shape != t.shape:
        raise ValueError("interaction must match one_body")
    if not np.all(np.isfinite(v)):
        raise ValueError("interaction must contain only finite values")
    if not np.allclose(v, v.T, rtol=0.0, atol=1.0e-12):
        raise ValueError("interaction must be symmetric")
    if definition.n_shells <= 0:
        raise ValueError("at least one occupied shell is required")
    return t, v


def _projector_tuple(
    projectors: Sequence[ArrayLike],
    *,
    dimension: int,
    n_shells: int,
) -> tuple[ComplexArray, ...]:
    if len(projectors) != n_shells:
        raise ValueError("projector count does not match state definition")
    checked: list[ComplexArray] = []
    for projector in projectors:
        p = np.asarray(projector, dtype=np.complex128)
        if p.shape != (dimension, dimension):
            raise ValueError("every shell projector must match one_body")
        if not np.all(np.isfinite(p)):
            raise ValueError("projectors must contain only finite values")
        checked.append(p)
    return tuple(checked)


def _physical_projectors(projectors: Sequence[ComplexArray]) -> tuple[ComplexArray, ...]:
    """Hermitianize only for evaluating the physical state-dependent functional.

    Exact dynamics remains on the Hermitian projector manifold.  Explicit ODE
    integrators may generate roundoff-scale anti-Hermitian components at
    intermediate stages.  Removing only that unphysical component from Fock and
    energy evaluations makes the RHS stable without reprojecting idempotency or
    mutual orthogonality; those numerical errors remain visible in the reported
    constraint metrics.
    """
    return tuple(
        np.asarray(0.5 * (p + p.conj().T), dtype=np.complex128)
        for p in projectors
    )


def _relaxed_shell_fock_matrices(
    one_body: ComplexArray,
    interaction: FloatArray,
    projectors: Sequence[ComplexArray],
    definition: OpenShellStateDefinition,
) -> tuple[ComplexArray, ...]:
    """Evaluate the validated S0 shell-Fock formula near the projector manifold."""
    ps = _physical_projectors(projectors)
    occupations = np.asarray(definition.occupations, dtype=np.float64)
    a = definition.a_matrix
    b = definition.b_matrix
    focks: list[ComplexArray] = []
    for mu in range(definition.n_shells):
        fock = np.array(one_body, copy=True, dtype=np.complex128)
        for nu, p_nu in enumerate(ps):
            density_nu = np.real(np.diag(p_nu))
            fock += occupations[nu] * a[mu, nu] * np.diag(
                interaction @ density_nu
            )
            fock -= (
                0.5
                * occupations[nu]
                * b[mu, nu]
                * (interaction * p_nu)
            )
        focks.append(
            np.asarray(0.5 * (fock + fock.conj().T), dtype=np.complex128)
        )
    return tuple(focks)


def spin_summed_rdm(
    projectors: Sequence[ArrayLike],
    definition: OpenShellStateDefinition,
) -> ComplexArray:
    """Return the spin-summed one-body RDM ``gamma = sum_mu n_mu P_mu``."""
    if len(projectors) != definition.n_shells:
        raise ValueError("projector count does not match state definition")
    first = np.asarray(projectors[0], dtype=np.complex128)
    gamma = np.zeros_like(first, dtype=np.complex128)
    for occupation, projector in zip(
        definition.occupations, projectors, strict=True
    ):
        gamma += occupation * np.asarray(projector, dtype=np.complex128)
    return np.asarray(0.5 * (gamma + gamma.conj().T), dtype=np.complex128)


def projector_energy(
    one_body: ArrayLike,
    interaction: ArrayLike,
    projectors: Sequence[ArrayLike],
    definition: OpenShellStateDefinition,
) -> float:
    """Evaluate the S0 open-shell energy without silently reprojecting the state."""
    t, v = _model_arrays(one_body, interaction, definition)
    ps = _projector_tuple(
        projectors,
        dimension=t.shape[0],
        n_shells=definition.n_shells,
    )
    ps = _physical_projectors(ps)
    occupations = np.asarray(definition.occupations, dtype=np.float64)
    a = definition.a_matrix
    b = definition.b_matrix

    energy = 0.0
    for occupation, projector in zip(occupations, ps, strict=True):
        energy += occupation * float(np.real(np.trace(projector @ t)))
    for mu, p_mu in enumerate(ps):
        density_mu = np.real(np.diag(p_mu))
        for nu, p_nu in enumerate(ps):
            density_nu = np.real(np.diag(p_nu))
            coulomb = float(density_mu @ v @ density_nu)
            exchange = float(np.real(np.sum(p_mu * p_nu.T * v)))
            energy += (
                0.25
                * occupations[mu]
                * occupations[nu]
                * (2.0 * a[mu, nu] * coulomb - b[mu, nu] * exchange)
            )
    return float(energy)


def variational_generator(
    one_body: ArrayLike,
    interaction: ArrayLike,
    projectors: Sequence[ArrayLike],
    definition: OpenShellStateDefinition,
) -> ComplexArray:
    """Return the common anti-Hermitian D0b orbital/projector generator.

    Equal-occupation blocks are set to zero.  This is a gauge choice for
    redundant rotations and is also the explicit restricted-tangent convention
    already used by the S0 orbital optimizer for distinct same-occupation
    shells in the minimal fixed-coefficient singlet ansatz.
    """
    t, v = _model_arrays(one_body, interaction, definition)
    ps = _projector_tuple(
        projectors,
        dimension=t.shape[0],
        n_shells=definition.n_shells,
    )
    physical = _physical_projectors(ps)
    focks = _relaxed_shell_fock_matrices(t, v, physical, definition)
    occupations = tuple(float(value) for value in definition.occupations)

    m = np.zeros_like(t, dtype=np.complex128)
    for occupation, projector, fock in zip(
        occupations, physical, focks, strict=True
    ):
        m += occupation * (projector @ fock - fock @ projector)
    m = np.asarray(0.5 * (m - m.conj().T), dtype=np.complex128)

    occupied_sum = np.sum(np.stack(physical, axis=0), axis=0)
    virtual = np.eye(t.shape[0], dtype=np.complex128) - occupied_sum
    subspaces = (*physical, virtual)
    subspace_occupations = (*occupations, 0.0)

    generator = np.zeros_like(t, dtype=np.complex128)
    for index_a, (projector_a, occupation_a) in enumerate(
        zip(subspaces, subspace_occupations, strict=True)
    ):
        for index_b, (projector_b, occupation_b) in enumerate(
            zip(subspaces, subspace_occupations, strict=True)
        ):
            if index_a == index_b or occupation_a == occupation_b:
                continue
            generator += (
                projector_a
                @ m
                @ projector_b
                / (1.0j * HBAR_EV_FS * (occupation_a - occupation_b))
            )
    return np.asarray(
        0.5 * (generator - generator.conj().T), dtype=np.complex128
    )


def projector_rhs(
    one_body: ArrayLike,
    interaction: ArrayLike,
    projectors: Sequence[ArrayLike],
    definition: OpenShellStateDefinition,
) -> tuple[ComplexArray, ...]:
    """Return ``dP_mu/dt = [K,P_mu]`` in inverse femtoseconds."""
    t, _ = _model_arrays(one_body, interaction, definition)
    ps = _projector_tuple(
        projectors,
        dimension=t.shape[0],
        n_shells=definition.n_shells,
    )
    generator = variational_generator(one_body, interaction, ps, definition)
    return tuple(
        np.asarray(generator @ p - p @ generator, dtype=np.complex128)
        for p in ps
    )


def projector_constraints(
    projectors: Sequence[ArrayLike],
    definition: OpenShellStateDefinition,
) -> ProjectorConstraintMetrics:
    """Return Hermiticity, idempotency, orthogonality and particle-number gates."""
    if len(projectors) != definition.n_shells:
        raise ValueError("projector count does not match state definition")
    ps = tuple(np.asarray(p, dtype=np.complex128) for p in projectors)
    if not ps:
        raise ValueError("at least one projector is required")
    hermiticity = max(float(np.linalg.norm(p - p.conj().T)) for p in ps)
    idempotency = max(float(np.linalg.norm(p @ p - p)) for p in ps)
    mutual = 0.0
    for mu in range(len(ps)):
        for nu in range(mu + 1, len(ps)):
            mutual = max(mutual, float(np.linalg.norm(ps[mu] @ ps[nu])))
    particle_number = float(np.real(np.trace(spin_summed_rdm(ps, definition))))
    return ProjectorConstraintMetrics(
        maximum_hermiticity_error=hermiticity,
        maximum_idempotency_error=idempotency,
        maximum_mutual_orthogonality_error=mutual,
        particle_number=particle_number,
    )


def validate_projector_manifold(
    projectors: Sequence[ArrayLike],
    definition: OpenShellStateDefinition,
    *,
    tolerance: float = 1.0e-9,
) -> None:
    """Require a valid initial shell-projector state."""
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    metrics = projector_constraints(projectors, definition)
    if metrics.maximum_hermiticity_error > tolerance:
        raise ValueError("initial projectors must be Hermitian")
    if metrics.maximum_idempotency_error > tolerance:
        raise ValueError("initial projectors must be idempotent")
    if metrics.maximum_mutual_orthogonality_error > tolerance:
        raise ValueError("initial shell projectors must be mutually orthogonal")


def _linear_combination(
    base: Sequence[ComplexArray],
    terms: Sequence[tuple[float, Sequence[ComplexArray]]],
) -> tuple[ComplexArray, ...]:
    result: list[ComplexArray] = []
    for shell_index, base_projector in enumerate(base):
        value = np.array(base_projector, copy=True, dtype=np.complex128)
        for coefficient, derivative in terms:
            value += coefficient * np.asarray(derivative[shell_index])
        result.append(value)
    return tuple(result)


def rk4_projector_step(
    one_body: ArrayLike,
    interaction: ArrayLike,
    projectors: Sequence[ArrayLike],
    definition: OpenShellStateDefinition,
    dt_fs: float,
) -> tuple[ComplexArray, ...]:
    """Classical RK4 step without post-step reprojecting or renormalization."""
    dt = float(dt_fs)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_fs must be finite and positive")
    t, _ = _model_arrays(one_body, interaction, definition)
    ps = _projector_tuple(
        projectors,
        dimension=t.shape[0],
        n_shells=definition.n_shells,
    )
    k1 = projector_rhs(one_body, interaction, ps, definition)
    k2 = projector_rhs(
        one_body,
        interaction,
        _linear_combination(ps, ((0.5 * dt, k1),)),
        definition,
    )
    k3 = projector_rhs(
        one_body,
        interaction,
        _linear_combination(ps, ((0.5 * dt, k2),)),
        definition,
    )
    k4 = projector_rhs(
        one_body,
        interaction,
        _linear_combination(ps, ((dt, k3),)),
        definition,
    )
    return _linear_combination(
        ps,
        (
            (dt / 6.0, k1),
            (dt / 3.0, k2),
            (dt / 3.0, k3),
            (dt / 6.0, k4),
        ),
    )


def predictor_exponential_midpoint_step(
    one_body: ArrayLike,
    interaction: ArrayLike,
    projectors: Sequence[ArrayLike],
    definition: OpenShellStateDefinition,
    dt_fs: float,
) -> tuple[ComplexArray, ...]:
    """Structure-preserving one-predictor exponential midpoint step.

    The first generator predicts a unitary half-step.  A second generator is
    evaluated at that midpoint and exponentiated for the full step.  This is a
    deliberately simple nonlinear exponential control for D0b; it is not
    assumed to inherit the D0a Krylov result or to be a production integrator.
    """
    dt = float(dt_fs)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_fs must be finite and positive")
    t, _ = _model_arrays(one_body, interaction, definition)
    ps = _projector_tuple(
        projectors,
        dimension=t.shape[0],
        n_shells=definition.n_shells,
    )
    k0 = variational_generator(one_body, interaction, ps, definition)
    half_unitary = expm(0.5 * dt * k0)
    midpoint = tuple(
        np.asarray(half_unitary @ p @ half_unitary.conj().T, dtype=np.complex128)
        for p in ps
    )
    kmid = variational_generator(one_body, interaction, midpoint, definition)
    unitary = expm(dt * kmid)
    return tuple(
        np.asarray(unitary @ p @ unitary.conj().T, dtype=np.complex128)
        for p in ps
    )


def integrate_rk4_projectors(
    one_body: ArrayLike,
    interaction: ArrayLike,
    projectors: Sequence[ArrayLike],
    definition: OpenShellStateDefinition,
    *,
    dt_fs: float,
    steps: int,
) -> ProjectorPropagationResult:
    """Integrate D0b with fixed-step RK4."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    validate_projector_manifold(projectors, definition)
    ps = tuple(np.asarray(p, dtype=np.complex128) for p in projectors)
    start = perf_counter()
    for _ in range(steps):
        ps = rk4_projector_step(
            one_body, interaction, ps, definition, dt_fs
        )
    elapsed = perf_counter() - start
    return ProjectorPropagationResult(
        projectors=ps,
        rhs_evaluations=4 * steps,
        elapsed_seconds=float(elapsed),
        accepted_steps=steps,
    )


def integrate_predictor_exponential_midpoint(
    one_body: ArrayLike,
    interaction: ArrayLike,
    projectors: Sequence[ArrayLike],
    definition: OpenShellStateDefinition,
    *,
    dt_fs: float,
    steps: int,
) -> ProjectorPropagationResult:
    """Integrate with the structure-preserving exponential midpoint control."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    validate_projector_manifold(projectors, definition)
    ps = tuple(np.asarray(p, dtype=np.complex128) for p in projectors)
    start = perf_counter()
    for _ in range(steps):
        ps = predictor_exponential_midpoint_step(
            one_body, interaction, ps, definition, dt_fs
        )
    elapsed = perf_counter() - start
    return ProjectorPropagationResult(
        projectors=ps,
        rhs_evaluations=2 * steps,
        elapsed_seconds=float(elapsed),
        accepted_steps=steps,
    )


def _pack_projectors(projectors: Sequence[ComplexArray]) -> ComplexArray:
    return np.concatenate(
        [np.asarray(p, dtype=np.complex128).ravel(order="C") for p in projectors]
    )


def _unpack_projectors(
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


def integrate_dop853_projectors(
    one_body: ArrayLike,
    interaction: ArrayLike,
    projectors: Sequence[ArrayLike],
    definition: OpenShellStateDefinition,
    *,
    final_time_fs: float,
    rtol: float = 1.0e-11,
    atol: float = 1.0e-13,
    max_step_fs: float = np.inf,
) -> ProjectorPropagationResult:
    """Adaptive eighth-order DOP853 reference for nonlinear D0b dynamics."""
    final_time = float(final_time_fs)
    if not np.isfinite(final_time) or final_time <= 0.0:
        raise ValueError("final_time_fs must be finite and positive")
    if rtol <= 0.0 or atol <= 0.0:
        raise ValueError("rtol and atol must be positive")
    if max_step_fs <= 0.0:
        raise ValueError("max_step_fs must be positive")
    t, _ = _model_arrays(one_body, interaction, definition)
    ps = _projector_tuple(
        projectors,
        dimension=t.shape[0],
        n_shells=definition.n_shells,
    )
    validate_projector_manifold(ps, definition)
    y0 = _pack_projectors(ps)

    def rhs(_time: float, values: ComplexArray) -> ComplexArray:
        current = _unpack_projectors(
            values,
            dimension=t.shape[0],
            n_shells=definition.n_shells,
        )
        return _pack_projectors(
            projector_rhs(one_body, interaction, current, definition)
        )

    start = perf_counter()
    solution = solve_ivp(
        rhs,
        (0.0, final_time),
        y0,
        method="DOP853",
        rtol=float(rtol),
        atol=float(atol),
        max_step=float(max_step_fs),
    )
    elapsed = perf_counter() - start
    final_projectors = _unpack_projectors(
        solution.y[:, -1],
        dimension=t.shape[0],
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


def projector_distance(
    reference: Sequence[ArrayLike],
    candidate: Sequence[ArrayLike],
) -> float:
    """Return normalized Frobenius distance between shell-projector states."""
    if len(reference) != len(candidate) or len(reference) == 0:
        raise ValueError("reference and candidate must contain the same shells")
    numerator = 0.0
    scale = 0.0
    for left, right in zip(reference, candidate, strict=True):
        a = np.asarray(left, dtype=np.complex128)
        b = np.asarray(right, dtype=np.complex128)
        if a.shape != b.shape:
            raise ValueError("projector shapes must match")
        numerator += float(np.linalg.norm(a - b) ** 2)
        scale += float(np.linalg.norm(a) ** 2)
    if scale == 0.0:
        raise ValueError("reference projectors must have non-zero norm")
    return float(np.sqrt(numerator / scale))


def compare_projector_states(
    one_body: ArrayLike,
    interaction: ArrayLike,
    initial: Sequence[ArrayLike],
    reference: Sequence[ArrayLike],
    candidate: Sequence[ArrayLike],
    definition: OpenShellStateDefinition,
) -> ProjectorComparisonMetrics:
    """Return D0b accuracy, energy-conservation and manifold metrics."""
    reference_rdm = spin_summed_rdm(reference, definition)
    candidate_rdm = spin_summed_rdm(candidate, definition)
    rdm_scale = float(np.linalg.norm(reference_rdm))
    if rdm_scale == 0.0:
        raise ValueError("reference RDM must have non-zero norm")
    initial_energy = projector_energy(one_body, interaction, initial, definition)
    final_energy = projector_energy(one_body, interaction, candidate, definition)
    return ProjectorComparisonMetrics(
        projector_distance=projector_distance(reference, candidate),
        rdm_distance=float(np.linalg.norm(candidate_rdm - reference_rdm) / rdm_scale),
        initial_energy=initial_energy,
        final_energy=final_energy,
        energy_drift=abs(final_energy - initial_energy),
        constraints=projector_constraints(candidate, definition),
    )
