"""Fourth-order structure-preserving RKMK integrator for D0b.

For the projector equation

    dP_mu/dt = [K(P), P_mu]

with anti-Hermitian generator ``K``, write the common unitary step as
``U = exp(Omega)``.  The local Lie-algebra equation is

    dOmega/dt = dexp_Omega^{-1}(K(P(Omega))).

For fourth order it is sufficient to retain the Bernoulli expansion through
second nested commutators,

    dexp^{-1}_Omega(A)
        = A - 1/2 [Omega,A] + 1/12 [Omega,[Omega,A]] + O(Omega^4).

Applying classical RK4 to this local algebra equation gives the standard
fourth-order Runge-Kutta-Munthe-Kaas construction.  The final projector update
is a common unitary similarity transformation, so Hermiticity, idempotency,
mutual shell orthogonality and particle number are preserved to floating-point
accuracy without any post-step repair.

The dense matrix exponential used here is intentionally a D0b reference
implementation for small spin-adapted controls.  A later production backend may
replace the exponential action without changing the RKMK equations.
"""

from __future__ import annotations

from time import perf_counter
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import expm

from ..spin_adapted.open_shell import OpenShellStateDefinition
from .spin_adapted import (
    ProjectorPropagationResult,
    validate_projector_manifold,
    variational_generator,
)

ComplexArray = NDArray[np.complex128]


def _commutator(left: ComplexArray, right: ComplexArray) -> ComplexArray:
    return np.asarray(left @ right - right @ left, dtype=np.complex128)


def _dexp_inverse_fourth_order(
    omega: ComplexArray,
    generator: ComplexArray,
) -> ComplexArray:
    """Return the fourth-order RKMK truncation of ``dexp^{-1}``."""
    first = _commutator(omega, generator)
    second = _commutator(omega, first)
    result = generator - 0.5 * first + (1.0 / 12.0) * second
    return np.asarray(0.5 * (result - result.conj().T), dtype=np.complex128)


def _apply_algebra_coordinate(
    projectors: Sequence[ComplexArray],
    omega: ComplexArray,
) -> tuple[ComplexArray, ...]:
    if np.linalg.norm(omega) == 0.0:
        return tuple(np.asarray(p, dtype=np.complex128) for p in projectors)
    unitary = expm(omega)
    return tuple(
        np.asarray(unitary @ p @ unitary.conj().T, dtype=np.complex128)
        for p in projectors
    )


def _algebra_rhs(
    one_body: ArrayLike,
    interaction: ArrayLike,
    base_projectors: Sequence[ComplexArray],
    definition: OpenShellStateDefinition,
    omega: ComplexArray,
) -> ComplexArray:
    current = _apply_algebra_coordinate(base_projectors, omega)
    generator = variational_generator(
        one_body,
        interaction,
        current,
        definition,
    )
    return _dexp_inverse_fourth_order(omega, generator)


def rkmk4_projector_step(
    one_body: ArrayLike,
    interaction: ArrayLike,
    projectors: Sequence[ArrayLike],
    definition: OpenShellStateDefinition,
    dt_fs: float,
) -> tuple[ComplexArray, ...]:
    """Advance one common-unitary fourth-order RKMK step."""
    dt = float(dt_fs)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_fs must be finite and positive")
    ps = tuple(np.asarray(p, dtype=np.complex128) for p in projectors)
    validate_projector_manifold(ps, definition, tolerance=1.0e-7)
    dimension = ps[0].shape[0]
    zero = np.zeros((dimension, dimension), dtype=np.complex128)

    k1 = _algebra_rhs(one_body, interaction, ps, definition, zero)
    k2 = _algebra_rhs(one_body, interaction, ps, definition, 0.5 * dt * k1)
    k3 = _algebra_rhs(one_body, interaction, ps, definition, 0.5 * dt * k2)
    k4 = _algebra_rhs(one_body, interaction, ps, definition, dt * k3)
    omega = (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    omega = np.asarray(0.5 * (omega - omega.conj().T), dtype=np.complex128)
    return _apply_algebra_coordinate(ps, omega)


def integrate_rkmk4_projectors(
    one_body: ArrayLike,
    interaction: ArrayLike,
    projectors: Sequence[ArrayLike],
    definition: OpenShellStateDefinition,
    *,
    dt_fs: float,
    steps: int,
) -> ProjectorPropagationResult:
    """Integrate D0b with fixed-step fourth-order RKMK."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    validate_projector_manifold(projectors, definition)
    ps = tuple(np.asarray(p, dtype=np.complex128) for p in projectors)
    start = perf_counter()
    for _ in range(steps):
        ps = rkmk4_projector_step(
            one_body,
            interaction,
            ps,
            definition,
            dt_fs,
        )
    elapsed = perf_counter() - start
    return ProjectorPropagationResult(
        projectors=ps,
        rhs_evaluations=4 * steps,
        elapsed_seconds=float(elapsed),
        accepted_steps=steps,
    )
