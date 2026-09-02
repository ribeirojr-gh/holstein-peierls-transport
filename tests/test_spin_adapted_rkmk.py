from __future__ import annotations

import numpy as np

from holstein_peierls.dynamics.frozen import HBAR_EV_FS
from holstein_peierls.dynamics.spin_adapted import (
    integrate_dop853_projectors,
    projector_constraints,
    projector_distance,
)
from holstein_peierls.dynamics.spin_adapted_rkmk import integrate_rkmk4_projectors
from holstein_peierls.spin_adapted.open_shell import (
    CLOSED_SHELL_SINGLET,
    HIGH_SPIN_TRIPLET,
    projector_from_orbitals,
)


def _unitary(dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(dimension, dimension)) + 1.0j * rng.normal(
        size=(dimension, dimension)
    )
    q, r = np.linalg.qr(matrix)
    diagonal = np.diag(r)
    phases = np.where(np.abs(diagonal) > 0.0, diagonal / np.abs(diagonal), 1.0)
    return np.asarray(q * phases.conj(), dtype=np.complex128)


def _projectors(orbitals: np.ndarray, sizes: tuple[int, ...]) -> tuple[np.ndarray, ...]:
    result: list[np.ndarray] = []
    start = 0
    for size in sizes:
        stop = start + size
        result.append(
            np.asarray(
                projector_from_orbitals(orbitals[:, start:stop]),
                dtype=np.complex128,
            )
        )
        start = stop
    return tuple(result)


def _model(dimension: int) -> tuple[np.ndarray, np.ndarray]:
    index = np.arange(dimension, dtype=np.float64)
    one_body = np.diag(-0.27 + 0.14 * index).astype(np.complex128)
    for site in range(dimension):
        neighbour = (site + 1) % dimension
        phase = np.exp(1.0j * 0.09 * (site + 1))
        one_body[site, neighbour] += -0.08 * phase
        one_body[neighbour, site] += -0.08 * phase.conjugate()
    one_body = 0.5 * (one_body + one_body.conj().T)

    interaction = np.zeros((dimension, dimension), dtype=np.float64)
    np.fill_diagonal(interaction, 0.38)
    for site in range(dimension):
        neighbour = (site + 1) % dimension
        interaction[site, neighbour] = 0.09
        interaction[neighbour, site] = 0.09
    return one_body, interaction


def test_rkmk4_preserves_projector_manifold_to_roundoff() -> None:
    one_body, interaction = _model(6)
    initial = _projectors(_unitary(6, seed=7), (1, 2))
    result = integrate_rkmk4_projectors(
        one_body,
        interaction,
        initial,
        HIGH_SPIN_TRIPLET,
        dt_fs=0.025,
        steps=16,
    )
    constraints = projector_constraints(result.projectors, HIGH_SPIN_TRIPLET)
    assert constraints.maximum_hermiticity_error < 3.0e-14
    assert constraints.maximum_idempotency_error < 5.0e-13
    assert constraints.maximum_mutual_orthogonality_error < 5.0e-13
    assert abs(constraints.particle_number - 4.0) < 5.0e-13


def test_rkmk4_has_fourth_order_convergence_in_exact_closed_shell_limit() -> None:
    one_body, _ = _model(4)
    interaction = np.zeros((4, 4), dtype=np.float64)
    initial = _projectors(_unitary(4, seed=12), (1,))
    final_time = 0.4

    eigenvalues, eigenvectors = np.linalg.eigh(one_body)
    unitary = eigenvectors @ np.diag(
        np.exp(-1.0j * eigenvalues * final_time / HBAR_EV_FS)
    ) @ eigenvectors.conj().T
    exact = (unitary @ initial[0] @ unitary.conj().T,)

    coarse = integrate_rkmk4_projectors(
        one_body,
        interaction,
        initial,
        CLOSED_SHELL_SINGLET,
        dt_fs=0.04,
        steps=10,
    )
    fine = integrate_rkmk4_projectors(
        one_body,
        interaction,
        initial,
        CLOSED_SHELL_SINGLET,
        dt_fs=0.02,
        steps=20,
    )
    coarse_error = projector_distance(exact, coarse.projectors)
    fine_error = projector_distance(exact, fine.projectors)
    assert fine_error < coarse_error / 12.0


def test_rkmk4_converges_fourth_order_to_nonlinear_dop853_reference() -> None:
    one_body, interaction = _model(5)
    initial = _projectors(_unitary(5, seed=23), (1, 1))
    final_time = 0.3
    reference = integrate_dop853_projectors(
        one_body,
        interaction,
        initial,
        HIGH_SPIN_TRIPLET,
        final_time_fs=final_time,
        rtol=5.0e-13,
        atol=5.0e-15,
        max_step_fs=0.01,
    )
    assert reference.success

    coarse = integrate_rkmk4_projectors(
        one_body,
        interaction,
        initial,
        HIGH_SPIN_TRIPLET,
        dt_fs=0.03,
        steps=10,
    )
    fine = integrate_rkmk4_projectors(
        one_body,
        interaction,
        initial,
        HIGH_SPIN_TRIPLET,
        dt_fs=0.015,
        steps=20,
    )
    coarse_error = projector_distance(reference.projectors, coarse.projectors)
    fine_error = projector_distance(reference.projectors, fine.projectors)
    assert fine_error < coarse_error / 10.0
    assert fine_error < 2.0e-8
