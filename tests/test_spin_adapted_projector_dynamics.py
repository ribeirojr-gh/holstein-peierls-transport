from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from holstein_peierls.dynamics.frozen import HBAR_EV_FS
from holstein_peierls.dynamics.spin_adapted import (
    compare_projector_states,
    integrate_dop853_projectors,
    integrate_predictor_exponential_midpoint,
    integrate_rk4_projectors,
    projector_constraints,
    projector_distance,
    projector_energy,
    projector_rhs,
    spin_summed_rdm,
    validate_projector_manifold,
    variational_generator,
)
from holstein_peierls.spin_adapted.open_shell import (
    CLOSED_SHELL_SINGLET,
    HIGH_SPIN_TRIPLET,
    OPEN_SHELL_SINGLET,
    general_open_shell_energy,
    projector_from_orbitals,
)


def _unitary(dimension: int, seed: int = 9) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(dimension, dimension)) + 1.0j * rng.normal(
        size=(dimension, dimension)
    )
    q, r = np.linalg.qr(matrix)
    phases = np.diag(r)
    phases = np.where(np.abs(phases) > 0.0, phases / np.abs(phases), 1.0)
    return np.asarray(q * phases.conj(), dtype=np.complex128)


def _projectors(orbitals: np.ndarray, shell_sizes: tuple[int, ...]) -> tuple[np.ndarray, ...]:
    projectors: list[np.ndarray] = []
    start = 0
    for size in shell_sizes:
        stop = start + size
        projectors.append(projector_from_orbitals(orbitals[:, start:stop]))
        start = stop
    return tuple(np.asarray(p, dtype=np.complex128) for p in projectors)


def _model(dimension: int) -> tuple[np.ndarray, np.ndarray]:
    index = np.arange(dimension, dtype=np.float64)
    one_body = np.diag(-0.32 + 0.17 * index).astype(np.complex128)
    for i in range(dimension):
        j = (i + 1) % dimension
        phase = np.exp(1.0j * 0.13 * (i + 1))
        one_body[i, j] += -0.09 * phase
        one_body[j, i] += -0.09 * phase.conjugate()
    one_body = 0.5 * (one_body + one_body.conj().T)

    interaction = np.zeros((dimension, dimension), dtype=np.float64)
    np.fill_diagonal(interaction, 0.45)
    for i in range(dimension):
        j = (i + 1) % dimension
        interaction[i, j] = 0.12
        interaction[j, i] = 0.12
    return one_body, interaction


def test_projector_energy_matches_validated_static_functional() -> None:
    one_body, interaction = _model(5)
    projectors = _projectors(_unitary(5), (1, 2))
    dynamic = projector_energy(
        one_body, interaction, projectors, HIGH_SPIN_TRIPLET
    )
    static = general_open_shell_energy(
        one_body, interaction, projectors, HIGH_SPIN_TRIPLET
    )
    assert abs(dynamic - static) < 1.0e-13


def test_variational_generator_is_antihermitian_and_rhs_tangent() -> None:
    one_body, interaction = _model(5)
    projectors = _projectors(_unitary(5), (1, 2))
    generator = variational_generator(
        one_body, interaction, projectors, HIGH_SPIN_TRIPLET
    )
    assert np.linalg.norm(generator + generator.conj().T) < 1.0e-13

    derivatives = projector_rhs(
        one_body, interaction, projectors, HIGH_SPIN_TRIPLET
    )
    for projector, derivative in zip(projectors, derivatives, strict=True):
        assert np.linalg.norm(derivative - derivative.conj().T) < 1.0e-12
        # Tangency to P^2=P: dP = dP P + P dP.
        residual = derivative - derivative @ projector - projector @ derivative
        assert np.linalg.norm(residual) < 1.0e-12

    rdm_derivative = sum(
        occupation * derivative
        for occupation, derivative in zip(
            HIGH_SPIN_TRIPLET.occupations, derivatives, strict=True
        )
    )
    assert abs(np.trace(rdm_derivative)) < 1.0e-12


def test_closed_shell_noninteracting_limit_matches_exact_tdse_projector() -> None:
    one_body, _ = _model(4)
    interaction = np.zeros((4, 4), dtype=np.float64)
    orbitals = _unitary(4, seed=13)
    initial = _projectors(orbitals, (1,))
    final_time = 0.35

    eigenvalues, eigenvectors = np.linalg.eigh(one_body)
    unitary = eigenvectors @ np.diag(
        np.exp(-1.0j * eigenvalues * final_time / HBAR_EV_FS)
    ) @ eigenvectors.conj().T
    exact = (unitary @ initial[0] @ unitary.conj().T,)

    adaptive = integrate_dop853_projectors(
        one_body,
        interaction,
        initial,
        CLOSED_SHELL_SINGLET,
        final_time_fs=final_time,
        rtol=1.0e-12,
        atol=1.0e-14,
        max_step_fs=0.03,
    )
    assert adaptive.success
    assert projector_distance(exact, adaptive.projectors) < 2.0e-11
    constraints = projector_constraints(adaptive.projectors, CLOSED_SHELL_SINGLET)
    assert constraints.maximum_idempotency_error < 2.0e-11
    assert abs(constraints.particle_number - 2.0) < 2.0e-11


def test_rk4_shows_fourth_order_convergence_in_closed_shell_limit() -> None:
    one_body, _ = _model(4)
    interaction = np.zeros((4, 4), dtype=np.float64)
    initial = _projectors(_unitary(4, seed=21), (1,))
    final_time = 0.4

    eigenvalues, eigenvectors = np.linalg.eigh(one_body)
    unitary = eigenvectors @ np.diag(
        np.exp(-1.0j * eigenvalues * final_time / HBAR_EV_FS)
    ) @ eigenvectors.conj().T
    exact = (unitary @ initial[0] @ unitary.conj().T,)

    coarse = integrate_rk4_projectors(
        one_body,
        interaction,
        initial,
        CLOSED_SHELL_SINGLET,
        dt_fs=0.04,
        steps=10,
    )
    fine = integrate_rk4_projectors(
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


def test_variational_flow_has_zero_first_order_energy_derivative() -> None:
    one_body, interaction = _model(5)
    initial = _projectors(_unitary(5, seed=5), (1, 2))
    generator = variational_generator(
        one_body, interaction, initial, HIGH_SPIN_TRIPLET
    )
    epsilon = 1.0e-6
    forward_u = expm(+epsilon * generator)
    backward_u = expm(-epsilon * generator)
    forward = tuple(forward_u @ p @ forward_u.conj().T for p in initial)
    backward = tuple(backward_u @ p @ backward_u.conj().T for p in initial)
    derivative = (
        projector_energy(one_body, interaction, forward, HIGH_SPIN_TRIPLET)
        - projector_energy(one_body, interaction, backward, HIGH_SPIN_TRIPLET)
    ) / (2.0 * epsilon)
    assert abs(derivative) < 2.0e-8


def test_predictor_exponential_midpoint_preserves_projector_manifold() -> None:
    one_body, interaction = _model(6)
    initial = _projectors(_unitary(6, seed=17), (1, 1, 1))
    propagated = integrate_predictor_exponential_midpoint(
        one_body,
        interaction,
        initial,
        OPEN_SHELL_SINGLET,
        dt_fs=0.05,
        steps=8,
    )
    constraints = projector_constraints(propagated.projectors, OPEN_SHELL_SINGLET)
    assert constraints.maximum_hermiticity_error < 2.0e-13
    assert constraints.maximum_idempotency_error < 2.0e-12
    assert constraints.maximum_mutual_orthogonality_error < 2.0e-12
    assert abs(constraints.particle_number - 4.0) < 2.0e-12


def test_equal_occupation_singlet_shell_rotation_is_gauge_fixed_to_zero() -> None:
    one_body, interaction = _model(6)
    initial = _projectors(_unitary(6, seed=31), (1, 1, 1))
    generator = variational_generator(
        one_body, interaction, initial, OPEN_SHELL_SINGLET
    )
    frontier_one = initial[1]
    frontier_two = initial[2]
    assert np.linalg.norm(frontier_one @ generator @ frontier_two) < 1.0e-12
    assert np.linalg.norm(frontier_two @ generator @ frontier_one) < 1.0e-12


def test_nonlinear_dop853_reference_conserves_energy_and_constraints() -> None:
    one_body, interaction = _model(6)
    initial = _projectors(_unitary(6, seed=41), (1, 2))
    propagated = integrate_dop853_projectors(
        one_body,
        interaction,
        initial,
        HIGH_SPIN_TRIPLET,
        final_time_fs=0.45,
        rtol=2.0e-12,
        atol=2.0e-14,
        max_step_fs=0.03,
    )
    assert propagated.success
    constraints = projector_constraints(propagated.projectors, HIGH_SPIN_TRIPLET)
    assert constraints.maximum_hermiticity_error < 2.0e-10
    assert constraints.maximum_idempotency_error < 2.0e-9
    assert constraints.maximum_mutual_orthogonality_error < 2.0e-9
    assert abs(constraints.particle_number - 4.0) < 2.0e-9
    initial_energy = projector_energy(
        one_body, interaction, initial, HIGH_SPIN_TRIPLET
    )
    final_energy = projector_energy(
        one_body, interaction, propagated.projectors, HIGH_SPIN_TRIPLET
    )
    assert abs(final_energy - initial_energy) < 2.0e-9


def test_nonlinear_rk4_converges_to_tight_dop853_reference() -> None:
    one_body, interaction = _model(5)
    initial = _projectors(_unitary(5, seed=53), (1, 1))
    final_time = 0.3
    reference = integrate_dop853_projectors(
        one_body,
        interaction,
        initial,
        HIGH_SPIN_TRIPLET,
        final_time_fs=final_time,
        rtol=5.0e-13,
        atol=5.0e-15,
        max_step_fs=0.015,
    )
    coarse = integrate_rk4_projectors(
        one_body,
        interaction,
        initial,
        HIGH_SPIN_TRIPLET,
        dt_fs=0.03,
        steps=10,
    )
    fine = integrate_rk4_projectors(
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

    metrics = compare_projector_states(
        one_body,
        interaction,
        initial,
        reference.projectors,
        fine.projectors,
        HIGH_SPIN_TRIPLET,
    )
    assert metrics.rdm_distance < 2.0e-7
    assert metrics.energy_drift < 2.0e-8


def test_invalid_initial_projector_state_is_rejected() -> None:
    projectors = _projectors(_unitary(4), (1,))
    invalid = (1.1 * projectors[0],)
    try:
        validate_projector_manifold(invalid, CLOSED_SHELL_SINGLET)
    except ValueError as error:
        assert "idempotent" in str(error)
    else:
        raise AssertionError("invalid projector state was accepted")
