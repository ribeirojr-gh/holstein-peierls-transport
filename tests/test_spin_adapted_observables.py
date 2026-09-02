import numpy as np
import pytest

from holstein_peierls.spin_adapted import (
    channel_projector,
    channel_yield_from_density,
    channel_yield_from_state,
    configuration_channel_yield,
    instantaneous_occupation_numbers,
    occupation_numbers_from_propagated_orbitals,
    one_rdm_from_orbitals,
    slater_determinant_overlap,
)


def _unitary(dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    trial = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension)
    )
    q, _ = np.linalg.qr(trial)
    return np.asarray(q, dtype=np.complex128)


def test_occupation_numbers_match_direct_overlap_formula() -> None:
    propagated = _unitary(5, 7)[:, :3]
    instantaneous = _unitary(5, 19)
    occupations = np.array([2.0, 1.0, 0.35])

    overlap = instantaneous.conj().T @ propagated
    direct = np.square(np.abs(overlap)) @ occupations
    via_rdm = occupation_numbers_from_propagated_orbitals(
        propagated,
        occupations,
        instantaneous,
    )

    np.testing.assert_allclose(via_rdm, direct, atol=2.0e-14, rtol=0.0)
    assert np.sum(via_rdm) == pytest.approx(np.sum(occupations), abs=2.0e-14)


def test_one_rdm_trace_and_static_occupation_pattern() -> None:
    basis = np.eye(4, dtype=np.complex128)
    gamma = one_rdm_from_orbitals(basis[:, :2], [2.0, 1.0])

    np.testing.assert_allclose(gamma, np.diag([2.0, 1.0, 0.0, 0.0]), atol=0.0)
    assert np.trace(gamma) == pytest.approx(3.0)
    occupations = instantaneous_occupation_numbers(gamma, basis)
    np.testing.assert_allclose(occupations, [2.0, 1.0, 0.0, 0.0], atol=0.0)


def test_occupations_are_invariant_to_orbital_phases() -> None:
    propagated = _unitary(4, 31)[:, :2]
    instantaneous = _unitary(4, 43)
    occupations = np.array([2.0, 0.75])
    reference = occupation_numbers_from_propagated_orbitals(
        propagated, occupations, instantaneous
    )

    propagated_phases = np.exp(1j * np.array([0.37, -1.24]))
    instantaneous_phases = np.exp(1j * np.array([0.2, -0.8, 1.7, 2.4]))
    transformed = occupation_numbers_from_propagated_orbitals(
        propagated * propagated_phases[np.newaxis, :],
        occupations,
        instantaneous * instantaneous_phases[np.newaxis, :],
    )
    np.testing.assert_allclose(transformed, reference, atol=2.0e-14, rtol=0.0)


def test_channel_projector_handles_nonorthogonal_and_redundant_states() -> None:
    e = np.eye(3, dtype=np.complex128)
    states = np.column_stack(
        [
            e[:, 0],
            (e[:, 0] + e[:, 1]) / np.sqrt(2.0),
            2.0 * e[:, 0],
        ]
    )
    projector = channel_projector(states)

    np.testing.assert_allclose(projector, projector.conj().T, atol=2.0e-14)
    np.testing.assert_allclose(projector @ projector, projector, atol=2.0e-14)
    np.testing.assert_allclose(
        projector,
        np.diag([1.0, 1.0, 0.0]),
        atol=2.0e-14,
    )
    assert np.trace(projector).real == pytest.approx(2.0, abs=2.0e-14)


def test_channel_yield_has_zero_one_and_fractional_controls() -> None:
    e = np.eye(3, dtype=np.complex128)
    projector = channel_projector(np.column_stack([e[:, 0], e[:, 1]]))

    assert channel_yield_from_state(e[:, 0], projector) == pytest.approx(1.0)
    assert channel_yield_from_state(e[:, 2], projector) == pytest.approx(0.0)
    state = (e[:, 0] + e[:, 2]) / np.sqrt(2.0)
    assert channel_yield_from_state(state, projector) == pytest.approx(0.5)

    rho = 0.25 * np.outer(e[:, 0], e[:, 0].conj()) + 0.75 * np.outer(
        e[:, 2], e[:, 2].conj()
    )
    assert channel_yield_from_density(rho, projector) == pytest.approx(0.25)


def test_multiconfigurational_channel_retains_destructive_interference() -> None:
    e = np.eye(2, dtype=np.complex128)
    propagated = (e[:, 0] + e[:, 1]) / np.sqrt(2.0)
    target_configuration = (e[:, 0] - e[:, 1]) / np.sqrt(2.0)

    yield_value = configuration_channel_yield(
        propagated,
        target_configuration[:, np.newaxis],
    )
    assert yield_value == pytest.approx(0.0, abs=2.0e-15)

    # Squaring determinant/configuration contributions before combining their
    # amplitudes would lose this cancellation and incorrectly give unity.
    naive_component_probability = (
        abs(np.vdot(e[:, 0], propagated)) ** 2
        + abs(np.vdot(e[:, 1], propagated)) ** 2
    )
    assert naive_component_probability == pytest.approx(1.0)


def test_slater_determinant_overlap_uses_determinant_of_orbital_overlaps() -> None:
    basis = np.eye(4, dtype=np.complex128)
    left = basis[:, :2]
    phases = np.diag(np.exp(1j * np.array([0.31, -0.72])))
    right_gauge = left @ phases

    overlap = slater_determinant_overlap(left, right_gauge)
    assert overlap == pytest.approx(np.linalg.det(phases), abs=2.0e-15)
    assert abs(overlap) == pytest.approx(1.0, abs=2.0e-15)

    right_orthogonal = np.column_stack([basis[:, 0], basis[:, 2]])
    assert slater_determinant_overlap(left, right_orthogonal) == pytest.approx(0.0)


def test_observable_validation_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="orthonormal"):
        one_rdm_from_orbitals(np.ones((3, 2)), [1.0, 1.0])
    with pytest.raises(ValueError, match="non-negative"):
        one_rdm_from_orbitals(np.eye(2), [1.0, -1.0])
    with pytest.raises(ValueError, match="normalized"):
        channel_yield_from_state([1.0, 1.0], np.eye(2))
    with pytest.raises(ValueError, match="at least one"):
        channel_projector(np.empty((3, 0)))
