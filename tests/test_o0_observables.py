import numpy as np

from holstein_peierls.observables import (
    channel_yield,
    configuration_expansion_overlap,
    configuration_expansion_norm,
    instantaneous_occupations,
    occupations_from_orbitals,
    slater_overlap,
    slater_yield,
)


def test_rdm_occupations_match_overlap_formula() -> None:
    propagated = np.eye(3, dtype=np.complex128)
    occupations = np.array([2.0, 1.0, 0.0])
    theta = 0.37
    instantaneous = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.complex128,
    )

    from_orbitals = occupations_from_orbitals(
        propagated,
        occupations,
        instantaneous,
    )
    gamma = (propagated * occupations[np.newaxis, :]) @ propagated.conj().T
    from_rdm = instantaneous_occupations(gamma, instantaneous)

    expected = np.array(
        [
            2.0 * np.cos(theta) ** 2 + np.sin(theta) ** 2,
            2.0 * np.sin(theta) ** 2 + np.cos(theta) ** 2,
            0.0,
        ]
    )
    np.testing.assert_allclose(from_orbitals, expected, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(from_rdm, expected, rtol=0.0, atol=1.0e-14)
    assert abs(np.sum(from_rdm) - np.trace(gamma).real) < 1.0e-14


def test_partial_instantaneous_window_returns_only_requested_levels() -> None:
    gamma = np.diag([1.0, 0.4, 0.0]).astype(np.complex128)
    window = np.eye(3, dtype=np.complex128)[:, :2]
    occupations = instantaneous_occupations(gamma, window)
    np.testing.assert_allclose(occupations, [1.0, 0.4], rtol=0.0, atol=1.0e-15)


def test_slater_overlap_is_invariant_to_occupied_unitary_rotation() -> None:
    reference = np.eye(3, dtype=np.complex128)[:, :2]
    phase = np.exp(0.31j)
    rotation = np.array(
        [[0.0, phase], [-np.conj(phase), 0.0]],
        dtype=np.complex128,
    )
    state = reference @ rotation

    overlap = slater_overlap(reference, state)
    assert abs(abs(overlap) - 1.0) < 1.0e-14
    assert abs(slater_yield(reference, state) - 1.0) < 1.0e-14


def test_orthogonal_slater_determinants_have_zero_yield() -> None:
    reference = np.eye(3, dtype=np.complex128)[:, [0, 1]]
    state = np.eye(3, dtype=np.complex128)[:, [0, 2]]
    assert abs(slater_overlap(reference, state)) < 1.0e-15
    assert slater_yield(reference, state) < 1.0e-30


def test_configuration_expansion_preserves_coherent_interference() -> None:
    basis = np.eye(2, dtype=np.complex128)
    states = [basis[:, [0]], basis[:, [1]]]
    coefficients = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    reference = np.array([[1.0], [1.0]], dtype=np.complex128) / np.sqrt(2.0)

    overlap = configuration_expansion_overlap(reference, states, coefficients)
    assert abs(overlap - 1.0) < 1.0e-14
    assert abs(configuration_expansion_norm(states, coefficients) - 1.0) < 1.0e-14
    assert abs(channel_yield([reference], states, coefficients) - 1.0) < 1.0e-14


def test_gram_projector_handles_nonorthogonal_and_redundant_channel_states() -> None:
    basis = np.eye(2, dtype=np.complex128)
    state = [basis[:, [0]]]
    coefficients = np.array([1.0])
    tilted = np.array([[1.0], [1.0]], dtype=np.complex128) / np.sqrt(2.0)

    assert abs(channel_yield([tilted], state, coefficients) - 0.5) < 1.0e-14
    assert abs(channel_yield([basis[:, [0]], tilted], state, coefficients) - 1.0) < 1.0e-14
    assert abs(channel_yield([basis[:, [0]], basis[:, [0]]], state, coefficients) - 1.0) < 1.0e-14
