"""Projection amplitudes and yields for one- and many-particle states.

For a normalized evolved state ``|Psi(t)>`` and a normalized reference
configuration ``|Phi_K>``, the relative projection yield is

    I_K(t) = |<Phi_K|Psi(t)>|^2.

The Slater-determinant functions implement the determinant-overlap formula used
by the supplied many-electron references. Generic Hilbert-space projectors are
also provided for states such as the distinguishable electron-hole wavefunction
``Psi(i_e, i_h)`` that are not represented internally as Slater determinants.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

ComplexArray = NDArray[np.complex128]


def _normalized_vector(values: ArrayLike, *, name: str) -> ComplexArray:
    vector = np.asarray(values, dtype=np.complex128).reshape(-1)
    if vector.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError(f"{name} must have non-zero norm")
    return np.asarray(vector / norm, dtype=np.complex128)


def hilbert_channel_yield(
    state: ArrayLike,
    reference_states: Sequence[ArrayLike],
    *,
    rcond: float = 1.0e-12,
    tolerance: float = 1.0e-10,
) -> float:
    """Project a state vector onto a possibly non-orthogonal reference subspace.

    If ``G_ab=<phi_a|phi_b>`` and ``v_a=<phi_a|psi>``, the projection
    probability is ``v^dagger G^+ v`` for normalized ``psi``. Reference vectors
    may be linearly dependent; the Moore-Penrose pseudoinverse prevents double
    counting.
    """
    if rcond <= 0.0 or tolerance <= 0.0:
        raise ValueError("rcond and tolerance must be positive")
    psi = _normalized_vector(state, name="state")
    if len(reference_states) == 0:
        raise ValueError("reference_states must contain at least one state")
    references = tuple(
        _normalized_vector(item, name=f"reference_states[{index}]")
        for index, item in enumerate(reference_states)
    )
    if any(item.shape != psi.shape for item in references):
        raise ValueError("state and reference vectors must have equal dimensions")

    reference_matrix = np.column_stack(references)
    gram = reference_matrix.conj().T @ reference_matrix
    overlaps = reference_matrix.conj().T @ psi
    metric = np.linalg.pinv(gram, rcond=rcond, hermitian=True)
    value_complex = complex(overlaps.conj() @ metric @ overlaps)
    if abs(value_complex.imag) > tolerance:
        raise FloatingPointError("Hilbert-space channel yield has a non-negligible imaginary part")
    value = float(value_complex.real)
    if value < -tolerance or value > 1.0 + tolerance:
        raise FloatingPointError("Hilbert-space channel yield lies outside [0, 1]")
    return float(np.clip(value, 0.0, 1.0))


def basis_mask_yield(state: ArrayLike, mask: ArrayLike) -> float:
    """Return probability in a subset of an orthonormal computational basis.

    ``state`` may have any shape (for example ``Psi[i_e, i_h]``); ``mask`` must
    have the same shape and selects basis configurations belonging to the
    physical channel. The state is normalized internally so the returned value
    is always a probability.
    """
    amplitudes = np.asarray(state, dtype=np.complex128)
    selected = np.asarray(mask, dtype=np.bool_)
    if amplitudes.shape != selected.shape or amplitudes.size == 0:
        raise ValueError("state and mask must be non-empty arrays of identical shape")
    if not np.all(np.isfinite(amplitudes)):
        raise ValueError("state amplitudes must be finite")
    norm2 = float(np.vdot(amplitudes.reshape(-1), amplitudes.reshape(-1)).real)
    if norm2 <= 0.0:
        raise ValueError("state must have positive norm")
    probability = np.abs(amplitudes) ** 2
    return float(np.sum(probability[selected]) / norm2)


def _configuration(values: ArrayLike, *, name: str) -> ComplexArray:
    orbitals = np.asarray(values, dtype=np.complex128)
    if orbitals.ndim != 2:
        raise ValueError(f"{name} must have shape (n_basis, n_occupied)")
    if orbitals.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one occupied orbital")
    overlap = orbitals.conj().T @ orbitals
    identity = np.eye(orbitals.shape[1], dtype=np.complex128)
    if not np.allclose(overlap, identity, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"{name} occupied orbitals must be orthonormal")
    return orbitals


def _configurations(values: Sequence[ArrayLike], *, name: str) -> tuple[ComplexArray, ...]:
    if len(values) == 0:
        raise ValueError(f"{name} must contain at least one configuration")
    result = tuple(
        _configuration(item, name=f"{name}[{index}]")
        for index, item in enumerate(values)
    )
    basis_sizes = {item.shape[0] for item in result}
    electron_counts = {item.shape[1] for item in result}
    if len(basis_sizes) != 1:
        raise ValueError(f"all {name} configurations must use the same one-particle basis")
    if len(electron_counts) != 1:
        raise ValueError(f"all {name} configurations must contain the same particle number")
    return result


def slater_overlap(bra_orbitals: ArrayLike, ket_orbitals: ArrayLike) -> complex:
    """Return ``<Phi_bra|Phi_ket>`` for normalized Slater determinants."""
    bra = _configuration(bra_orbitals, name="bra_orbitals")
    ket = _configuration(ket_orbitals, name="ket_orbitals")
    if bra.shape != ket.shape:
        raise ValueError("Slater determinants must use equal basis and particle dimensions")
    return complex(np.linalg.det(bra.conj().T @ ket))


def slater_yield(reference_orbitals: ArrayLike, state_orbitals: ArrayLike) -> float:
    """Return ``|<Phi_reference|Phi_state>|^2`` for two determinants."""
    overlap = slater_overlap(reference_orbitals, state_orbitals)
    return float(abs(overlap) ** 2)


def slater_gram_matrix(configurations: Sequence[ArrayLike]) -> ComplexArray:
    """Return the Gram matrix of normalized Slater determinants."""
    configs = _configurations(configurations, name="configurations")
    count = len(configs)
    gram = np.empty((count, count), dtype=np.complex128)
    for i, bra in enumerate(configs):
        for j in range(i, count):
            value = complex(np.linalg.det(bra.conj().T @ configs[j]))
            gram[i, j] = value
            gram[j, i] = np.conj(value)
    return gram


def configuration_expansion_overlap(
    reference_configuration: ArrayLike,
    state_configurations: Sequence[ArrayLike],
    state_coefficients: ArrayLike,
) -> complex:
    """Return the coherent overlap of a determinant with a configuration expansion."""
    reference = _configuration(reference_configuration, name="reference_configuration")
    states = _configurations(state_configurations, name="state_configurations")
    if any(item.shape != reference.shape for item in states):
        raise ValueError("reference and state configurations must have equal dimensions")
    coefficients = np.asarray(state_coefficients, dtype=np.complex128)
    if coefficients.ndim != 1 or coefficients.shape[0] != len(states):
        raise ValueError("state_coefficients must match the number of configurations")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("state_coefficients must be finite")
    amplitudes = np.asarray(
        [np.linalg.det(reference.conj().T @ item) for item in states],
        dtype=np.complex128,
    )
    return complex(amplitudes @ coefficients)


def configuration_expansion_norm(
    configurations: Sequence[ArrayLike], coefficients: ArrayLike
) -> float:
    """Return ``<Psi|Psi>`` for a determinant expansion."""
    configs = _configurations(configurations, name="configurations")
    coeff = np.asarray(coefficients, dtype=np.complex128)
    if coeff.ndim != 1 or coeff.shape[0] != len(configs):
        raise ValueError("coefficients must match the number of configurations")
    if not np.all(np.isfinite(coeff)):
        raise ValueError("coefficients must be finite")
    gram = slater_gram_matrix(configs)
    norm = complex(coeff.conj() @ gram @ coeff)
    if abs(norm.imag) > 1.0e-10:
        raise FloatingPointError("configuration-expansion norm has a non-negligible imaginary part")
    value = float(norm.real)
    if value <= 0.0:
        raise ValueError("configuration expansion must have positive norm")
    return value


def channel_yield(
    reference_configurations: Sequence[ArrayLike],
    state_configurations: Sequence[ArrayLike],
    state_coefficients: ArrayLike,
    *,
    rcond: float = 1.0e-12,
    tolerance: float = 1.0e-10,
) -> float:
    """Project a determinant expansion onto a reference-configuration channel."""
    if rcond <= 0.0 or tolerance <= 0.0:
        raise ValueError("rcond and tolerance must be positive")
    references = _configurations(reference_configurations, name="reference_configurations")
    states = _configurations(state_configurations, name="state_configurations")
    if references[0].shape != states[0].shape:
        raise ValueError("reference and state configurations must have equal dimensions")
    coefficients = np.asarray(state_coefficients, dtype=np.complex128)
    if coefficients.ndim != 1 or coefficients.shape[0] != len(states):
        raise ValueError("state_coefficients must match the state configuration count")
    overlaps = np.asarray(
        [configuration_expansion_overlap(reference, states, coefficients) for reference in references],
        dtype=np.complex128,
    )
    gram = slater_gram_matrix(references)
    projector_metric = np.linalg.pinv(gram, rcond=rcond, hermitian=True)
    numerator = complex(overlaps.conj() @ projector_metric @ overlaps)
    if abs(numerator.imag) > tolerance:
        raise FloatingPointError("channel projection has a non-negligible imaginary part")
    denominator = configuration_expansion_norm(states, coefficients)
    value = float(numerator.real / denominator)
    if value < -tolerance or value > 1.0 + tolerance:
        raise FloatingPointError("channel yield lies outside the normalized probability interval")
    return float(np.clip(value, 0.0, 1.0))
