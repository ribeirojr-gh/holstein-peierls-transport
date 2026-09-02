"""Many-electron projection amplitudes and yields.

For a normalized evolved many-electron state ``|Psi(t)>`` and a normalized
configuration ``|Phi_K>``, the literature defines the relative yield as

    I_K(t) = |<Phi_K|Psi(t)>|^2.

For Slater determinants, the overlap is the determinant of the occupied-orbital
overlap matrix.  This module also supports coherent multiconfigurational
expansions and projection onto a channel spanned by several, possibly
non-orthogonal, reference configurations.  The latter uses the Gram-matrix
projector rather than blindly summing probabilities.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

ComplexArray = NDArray[np.complex128]


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
    result = tuple(_configuration(item, name=f"{name}[{index}]") for index, item in enumerate(values))
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
    """Return the Gram matrix of a set of normalized Slater determinants."""
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
    """Return the coherent overlap of one determinant with a configuration expansion.

    The state is

        |Psi> = sum_beta c_beta |D_beta>.

    Coefficients are summed at the amplitude level before taking any modulus,
    preserving interference between configurations.
    """
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
        [np.linalg.det(reference.conj().T @ state) for state in states],
        dtype=np.complex128,
    )
    return complex(amplitudes @ coefficients)


def configuration_expansion_norm(
    configurations: Sequence[ArrayLike],
    coefficients: ArrayLike,
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
    """Project a configuration expansion onto the span of a reference channel.

    Let ``G_ab=<Phi_a|Phi_b>`` be the Gram matrix of the reference
    configurations and ``v_a=<Phi_a|Psi>``.  The projector onto their span is

        P = sum_ab |Phi_a> (G^+)_{ab} <Phi_b|,

    where ``G^+`` is the Moore-Penrose pseudoinverse.  The normalized channel
    yield is therefore ``v^dagger G^+ v / <Psi|Psi>``.  This reduces to a sum
    of individual configuration yields when the channel references are
    orthonormal, but remains correct for redundant or non-orthogonal references.
    """
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
        [
            configuration_expansion_overlap(reference, states, coefficients)
            for reference in references
        ],
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
    if value < 0.0:
        value = 0.0
    elif value > 1.0:
        value = 1.0
    return value
