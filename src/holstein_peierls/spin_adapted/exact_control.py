"""Tiny exact many-body controls for validating the S0 open-shell functional.

These routines are intentionally exponential and are never used for production
lattices.  They construct the fixed-electron-number Fock-space Hamiltonian for
small site models so that the general open-shell energy can be checked against
a direct many-body expectation value.

Spatial sites carry alpha/beta spin orbitals.  The two-body control is the
spin-independent density-density interaction

    H_ee = 1/2 sum_ij,sum_sigma,tau V_ij
           c^dagger_(i,sigma) c^dagger_(j,tau)
           c_(j,tau) c_(i,sigma),

which gives the usual onsite Hubbard term for ``V_ii=U`` and intersite density
repulsion for ``V_ij``.  This is exactly the site-basis interaction assumed by
the open-shell contraction formulas.
"""

from __future__ import annotations

from itertools import combinations
import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def fixed_particle_basis(n_spatial: int, n_electrons: int) -> tuple[int, ...]:
    """Return bit masks for the fixed-N sector of ``2*n_spatial`` spin orbitals."""
    n_spin = 2 * n_spatial
    if not 0 <= n_electrons <= n_spin:
        raise ValueError("electron number is outside the spin-orbital space")
    result: list[int] = []
    for occupied in combinations(range(n_spin), n_electrons):
        mask = 0
        for index in occupied:
            mask |= 1 << index
        result.append(mask)
    return tuple(result)


def _fermion_sign(mask: int, orbital: int) -> float:
    lower = mask & ((1 << orbital) - 1)
    return -1.0 if lower.bit_count() % 2 else 1.0


def _apply_creation_annihilation(
    mask: int,
    create: int,
    annihilate: int,
) -> tuple[int, float] | None:
    if not (mask & (1 << annihilate)):
        return None
    sign = _fermion_sign(mask, annihilate)
    after = mask ^ (1 << annihilate)
    if after & (1 << create):
        return None
    sign *= _fermion_sign(after, create)
    after |= 1 << create
    return after, sign


def exact_density_density_hamiltonian(
    one_body: FloatArray,
    interaction: FloatArray,
    n_electrons: int,
) -> tuple[FloatArray, tuple[int, ...]]:
    """Build the exact small-system Hamiltonian in a fixed-particle Fock basis."""
    t = np.asarray(one_body, dtype=np.float64)
    v = np.asarray(interaction, dtype=np.float64)
    if t.ndim != 2 or t.shape[0] != t.shape[1]:
        raise ValueError("one_body must be square")
    n = t.shape[0]
    if v.shape != (n, n):
        raise ValueError("interaction must match one_body")
    if not np.allclose(t, t.T):
        raise ValueError("one_body must be real symmetric in this exact control")
    if not np.allclose(v, v.T):
        raise ValueError("interaction must be symmetric")

    basis = fixed_particle_basis(n, n_electrons)
    lookup = {mask: index for index, mask in enumerate(basis)}
    h = np.zeros((len(basis), len(basis)), dtype=np.float64)

    for column, mask in enumerate(basis):
        site_occupations = np.zeros(n, dtype=np.float64)
        for site in range(n):
            site_occupations[site] = float(
                ((mask >> (2 * site)) & 1) + ((mask >> (2 * site + 1)) & 1)
            )
        h[column, column] += 0.5 * float(
            site_occupations @ v @ site_occupations
            - np.dot(np.diag(v), site_occupations)
        )

        for spin in (0, 1):
            for i in range(n):
                p = 2 * i + spin
                for j in range(n):
                    q = 2 * j + spin
                    value = t[i, j]
                    if value == 0.0:
                        continue
                    applied = _apply_creation_annihilation(mask, p, q)
                    if applied is None:
                        continue
                    target, sign = applied
                    row = lookup[target]
                    h[row, column] += value * sign

    h = 0.5 * (h + h.T)
    return h, basis


def _spin_orbital(spatial: FloatArray, spin: int) -> FloatArray:
    n = spatial.size
    result = np.zeros(2 * n, dtype=np.float64)
    result[spin::2] = spatial
    return result


def determinant_state(
    occupied_spin_orbitals: FloatArray,
    basis: tuple[int, ...],
) -> FloatArray:
    """Expand one Slater determinant in the supplied site-spin Fock basis."""
    coefficients = np.asarray(occupied_spin_orbitals, dtype=np.float64)
    if coefficients.ndim != 2:
        raise ValueError("occupied spin orbitals must be columns of a matrix")
    n_spin, n_electrons = coefficients.shape
    if any(mask.bit_count() != n_electrons for mask in basis):
        raise ValueError("basis electron number does not match determinant")
    state = np.zeros(len(basis), dtype=np.float64)
    for index, mask in enumerate(basis):
        occupied = [orbital for orbital in range(n_spin) if mask & (1 << orbital)]
        state[index] = float(np.linalg.det(coefficients[occupied, :]))
    norm = float(np.linalg.norm(state))
    if norm < 1.0e-14:
        raise RuntimeError("constructed determinant has zero norm")
    return state / norm


def open_shell_singlet_state(
    spatial_orbitals: FloatArray,
    n_closed: int,
    basis: tuple[int, ...],
) -> FloatArray:
    """Build the minimal two-determinant open-shell singlet.

    Spatial columns are ordered as closed orbitals, then the two singly
    occupied orbitals.  The determinant ordering follows Miranda's
    ``|... v alpha, c beta> + |... c alpha, v beta>`` convention.
    """
    c = np.asarray(spatial_orbitals, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] < n_closed + 2:
        raise ValueError("not enough spatial orbitals for the open-shell singlet")

    common: list[FloatArray] = []
    for index in range(n_closed):
        common.extend((_spin_orbital(c[:, index], 0), _spin_orbital(c[:, index], 1)))
    valence = c[:, n_closed]
    conduction = c[:, n_closed + 1]
    determinant_1 = np.column_stack(
        common + [_spin_orbital(valence, 0), _spin_orbital(conduction, 1)]
    )
    determinant_2 = np.column_stack(
        common + [_spin_orbital(conduction, 0), _spin_orbital(valence, 1)]
    )
    state_1 = determinant_state(determinant_1, basis)
    state_2 = determinant_state(determinant_2, basis)
    state = (state_1 + state_2) / math.sqrt(2.0)
    norm = float(np.linalg.norm(state))
    if norm < 1.0e-14:
        raise RuntimeError("open-shell singlet construction produced zero norm")
    return state / norm


def high_spin_triplet_state(
    spatial_orbitals: FloatArray,
    n_closed: int,
    basis: tuple[int, ...],
) -> FloatArray:
    """Build the M_S=1 high-spin triplet determinant."""
    c = np.asarray(spatial_orbitals, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] < n_closed + 2:
        raise ValueError("not enough spatial orbitals for the triplet")
    occupied: list[FloatArray] = []
    for index in range(n_closed):
        occupied.extend((_spin_orbital(c[:, index], 0), _spin_orbital(c[:, index], 1)))
    occupied.extend(
        (
            _spin_orbital(c[:, n_closed], 0),
            _spin_orbital(c[:, n_closed + 1], 0),
        )
    )
    return determinant_state(np.column_stack(occupied), basis)


def expectation_value(state: FloatArray, hamiltonian: FloatArray) -> float:
    psi = np.asarray(state, dtype=np.float64)
    h = np.asarray(hamiltonian, dtype=np.float64)
    if h.shape != (psi.size, psi.size):
        raise ValueError("Hamiltonian dimension does not match state")
    return float(psi @ h @ psi)
