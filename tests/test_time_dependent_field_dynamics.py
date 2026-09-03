from __future__ import annotations

import numpy as np
import pytest

from holstein_peierls.dynamics.field import (
    UniformElectricField2D,
    build_dense_field_hamiltonian,
)
from holstein_peierls.dynamics.frozen import exact_spectral_step
from holstein_peierls.dynamics.time_dependent import (
    cfm4_lanczos_step,
    compare_time_dependent_states,
    integrate_cfm4_lanczos,
    integrate_dop853_time_dependent,
    integrate_rk4_time_dependent,
)
from holstein_peierls.lattice import LatticeState
from holstein_peierls.spin_adapted.isotropic import IsotropicControlParameters


def _ring_hamiltonian(time_fs: float) -> np.ndarray:
    """Four-site noncommuting D1 control with a deliberately strong field."""
    n = 4
    hopping = 0.1
    hbar_ev_fs = 0.6582119569
    phase = -(3.0 * 0.2 / hbar_ev_fs) * time_fs
    matrix = np.zeros((n, n), dtype=np.complex128)
    for site in range(n):
        neighbour = (site + 1) % n
        forward = -hopping * np.exp(1.0j * phase)
        matrix[site, neighbour] += forward
        matrix[neighbour, site] += np.conj(forward)
    # A small static diagonal term removes accidental symmetries without changing
    # the fourth-order convergence target.
    matrix += np.diag(np.asarray((-0.03, 0.01, 0.02, 0.0)))
    return matrix


def _initial_state() -> np.ndarray:
    state = np.asarray((1.0, 0.2j, -0.1, 0.05j), dtype=np.complex128)
    return state / np.linalg.norm(state)


def test_cfm4_constant_hamiltonian_reduces_to_exact_exponential() -> None:
    matrix = np.asarray(
        [
            [0.1, -0.07 + 0.02j, 0.0],
            [-0.07 - 0.02j, -0.03, 0.04],
            [0.0, 0.04, 0.02],
        ],
        dtype=np.complex128,
    )
    state = np.asarray((1.0, 0.3j, -0.2), dtype=np.complex128)
    state /= np.linalg.norm(state)
    dt = 0.17
    exact = exact_spectral_step(matrix, state, dt).state
    candidate = cfm4_lanczos_step(
        lambda _time: matrix,
        state,
        0.0,
        dt,
        krylov_dimension=3,
    ).state
    metrics = compare_time_dependent_states(exact, candidate)
    assert metrics.phase_aligned_state_error < 2.0e-14
    assert metrics.norm_error < 2.0e-14


@pytest.mark.parametrize("method", ["rk4", "cfm4"])
def test_time_dependent_methods_show_fourth_order_global_convergence(method: str) -> None:
    state = _initial_state()
    final_time = 2.0
    reference = integrate_dop853_time_dependent(
        _ring_hamiltonian,
        state,
        final_time_fs=final_time,
        rtol=2.0e-13,
        atol=2.0e-15,
        max_step_fs=0.005,
    )
    assert reference.success

    errors: list[float] = []
    for dt in (0.2, 0.1, 0.05):
        steps = int(round(final_time / dt))
        if method == "rk4":
            result = integrate_rk4_time_dependent(
                _ring_hamiltonian,
                state,
                dt_fs=dt,
                steps=steps,
            )
        else:
            result = integrate_cfm4_lanczos(
                _ring_hamiltonian,
                state,
                dt_fs=dt,
                steps=steps,
                krylov_dimension=4,
            )
        errors.append(
            compare_time_dependent_states(
                reference.state,
                result.state,
            ).phase_aligned_state_error
        )

    order_coarse = np.log2(errors[0] / errors[1])
    order_fine = np.log2(errors[1] / errors[2])
    assert order_coarse > 3.7
    assert order_fine > 3.7


def test_cfm4_preserves_norm_to_roundoff_for_time_dependent_field() -> None:
    state = _initial_state()
    result = integrate_cfm4_lanczos(
        _ring_hamiltonian,
        state,
        dt_fs=0.1,
        steps=20,
        krylov_dimension=4,
    )
    assert abs(np.linalg.norm(result.state) - 1.0) < 2.0e-14


def test_holstein_peierls_field_control_agrees_with_tight_reference() -> None:
    parameters = IsotropicControlParameters().to_polaron_parameters(nx=4, ny=4)
    lattice = LatticeState.zeros(4, 4)
    field = UniformElectricField2D.from_millivolt_per_angstrom(2.0, 0.0)

    def hamiltonian_at(time_fs: float) -> np.ndarray:
        return build_dense_field_hamiltonian(lattice, parameters, field, time_fs)

    _, eigenvectors = np.linalg.eigh(hamiltonian_at(0.0))
    state = np.asarray(
        (eigenvectors[:, 0] + 0.35j * eigenvectors[:, 3]),
        dtype=np.complex128,
    )
    state /= np.linalg.norm(state)
    final_time = 1.0
    reference = integrate_dop853_time_dependent(
        hamiltonian_at,
        state,
        final_time_fs=final_time,
        rtol=2.0e-12,
        atol=2.0e-14,
        max_step_fs=0.005,
    )
    candidate = integrate_cfm4_lanczos(
        hamiltonian_at,
        state,
        dt_fs=0.05,
        steps=20,
        krylov_dimension=8,
    )
    metrics = compare_time_dependent_states(reference.state, candidate.state)
    assert reference.success
    assert metrics.phase_aligned_state_error < 1.0e-9
    assert metrics.norm_error < 2.0e-13
    assert candidate.hamiltonian_evaluations == 40
    assert candidate.hamiltonian_applications <= 320
