import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix

from holstein_peierls.dynamics import (
    HBAR_EV_FS,
    CountingMatrixHamiltonian,
    benchmark_frozen_hamiltonian,
    cfm4_frozen_limit_step,
    compare_to_reference,
    crank_nicolson_step,
    exact_spectral_step,
    lanczos_exponential_step,
    rk4_step,
    rkf78_step,
)


def _two_level_control() -> tuple[np.ndarray, np.ndarray]:
    hamiltonian = np.array(
        [
            [0.20, 0.07 - 0.03j],
            [0.07 + 0.03j, -0.11],
        ],
        dtype=np.complex128,
    )
    state = np.array([1.0, 0.4 + 0.8j], dtype=np.complex128)
    state /= np.linalg.norm(state)
    return hamiltonian, state


def test_exact_spectral_step_matches_dense_matrix_exponential() -> None:
    hamiltonian, state = _two_level_control()
    dt_fs = 0.73
    expected = expm((-1.0j * dt_fs / HBAR_EV_FS) * hamiltonian) @ state
    result = exact_spectral_step(hamiltonian, state, dt_fs)

    np.testing.assert_allclose(result.state, expected, rtol=0.0, atol=2.0e-15)
    assert result.eigendecompositions == 1
    assert result.hamiltonian_applications == 0
    assert abs(np.linalg.norm(result.state) - 1.0) < 2.0e-15


def test_rk4_uses_four_hamiltonian_actions_and_converges_under_step_halving() -> None:
    hamiltonian, state = _two_level_control()
    sparse = csr_matrix(hamiltonian)
    action = CountingMatrixHamiltonian(sparse)

    coarse = rk4_step(action, state, 1.0)
    assert coarse.hamiltonian_applications == 4
    assert action.applications == 4
    coarse_reference = exact_spectral_step(hamiltonian, state, 1.0).state
    coarse_error = np.linalg.norm(coarse.state - coarse_reference)

    action.reset()
    half_one = rk4_step(action, state, 0.5)
    half_two = rk4_step(action, half_one.state, 0.5)
    fine_reference = coarse_reference
    fine_error = np.linalg.norm(half_two.state - fine_reference)
    assert action.applications == 8
    assert fine_error < coarse_error / 10.0


def test_rkf78_order8_is_high_accuracy_and_reports_embedded_error() -> None:
    hamiltonian, state = _two_level_control()
    action = CountingMatrixHamiltonian(csr_matrix(hamiltonian))
    dt_fs = 1.0

    rk4 = rk4_step(action, state, dt_fs)
    action.reset()
    rkf = rkf78_step(action, state, dt_fs)
    reference = exact_spectral_step(hamiltonian, state, dt_fs).state

    assert rkf.hamiltonian_applications == 13
    assert action.applications == 13
    assert rkf.embedded_error_estimate is not None
    assert rkf.embedded_error_estimate > 0.0
    rk4_error = np.linalg.norm(rk4.state - reference)
    rkf_error = np.linalg.norm(rkf.state - reference)
    assert rkf_error < rk4_error * 1.0e-3
    assert rkf_error < 1.0e-9


def test_full_dimension_lanczos_and_frozen_cfm_match_exact_two_level_result() -> None:
    hamiltonian, state = _two_level_control()
    sparse = csr_matrix(hamiltonian)
    reference = exact_spectral_step(hamiltonian, state, 0.91).state

    lanczos_action = CountingMatrixHamiltonian(sparse)
    lanczos = lanczos_exponential_step(
        lanczos_action,
        state,
        0.91,
        krylov_dimension=2,
    )
    np.testing.assert_allclose(lanczos.state, reference, rtol=0.0, atol=2.0e-13)
    assert lanczos.hamiltonian_applications == 2
    assert lanczos.krylov_dimension_used == 2
    assert abs(np.linalg.norm(lanczos.state) - 1.0) < 2.0e-13

    cfm_action = CountingMatrixHamiltonian(sparse)
    cfm = cfm4_frozen_limit_step(
        cfm_action,
        state,
        0.91,
        krylov_dimension=2,
    )
    np.testing.assert_allclose(cfm.state, reference, rtol=0.0, atol=4.0e-13)
    assert cfm.hamiltonian_applications == 4
    assert abs(np.linalg.norm(cfm.state) - 1.0) < 4.0e-13


def test_crank_nicolson_preserves_norm_without_manual_renormalization() -> None:
    hamiltonian, state = _two_level_control()
    result = crank_nicolson_step(csr_matrix(hamiltonian), state, 1.5)
    reference = exact_spectral_step(hamiltonian, state, 1.5).state

    assert result.linear_solves == 1
    assert result.hamiltonian_applications == 1
    assert abs(np.linalg.norm(result.state) - np.linalg.norm(state)) < 2.0e-15
    assert np.linalg.norm(result.state - reference) < 2.0e-2


def test_phase_aligned_metrics_ignore_global_phase_but_raw_error_does_not() -> None:
    hamiltonian, state = _two_level_control()
    candidate = np.exp(0.73j) * state
    metrics = compare_to_reference(state, candidate, hamiltonian)

    assert abs(metrics.fidelity - 1.0) < 2.0e-15
    assert metrics.phase_aligned_state_error < 2.0e-15
    assert metrics.raw_state_error > 0.5
    assert metrics.energy_error < 2.0e-15
    assert metrics.norm_error < 2.0e-15


def test_common_benchmark_counts_work_and_uses_same_frozen_operator() -> None:
    hamiltonian = np.array(
        [
            [0.21, 0.04 + 0.01j, -0.02j],
            [0.04 - 0.01j, -0.08, 0.06],
            [0.02j, 0.06, 0.13],
        ],
        dtype=np.complex128,
    )
    state = np.array([1.0, 0.3j, -0.4 + 0.2j], dtype=np.complex128)
    state /= np.linalg.norm(state)
    steps = 3
    suite = benchmark_frozen_hamiltonian(
        hamiltonian,
        csr_matrix(hamiltonian),
        state,
        dt_fs=0.1,
        steps=steps,
        krylov_dimension=3,
    )

    records = {record.method: record for record in suite.records}
    assert tuple(records) == (
        "legacy_spectral",
        "rk4",
        "rkf78_order8",
        "lanczos_m3",
        "cfm4_frozen_m3",
        "crank_nicolson",
    )
    assert records["legacy_spectral"].eigendecompositions == steps
    assert records["rk4"].hamiltonian_applications == 4 * steps
    assert records["rkf78_order8"].hamiltonian_applications == 13 * steps
    assert records["lanczos_m3"].hamiltonian_applications == 3 * steps
    assert records["cfm4_frozen_m3"].hamiltonian_applications == 6 * steps
    assert records["crank_nicolson"].linear_solves == steps
    assert records["crank_nicolson"].hamiltonian_applications == steps

    assert records["legacy_spectral"].metrics.phase_aligned_state_error < 1.0e-13
    assert records["lanczos_m3"].metrics.phase_aligned_state_error < 1.0e-12
    assert records["cfm4_frozen_m3"].metrics.phase_aligned_state_error < 2.0e-12
    assert records["rkf78_order8"].metrics.phase_aligned_state_error < 1.0e-12
