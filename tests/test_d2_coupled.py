"""First coupled electron-lattice conservation and convergence gates for D2."""

from __future__ import annotations

import numpy as np

from holstein_peierls.dynamics.coupled import (
    CoupledEhrenfestState,
    coupled_verlet_step,
    integrate_coupled_dop853,
    integrate_coupled_verlet,
    interpolate_lattice,
)
from holstein_peierls.dynamics.ehrenfest import (
    LatticeVelocity,
    dynamic_total_energy,
)
from holstein_peierls.electronic import solve_ground_state
from holstein_peierls.lattice import LatticeState
from holstein_peierls.parameters import StaticPolaronParameters


def _initial_control() -> tuple[CoupledEhrenfestState, StaticPolaronParameters]:
    parameters = StaticPolaronParameters(nx=3, ny=3, polaron_position=5)
    lattice = LatticeState.zeros(3, 3)
    lattice.u[1, 1] = -0.035
    lattice.vx[1, 1] = 0.012
    lattice.vy[1, 1] = -0.008
    ground = solve_ground_state(lattice, parameters, solver="dense_lowest")
    velocity = LatticeVelocity.zeros(3, 3)
    velocity.u[1, 1] = 8.0e-4
    velocity.vx[1, 1] = -4.0e-4
    velocity.vy[1, 1] = 3.0e-4
    return (
        CoupledEhrenfestState(
            lattice,
            velocity,
            np.asarray(ground.wavefunction, dtype=np.complex128),
        ),
        parameters,
    )


def _phase_aligned_electronic_error(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> float:
    overlap = np.vdot(reference, candidate)
    aligned = candidate
    if abs(overlap) > 0.0:
        aligned = candidate * np.conj(overlap) / abs(overlap)
    return float(np.linalg.norm(aligned - reference) / np.linalg.norm(reference))


def _lattice_error(reference: LatticeState, candidate: LatticeState) -> float:
    return float(
        np.sqrt(
            np.linalg.norm(candidate.u - reference.u) ** 2
            + np.linalg.norm(candidate.vx - reference.vx) ** 2
            + np.linalg.norm(candidate.vy - reference.vy) ** 2
        )
    )


def test_lattice_interpolation_reproduces_endpoints() -> None:
    initial, _ = _initial_control()
    end = initial.lattice.copy()
    end.u += 0.1
    end.vx -= 0.03
    end.vy += 0.04
    at_zero = interpolate_lattice(initial.lattice, end, 0.0)
    at_one = interpolate_lattice(initial.lattice, end, 1.0)
    np.testing.assert_allclose(at_zero.u, initial.lattice.u)
    np.testing.assert_allclose(at_zero.vx, initial.lattice.vx)
    np.testing.assert_allclose(at_zero.vy, initial.lattice.vy)
    np.testing.assert_allclose(at_one.u, end.u)
    np.testing.assert_allclose(at_one.vx, end.vx)
    np.testing.assert_allclose(at_one.vy, end.vy)


def test_tight_coupled_dop853_conserves_total_energy() -> None:
    initial, parameters = _initial_control()
    energy_initial = dynamic_total_energy(
        initial.lattice,
        initial.velocity,
        parameters,
        initial.electronic_state,
    ).total
    reference = integrate_coupled_dop853(
        initial,
        parameters,
        final_time_fs=2.0,
        rtol=2.0e-11,
        atol=2.0e-13,
        max_step_fs=0.01,
    )
    assert reference.success
    energy_final = dynamic_total_energy(
        reference.state.lattice,
        reference.state.velocity,
        parameters,
        reference.state.electronic_state,
    ).total
    assert abs(energy_final - energy_initial) < 2.0e-9


def test_coupled_verlet_cfm4_converges_to_full_dop853_reference() -> None:
    initial, parameters = _initial_control()
    final_time = 1.0
    reference = integrate_coupled_dop853(
        initial,
        parameters,
        final_time_fs=final_time,
        rtol=2.0e-12,
        atol=2.0e-14,
        max_step_fs=0.005,
    )
    assert reference.success

    electronic_errors = []
    lattice_errors = []
    for dt in (0.1, 0.05, 0.025):
        result = integrate_coupled_verlet(
            initial,
            parameters,
            dt_fs=dt,
            steps=int(round(final_time / dt)),
            electronic_method="cfm4_lanczos",
            krylov_dimension=6,
        )
        electronic_errors.append(
            _phase_aligned_electronic_error(
                reference.state.electronic_state,
                result.state.electronic_state,
            )
        )
        lattice_errors.append(
            _lattice_error(reference.state.lattice, result.state.lattice)
        )

    electronic_orders = [
        np.log2(electronic_errors[i] / electronic_errors[i + 1])
        for i in range(2)
    ]
    lattice_orders = [
        np.log2(lattice_errors[i] / lattice_errors[i + 1])
        for i in range(2)
    ]
    assert min(electronic_orders) > 1.7
    assert min(lattice_orders) > 1.7


def _maximum_energy_deviation(dt_fs: float, final_time_fs: float) -> float:
    initial, parameters = _initial_control()
    current = initial
    initial_energy = dynamic_total_energy(
        current.lattice,
        current.velocity,
        parameters,
        current.electronic_state,
    ).total
    maximum = 0.0
    steps = int(round(final_time_fs / dt_fs))
    for _ in range(steps):
        current, _, _ = coupled_verlet_step(
            current,
            parameters,
            dt_fs,
            electronic_method="cfm4_lanczos",
            krylov_dimension=6,
        )
        energy = dynamic_total_energy(
            current.lattice,
            current.velocity,
            parameters,
            current.electronic_state,
        ).total
        maximum = max(maximum, abs(energy - initial_energy))
    return maximum


def test_coupled_total_energy_error_decreases_with_time_step() -> None:
    coarse = _maximum_energy_deviation(0.2, 4.0)
    medium = _maximum_energy_deviation(0.1, 4.0)
    fine = _maximum_energy_deviation(0.05, 4.0)
    assert medium < 0.4 * coarse
    assert fine < 0.4 * medium


def test_cfm4_split_preserves_electronic_norm_better_than_rk4() -> None:
    initial, parameters = _initial_control()
    cfm4 = integrate_coupled_verlet(
        initial,
        parameters,
        dt_fs=0.2,
        steps=20,
        electronic_method="cfm4_lanczos",
        krylov_dimension=6,
    )
    rk4 = integrate_coupled_verlet(
        initial,
        parameters,
        dt_fs=0.2,
        steps=20,
        electronic_method="rk4",
    )
    initial_norm = np.linalg.norm(initial.electronic_state)
    cfm4_error = abs(np.linalg.norm(cfm4.state.electronic_state) - initial_norm)
    rk4_error = abs(np.linalg.norm(rk4.state.electronic_state) - initial_norm)
    assert cfm4_error < 2.0e-13
    assert cfm4_error < rk4_error
