"""Independent full-system reference gates for D3."""

from __future__ import annotations

import numpy as np

from holstein_peierls.dynamics.coupled import (
    CoupledEhrenfestState,
    integrate_coupled_dop853,
)
from holstein_peierls.dynamics.driven import (
    field_dynamic_total_energy,
    integrate_coupled_field_verlet,
)
from holstein_peierls.dynamics.driven_reference import integrate_coupled_field_dop853
from holstein_peierls.dynamics.ehrenfest import LatticeVelocity
from holstein_peierls.dynamics.field import UniformElectricField2D
from holstein_peierls.lattice import LatticeState
from holstein_peierls.parameters import StaticPolaronParameters


def _control():
    parameters = StaticPolaronParameters(nx=4, ny=4, polaron_position=6)
    rng = np.random.default_rng(271828)
    lattice = LatticeState(
        8.0e-3 * rng.normal(size=(4, 4)),
        8.0e-3 * rng.normal(size=(4, 4)),
        8.0e-3 * rng.normal(size=(4, 4)),
    )
    velocity = LatticeVelocity(
        2.0e-4 * rng.normal(size=(4, 4)),
        1.0e-4 * rng.normal(size=(4, 4)),
        1.0e-4 * rng.normal(size=(4, 4)),
    )
    psi = rng.normal(size=16) + 1.0j * rng.normal(size=16)
    psi = np.asarray(psi / np.linalg.norm(psi), dtype=np.complex128)
    return parameters, CoupledEhrenfestState(lattice, velocity, psi)


def _phase_aligned_state_error(reference, candidate):
    overlap = np.vdot(reference, candidate)
    if abs(overlap) > 0.0:
        candidate = candidate * (np.conj(overlap) / abs(overlap))
    return float(np.linalg.norm(candidate - reference) / np.linalg.norm(reference))


def _lattice_distance(reference, candidate):
    differences = (
        candidate.u - reference.u,
        candidate.vx - reference.vx,
        candidate.vy - reference.vy,
    )
    return float(np.sqrt(sum(np.vdot(d, d).real for d in differences)))


def test_zero_field_reference_reduces_to_d2_full_system_dop853():
    parameters, initial = _control()
    field = UniformElectricField2D()
    final_time = 0.3

    d2 = integrate_coupled_dop853(
        initial,
        parameters,
        final_time_fs=final_time,
        rtol=2.0e-11,
        atol=2.0e-13,
        max_step_fs=0.01,
    )
    d3 = integrate_coupled_field_dop853(
        initial,
        parameters,
        field,
        final_time_fs=final_time,
        rtol=2.0e-11,
        atol=2.0e-13,
        max_step_fs=0.01,
    )

    assert d2.success and d3.success
    assert abs(d3.field_work_eV) < 1.0e-16
    assert _phase_aligned_state_error(
        d2.state.electronic_state,
        d3.state.electronic_state,
    ) < 2.0e-11
    assert _lattice_distance(d2.state.lattice, d3.state.lattice) < 2.0e-11


def test_tight_driven_dop853_satisfies_energy_work_balance():
    parameters, initial = _control()
    field = UniformElectricField2D.from_millivolt_per_angstrom(
        5.0,
        0.31,
        ax_angstrom=3.0,
        ay_angstrom=3.4,
    )
    final_time = 0.5
    initial_energy = field_dynamic_total_energy(
        initial.lattice,
        initial.velocity,
        parameters,
        initial.electronic_state,
        field,
        0.0,
    ).total

    reference = integrate_coupled_field_dop853(
        initial,
        parameters,
        field,
        final_time_fs=final_time,
        rtol=2.0e-11,
        atol=2.0e-13,
        max_step_fs=0.005,
    )
    assert reference.success
    final_energy = field_dynamic_total_energy(
        reference.state.lattice,
        reference.state.velocity,
        parameters,
        reference.state.electronic_state,
        field,
        final_time,
    ).total
    residual = abs((final_energy - initial_energy) - reference.field_work_eV)
    assert residual < 2.0e-10
    assert abs(np.linalg.norm(reference.state.electronic_state) - 1.0) < 2.0e-11


def test_verlet_cf4_converges_to_full_driven_reference_with_second_order():
    parameters, initial = _control()
    field = UniformElectricField2D.from_millivolt_per_angstrom(
        5.0,
        -0.27,
        ax_angstrom=3.0,
        ay_angstrom=3.2,
    )
    final_time = 0.4
    reference = integrate_coupled_field_dop853(
        initial,
        parameters,
        field,
        final_time_fs=final_time,
        rtol=1.0e-11,
        atol=1.0e-13,
        max_step_fs=0.005,
    )
    assert reference.success

    combined_errors = []
    for dt in (0.1, 0.05, 0.025):
        result = integrate_coupled_field_verlet(
            initial,
            parameters,
            field,
            dt_fs=dt,
            steps=int(round(final_time / dt)),
            electronic_method="cfm4_lanczos",
            krylov_dimension=6,
        )
        electronic_error = _phase_aligned_state_error(
            reference.state.electronic_state,
            result.state.electronic_state,
        )
        lattice_error = _lattice_distance(
            reference.state.lattice,
            result.state.lattice,
        )
        combined_errors.append(electronic_error + lattice_error)

    # Velocity-Verlet controls the global order: each halving should approach
    # the expected factor-of-four reduction without overfitting roundoff.
    assert combined_errors[1] < 0.4 * combined_errors[0]
    assert combined_errors[2] < 0.4 * combined_errors[1]
