"""TP0 tests for finite-temperature field-driven one-polaron IDC dynamics."""

from __future__ import annotations

import numpy as np

from holstein_peierls.dynamics.coupled import CoupledEhrenfestState
from holstein_peierls.dynamics.driven import (
    field_dynamic_total_energy,
    field_electronic_energy_expectation,
)
from holstein_peierls.dynamics.driven_thermal_decoherence import (
    apply_field_instantaneous_decoherence,
    integrate_coupled_baoab_idc_field,
)
from holstein_peierls.dynamics.ehrenfest import LatticeVelocity
from holstein_peierls.dynamics.electronic_decoherence import (
    apply_instantaneous_decoherence,
)
from holstein_peierls.dynamics.field import (
    UniformElectricField2D,
    build_dense_field_hamiltonian,
)
from holstein_peierls.dynamics.langevin import LangevinBath
from holstein_peierls.dynamics.thermal import project_inter_molecular_zero_modes
from holstein_peierls.lattice import LatticeState
from holstein_peierls.parameters import StaticPolaronParameters


def _control_state() -> tuple[CoupledEhrenfestState, StaticPolaronParameters]:
    parameters = StaticPolaronParameters(nx=3, ny=3, polaron_position=5)
    y, x = np.indices((3, 3), dtype=np.float64)
    lattice = LatticeState(
        u=7.0e-3 * np.cos(0.67 * x + 0.41 * y),
        vx=3.0e-3 * np.sin(0.83 * x - 0.29 * y),
        vy=2.0e-3 * np.cos(0.37 * x + 0.71 * y),
    )
    zero = UniformElectricField2D()
    hamiltonian = build_dense_field_hamiltonian(lattice, parameters, zero, 0.0)
    _, vectors = np.linalg.eigh(hamiltonian)
    psi = (
        0.91 * vectors[:, 0]
        + (0.21 + 0.13j) * vectors[:, 1]
        + (-0.08 + 0.17j) * vectors[:, 3]
    )
    psi = np.asarray(psi / np.linalg.norm(psi), dtype=np.complex128)
    return (
        CoupledEhrenfestState(
            lattice,
            LatticeVelocity.zeros(3, 3),
            psi,
        ),
        parameters,
    )


def _phase_aligned_error(first: np.ndarray, second: np.ndarray) -> float:
    overlap = np.vdot(first, second)
    if abs(overlap) == 0.0:
        return float(np.linalg.norm(first - second))
    phase = overlap / abs(overlap)
    return float(np.linalg.norm(first - phase.conjugate() * second))


def test_field_idc_zero_field_reproduces_d5_event():
    state, parameters = _control_state()
    zero = UniformElectricField2D()
    old_rng = np.random.default_rng(20260905)
    new_rng = np.random.default_rng(20260905)

    old = apply_instantaneous_decoherence(
        state.lattice,
        parameters,
        state.electronic_state,
        300.0,
        "bm",
        old_rng,
    )
    new = apply_field_instantaneous_decoherence(
        state,
        parameters,
        zero,
        37.0,
        300.0,
        "bm",
        new_rng,
    )

    assert old.selected_state_index == new.selected_state_index
    assert np.allclose(old.adiabatic_energies_eV, new.adiabatic_energies_eV, atol=1e-13)
    assert np.allclose(
        old.adiabatic_populations_before,
        new.adiabatic_populations_before,
        atol=2e-12,
    )
    assert np.allclose(old.collapse_probabilities, new.collapse_probabilities, atol=2e-12)
    assert np.isclose(
        old.electronic_environment_exchange_eV,
        new.electronic_environment_exchange_eV,
        atol=2e-12,
    )
    assert _phase_aligned_error(old.electronic_state, new.electronic_state) < 2e-11


def test_field_idc_energy_jump_matches_direct_expectations():
    state, parameters = _control_state()
    field = UniformElectricField2D.from_millivolt_per_angstrom(
        2.0,
        ax_angstrom=3.0,
        ay_angstrom=3.0,
    )
    time_fs = 53.0
    event = apply_field_instantaneous_decoherence(
        state,
        parameters,
        field,
        time_fs,
        300.0,
        "bm",
        np.random.default_rng(17),
    )

    before = field_electronic_energy_expectation(
        state.lattice,
        parameters,
        state.electronic_state,
        field,
        time_fs,
    )
    after = field_electronic_energy_expectation(
        state.lattice,
        parameters,
        event.electronic_state,
        field,
        time_fs,
    )
    assert np.isclose(event.electronic_energy_before_eV, before, atol=2e-12)
    assert np.isclose(event.electronic_energy_after_eV, after, atol=2e-12)
    assert np.isclose(
        event.electronic_environment_exchange_eV,
        after - before,
        atol=3e-12,
    )


def test_zero_field_tp0_has_zero_work_and_generalized_balance():
    state, parameters = _control_state()
    state = project_inter_molecular_zero_modes(state)
    bath = LangevinBath(300.0, 0.01, 0.01)
    field = UniformElectricField2D()
    dt_fs = 0.2
    steps = 50
    initial = field_dynamic_total_energy(
        state.lattice,
        state.velocity,
        parameters,
        state.electronic_state,
        field,
        0.0,
    ).total

    result = integrate_coupled_baoab_idc_field(
        state,
        parameters,
        bath,
        field,
        np.random.default_rng(101),
        np.random.default_rng(202),
        dt_fs=dt_fs,
        steps=steps,
        decoherence_interval_fs=5.0,
        scheme="bm",
        zero_mode_policy="project",
        krylov_dimension=6,
    )
    final = field_dynamic_total_energy(
        result.state.lattice,
        result.state.velocity,
        parameters,
        result.state.electronic_state,
        field,
        steps * dt_fs,
    ).total
    residual = (
        (final - initial)
        - result.lattice_bath_heat_eV
        - result.electronic_environment_exchange_eV
        - result.field_work_eV
    )

    assert result.decoherence_events == 2
    assert result.field_work_eV == 0.0
    assert abs(residual) < 5e-5
    assert abs(np.linalg.norm(result.state.electronic_state) - 1.0) < 1e-12


def test_finite_field_tp0_closes_full_energy_balance():
    state, parameters = _control_state()
    state = project_inter_molecular_zero_modes(state)
    bath = LangevinBath(300.0, 0.01, 0.01)
    field = UniformElectricField2D.from_millivolt_per_angstrom(
        2.0,
        ax_angstrom=3.0,
        ay_angstrom=3.0,
    )
    dt_fs = 0.2
    steps = 100
    initial = field_dynamic_total_energy(
        state.lattice,
        state.velocity,
        parameters,
        state.electronic_state,
        field,
        0.0,
    ).total

    result = integrate_coupled_baoab_idc_field(
        state,
        parameters,
        bath,
        field,
        np.random.default_rng(303),
        np.random.default_rng(404),
        dt_fs=dt_fs,
        steps=steps,
        decoherence_interval_fs=10.0,
        scheme="bm",
        zero_mode_policy="project",
        krylov_dimension=6,
    )
    final_time = steps * dt_fs
    final = field_dynamic_total_energy(
        result.state.lattice,
        result.state.velocity,
        parameters,
        result.state.electronic_state,
        field,
        final_time,
    ).total
    residual = (
        (final - initial)
        - result.lattice_bath_heat_eV
        - result.electronic_environment_exchange_eV
        - result.field_work_eV
    )

    assert result.decoherence_events == 2
    assert np.isfinite(result.field_work_eV)
    assert abs(residual) < 8e-5
    assert abs(np.linalg.norm(result.state.electronic_state) - 1.0) < 1e-12
