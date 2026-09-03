"""Validation gates for D4b coupled finite-temperature Ehrenfest dynamics."""

from __future__ import annotations

import numpy as np

from holstein_peierls.dynamics.coupled import CoupledEhrenfestState, coupled_verlet_step
from holstein_peierls.dynamics.driven import coupled_field_verlet_step
from holstein_peierls.dynamics.ehrenfest import LatticeVelocity
from holstein_peierls.dynamics.field import UniformElectricField2D
from holstein_peierls.dynamics.langevin import LangevinBath
from holstein_peierls.dynamics.thermal import (
    coupled_baoab_step,
    integrate_coupled_baoab,
    project_inter_molecular_zero_modes,
    thermostatted_kinetic_degrees_of_freedom,
    zero_mode_means,
)
from holstein_peierls.lattice import LatticeState
from holstein_peierls.parameters import StaticPolaronParameters


def _control() -> StaticPolaronParameters:
    return StaticPolaronParameters(nx=4, ny=4, polaron_position=1)


def _state(seed: int = 123) -> CoupledEhrenfestState:
    rng = np.random.default_rng(seed)
    shape = (4, 4)
    lattice = LatticeState(
        2.0e-3 * rng.normal(size=shape),
        2.0e-3 * rng.normal(size=shape),
        2.0e-3 * rng.normal(size=shape),
    )
    velocity = LatticeVelocity(
        2.0e-4 * rng.normal(size=shape),
        2.0e-4 * rng.normal(size=shape),
        2.0e-4 * rng.normal(size=shape),
    )
    psi = rng.normal(size=16) + 1.0j * rng.normal(size=16)
    psi = np.asarray(psi / np.linalg.norm(psi), dtype=np.complex128)
    return CoupledEhrenfestState(lattice, velocity, psi)


def _assert_states_equal(first: CoupledEhrenfestState, second: CoupledEhrenfestState) -> None:
    for a, b in (
        (first.lattice.u, second.lattice.u),
        (first.lattice.vx, second.lattice.vx),
        (first.lattice.vy, second.lattice.vy),
        (first.velocity.u, second.velocity.u),
        (first.velocity.vx, second.velocity.vx),
        (first.velocity.vy, second.velocity.vy),
        (first.electronic_state, second.electronic_state),
    ):
        assert np.array_equal(a, b)


def _bond_differences(values: np.ndarray, axis: int) -> np.ndarray:
    return np.roll(values, -1, axis=axis) - values


def test_frictionless_retain_step_reduces_exactly_to_d2_and_does_not_consume_rng():
    parameters = _control()
    initial = _state()
    dt = 0.2
    reference, ref_eval, ref_apply = coupled_verlet_step(initial, parameters, dt)

    rng = np.random.default_rng(2026)
    expected_next = np.random.default_rng(2026).standard_normal()
    result = coupled_baoab_step(
        initial,
        parameters,
        LangevinBath(temperature_K=300.0, gamma_u_per_fs=0.0, gamma_v_per_fs=0.0),
        rng,
        0.0,
        dt,
        zero_mode_policy="retain",
    )

    _assert_states_equal(result.state, reference)
    assert result.bath_heat_eV == 0.0
    assert result.field_work_eV == 0.0
    assert result.hamiltonian_evaluations == ref_eval
    assert result.hamiltonian_applications == ref_apply
    assert rng.standard_normal() == expected_next


def test_frictionless_retain_step_reduces_exactly_to_d3():
    parameters = _control()
    initial = _state(456)
    field = UniformElectricField2D.from_millivolt_per_angstrom(2.0)
    time = 0.37
    dt = 0.2
    reference, work, ref_eval, ref_apply = coupled_field_verlet_step(
        initial,
        parameters,
        field,
        time,
        dt,
    )
    result = coupled_baoab_step(
        initial,
        parameters,
        LangevinBath(temperature_K=300.0, gamma_u_per_fs=0.0, gamma_v_per_fs=0.0),
        np.random.default_rng(99),
        time,
        dt,
        field=field,
        zero_mode_policy="retain",
    )

    _assert_states_equal(result.state, reference)
    assert result.bath_heat_eV == 0.0
    assert result.field_work_eV == work
    assert result.hamiltonian_evaluations == ref_eval
    assert result.hamiltonian_applications == ref_apply


def test_finite_temperature_coupled_step_is_seed_reproducible():
    parameters = _control()
    initial = _state(789)
    bath = LangevinBath(temperature_K=300.0, gamma_u_per_fs=0.01, gamma_v_per_fs=0.015)

    first = coupled_baoab_step(
        initial,
        parameters,
        bath,
        np.random.default_rng(31415),
        0.0,
        0.2,
    )
    second = coupled_baoab_step(
        initial,
        parameters,
        bath,
        np.random.default_rng(31415),
        0.0,
        0.2,
    )

    _assert_states_equal(first.state, second.state)
    assert first.bath_heat_eV == second.bath_heat_eV
    assert first.field_work_eV == second.field_work_eV


def test_cf4_coupled_bath_preserves_electronic_norm_without_renormalization():
    parameters = _control()
    initial = _state(2468)
    initial_norm = float(np.linalg.norm(initial.electronic_state))
    result = integrate_coupled_baoab(
        initial,
        parameters,
        LangevinBath(temperature_K=300.0, gamma_u_per_fs=0.01, gamma_v_per_fs=0.01),
        np.random.default_rng(13579),
        dt_fs=0.2,
        steps=50,
        zero_mode_policy="retain",
        electronic_method="cfm4_lanczos",
        krylov_dimension=6,
    )
    final_norm = float(np.linalg.norm(result.state.electronic_state))
    assert abs(final_norm - initial_norm) < 2.0e-12


def test_projected_policy_preserves_collective_constraints_and_reduced_dof():
    parameters = _control()
    initial = project_inter_molecular_zero_modes(_state(1122))
    result = integrate_coupled_baoab(
        initial,
        parameters,
        LangevinBath(temperature_K=300.0, gamma_u_per_fs=0.012, gamma_v_per_fs=0.018),
        np.random.default_rng(778899),
        dt_fs=0.2,
        steps=30,
        zero_mode_policy="project",
    )
    means = zero_mode_means(result.state)
    assert max(abs(value) for value in means.values()) < 2.0e-15
    assert thermostatted_kinetic_degrees_of_freedom(parameters, "retain") == 48
    assert thermostatted_kinetic_degrees_of_freedom(parameters, "project") == 46


def test_retain_and_project_policies_have_same_internal_v_differences_and_electronic_state():
    parameters = _control()
    initial = project_inter_molecular_zero_modes(_state(4455))
    bath = LangevinBath(temperature_K=300.0, gamma_u_per_fs=0.01, gamma_v_per_fs=0.02)

    retained = coupled_baoab_step(
        initial,
        parameters,
        bath,
        np.random.default_rng(54321),
        0.0,
        0.2,
        zero_mode_policy="retain",
    ).state
    projected = coupled_baoab_step(
        initial,
        parameters,
        bath,
        np.random.default_rng(54321),
        0.0,
        0.2,
        zero_mode_policy="project",
    ).state

    assert np.allclose(retained.lattice.u, projected.lattice.u, rtol=0.0, atol=2.0e-15)
    assert np.allclose(
        _bond_differences(retained.lattice.vx, axis=1),
        _bond_differences(projected.lattice.vx, axis=1),
        rtol=0.0,
        atol=2.0e-15,
    )
    assert np.allclose(
        _bond_differences(retained.lattice.vy, axis=0),
        _bond_differences(projected.lattice.vy, axis=0),
        rtol=0.0,
        atol=2.0e-15,
    )
    overlap = np.vdot(retained.electronic_state, projected.electronic_state)
    aligned = projected.electronic_state
    if abs(overlap) > 0.0:
        aligned = aligned * np.conj(overlap) / abs(overlap)
    assert np.linalg.norm(aligned - retained.electronic_state) < 2.0e-12
