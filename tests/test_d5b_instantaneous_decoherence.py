"""Validation tests for D5b instantaneous decoherence controls."""

from __future__ import annotations

import numpy as np

from holstein_peierls.dynamics.coupled import CoupledEhrenfestState
from holstein_peierls.dynamics.ehrenfest import LatticeVelocity, dynamic_total_energy
from holstein_peierls.dynamics.electronic_decoherence import (
    apply_instantaneous_decoherence,
    idc_collapse_probabilities,
)
from holstein_peierls.dynamics.langevin import BOLTZMANN_EV_PER_K, LangevinBath
from holstein_peierls.dynamics.thermal_decoherence import (
    apply_idc_to_coupled_state,
    decoherence_interval_steps,
    integrate_coupled_baoab_idc,
)
from holstein_peierls.hamiltonian import build_dense_hamiltonian
from holstein_peierls.lattice import LatticeState
from holstein_peierls.parameters import StaticPolaronParameters


def _control() -> tuple[CoupledEhrenfestState, StaticPolaronParameters]:
    parameters = StaticPolaronParameters(nx=4, ny=4, polaron_position=6)
    lattice = LatticeState.zeros(4, 4)
    velocity = LatticeVelocity.zeros(4, 4)
    rng = np.random.default_rng(99)
    psi = rng.normal(size=parameters.n_sites) + 1.0j * rng.normal(size=parameters.n_sites)
    psi = np.asarray(psi / np.linalg.norm(psi), dtype=np.complex128)
    return CoupledEhrenfestState(lattice, velocity, psi), parameters


def test_dp_probabilities_equal_precollapse_populations():
    energies = np.array([-0.4, -0.1, 0.2, 0.7])
    population = np.array([0.1, 0.2, 0.3, 0.4])
    result = idc_collapse_probabilities(energies, population, 300.0, "dp")
    assert np.allclose(result, population, rtol=0.0, atol=2.0e-16)


def test_bm_probabilities_follow_boltzmann_reweighting_and_are_shift_invariant():
    energies = np.array([-0.4, -0.1, 0.2, 0.7])
    population = np.array([0.1, 0.2, 0.3, 0.4])
    temperature = 300.0
    beta = 1.0 / (BOLTZMANN_EV_PER_K * temperature)
    expected = population * np.exp(-beta * (energies - np.min(energies)))
    expected /= np.sum(expected)
    first = idc_collapse_probabilities(energies, population, temperature, "bm")
    second = idc_collapse_probabilities(energies + 9.3, population, temperature, "bm")
    assert np.allclose(first, expected, rtol=0.0, atol=2.0e-15)
    assert np.allclose(first, second, rtol=0.0, atol=2.0e-15)
    assert first[0] > population[0]
    assert first[-1] < population[-1]


def test_ma_probabilities_suppress_only_states_above_mean_energy():
    energies = np.array([-0.3, -0.1, 0.2, 0.8])
    population = np.array([0.2, 0.3, 0.1, 0.4])
    temperature = 300.0
    beta = 1.0 / (BOLTZMANN_EV_PER_K * temperature)
    mean_energy = float(np.dot(population, energies))
    expected = population * np.exp(-beta * np.maximum(energies - mean_energy, 0.0))
    expected /= np.sum(expected)
    result = idc_collapse_probabilities(energies, population, temperature, "ma")
    assert np.allclose(result, expected, rtol=0.0, atol=2.0e-15)


def test_seeded_collapse_is_reproducible_and_normalized():
    state, parameters = _control()
    first = apply_instantaneous_decoherence(
        state.lattice, parameters, state.electronic_state, 300.0, "bm", np.random.default_rng(17)
    )
    second = apply_instantaneous_decoherence(
        state.lattice, parameters, state.electronic_state, 300.0, "bm", np.random.default_rng(17)
    )
    assert first.selected_state_index == second.selected_state_index
    assert np.allclose(first.electronic_state, second.electronic_state, rtol=0.0, atol=0.0)
    assert np.isclose(np.linalg.norm(first.electronic_state), 1.0, rtol=0.0, atol=2.0e-15)
    assert np.isclose(np.sum(first.collapse_probabilities), 1.0, rtol=0.0, atol=2.0e-15)


def test_collapse_energy_exchange_equals_matter_energy_jump_at_fixed_lattice():
    state, parameters = _control()
    energy_before = dynamic_total_energy(
        state.lattice, state.velocity, parameters, state.electronic_state
    ).total
    collapsed, event = apply_idc_to_coupled_state(
        state, parameters, 300.0, "ma", np.random.default_rng(21)
    )
    energy_after = dynamic_total_energy(
        collapsed.lattice, collapsed.velocity, parameters, collapsed.electronic_state
    ).total
    assert np.isclose(
        energy_after - energy_before,
        event.electronic_environment_exchange_eV,
        rtol=0.0,
        atol=2.0e-13,
    )
    assert np.allclose(collapsed.lattice.u, state.lattice.u, rtol=0.0, atol=0.0)
    assert np.allclose(collapsed.velocity.u, state.velocity.u, rtol=0.0, atol=0.0)


def test_decoherence_interval_requires_integer_multiple_of_dt():
    assert decoherence_interval_steps(10.0, 0.2) == 50
    assert decoherence_interval_steps(0.6, 0.2) == 3
    try:
        decoherence_interval_steps(1.0, 0.3)
    except ValueError:
        pass
    else:
        raise AssertionError("non-commensurate decoherence interval should fail")


def test_short_coupled_idc_trajectory_is_seed_reproducible_and_counts_events():
    state, parameters = _control()
    bath = LangevinBath(temperature_K=300.0, gamma_u_per_fs=0.01, gamma_v_per_fs=0.01)

    def run():
        return integrate_coupled_baoab_idc(
            state,
            parameters,
            bath,
            np.random.default_rng(123),
            np.random.default_rng(456),
            dt_fs=0.2,
            steps=20,
            decoherence_interval_fs=1.0,
            scheme="bm",
            zero_mode_policy="project",
            krylov_dimension=6,
        )

    first = run()
    second = run()
    assert first.decoherence_events == 4
    assert first.decoherence_diagonalizations == 4
    assert second.decoherence_events == first.decoherence_events
    assert np.allclose(first.state.electronic_state, second.state.electronic_state, rtol=0.0, atol=0.0)
    assert np.allclose(first.state.lattice.u, second.state.lattice.u, rtol=0.0, atol=0.0)
    assert np.allclose(first.state.velocity.u, second.state.velocity.u, rtol=0.0, atol=0.0)
    assert first.lattice_bath_heat_eV == second.lattice_bath_heat_eV
    assert first.electronic_environment_exchange_eV == second.electronic_environment_exchange_eV
