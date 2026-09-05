from __future__ import annotations

import numpy as np

from holstein_peierls.dynamics.ehrenfest import LatticeVelocity
from holstein_peierls.dynamics.langevin import LangevinBath
from holstein_peierls.dynamics.pair_coupled import (
    PairCoupledState,
    PairLatticeMasses,
    pair_dynamic_total_energy,
)
from holstein_peierls.dynamics.pair_decoherence import apply_pair_instantaneous_decoherence
from holstein_peierls.dynamics.pair_frozen import (
    bipolaron_exchange_symmetry_error,
    bipolaron_one_body_density_matrix,
    exciton_one_body_density_matrices,
)
from holstein_peierls.dynamics.pair_thermal import project_pair_zero_modes
from holstein_peierls.dynamics.pair_thermal_decoherence import (
    apply_idc_to_pair_state,
    decoherence_interval_steps,
    integrate_pair_coupled_baoab_idc,
)
from holstein_peierls.exciton.parameters import ExcitonParameters
from holstein_peierls.exciton.solver import solve_exciton_ground_state
from holstein_peierls.lattice import LatticeState
from holstein_peierls.two_particle.parameters import BipolaronParameters
from holstein_peierls.two_particle.peierls import solve_holstein_peierls_ground_state


def _lattice() -> LatticeState:
    y, x = np.indices((3, 3), dtype=np.float64)
    return LatticeState(
        6.0e-3 * np.cos(0.7 * x + 0.3 * y),
        2.0e-3 * np.sin(0.8 * x - 0.2 * y),
        2.5e-3 * np.cos(0.4 * x + 0.6 * y),
    )


def _bp() -> BipolaronParameters:
    return BipolaronParameters(nx=3, ny=3, pair_position=5, hubbard_u=0.22, nearest_neighbor_v=0.04)


def _ex() -> ExcitonParameters:
    return ExcitonParameters(
        nx=3, ny=3, exciton_position=5,
        electron_j0x=0.100, electron_j0y=0.015,
        hole_j0x=0.082, hole_j0y=0.021,
        electron_alpha_intra=3.0, hole_alpha_intra=2.4,
        electron_alpha_interx=0.4, electron_alpha_intery=0.4,
        hole_alpha_interx=0.31, hole_alpha_intery=0.28,
        onsite_attraction=0.30, nearest_neighbor_attraction=0.05,
    )


def _state(sector: str):
    lattice = _lattice()
    if sector == "bipolaron":
        parameters = _bp()
        ground = solve_holstein_peierls_ground_state(lattice, parameters)
    else:
        parameters = _ex()
        ground = solve_exciton_ground_state(lattice, parameters)
    raw = PairCoupledState(
        lattice,
        LatticeVelocity.zeros(3, 3),
        np.asarray(ground.wavefunction, dtype=np.complex128).ravel(order="C"),
    )
    return project_pair_zero_modes(raw), parameters


def test_bipolaron_idc_stays_in_symmetric_singlet_sector() -> None:
    state, parameters = _state("bipolaron")
    event = apply_pair_instantaneous_decoherence(
        state.lattice, "bipolaron", parameters, state.electronic_state,
        300.0, "bm", np.random.default_rng(11),
    )
    assert event.physical_sector_dimension == 45
    assert bipolaron_exchange_symmetry_error(event.electronic_state, 9) < 2.0e-13
    gamma = bipolaron_one_body_density_matrix(event.electronic_state, 9)
    np.testing.assert_allclose(np.trace(gamma), 2.0, rtol=0.0, atol=2.0e-13)
    assert abs(np.linalg.norm(event.electronic_state) - 1.0) < 2.0e-13


def test_exciton_idc_uses_full_distinguishable_sector() -> None:
    state, parameters = _state("exciton")
    event = apply_pair_instantaneous_decoherence(
        state.lattice, "exciton", parameters, state.electronic_state,
        300.0, "ma", np.random.default_rng(12),
    )
    assert event.physical_sector_dimension == 81
    gamma_e, gamma_h = exciton_one_body_density_matrices(event.electronic_state, 9)
    np.testing.assert_allclose(np.trace(gamma_e), 1.0, rtol=0.0, atol=2.0e-13)
    np.testing.assert_allclose(np.trace(gamma_h), 1.0, rtol=0.0, atol=2.0e-13)


def test_pair_idc_seeded_collapse_is_reproducible() -> None:
    state, parameters = _state("exciton")
    first = apply_pair_instantaneous_decoherence(
        state.lattice, "exciton", parameters, state.electronic_state,
        300.0, "dp", np.random.default_rng(77),
    )
    second = apply_pair_instantaneous_decoherence(
        state.lattice, "exciton", parameters, state.electronic_state,
        300.0, "dp", np.random.default_rng(77),
    )
    assert first.selected_state_index == second.selected_state_index
    np.testing.assert_allclose(first.electronic_state, second.electronic_state, rtol=0.0, atol=0.0)


def test_pair_idc_energy_exchange_equals_full_matter_jump_at_fixed_lattice() -> None:
    masses = PairLatticeMasses(75000.0, 150000.0)
    for sector in ("bipolaron", "exciton"):
        state, parameters = _state(sector)
        before = pair_dynamic_total_energy(state, sector, parameters, masses).total
        collapsed, event = apply_idc_to_pair_state(
            state, sector, parameters, 300.0, "bm", np.random.default_rng(91)
        )
        after = pair_dynamic_total_energy(collapsed, sector, parameters, masses).total
        np.testing.assert_allclose(
            after - before,
            event.electronic_environment_exchange_eV,
            rtol=0.0,
            atol=3.0e-12,
        )


def test_decoherence_interval_requires_integer_multiple_of_dt() -> None:
    assert decoherence_interval_steps(100.0, 0.2) == 500
    try:
        decoherence_interval_steps(1.0, 0.3)
    except ValueError:
        pass
    else:
        raise AssertionError("noncommensurate interval should fail")


def test_short_pair_idc_trajectory_is_reproducible_and_counts_events() -> None:
    state, parameters = _state("bipolaron")
    masses = PairLatticeMasses(75000.0, 150000.0)
    bath = LangevinBath(300.0, 0.01, 0.01)
    kwargs = dict(
        initial_state=state,
        sector="bipolaron",
        parameters=parameters,
        masses=masses,
        bath=bath,
        dt_fs=0.2,
        steps=10,
        decoherence_interval_fs=1.0,
        scheme="bm",
        zero_mode_policy="project",
        krylov_dimension=8,
    )
    first = integrate_pair_coupled_baoab_idc(
        **kwargs,
        lattice_rng=np.random.default_rng(101),
        decoherence_rng=np.random.default_rng(202),
    )
    second = integrate_pair_coupled_baoab_idc(
        **kwargs,
        lattice_rng=np.random.default_rng(101),
        decoherence_rng=np.random.default_rng(202),
    )
    assert first.idc_events == 2
    np.testing.assert_allclose(first.state.electronic_state, second.state.electronic_state, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(first.state.lattice.u, second.state.lattice.u, rtol=0.0, atol=0.0)
    assert first.lattice_bath_heat_eV == second.lattice_bath_heat_eV
    assert first.electronic_environment_exchange_eV == second.electronic_environment_exchange_eV


def test_bm_and_ma_probabilities_are_normalized_in_pair_events() -> None:
    state, parameters = _state("bipolaron")
    for scheme in ("bm", "ma"):
        event = apply_pair_instantaneous_decoherence(
            state.lattice, "bipolaron", parameters, state.electronic_state,
            300.0, scheme, np.random.default_rng(303),
        )
        np.testing.assert_allclose(np.sum(event.collapse_probabilities), 1.0, rtol=0.0, atol=2.0e-15)
        assert np.all(event.collapse_probabilities >= 0.0)
