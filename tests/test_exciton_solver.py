import numpy as np
import pytest

from holstein_peierls.exciton import (
    ExcitonParameters,
    exciton_observables,
    initial_wavefunction,
    solve_exciton_ground_state,
)
from holstein_peierls.lattice import LatticeState


def test_noninteracting_energy_is_sum_of_one_particle_band_minima() -> None:
    parameters = ExcitonParameters(
        nx=3,
        ny=3,
        exciton_position=5,
        electron_j0x=0.10,
        electron_j0y=0.02,
        hole_j0x=0.07,
        hole_j0y=0.03,
        electron_alpha_intra=0.0,
        electron_alpha_interx=0.0,
        electron_alpha_intery=0.0,
        hole_alpha_intra=0.0,
        hole_alpha_interx=0.0,
        hole_alpha_intery=0.0,
    )
    state = solve_exciton_ground_state(LatticeState.zeros(3, 3), parameters)
    expected = -2.0 * (0.10 + 0.02 + 0.07 + 0.03)
    assert state.energy == pytest.approx(expected, abs=1.0e-11)


def test_reduced_density_matrices_each_have_trace_one() -> None:
    parameters = ExcitonParameters(
        nx=3,
        ny=3,
        exciton_position=5,
        onsite_attraction=0.35,
    )
    state = solve_exciton_ground_state(LatticeState.zeros(3, 3), parameters)
    assert np.trace(state.electron_density_matrix) == pytest.approx(1.0, abs=1.0e-12)
    assert np.trace(state.hole_density_matrix) == pytest.approx(1.0, abs=1.0e-12)
    assert np.sum(state.electron_density) == pytest.approx(1.0, abs=1.0e-12)
    assert np.sum(state.hole_density) == pytest.approx(1.0, abs=1.0e-12)


def test_strong_atomic_onsite_attraction_produces_frenkel_pair() -> None:
    parameters = ExcitonParameters(
        nx=2,
        ny=2,
        exciton_position=1,
        electron_j0x=0.0,
        electron_j0y=0.0,
        hole_j0x=0.0,
        hole_j0y=0.0,
        electron_alpha_intra=0.0,
        electron_alpha_interx=0.0,
        electron_alpha_intery=0.0,
        hole_alpha_intra=0.0,
        hole_alpha_interx=0.0,
        hole_alpha_intery=0.0,
        onsite_attraction=0.7,
    )
    state = solve_exciton_ground_state(LatticeState.zeros(2, 2), parameters)
    assert state.energy == pytest.approx(-0.7, abs=1.0e-12)
    assert state.onsite_probability == pytest.approx(1.0, abs=1.0e-12)


def test_atomic_nearest_neighbor_attraction_produces_ct_pair() -> None:
    parameters = ExcitonParameters(
        nx=4,
        ny=4,
        exciton_position=6,
        electron_j0x=0.0,
        electron_j0y=0.0,
        hole_j0x=0.0,
        hole_j0y=0.0,
        electron_alpha_intra=0.0,
        electron_alpha_interx=0.0,
        electron_alpha_intery=0.0,
        hole_alpha_intra=0.0,
        hole_alpha_interx=0.0,
        hole_alpha_intery=0.0,
        nearest_neighbor_attraction=0.4,
    )
    seed = initial_wavefunction(parameters, "ct_x")
    state = solve_exciton_ground_state(
        LatticeState.zeros(4, 4), parameters, initial_state=seed
    )
    observables = exciton_observables(state, parameters)
    assert state.energy == pytest.approx(-0.4, abs=1.0e-12)
    assert state.onsite_probability == pytest.approx(0.0, abs=1.0e-12)
    assert observables.mean_separation_sites == pytest.approx(1.0, abs=1.0e-12)


def test_equal_carrier_control_has_equal_electron_and_hole_densities() -> None:
    parameters = ExcitonParameters(
        nx=3, ny=3, exciton_position=5, onsite_attraction=0.525
    )
    state = solve_exciton_ground_state(LatticeState.zeros(3, 3), parameters)
    assert np.allclose(state.electron_density, state.hole_density, rtol=0.0, atol=1.0e-11)


def test_seed_topologies_place_distinguishable_carriers_as_requested() -> None:
    parameters = ExcitonParameters(nx=6, ny=6, exciton_position=15)
    center = parameters.exciton_index
    cy, cx = divmod(center, parameters.nx)
    expected_holes = {
        "frenkel": center,
        "ct_x": cy * parameters.nx + (cx + 1) % parameters.nx,
        "ct_y": ((cy + 1) % parameters.ny) * parameters.nx + cx,
        "diagonal": ((cy + 1) % parameters.ny) * parameters.nx
        + (cx + 1) % parameters.nx,
        "separated": ((cy + parameters.ny // 2) % parameters.ny) * parameters.nx
        + (cx + parameters.nx // 2) % parameters.nx,
    }
    for mode, hole_site in expected_holes.items():
        psi = initial_wavefunction(parameters, mode)
        assert psi[center, hole_site] == 1.0
        assert np.count_nonzero(psi) == 1
