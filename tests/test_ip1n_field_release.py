import numpy as np
import pytest

from holstein_peierls.dynamics.driven import field_power
from holstein_peierls.dynamics.field import UniformElectricField2D, build_sparse_field_hamiltonian
from holstein_peierls.dynamics.field_release import (
    HeldPeierlsPhase2D,
    sliding_amplitude_windows,
    sustained_pattern_gate,
    trailing_amplitude_series,
    trailing_outward_matrix,
)
from holstein_peierls.lattice import LatticeState
from holstein_peierls.parameters import StaticPolaronParameters


def test_held_phase_matches_driving_hamiltonian_at_switch_and_has_zero_rate():
    parameters = StaticPolaronParameters(nx=3, ny=3)
    lattice = LatticeState.zeros(3, 3)
    field = UniformElectricField2D.from_millivolt_per_angstrom(10.0)
    switch = 2874.0
    held = HeldPeierlsPhase2D.from_driving_field(field, switch)

    assert held.phases(switch) == pytest.approx(field.phases(switch))
    assert held.phases(switch + 1000.0) == pytest.approx(field.phases(switch))
    assert held.phase_rates_per_fs() == pytest.approx((0.0, 0.0))
    assert held.physical_field_is_zero

    h_drive = build_sparse_field_hamiltonian(lattice, parameters, field, switch).toarray()
    h_held = build_sparse_field_hamiltonian(lattice, parameters, held, switch + 1000.0).toarray()
    assert np.max(np.abs(h_drive - h_held)) < 1.0e-14


def test_held_phase_field_power_is_zero():
    parameters = StaticPolaronParameters(nx=3, ny=3)
    lattice = LatticeState.zeros(3, 3)
    field = UniformElectricField2D.from_millivolt_per_angstrom(10.0)
    held = HeldPeierlsPhase2D.from_driving_field(field, 100.0)
    psi = np.ones(parameters.n_sites, dtype=np.complex128)
    psi /= np.linalg.norm(psi)
    assert field_power(lattice, parameters, psi, held, 500.0) == pytest.approx(0.0)


def test_trailing_amplitude_handles_alternating_boundary_signs():
    s = np.array([-2, -1, 0, 1, 2])
    profiles = np.array(
        [
            [-2.0, 1.0, 0.0, 0.0, 0.0],
            [-4.0, 2.0, 0.0, 0.0, 0.0],
        ]
    )
    matrix = trailing_outward_matrix(profiles, s, distances_sites=(1, 2))
    # backward outward current is -j(-d)
    assert matrix.tolist() == pytest.approx([[-1.0, 2.0], [-2.0, 4.0]])
    amplitude = trailing_amplitude_series(matrix)
    assert amplitude[0] == pytest.approx(np.sqrt(2.5))
    assert amplitude[1] == pytest.approx(np.sqrt(10.0))


def test_sliding_amplitude_windows_and_sustained_gate():
    times = np.arange(0.0, 2000.0 + 20.0, 20.0)
    amplitude = np.ones_like(times)
    windows = sliding_amplitude_windows(
        times,
        amplitude,
        0.0,
        400.0,
        width_fs=100.0,
        step_fs=20.0,
    )
    assert len(windows) == 16
    assert windows[0].rms_amplitude_eV_per_fs == pytest.approx(1.0)

    records = [
        {"center_fs": 1100.0, "rms_percentile": 100.0, "peak_percentile": 100.0},
        {"center_fs": 1300.0, "rms_percentile": 100.0, "peak_percentile": 100.0},
        {"center_fs": 1600.0, "rms_percentile": 100.0, "peak_percentile": 100.0},
    ]
    gate = sustained_pattern_gate(records, switch_time_fs=0.0)
    assert gate["late_passing_bin_count"] == 3
    assert gate["latest_passing_center_fs"] == pytest.approx(1600.0)
    assert gate["sustained_pattern_memory"]
