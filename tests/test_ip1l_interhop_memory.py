import numpy as np
import pytest

from holstein_peierls.dynamics.interhop_memory import (
    empirical_percentile,
    flux_window_metrics,
    incremental_pulse_gate,
    memory_gate,
    nonoverlapping_flux_bins,
    sliding_flux_windows,
)


def test_flux_window_metrics_integrates_positive_and_signed_flux():
    times = np.arange(0.0, 12.0, 2.0)
    flux = np.array([0.0, 1.0, -1.0, 2.0, -2.0, 0.0])
    record = flux_window_metrics(times, flux, 0.0, 10.0)
    expected_positive = np.trapezoid(np.clip(flux, 0.0, None), times)
    expected_net = np.trapezoid(flux, times)
    assert record.positive_energy_eV == pytest.approx(expected_positive)
    assert record.net_energy_eV == pytest.approx(expected_net)
    assert record.peak_outward_flux_eV_per_fs == pytest.approx(2.0)


def test_sliding_and_nonoverlapping_windows_have_expected_centers():
    times = np.arange(0.0, 1000.0 + 20.0, 20.0)
    flux = np.ones_like(times)
    sliding = sliding_flux_windows(
        times, flux, 0.0, 600.0, width_fs=100.0, step_fs=20.0
    )
    assert len(sliding) == 26
    assert sliding[0].center_fs == pytest.approx(50.0)
    assert sliding[-1].center_fs == pytest.approx(550.0)

    bins = nonoverlapping_flux_bins(times, flux, 0.0, 450.0, width_fs=100.0)
    assert [record.center_fs for record in bins] == pytest.approx([50.0, 150.0, 250.0, 350.0])


def test_empirical_percentile_is_inclusive():
    background = np.array([1.0, 2.0, 3.0, 4.0])
    assert empirical_percentile(0.5, background) == pytest.approx(0.0)
    assert empirical_percentile(2.0, background) == pytest.approx(50.0)
    assert empirical_percentile(4.0, background) == pytest.approx(100.0)


def test_memory_gate_requires_joint_late_energy_and_peak_excess():
    records = [
        {"center_fs": 900.0, "positive_energy_percentile": 100.0, "peak_flux_percentile": 100.0},
        {"center_fs": 1100.0, "positive_energy_percentile": 100.0, "peak_flux_percentile": 90.0},
        {"center_fs": 1200.0, "positive_energy_percentile": 96.0, "peak_flux_percentile": 99.0},
        {"center_fs": 1300.0, "positive_energy_percentile": 80.0, "peak_flux_percentile": 100.0},
    ]
    gate = memory_gate(records, first_event_time_fs=0.0)
    assert gate["late_bin_count"] == 3
    assert gate["late_passing_bin_count"] == 1
    assert gate["latest_passing_center_fs"] == pytest.approx(1200.0)
    assert gate["late_memory_gate_pass"]


def test_incremental_pulse_gate_keeps_packet_peak_and_positive_energy_requirements():
    assert incremental_pulse_gate(
        packet_qualified=True,
        peak_percentile=100.0,
        positive_excess_energy_eV=1.0e-4,
    )
    assert not incremental_pulse_gate(
        packet_qualified=False,
        peak_percentile=100.0,
        positive_excess_energy_eV=1.0e-4,
    )
    assert not incremental_pulse_gate(
        packet_qualified=True,
        peak_percentile=90.0,
        positive_excess_energy_eV=1.0e-4,
    )
    assert not incremental_pulse_gate(
        packet_qualified=True,
        peak_percentile=100.0,
        positive_excess_energy_eV=0.0,
    )
