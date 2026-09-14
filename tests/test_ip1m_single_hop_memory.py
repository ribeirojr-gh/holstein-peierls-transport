import pytest

from holstein_peierls.dynamics.single_hop_memory import (
    continuation_neighbor,
    jointly_background_separated,
    ordered_envelope_fit,
    sustained_memory_gate,
)


def _record(center, energy, peak):
    return {
        "center_fs": float(center),
        "positive_energy_percentile": float(energy),
        "peak_flux_percentile": float(peak),
    }


def test_continuation_neighbor_follows_event_direction_and_wraps():
    assert continuation_neighbor(819, dx_sites=-1, dy_sites=0, nx=40, ny=40) == 818
    assert continuation_neighbor(800, dx_sites=-1, dy_sites=0, nx=40, ny=40) == 839
    assert continuation_neighbor(39, dx_sites=1, dy_sites=0, nx=40, ny=40) == 0
    assert continuation_neighbor(0, dx_sites=0, dy_sites=-1, nx=40, ny=40) == 1560


def test_continuation_neighbor_rejects_non_nearest_event():
    with pytest.raises(ValueError):
        continuation_neighbor(10, dx_sites=1, dy_sites=1, nx=40, ny=40)


def test_joint_background_separation_requires_both_percentiles():
    assert jointly_background_separated(_record(0, 95, 99))
    assert not jointly_background_separated(_record(0, 94.9, 100))
    assert not jointly_background_separated(_record(0, 100, 94.9))


def test_sustained_memory_gate_distinguishes_late_from_sustained():
    t0 = 2800.0
    records = [
        _record(3850, 100, 100),
        _record(3950, 100, 100),
        _record(4050, 100, 100),
        _record(4350, 100, 100),
    ]
    gate = sustained_memory_gate(records, first_event_time_fs=t0)
    assert gate["late_memory_present"]
    assert gate["late_passing_bin_count"] == 4
    assert gate["latest_passing_center_fs"] == pytest.approx(4350.0)
    assert gate["sustained_late_memory"]

    weak = sustained_memory_gate(
        [_record(3850, 100, 100)],
        first_event_time_fs=t0,
    )
    assert weak["late_memory_present"]
    assert not weak["sustained_late_memory"]


def test_ordered_envelope_fit_returns_speed_for_monotonic_arrivals():
    fit = ordered_envelope_fit({1: 100.0, 2: 650.0, 3: 1200.0, 4: 1700.0})
    assert fit["available"]
    assert fit["ordered_primary_arrivals"]
    assert fit["slope_fs_per_site"] == pytest.approx(550.0)
    assert fit["speed_sites_per_ps"] == pytest.approx(1000.0 / 550.0)
    assert fit["r_squared"] == pytest.approx(1.0)


def test_ordered_envelope_fit_rejects_missing_or_unordered_primary_arrivals():
    assert not ordered_envelope_fit({1: 100.0, 2: None, 3: 900.0})["available"]
    result = ordered_envelope_fit({1: 100.0, 2: 90.0, 3: 900.0})
    assert not result["available"]
    assert not result["ordered_primary_arrivals"]
