import numpy as np
import pytest

from holstein_peierls.dynamics.event_background import (
    background_separated,
    complete_real_event_times,
    empirical_percentile,
    matched_window,
    packet_gate,
    pseudo_event_times,
    window_contains_event,
)


def test_matched_window_uses_complete_baseline_and_post_interval():
    times = np.arange(0.0, 5000.0 + 2.0, 2.0)
    window = matched_window(times, 2500.0)
    assert window.baseline_start_fs == pytest.approx(1800.0)
    assert window.baseline_end_fs == pytest.approx(2400.0)
    assert window.post_end_fs == pytest.approx(3300.0)
    assert times[window.baseline_mask][0] == pytest.approx(1800.0)
    assert times[window.post_mask][-1] == pytest.approx(3300.0)


def test_complete_real_events_reject_other_event_starts_inside_window():
    times = np.arange(0.0, 5000.0 + 2.0, 2.0)
    events = np.array([600.0, 1600.0, 2200.0, 2496.0, 4108.0])
    selected = complete_real_event_times(times, events)
    # 600 fs lacks the required 700 fs prehistory. The 1600/2200/2496
    # cluster contaminates each local [-700,+800] fs window with another
    # persistent-event start. 4108 fs remains complete and isolated.
    assert selected.tolist() == pytest.approx([4108.0])


def test_window_contains_event_can_ignore_the_center_event():
    events = np.array([2496.0, 4108.0])
    assert window_contains_event(2496.0, events, exclude_center_event=False)
    assert not window_contains_event(2496.0, events, exclude_center_event=True)


def test_pseudo_event_times_exclude_windows_containing_real_events():
    times = np.arange(0.0, 5000.0 + 2.0, 2.0)
    events = np.array([2496.0, 4108.0])
    pseudo = pseudo_event_times(times, events, cadence_fs=100.0)
    assert pseudo.size >= 5
    assert np.all(pseudo >= 700.0)
    assert np.all(pseudo <= 4200.0)
    for center in pseudo:
        assert not window_contains_event(center, events)


def test_empirical_percentile_is_inclusive_and_deterministic():
    background = np.array([1.0, 2.0, 3.0, 4.0])
    assert empirical_percentile(0.5, background) == pytest.approx(0.0)
    assert empirical_percentile(2.0, background) == pytest.approx(50.0)
    assert empirical_percentile(5.0, background) == pytest.approx(100.0)


def test_packet_gate_rejects_search_boundary_and_superharmonic_speed():
    assert packet_gate(
        lag_fs=584.0,
        correlation=0.99,
        speed_sites_per_ps=1.71,
        vmax_sites_per_ps=1.84,
    )
    assert not packet_gate(
        lag_fs=300.0,
        correlation=0.99,
        speed_sites_per_ps=1.71,
        vmax_sites_per_ps=1.84,
    )
    assert not packet_gate(
        lag_fs=584.0,
        correlation=0.99,
        speed_sites_per_ps=2.2,
        vmax_sites_per_ps=1.84,
    )


def test_background_separation_requires_both_amplitude_percentiles():
    assert background_separated(
        packet_qualified=True,
        energy_percentile=100.0,
        peak_percentile=98.0,
    )
    assert not background_separated(
        packet_qualified=True,
        energy_percentile=95.0,
        peak_percentile=100.0,
    )
    assert not background_separated(
        packet_qualified=False,
        energy_percentile=100.0,
        peak_percentile=100.0,
    )
