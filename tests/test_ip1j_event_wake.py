import numpy as np
import pytest

from holstein_peierls.dynamics.event_wake import (
    baseline_subtracted_profiles,
    event_wake_window,
    positive_energy_directionality,
)


def test_event_wake_window_truncates_before_next_event():
    times = np.arange(0.0, 5000.0 + 2.0, 2.0)
    window = event_wake_window(
        times,
        2496.0,
        max_post_fs=1500.0,
        next_event_time_fs=4108.0,
        next_event_guard_fs=50.0,
    )
    assert window.baseline_start_fs == pytest.approx(1996.0)
    assert window.baseline_end_fs == pytest.approx(2396.0)
    assert window.post_end_fs == pytest.approx(3996.0)
    assert times[window.post_mask][-1] == pytest.approx(3996.0)


def test_event_wake_window_uses_max_post_when_next_event_is_later():
    times = np.arange(0.0, 5000.0 + 2.0, 2.0)
    window = event_wake_window(times, 2826.0, max_post_fs=1500.0)
    assert window.post_end_fs == pytest.approx(4326.0)
    assert times[window.post_mask][0] == pytest.approx(2826.0)
    assert times[window.post_mask][-1] == pytest.approx(4326.0)


def test_baseline_subtraction_removes_pre_event_offset_columnwise():
    times = np.arange(0.0, 10.0, 1.0)
    mask = times < 4.0
    profiles = np.column_stack((2.0 + times, -3.0 + 0.5 * times))
    shifted = baseline_subtracted_profiles(profiles, mask)
    assert np.allclose(np.mean(shifted[mask], axis=0), 0.0, atol=1.0e-14)
    assert shifted.shape == profiles.shape


def test_positive_energy_directionality_sign_and_limits():
    assert positive_energy_directionality(3.0, 1.0) == pytest.approx(0.5)
    assert positive_energy_directionality(1.0, 3.0) == pytest.approx(-0.5)
    assert positive_energy_directionality(0.0, 0.0) == 0.0
    with pytest.raises(ValueError):
        positive_energy_directionality(-1.0, 2.0)
