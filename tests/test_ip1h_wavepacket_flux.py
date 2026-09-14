"""Tests for IP1h boundary-flux wavepacket diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from holstein_peierls.dynamics.wavepacket_flux import (
    boundary_chain_summary,
    integrated_outward_energy_eV,
    normalized_flux_delay,
    outward_boundary_flux_series,
)


def test_boundary_flux_uses_true_boundary_not_half_space_sum() -> None:
    s = np.arange(-5, 6, dtype=np.int64)
    flux = np.zeros((3, s.size), dtype=np.float64)
    flux[:, s == -2] = -2.0
    flux[:, s == -3] = -100.0
    flux[:, s == 2] = 3.0
    back = outward_boundary_flux_series(flux, s, 2, side="backward")
    front = outward_boundary_flux_series(flux, s, 2, side="forward")
    assert np.all(back == 2.0)
    assert np.all(front == 3.0)


def test_integrated_outward_energy_clips_inward_segments_when_requested() -> None:
    times = np.array([0.0, 1.0, 2.0, 3.0])
    flux = np.array([1.0, -1.0, 1.0, 1.0])
    positive = integrated_outward_energy_eV(times, flux, positive_only=True)
    net = integrated_outward_energy_eV(times, flux, positive_only=False)
    assert positive == pytest.approx(np.trapezoid(np.clip(flux, 0.0, None), times))
    assert net == pytest.approx(np.trapezoid(flux, times))
    assert positive > net


def test_normalized_delay_recovers_known_shifted_pulse() -> None:
    times = np.arange(0.0, 3000.0 + 2.0, 2.0)
    center = 800.0
    width = 120.0
    lag = 600.0
    upstream = np.exp(-0.5 * ((times - center) / width) ** 2)
    downstream = 0.7 * np.exp(-0.5 * ((times - (center + lag)) / width) ** 2)
    estimate = normalized_flux_delay(
        times,
        upstream,
        downstream,
        lag_min_fs=300.0,
        lag_max_fs=900.0,
    )
    assert estimate.lag_fs == pytest.approx(lag, abs=4.0)
    assert estimate.speed_sites_per_ps == pytest.approx(1000.0 / lag, rel=0.01)
    assert estimate.correlation > 0.999


def test_boundary_chain_summary_recovers_propagating_packet() -> None:
    times = np.arange(0.0, 4000.0 + 2.0, 2.0)
    s = np.arange(-10, 11, dtype=np.int64)
    flux = np.zeros((times.size, s.size), dtype=np.float64)
    lag_per_site = 600.0
    for distance in range(2, 7):
        pulse = np.exp(-0.5 * ((times - (300.0 + distance * lag_per_site)) / 120.0) ** 2)
        flux[:, s == -distance] = -pulse[:, None]
    summary = boundary_chain_summary(
        times,
        flux,
        s,
        side="backward",
        distances_sites=(2, 3, 4, 5, 6),
        lag_min_fs=300.0,
        lag_max_fs=900.0,
    )
    assert summary["median_speed_sites_per_ps"] == pytest.approx(1000.0 / lag_per_site, rel=0.02)
    assert summary["minimum_correlation"] > 0.99
    assert all(item["positive_outward_energy_eV"] > 0.0 for item in summary["boundaries"])
