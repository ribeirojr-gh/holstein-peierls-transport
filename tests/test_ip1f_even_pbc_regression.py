"""Regression tests for even-cell IP1f event-aligned PBC coordinates."""

from __future__ import annotations

import numpy as np

from holstein_peierls.dynamics.phonon_wake import (
    event_aligned_coordinates,
    longitudinal_profile,
)


def test_even_cell_all_hop_directions_keep_exact_periodic_profile_length() -> None:
    size = 40
    source_x = source_y = 20
    source = source_y * size + source_x
    targets = (
        source + 1,
        source - 1,
        source + size,
        source - size,
    )
    values = np.ones((size, size), dtype=np.float64)

    for target in targets:
        coords = event_aligned_coordinates(source, target, size, size)
        s_axis, profile = longitudinal_profile(values, coords)
        assert s_axis.shape == (size,)
        assert profile.shape == (size,)
        assert int(np.min(coords.s_sites)) == -(size // 2)
        assert int(np.max(coords.s_sites)) == size // 2 - 1
        assert int(np.min(coords.p_sites)) == -(size // 2)
        assert int(np.max(coords.p_sites)) == size // 2 - 1
        assert float(np.sum(profile)) == float(np.sum(values))
