"""Tests for finite-size numerical validation helpers."""

from __future__ import annotations

import pytest

from holstein_peierls.dynamics.numerical_validation import (
    extensive_energy_balance_passes,
    size_scaled_energy_balance_tolerance_eV,
)


def test_size_scaled_energy_balance_preserves_20x20_gate() -> None:
    assert size_scaled_energy_balance_tolerance_eV(400) == pytest.approx(5.0e-5)


def test_size_scaled_energy_balance_scales_extensively() -> None:
    assert size_scaled_energy_balance_tolerance_eV(1600) == pytest.approx(2.0e-4)
    assert extensive_energy_balance_passes(5.4e-5, 1600)
    assert not extensive_energy_balance_passes(2.1e-4, 1600)


def test_size_scaled_energy_balance_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        size_scaled_energy_balance_tolerance_eV(0)
    with pytest.raises(ValueError):
        size_scaled_energy_balance_tolerance_eV(400, reference_sites=0)
    with pytest.raises(ValueError):
        size_scaled_energy_balance_tolerance_eV(400, reference_tolerance_eV=0.0)
    assert not extensive_energy_balance_passes(float("nan"), 400)
    assert not extensive_energy_balance_passes(-1.0e-8, 400)
