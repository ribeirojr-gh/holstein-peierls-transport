"""Regression tests for D5c decoherence-interval sensitivity aggregation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_module():
    path = Path(__file__).resolve().parents[1] / "experiments" / "d5c_decoherence_interval_sweep.py"
    spec = importlib.util.spec_from_file_location("_d5c_test_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(scheme: str, interval: float, seed: int, shift: float) -> dict:
    return {
        "scheme": scheme,
        "decoherence_interval_fs": interval,
        "lattice_seed": seed,
        "post_burn_samples": 20,
        "temperature_mean_K": 300.0 + shift,
        "pre_heating_coordinate_mean": 0.01 + shift,
        "pre_heating_coordinate_late_mean": 0.02 + shift,
        "pre_heating_coordinate_slope_per_ps": 0.003 + shift,
        "pre_tv_to_canonical_mean": 0.10 + shift,
        "pre_beta_eff_over_beta_bath_mean_finite": 1.1 + shift,
        "pre_ground_manifold_population_mean": 0.80 + shift,
        "canonical_ground_manifold_population_mean": 0.82,
        "expected_post_heating_coordinate_mean": -0.01 + shift,
        "accumulated_electronic_environment_exchange_eV": -0.06 + shift,
        "final_time_fs": 6000.0,
        "maximum_event_abs_generalized_energy_residual_eV": 1.0e-5 + 1.0e-7 * seed,
        "maximum_electronic_norm_error": 1.0e-13,
    }


def test_d5c_aggregate_groups_scheme_and_interval_without_weighted_score():
    module = _load_module()
    records = []
    for scheme in ("bm", "ma"):
        for interval in (50.0, 100.0):
            records.append(_record(scheme, interval, 1, 0.0))
            records.append(_record(scheme, interval, 2, 0.02))

    rows = module._aggregate(records)

    assert len(rows) == 4
    assert {(row["scheme"], row["decoherence_interval_fs"]) for row in rows} == {
        ("bm", 50.0),
        ("bm", 100.0),
        ("ma", 50.0),
        ("ma", 100.0),
    }
    first = rows[0]
    assert first["trajectory_count"] == 2
    assert first["post_burn_samples"] == 40
    assert np.isclose(first["temperature_mean_K"], 300.01)
    assert np.isclose(first["pre_heating_coordinate_mean"], 0.02)
    assert np.isclose(first["ground_manifold_absolute_mismatch_mean"], 0.01)
    assert "score" not in first


def test_d5c_electronic_exchange_is_reported_as_rate():
    module = _load_module()
    records = [
        _record("bm", 100.0, 1, 0.0),
        _record("bm", 100.0, 2, 0.0),
    ]
    rows = module._aggregate(records)
    assert len(rows) == 1
    assert np.isclose(rows[0]["electronic_environment_exchange_rate_eV_per_ps"], -0.01)
