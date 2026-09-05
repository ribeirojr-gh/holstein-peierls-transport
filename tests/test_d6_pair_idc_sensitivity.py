"""Regression tests for D6f pair IDC interval-sensitivity aggregation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_module():
    path = Path(__file__).resolve().parents[1] / "experiments" / "d6f_pair_idc_interval_sweep.py"
    spec = importlib.util.spec_from_file_location("_d6f_test_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(sector: str, scheme: str, interval: float, shift: float) -> dict:
    return {
        "sector": sector,
        "scheme": scheme,
        "decoherence_interval_fs": interval,
        "diagnostic_samples": 10,
        "mean_lattice_temperature_K": 300.0 + shift,
        "mean_pre_heating": 0.01 + shift,
        "early_pre_heating": 0.02 + shift,
        "late_pre_heating": 0.03 + shift,
        "heating_slope_per_ps": 0.004 + shift,
        "mean_tv_canonical": 0.10 + shift,
        "mean_tv_uniform": 0.90 - shift,
        "mean_ground_mismatch": 0.05 + shift,
        "mean_beta_ratio_finite": 0.8 + shift,
        "mean_expected_post_heating": -0.01 + shift,
        "mean_realized_post_heating": -0.02 + shift,
        "electronic_exchange_rate_eV_per_ps": -0.03 + shift,
        "maximum_generalized_balance_residual_eV": 1.0e-6 + shift * 1.0e-8,
        "maximum_norm_error": 1.0e-13,
        "maximum_sector_constraint_error": 2.0e-13,
    }


def test_d6f_aggregate_groups_sector_scheme_and_interval_without_score() -> None:
    module = _load_module()
    records = []
    for sector in ("bipolaron", "exciton"):
        for scheme in ("dp", "bm"):
            for interval in (50.0, 100.0):
                records.append(_record(sector, scheme, interval, 0.0))
                records.append(_record(sector, scheme, interval, 0.02))

    rows = module.aggregate(records)
    assert len(rows) == 8
    keys = {(r["sector"], r["scheme"], r["decoherence_interval_fs"]) for r in rows}
    assert ("bipolaron", "dp", 50.0) in keys
    assert ("exciton", "bm", 100.0) in keys
    first = rows[0]
    assert first["trajectory_count"] == 2
    assert first["diagnostic_samples"] == 20
    assert np.isclose(first["mean_lattice_temperature_K"], 300.01)
    assert np.isclose(first["mean_pre_heating"], 0.02)
    assert "score" not in first


def test_d6f_aggregate_reports_exchange_rate_and_worst_numerical_errors() -> None:
    module = _load_module()
    records = [
        _record("bipolaron", "dp", 100.0, 0.0),
        _record("bipolaron", "dp", 100.0, 0.02),
    ]
    records[1]["maximum_generalized_balance_residual_eV"] = 9.0e-6
    records[1]["maximum_norm_error"] = 7.0e-13
    rows = module.aggregate(records)
    assert len(rows) == 1
    row = rows[0]
    assert np.isclose(row["electronic_exchange_rate_eV_per_ps"], -0.02)
    assert np.isclose(row["maximum_generalized_balance_residual_eV"], 9.0e-6)
    assert np.isclose(row["maximum_norm_error"], 7.0e-13)
