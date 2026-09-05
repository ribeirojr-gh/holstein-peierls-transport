#!/usr/bin/env python3
"""D6g final 10 ps closure for selected pair IDC controls.

The D6f sensitivity sweep selected a sector-specific, deliberately minimal
reference model:

- bipolaron: BM instantaneous decoherence;
- distinguishable e-h exciton: DP instantaneous decoherence.

Both use t_d=100 fs only as a common numerical-control interval.  This script
pre-registers the final 10 ps closure criteria.  It does not calibrate a
material decoherence time, apply an electric field, or fit transport data.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from holstein_peierls.dynamics.langevin import LangevinBath
from holstein_peierls.dynamics.pair_coupled import PairLatticeMasses


def _load_d6e():
    path = Path(__file__).with_name("d6e_pair_idc_screen.py")
    spec = importlib.util.spec_from_file_location("d6e_closure_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load D6e helper module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mean(values: list[float]) -> float:
    return float(np.mean(values))


def _std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _aggregate(sector: str, scheme: str, runs: list[dict]) -> dict:
    keys = [
        "mean_lattice_temperature_K",
        "mean_pre_heating",
        "early_pre_heating",
        "late_pre_heating",
        "heating_slope_per_ps",
        "mean_tv_canonical",
        "mean_tv_uniform",
        "mean_ground_mismatch",
        "mean_beta_ratio_finite",
        "mean_expected_post_heating",
        "mean_realized_post_heating",
        "electronic_exchange_rate_eV_per_ps",
    ]
    item = {
        "sector": sector,
        "scheme": scheme,
        "trajectory_count": len(runs),
        "diagnostic_samples": int(sum(r["diagnostic_samples"] for r in runs)),
    }
    for key in keys:
        values = [float(r[key]) for r in runs]
        item[key] = _mean(values)
        item[key + "_std"] = _std(values)
    item["maximum_generalized_balance_residual_eV"] = max(
        float(r["maximum_generalized_balance_residual_eV"]) for r in runs
    )
    item["maximum_norm_error"] = max(float(r["maximum_norm_error"]) for r in runs)
    item["maximum_sector_constraint_error"] = max(
        float(r["maximum_sector_constraint_error"]) for r in runs
    )
    return item


def _closure_checks(item: dict) -> dict[str, bool]:
    prefix = item["sector"]
    return {
        prefix + "_temperature": 240.0 <= item["mean_lattice_temperature_K"] <= 360.0,
        prefix + "_mean_heating": abs(item["mean_pre_heating"]) < 0.02,
        prefix + "_late_heating": abs(item["late_pre_heating"]) < 0.04,
        prefix + "_heating_slope": abs(item["heating_slope_per_ps"]) < 0.01,
        prefix + "_canonical_tv": item["mean_tv_canonical"] < 0.05,
        prefix + "_ground_mismatch": item["mean_ground_mismatch"] < 0.05,
        prefix + "_expected_post_heating": abs(item["mean_expected_post_heating"]) < 0.02,
        prefix + "_generalized_balance": item["maximum_generalized_balance_residual_eV"] < 1.0e-4,
        prefix + "_norm": item["maximum_norm_error"] < 1.0e-10,
        prefix + "_sector_constraints": item["maximum_sector_constraint_error"] < 1.0e-10,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--temperature-K", type=float, default=300.0)
    parser.add_argument("--gamma-u-per-fs", type=float, default=0.01)
    parser.add_argument("--gamma-v-per-fs", type=float, default=0.01)
    parser.add_argument("--dt-fs", type=float, default=0.2)
    parser.add_argument("--final-time-fs", type=float, default=10000.0)
    parser.add_argument("--burn-in-fs", type=float, default=2000.0)
    parser.add_argument("--decoherence-interval-fs", type=float, default=100.0)
    parser.add_argument(
        "--seeds", nargs="+", type=int,
        default=[20260905, 20260906, 20260907, 20260908],
    )
    parser.add_argument("--krylov-dimension", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    d6e = _load_d6e()
    masses = PairLatticeMasses(75000.0, 150000.0)
    bath = LangevinBath(args.temperature_K, args.gamma_u_per_fs, args.gamma_v_per_fs)
    selections = (("bipolaron", "bm"), ("exciton", "dp"))

    runs: list[dict] = []
    for sector, scheme in selections:
        for seed in args.seeds:
            runs.append(
                d6e.run_trajectory(
                    sector,
                    scheme,
                    args.size,
                    masses,
                    bath,
                    seed,
                    args.dt_fs,
                    args.final_time_fs,
                    args.burn_in_fs,
                    args.decoherence_interval_fs,
                    args.krylov_dimension,
                )
            )

    aggregates = []
    checks: dict[str, bool] = {}
    for sector, scheme in selections:
        group = [r for r in runs if r["sector"] == sector and r["scheme"] == scheme]
        item = _aggregate(sector, scheme, group)
        aggregates.append(item)
        checks.update(_closure_checks(item))

    closure_pass = all(checks.values())
    payload = {
        "size": args.size,
        "temperature_K": args.temperature_K,
        "gamma_u_per_fs": args.gamma_u_per_fs,
        "gamma_v_per_fs": args.gamma_v_per_fs,
        "dt_fs": args.dt_fs,
        "final_time_fs": args.final_time_fs,
        "burn_in_fs": args.burn_in_fs,
        "decoherence_interval_fs": args.decoherence_interval_fs,
        "seeds": args.seeds,
        "krylov_dimension": args.krylov_dimension,
        "selected_schemes": {"bipolaron": "bm", "exciton": "dp"},
        "interval_is_numerical_control_only": True,
        "runs": runs,
        "aggregates": aggregates,
        "closure_checks": checks,
        "closure_pass": closure_pass,
        "interpretation": (
            "D6g is the final zero-field finite-temperature pair-dynamics closure. "
            "It validates the selected numerical controls but does not calibrate a "
            "material-specific decoherence time or establish mobility."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# D6g final pair-dynamics closure",
        "",
        f"Control: {args.size}x{args.size}, {args.temperature_K:g} K, dt={args.dt_fs:g} fs, "
        f"time={args.final_time_fs/1000:g} ps, burn-in={args.burn_in_fs/1000:g} ps, "
        f"t_d={args.decoherence_interval_fs:g} fs (numerical control only).",
        "",
        "Selected schemes: bipolaron=BM; distinguishable e-h exciton=DP.",
        "",
        "| sector | scheme | <T> [K] | <pre heat> | late | slope/ps | TV can | ground mismatch | expected post heat | Qe rate [eV/ps] | max balance [eV] |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in aggregates:
        lines.append(
            f"| {item['sector']} | {item['scheme']} | {item['mean_lattice_temperature_K']:.3f} | "
            f"{item['mean_pre_heating']:.6g} | {item['late_pre_heating']:.6g} | "
            f"{item['heating_slope_per_ps']:.3e} | {item['mean_tv_canonical']:.6g} | "
            f"{item['mean_ground_mismatch']:.6g} | {item['mean_expected_post_heating']:.6g} | "
            f"{item['electronic_exchange_rate_eV_per_ps']:.3e} | "
            f"{item['maximum_generalized_balance_residual_eV']:.3e} |"
        )
    lines += ["", "## Pre-registered closure checks", ""]
    lines += [f"- {name}: {'PASS' if value else 'FAIL'}" for name, value in checks.items()]
    lines += ["", f"Overall: {'PASS' if closure_pass else 'FAIL'}", ""]
    args.markdown.write_text("\n".join(lines), encoding="utf-8")

    if not closure_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
