#!/usr/bin/env python3
"""D6f pair IDC decoherence-interval sensitivity sweep.

D6e showed that both DP and BM suppress coherent Ehrenfest overheating at the
single numerical-control interval t_d=100 fs. D6f asks whether that result is
robust over a nontrivial interval range. MA is intentionally excluded from the
primary sweep because D6e found a large bipolaron outlier while DP and BM were
both well behaved.

This benchmark reuses the validated D6e trajectory kernel. It changes no
propagation physics. DP and BM are compared separately for the symmetric
singlet bipolaron and distinguishable e-h exciton sectors.

The interval remains a phenomenological model parameter. No material-specific
value is inferred by this benchmark and no weighted selection score is used.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from holstein_peierls.dynamics.langevin import LangevinBath
from holstein_peierls.dynamics.pair_coupled import PairLatticeMasses


def _load_d6e_helpers():
    path = Path(__file__).with_name("d6e_pair_idc_screen.py")
    spec = importlib.util.spec_from_file_location("d6e_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load D6e helper module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def aggregate(runs: list[dict]) -> list[dict]:
    """Aggregate trajectory metrics by sector, scheme and decoherence interval."""
    result: list[dict] = []
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
    sectors = ("bipolaron", "exciton")
    schemes = ("dp", "bm")
    intervals = sorted({float(r["decoherence_interval_fs"]) for r in runs})
    for sector in sectors:
        for scheme in schemes:
            for interval in intervals:
                group = [
                    r
                    for r in runs
                    if r["sector"] == sector
                    and r["scheme"] == scheme
                    and np.isclose(r["decoherence_interval_fs"], interval)
                ]
                if not group:
                    continue
                item: dict[str, float | int | str] = {
                    "sector": sector,
                    "scheme": scheme,
                    "decoherence_interval_fs": float(interval),
                    "trajectory_count": len(group),
                    "diagnostic_samples": int(sum(r["diagnostic_samples"] for r in group)),
                }
                for key in keys:
                    values = [float(r[key]) for r in group]
                    item[key] = _mean(values)
                    item[f"{key}_std"] = _std(values)
                item["maximum_generalized_balance_residual_eV"] = max(
                    float(r["maximum_generalized_balance_residual_eV"]) for r in group
                )
                item["maximum_norm_error"] = max(float(r["maximum_norm_error"]) for r in group)
                item["maximum_sector_constraint_error"] = max(
                    float(r["maximum_sector_constraint_error"]) for r in group
                )
                result.append(item)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=4)
    p.add_argument("--temperature-K", type=float, default=300.0)
    p.add_argument("--gamma-u-per-fs", type=float, default=0.01)
    p.add_argument("--gamma-v-per-fs", type=float, default=0.01)
    p.add_argument("--dt-fs", type=float, default=0.2)
    p.add_argument("--final-time-fs", type=float, default=4000.0)
    p.add_argument("--burn-in-fs", type=float, default=1000.0)
    p.add_argument(
        "--decoherence-intervals-fs",
        nargs="+",
        type=float,
        default=[50.0, 100.0, 180.0, 250.0, 500.0],
    )
    p.add_argument(
        "--seeds", nargs="+", type=int, default=[20260905, 20260906, 20260907, 20260908]
    )
    p.add_argument("--krylov-dimension", type=int, default=8)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--markdown", type=Path, required=True)
    args = p.parse_args()

    if args.final_time_fs <= args.burn_in_fs:
        raise ValueError("final_time_fs must exceed burn_in_fs")
    intervals = [float(value) for value in args.decoherence_intervals_fs]
    if len(set(intervals)) != len(intervals) or any(value <= 0.0 for value in intervals):
        raise ValueError("decoherence intervals must be positive and unique")
    for interval in intervals:
        steps = interval / args.dt_fs
        if not np.isclose(steps, round(steps), rtol=0.0, atol=1.0e-10):
            raise ValueError("every decoherence interval must be an integer multiple of dt")

    helpers = _load_d6e_helpers()
    masses = PairLatticeMasses(75000.0, 150000.0)
    bath = LangevinBath(args.temperature_K, args.gamma_u_per_fs, args.gamma_v_per_fs)

    runs: list[dict] = []
    for sector in ("bipolaron", "exciton"):
        for scheme in ("dp", "bm"):
            for interval in intervals:
                for seed in args.seeds:
                    record = helpers.run_trajectory(
                        sector,
                        scheme,
                        args.size,
                        masses,
                        bath,
                        seed,
                        args.dt_fs,
                        args.final_time_fs,
                        args.burn_in_fs,
                        interval,
                        args.krylov_dimension,
                    )
                    record["decoherence_interval_fs"] = float(interval)
                    runs.append(record)

    aggregates = aggregate(runs)
    checks: dict[str, bool] = {}
    for item in aggregates:
        prefix = f"{item['sector']}_{item['scheme']}_{item['decoherence_interval_fs']:g}fs"
        checks[prefix + "_temperature"] = 240.0 <= float(item["mean_lattice_temperature_K"]) <= 360.0
        checks[prefix + "_balance"] = float(item["maximum_generalized_balance_residual_eV"]) < 1.0e-4
        checks[prefix + "_norm"] = float(item["maximum_norm_error"]) < 1.0e-10
        checks[prefix + "_constraints"] = float(item["maximum_sector_constraint_error"]) < 1.0e-10
    numerical_pass = all(checks.values())

    payload = {
        "size": args.size,
        "temperature_K": args.temperature_K,
        "gamma_u_per_fs": args.gamma_u_per_fs,
        "gamma_v_per_fs": args.gamma_v_per_fs,
        "dt_fs": args.dt_fs,
        "final_time_fs": args.final_time_fs,
        "burn_in_fs": args.burn_in_fs,
        "decoherence_intervals_fs": intervals,
        "seeds": args.seeds,
        "krylov_dimension": args.krylov_dimension,
        "schemes": ["dp", "bm"],
        "sectors": ["bipolaron", "exciton"],
        "runs": runs,
        "aggregates": aggregates,
        "numerical_checks": checks,
        "numerical_pass": numerical_pass,
        "physical_selection_is_manual": True,
        "no_weighted_selection_score": True,
        "interpretation": (
            "D6f is a phenomenological decoherence-interval sensitivity sweep. "
            "Numerical PASS does not validate a material-specific interval or select a model."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# D6f pair IDC interval sensitivity",
        "",
        f"Control: {args.size}x{args.size}, {args.temperature_K:g} K, dt={args.dt_fs:g} fs, "
        f"time={args.final_time_fs/1000:g} ps, burn-in={args.burn_in_fs/1000:g} ps.",
        "",
        "DP and BM are compared over the same interval grid. No weighted selection score is used.",
        "",
        "| sector | scheme | t_d [fs] | <T> [K] | <pre heat> | late | slope/ps | TV can | ground mismatch | expected post heat | Qe rate [eV/ps] | max balance [eV] |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in aggregates:
        lines.append(
            f"| {item['sector']} | {item['scheme']} | {item['decoherence_interval_fs']:g} | "
            f"{item['mean_lattice_temperature_K']:.3f} | {item['mean_pre_heating']:.5g} | "
            f"{item['late_pre_heating']:.5g} | {item['heating_slope_per_ps']:.3e} | "
            f"{item['mean_tv_canonical']:.5g} | {item['mean_ground_mismatch']:.5g} | "
            f"{item['mean_expected_post_heating']:.5g} | "
            f"{item['electronic_exchange_rate_eV_per_ps']:.3e} | "
            f"{item['maximum_generalized_balance_residual_eV']:.3e} |"
        )
    lines += [
        "",
        "Numerical gates: " + ("PASS" if numerical_pass else "FAIL"),
        "",
        "Physical selection remains manual. A useful candidate should remain well behaved over a nontrivial interval range rather than only at one tuned point.",
        "",
    ]
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    if not numerical_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
