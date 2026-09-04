"""D5d final zero-field finite-temperature IDC-BM closure benchmark.

This is the final D5 reference gate after the D5a overheating diagnosis, D5b
IDC comparison, and D5c decoherence-interval sensitivity sweep.  The benchmark
uses IDC-BM with ``t_d = 180 fs`` only as a numerical reference control.  The
interval is not a material-specific prediction and remains an explicit model
parameter.

Four independent lattice/decoherence trajectories are propagated for 10 ps on
the validated 20x20, 300 K D4 control.  The physical closure criteria below were
fixed before this 10 ps run.  They are deliberately broad enough to test that
the D5 correction removes the severe Ehrenfest overheating without overfitting
the finite stochastic ensemble.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from d5b_instantaneous_decoherence import _prepare_initial_state, _run_scheme_trajectory
from holstein_peierls.dynamics.langevin import LangevinBath


CLOSURE_THRESHOLDS = {
    "temperature_min_K": 285.0,
    "temperature_max_K": 315.0,
    "abs_mean_heating_coordinate_max": 0.05,
    "abs_late_heating_coordinate_max": 0.08,
    "positive_heating_slope_max_per_ps": 0.02,
    "tv_to_canonical_mean_max": 0.25,
    "ground_manifold_absolute_mismatch_max": 0.15,
    "abs_expected_post_heating_coordinate_max": 0.08,
    "maximum_generalized_energy_residual_eV": 5.0e-5,
    "maximum_electronic_norm_error": 1.0e-10,
}


def _mean(records: list[dict], key: str) -> float:
    return float(np.mean([float(record[key]) for record in records]))


def _std(records: list[dict], key: str) -> float:
    values = np.asarray([float(record[key]) for record in records], dtype=np.float64)
    return float(np.std(values, ddof=1)) if values.size > 1 else 0.0


def _aggregate(records: list[dict], final_time_fs: float) -> dict[str, float | int]:
    ground_mismatch = np.asarray(
        [
            abs(
                float(record["pre_ground_manifold_population_mean"])
                - float(record["canonical_ground_manifold_population_mean"])
            )
            for record in records
        ],
        dtype=np.float64,
    )
    duration_ps = float(final_time_fs) / 1000.0
    electronic_exchange_rates = np.asarray(
        [
            float(record["accumulated_electronic_environment_exchange_eV"])
            / duration_ps
            for record in records
        ],
        dtype=np.float64,
    )
    return {
        "trajectory_count": len(records),
        "total_post_burn_samples": int(
            sum(int(record["post_burn_samples"]) for record in records)
        ),
        "temperature_mean_K": _mean(records, "temperature_mean_K"),
        "temperature_std_across_trajectories_K": _std(records, "temperature_mean_K"),
        "pre_heating_coordinate_mean": _mean(records, "pre_heating_coordinate_mean"),
        "pre_heating_coordinate_late_mean": _mean(
            records, "pre_heating_coordinate_late_mean"
        ),
        "pre_heating_coordinate_slope_per_ps": _mean(
            records, "pre_heating_coordinate_slope_per_ps"
        ),
        "pre_tv_to_canonical_mean": _mean(records, "pre_tv_to_canonical_mean"),
        "pre_beta_eff_over_beta_bath_mean_finite": _mean(
            records, "pre_beta_eff_over_beta_bath_mean_finite"
        ),
        "ground_manifold_absolute_mismatch_mean": float(np.mean(ground_mismatch)),
        "ground_manifold_absolute_mismatch_max": float(np.max(ground_mismatch)),
        "expected_post_heating_coordinate_mean": _mean(
            records, "expected_post_heating_coordinate_mean"
        ),
        "electronic_environment_exchange_rate_eV_per_ps": float(
            np.mean(electronic_exchange_rates)
        ),
        "maximum_generalized_energy_residual_eV": float(
            max(float(record["maximum_event_abs_generalized_energy_residual_eV"]) for record in records)
        ),
        "maximum_electronic_norm_error": float(
            max(float(record["maximum_electronic_norm_error"]) for record in records)
        ),
    }


def _evaluate_closure(aggregate: dict[str, float | int]) -> dict[str, bool]:
    t = CLOSURE_THRESHOLDS
    checks = {
        "temperature": (
            t["temperature_min_K"]
            <= float(aggregate["temperature_mean_K"])
            <= t["temperature_max_K"]
        ),
        "mean_heating": abs(float(aggregate["pre_heating_coordinate_mean"]))
        <= t["abs_mean_heating_coordinate_max"],
        "late_heating": abs(float(aggregate["pre_heating_coordinate_late_mean"]))
        <= t["abs_late_heating_coordinate_max"],
        "heating_slope": float(aggregate["pre_heating_coordinate_slope_per_ps"])
        <= t["positive_heating_slope_max_per_ps"],
        "canonical_tv": float(aggregate["pre_tv_to_canonical_mean"])
        <= t["tv_to_canonical_mean_max"],
        "ground_manifold": float(aggregate["ground_manifold_absolute_mismatch_mean"])
        <= t["ground_manifold_absolute_mismatch_max"],
        "expected_post_heating": abs(
            float(aggregate["expected_post_heating_coordinate_mean"])
        )
        <= t["abs_expected_post_heating_coordinate_max"],
        "energy_balance": float(aggregate["maximum_generalized_energy_residual_eV"])
        <= t["maximum_generalized_energy_residual_eV"],
        "electronic_norm": float(aggregate["maximum_electronic_norm_error"])
        <= t["maximum_electronic_norm_error"],
    }
    return checks


def _markdown(payload: dict) -> str:
    a = payload["aggregate"]
    checks = payload["closure_checks"]
    lines = [
        "# D5d IDC-BM 10 ps closure",
        "",
        f"Lattice: {payload['size']}x{payload['size']}; T={payload['temperature_K']} K; dt={payload['dt_fs']} fs",
        f"Scheme: IDC-BM; td={payload['decoherence_interval_fs']} fs (numerical control, not material calibration)",
        f"Trajectories: {a['trajectory_count']}; post-burn samples: {a['total_post_burn_samples']}",
        "",
        "| metric | result |",
        "|---|---:|",
        f"| mean lattice temperature [K] | {a['temperature_mean_K']:.6f} |",
        f"| mean pre-collapse heating coordinate | {a['pre_heating_coordinate_mean']:.6f} |",
        f"| late pre-collapse heating coordinate | {a['pre_heating_coordinate_late_mean']:.6f} |",
        f"| heating slope [ps^-1] | {a['pre_heating_coordinate_slope_per_ps']:.6f} |",
        f"| TV distance to canonical | {a['pre_tv_to_canonical_mean']:.6f} |",
        f"| ground-manifold absolute mismatch | {a['ground_manifold_absolute_mismatch_mean']:.6f} |",
        f"| expected post-collapse heating coordinate | {a['expected_post_heating_coordinate_mean']:.6f} |",
        f"| electronic-environment exchange rate [eV/ps] | {a['electronic_environment_exchange_rate_eV_per_ps']:.6e} |",
        f"| max generalized energy residual [eV] | {a['maximum_generalized_energy_residual_eV']:.6e} |",
        f"| max electronic norm error | {a['maximum_electronic_norm_error']:.6e} |",
        "",
        "## Pre-registered closure checks",
        "",
    ]
    for name, passed in checks.items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    lines.extend(
        [
            "",
            f"Overall physical closure: {'PASS' if payload['closure_pass'] else 'FAIL'}",
            "",
            "`beta_eff/beta_bath` is diagnostic only and is not a stand-alone closure criterion.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--temperature-K", type=float, default=300.0)
    parser.add_argument("--gamma-u-per-fs", type=float, default=0.01)
    parser.add_argument("--gamma-v-per-fs", type=float, default=0.01)
    parser.add_argument("--dt-fs", type=float, default=0.2)
    parser.add_argument("--final-time-fs", type=float, default=10000.0)
    parser.add_argument("--burn-in-fs", type=float, default=2000.0)
    parser.add_argument("--decoherence-interval-fs", type=float, default=180.0)
    parser.add_argument(
        "--lattice-seeds",
        type=int,
        nargs="+",
        default=[20260903, 20260904, 20260905, 20260906],
    )
    parser.add_argument("--decoherence-seed-offset", type=int, default=3000000)
    parser.add_argument("--krylov-dimension", type=int, default=6)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    bath = LangevinBath(
        temperature_K=args.temperature_K,
        gamma_u_per_fs=args.gamma_u_per_fs,
        gamma_v_per_fs=args.gamma_v_per_fs,
    )
    initial, parameters, static_metadata = _prepare_initial_state(args.size)
    trajectories = []
    for seed in args.lattice_seeds:
        trajectories.append(
            _run_scheme_trajectory(
                initial,
                parameters,
                bath,
                scheme="bm",
                dt_fs=args.dt_fs,
                final_time_fs=args.final_time_fs,
                burn_in_fs=args.burn_in_fs,
                decoherence_interval_fs=args.decoherence_interval_fs,
                lattice_seed=seed,
                decoherence_seed=args.decoherence_seed_offset + seed,
                krylov_dimension=args.krylov_dimension,
            )
        )

    aggregate = _aggregate(trajectories, args.final_time_fs)
    checks = _evaluate_closure(aggregate)
    payload = {
        "size": args.size,
        "temperature_K": args.temperature_K,
        "gamma_u_per_fs": args.gamma_u_per_fs,
        "gamma_v_per_fs": args.gamma_v_per_fs,
        "dt_fs": args.dt_fs,
        "final_time_fs": args.final_time_fs,
        "burn_in_fs": args.burn_in_fs,
        "scheme": "bm",
        "decoherence_interval_fs": args.decoherence_interval_fs,
        "decoherence_interval_is_material_calibrated": False,
        "lattice_seeds": args.lattice_seeds,
        "krylov_dimension": args.krylov_dimension,
        "static": static_metadata,
        "thresholds": CLOSURE_THRESHOLDS,
        "trajectories": trajectories,
        "aggregate": aggregate,
        "closure_checks": checks,
        "closure_pass": bool(all(checks.values())),
    }
    text = _markdown(payload)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(text, encoding="utf-8")

    if not payload["closure_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
