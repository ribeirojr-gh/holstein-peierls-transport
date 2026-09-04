"""D5c sensitivity sweep for the instantaneous-decoherence interval.

D5b validated IDC-DP/BM/MA at one numerical-control interval, 100 fs.  D5c does
not introduce new dynamics.  It reuses the validated D5b trajectory kernel to
measure how the two energy-relaxing candidates, IDC-BM and IDC-MA, depend on the
phenomenological decoherence interval ``t_d``.

No single weighted score is used to choose a method.  The output reports the
independent physical diagnostics needed for a decision: electronic heating,
late-time drift, distance to the instantaneous canonical distribution,
ground-manifold mismatch, lattice temperature, energy exchange and generalized
energy balance.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np

from holstein_peierls.dynamics.langevin import LangevinBath


def _load_d5b_module() -> ModuleType:
    path = Path(__file__).with_name("d5b_instantaneous_decoherence.py")
    spec = importlib.util.spec_from_file_location("_d5b_benchmark", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the D5b benchmark module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return (
        float(np.mean(array)),
        float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
    )


def _aggregate(records: list[dict]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for scheme in ("bm", "ma"):
        intervals = sorted(
            {float(record["decoherence_interval_fs"]) for record in records if record["scheme"] == scheme}
        )
        for interval in intervals:
            subset = [
                record
                for record in records
                if record["scheme"] == scheme
                and float(record["decoherence_interval_fs"]) == interval
            ]
            temperature, temperature_std = _mean_std(
                [float(record["temperature_mean_K"]) for record in subset]
            )
            heating, heating_std = _mean_std(
                [float(record["pre_heating_coordinate_mean"]) for record in subset]
            )
            late, late_std = _mean_std(
                [float(record["pre_heating_coordinate_late_mean"]) for record in subset]
            )
            slope, slope_std = _mean_std(
                [float(record["pre_heating_coordinate_slope_per_ps"]) for record in subset]
            )
            tv, tv_std = _mean_std(
                [float(record["pre_tv_to_canonical_mean"]) for record in subset]
            )
            beta, beta_std = _mean_std(
                [float(record["pre_beta_eff_over_beta_bath_mean_finite"]) for record in subset]
            )
            ground_error, ground_error_std = _mean_std(
                [
                    abs(
                        float(record["pre_ground_manifold_population_mean"])
                        - float(record["canonical_ground_manifold_population_mean"])
                    )
                    for record in subset
                ]
            )
            expected_post, expected_post_std = _mean_std(
                [float(record["expected_post_heating_coordinate_mean"]) for record in subset]
            )
            electronic_exchange_rate, electronic_exchange_rate_std = _mean_std(
                [
                    float(record["accumulated_electronic_environment_exchange_eV"])
                    / (float(record["final_time_fs"]) / 1000.0)
                    for record in subset
                ]
            )
            balance, balance_std = _mean_std(
                [float(record["maximum_event_abs_generalized_energy_residual_eV"]) for record in subset]
            )
            norm, norm_std = _mean_std(
                [float(record["maximum_electronic_norm_error"]) for record in subset]
            )
            rows.append(
                {
                    "scheme": scheme,
                    "decoherence_interval_fs": interval,
                    "trajectory_count": len(subset),
                    "post_burn_samples": int(sum(int(record["post_burn_samples"]) for record in subset)),
                    "temperature_mean_K": temperature,
                    "temperature_std_across_trajectories_K": temperature_std,
                    "pre_heating_coordinate_mean": heating,
                    "pre_heating_coordinate_std_across_trajectories": heating_std,
                    "pre_heating_coordinate_late_mean": late,
                    "pre_heating_coordinate_late_std_across_trajectories": late_std,
                    "pre_heating_coordinate_slope_per_ps": slope,
                    "pre_heating_coordinate_slope_std_across_trajectories_per_ps": slope_std,
                    "pre_tv_to_canonical_mean": tv,
                    "pre_tv_to_canonical_std_across_trajectories": tv_std,
                    "pre_beta_eff_over_beta_bath_mean_finite": beta,
                    "pre_beta_eff_over_beta_bath_std_across_trajectories": beta_std,
                    "ground_manifold_absolute_mismatch_mean": ground_error,
                    "ground_manifold_absolute_mismatch_std_across_trajectories": ground_error_std,
                    "expected_post_heating_coordinate_mean": expected_post,
                    "expected_post_heating_coordinate_std_across_trajectories": expected_post_std,
                    "electronic_environment_exchange_rate_eV_per_ps": electronic_exchange_rate,
                    "electronic_environment_exchange_rate_std_across_trajectories_eV_per_ps": electronic_exchange_rate_std,
                    "maximum_generalized_energy_residual_eV_mean": balance,
                    "maximum_generalized_energy_residual_eV_std": balance_std,
                    "maximum_electronic_norm_error_mean": norm,
                    "maximum_electronic_norm_error_std": norm_std,
                }
            )
    return rows


def _markdown(payload: dict) -> str:
    lines = [
        "# D5c decoherence-interval sensitivity sweep",
        "",
        f"Lattice: {payload['size']}x{payload['size']}; T={payload['temperature_K']} K; dt={payload['dt_fs']} fs",
        f"Trajectory length: {payload['final_time_fs']} fs; burn-in: {payload['burn_in_fs']} fs",
        "Zero-mode policy: project; propagation between IDC events: CF4-Lanczos",
        "",
        "`beta_eff/beta_bath` is retained as a secondary nonlinear diagnostic; it is not a stand-alone selection criterion.",
        "",
        "| scheme | td [fs] | traj | samples | <T> [K] | mean heat | late heat | slope [ps^-1] | TV canonical | ground mismatch | expected post heat | Qe rate [eV/ps] | max balance residual [eV] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["aggregate"]:
        lines.append(
            "| {scheme} | {td:.0f} | {n} | {samples} | {temp:.3f} | {heat:.4f} | {late:.4f} | {slope:.4f} | {tv:.4f} | {ground:.4f} | {post:.4f} | {qe:.4e} | {balance:.3e} |".format(
                scheme=str(row["scheme"]).upper(),
                td=float(row["decoherence_interval_fs"]),
                n=int(row["trajectory_count"]),
                samples=int(row["post_burn_samples"]),
                temp=float(row["temperature_mean_K"]),
                heat=float(row["pre_heating_coordinate_mean"]),
                late=float(row["pre_heating_coordinate_late_mean"]),
                slope=float(row["pre_heating_coordinate_slope_per_ps"]),
                tv=float(row["pre_tv_to_canonical_mean"]),
                ground=float(row["ground_manifold_absolute_mismatch_mean"]),
                post=float(row["expected_post_heating_coordinate_mean"]),
                qe=float(row["electronic_environment_exchange_rate_eV_per_ps"]),
                balance=float(row["maximum_generalized_energy_residual_eV_mean"]),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--temperature-K", type=float, default=300.0)
    parser.add_argument("--gamma-u-per-fs", type=float, default=0.01)
    parser.add_argument("--gamma-v-per-fs", type=float, default=0.01)
    parser.add_argument("--dt-fs", type=float, default=0.2)
    parser.add_argument("--final-time-fs", type=float, default=6000.0)
    parser.add_argument("--burn-in-fs", type=float, default=2000.0)
    parser.add_argument(
        "--decoherence-intervals-fs",
        type=float,
        nargs="+",
        default=[50.0, 100.0, 180.0, 250.0, 500.0],
    )
    parser.add_argument(
        "--lattice-seeds",
        type=int,
        nargs="+",
        default=[20260903, 20260904, 20260905, 20260906],
    )
    parser.add_argument("--decoherence-seed-offset", type=int, default=2000000)
    parser.add_argument("--krylov-dimension", type=int, default=6)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    if args.final_time_fs <= args.burn_in_fs:
        raise ValueError("final_time_fs must exceed burn_in_fs")
    if any(interval <= 0.0 for interval in args.decoherence_intervals_fs):
        raise ValueError("all decoherence intervals must be positive")

    d5b = _load_d5b_module()
    bath = LangevinBath(
        temperature_K=args.temperature_K,
        gamma_u_per_fs=args.gamma_u_per_fs,
        gamma_v_per_fs=args.gamma_v_per_fs,
    )
    initial, parameters, static_metadata = d5b._prepare_initial_state(args.size)

    trajectories: list[dict] = []
    for scheme_index, scheme in enumerate(("bm", "ma")):
        for interval_index, interval in enumerate(args.decoherence_intervals_fs):
            for seed in args.lattice_seeds:
                decoherence_seed = (
                    args.decoherence_seed_offset
                    + 100000 * scheme_index
                    + 1000 * interval_index
                    + seed
                )
                trajectories.append(
                    d5b._run_scheme_trajectory(
                        initial,
                        parameters,
                        bath,
                        scheme=scheme,
                        dt_fs=args.dt_fs,
                        final_time_fs=args.final_time_fs,
                        burn_in_fs=args.burn_in_fs,
                        decoherence_interval_fs=interval,
                        lattice_seed=seed,
                        decoherence_seed=decoherence_seed,
                        krylov_dimension=args.krylov_dimension,
                    )
                )

    payload = {
        "size": args.size,
        "temperature_K": args.temperature_K,
        "gamma_u_per_fs": args.gamma_u_per_fs,
        "gamma_v_per_fs": args.gamma_v_per_fs,
        "dt_fs": args.dt_fs,
        "final_time_fs": args.final_time_fs,
        "burn_in_fs": args.burn_in_fs,
        "decoherence_intervals_fs": args.decoherence_intervals_fs,
        "lattice_seeds": args.lattice_seeds,
        "krylov_dimension": args.krylov_dimension,
        "static": static_metadata,
        "interpretation": {
            "decoherence_intervals_are_numerical_controls": True,
            "no_weighted_selection_score": True,
            "beta_eff_ratio_is_secondary_nonlinear_diagnostic": True,
        },
        "trajectories": trajectories,
        "aggregate": _aggregate(trajectories),
    }
    text = _markdown(payload)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
