#!/usr/bin/env python3
"""TP2b long-trajectory paired-field convergence benchmark.

TP2a established numerically correct paired +/- field transport observables but
not a statistically converged mobility. TP2b increases both independent-seed
count and trajectory duration. Several cumulative post-burn-in checkpoints are
extracted from each *single* long trajectory, avoiding the cost and statistical
ambiguity of independent reruns at multiple final times.

The 2 mV/A field is retained as a nonlinearity diagnostic. No field range is
assumed a priori to be in linear response and no reported mobility is promoted
to a material value unless the pre-registered readiness diagnostics pass.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from holstein_peierls.dynamics.coupled import CoupledEhrenfestState
from holstein_peierls.dynamics.driven import field_dynamic_total_energy
from holstein_peierls.dynamics.driven_thermal_decoherence import apply_field_idc_to_coupled_state
from holstein_peierls.dynamics.ehrenfest import LatticeVelocity
from holstein_peierls.dynamics.field import UniformElectricField2D
from holstein_peierls.dynamics.langevin import LangevinBath, kinetic_temperature_K
from holstein_peierls.dynamics.linear_response import analyze_paired_field_ensemble
from holstein_peierls.dynamics.thermal import (
    coupled_baoab_step,
    project_inter_molecular_zero_modes,
    thermostatted_kinetic_degrees_of_freedom,
    zero_mode_means,
)
from holstein_peierls.dynamics.thermal_decoherence import decoherence_interval_steps
from holstein_peierls.dynamics.transport_observables import (
    field_power_from_particle_velocity,
    transport_kinematics,
    trapezoidal_displacement_increment,
)
from holstein_peierls.electronic import solve_ground_state
from holstein_peierls.parameters import StaticPolaronParameters
from holstein_peierls.polaron import solve_static_polaron


def _prepare_initial_state(size: int):
    center = (size // 2) * size + (size // 2) + 1
    parameters = StaticPolaronParameters(nx=size, ny=size, polaron_position=center)
    static = solve_static_polaron(
        parameters,
        solver="sparse",
        gradient_mode="optimized",
        legacy_convergence=False,
    )
    ground = solve_ground_state(static.state, parameters, solver="sparse")
    raw = CoupledEhrenfestState(
        static.state.copy(),
        LatticeVelocity.zeros(size, size),
        np.asarray(ground.wavefunction, dtype=np.complex128),
    )
    return project_inter_molecular_zero_modes(raw), parameters, {
        "static_iterations": int(static.diagnostics.iterations),
        "static_converged": bool(static.diagnostics.converged),
        "static_total_energy_eV": float(static.total_energy),
        "static_formation_energy_eV": float(static.formation_energy),
    }


def _run_trajectory(
    initial: CoupledEhrenfestState,
    parameters: StaticPolaronParameters,
    bath: LangevinBath,
    *,
    field_magnitude_mV_per_A: float,
    field_sign: int,
    seed: int,
    dt_fs: float,
    final_time_fs: float,
    burn_in_fs: float,
    checkpoint_times_fs: list[float],
    temperature_sample_interval_fs: float,
    decoherence_interval_fs: float,
    krylov_dimension: int,
) -> dict:
    if field_sign not in (-1, +1):
        raise ValueError("field_sign must be -1 or +1")
    magnitude = float(field_magnitude_mV_per_A)
    if not np.isfinite(magnitude) or magnitude <= 0.0:
        raise ValueError("field magnitude must be finite and positive")

    steps = int(round(final_time_fs / dt_fs))
    burn_steps = int(round(burn_in_fs / dt_fs))
    sample_stride = int(round(temperature_sample_interval_fs / dt_fs))
    event_stride = decoherence_interval_steps(decoherence_interval_fs, dt_fs)
    checkpoint_steps = [int(round(value / dt_fs)) for value in checkpoint_times_fs]
    if steps <= 0 or not np.isclose(steps * dt_fs, final_time_fs):
        raise ValueError("final_time_fs must be an integer multiple of dt_fs")
    if burn_steps < 0 or burn_steps >= steps or not np.isclose(burn_steps * dt_fs, burn_in_fs):
        raise ValueError("burn_in_fs must be an integer multiple of dt_fs in [0, final)")
    if sample_stride <= 0 or not np.isclose(sample_stride * dt_fs, temperature_sample_interval_fs):
        raise ValueError("temperature sample interval must be an integer multiple of dt")
    for value, checkpoint_step in zip(checkpoint_times_fs, checkpoint_steps):
        if value <= burn_in_fs or value > final_time_fs:
            raise ValueError("checkpoint times must lie in (burn_in_fs, final_time_fs]")
        if not np.isclose(checkpoint_step * dt_fs, value):
            raise ValueError("checkpoint times must be integer multiples of dt_fs")

    signed_field_mV = field_sign * magnitude
    field = UniformElectricField2D.from_millivolt_per_angstrom(
        signed_field_mV,
        angle_radians=0.0,
        ax_angstrom=3.0,
        ay_angstrom=3.0,
    )
    current = CoupledEhrenfestState(
        initial.lattice.copy(), initial.velocity.copy(), initial.electronic_state.copy()
    )
    lattice_rng = np.random.default_rng(seed)
    idc_rng = np.random.default_rng(seed + 100_000)
    dof = thermostatted_kinetic_degrees_of_freedom(parameters, "project")
    initial_energy = field_dynamic_total_energy(
        current.lattice,
        current.velocity,
        parameters,
        current.electronic_state,
        field,
        0.0,
    ).total

    lattice_heat = 0.0
    electronic_exchange = 0.0
    field_work = 0.0
    field_work_from_displacement = 0.0
    postburn_dx = 0.0
    postburn_dy = 0.0
    event_count = 0
    temperatures: list[float] = []
    residuals: list[float] = []
    max_norm_error = 0.0
    max_zero_mode = 0.0
    max_power_identity_error = 0.0
    h_eval = 0
    h_apply = 0
    checkpoint_velocity_x: dict[str, float] = {}
    checkpoint_velocity_y: dict[str, float] = {}
    checkpoint_set = set(checkpoint_steps)
    start = perf_counter()

    for step in range(steps):
        time_fs = step * dt_fs
        kin_old = transport_kinematics(
            current.lattice,
            parameters,
            current.electronic_state,
            field=field,
            time_fs=time_fs,
        )
        direct_power = field_power_from_particle_velocity(field, kin_old)
        thermal = coupled_baoab_step(
            current,
            parameters,
            bath,
            lattice_rng,
            time_fs,
            dt_fs,
            field=field,
            zero_mode_policy="project",
            electronic_method="cfm4_lanczos",
            krylov_dimension=krylov_dimension,
        )
        pre_idc = thermal.state
        new_time_fs = (step + 1) * dt_fs
        kin_new = transport_kinematics(
            pre_idc.lattice,
            parameters,
            pre_idc.electronic_state,
            field=field,
            time_fs=new_time_fs,
        )
        dx, dy = trapezoidal_displacement_increment(kin_old, kin_new, dt_fs)
        if step >= burn_steps:
            postburn_dx += dx
            postburn_dy += dy

        lattice_heat += thermal.bath_heat_eV
        electronic_exchange += 0.0
        field_work += thermal.field_work_eV
        field_work_from_displacement += -field.ex_v_per_angstrom * dx
        h_eval += thermal.hamiltonian_evaluations
        h_apply += thermal.hamiltonian_applications
        max_power_identity_error = max(
            max_power_identity_error,
            abs(direct_power + field.ex_v_per_angstrom * kin_old.velocity_x_A_per_fs),
        )
        current = pre_idc

        if (step + 1) % event_stride == 0:
            current, event = apply_field_idc_to_coupled_state(
                current,
                parameters,
                field,
                new_time_fs,
                bath.temperature_K,
                "bm",
                idc_rng,
            )
            electronic_exchange += event.electronic_environment_exchange_eV
            event_count += 1

        max_norm_error = max(
            max_norm_error,
            abs(float(np.linalg.norm(current.electronic_state)) - 1.0),
        )
        max_zero_mode = max(
            max_zero_mode,
            max(abs(value) for value in zero_mode_means(current).values()),
        )

        completed_steps = step + 1
        if completed_steps in checkpoint_set:
            window_fs = new_time_fs - burn_in_fs
            key = f"{new_time_fs:.6f}"
            checkpoint_velocity_x[key] = float(postburn_dx / window_fs)
            checkpoint_velocity_y[key] = float(postburn_dy / window_fs)

        if completed_steps % sample_stride == 0 or completed_steps == steps:
            if completed_steps > burn_steps:
                temperatures.append(
                    kinetic_temperature_K(
                        current.velocity,
                        parameters,
                        degrees_of_freedom=dof,
                    )
                )
            energy = field_dynamic_total_energy(
                current.lattice,
                current.velocity,
                parameters,
                current.electronic_state,
                field,
                new_time_fs,
            ).total
            residuals.append(
                float(
                    (energy - initial_energy)
                    - lattice_heat
                    - electronic_exchange
                    - field_work
                )
            )

    if not temperatures or not residuals:
        raise RuntimeError("TP2b produced no post-burn samples")
    if len(checkpoint_velocity_x) != len(checkpoint_steps):
        raise RuntimeError("TP2b did not record every requested checkpoint")

    postburn_time_fs = final_time_fs - burn_in_fs
    return {
        "seed": int(seed),
        "field_sign": int(field_sign),
        "field_magnitude_mV_per_A": magnitude,
        "signed_field_mV_per_A": signed_field_mV,
        "steps": int(steps),
        "elapsed_seconds": float(perf_counter() - start),
        "idc_events": int(event_count),
        "expected_idc_events": int(steps // event_stride),
        "mean_lattice_temperature_K": float(np.mean(temperatures)),
        "std_lattice_temperature_K": float(np.std(temperatures)),
        "postburn_displacement_x_A": float(postburn_dx),
        "postburn_displacement_y_A": float(postburn_dy),
        "postburn_mean_velocity_x_A_per_fs": float(postburn_dx / postburn_time_fs),
        "postburn_mean_velocity_y_A_per_fs": float(postburn_dy / postburn_time_fs),
        "checkpoint_mean_velocity_x_A_per_fs": checkpoint_velocity_x,
        "checkpoint_mean_velocity_y_A_per_fs": checkpoint_velocity_y,
        "lattice_bath_heat_eV": float(lattice_heat),
        "electronic_environment_exchange_eV": float(electronic_exchange),
        "field_work_eV": float(field_work),
        "field_work_from_displacement_eV": float(field_work_from_displacement),
        "field_work_displacement_difference_eV": float(field_work - field_work_from_displacement),
        "maximum_power_velocity_identity_error_eV_per_fs": float(max_power_identity_error),
        "maximum_sampled_abs_generalized_balance_residual_eV": float(np.max(np.abs(residuals))),
        "maximum_norm_error": float(max_norm_error),
        "maximum_zero_mode_mean": float(max_zero_mode),
        "hamiltonian_evaluations": int(h_eval),
        "hamiltonian_applications": int(h_apply),
    }


def _response_dict(response, fields: np.ndarray) -> dict:
    field_summaries = []
    for index, magnitude in enumerate(fields):
        field_summaries.append({
            "field_magnitude_mV_per_A": float(magnitude),
            "odd_velocity_ci_A_per_fs": asdict(response.field_odd_velocity_ci[index]),
            "even_velocity_ci_A_per_fs": asdict(response.field_even_velocity_ci[index]),
            "electron_mobility_ci_cm2_per_V_s": asdict(response.field_mobility_ci[index]),
            "odd_velocity_samples_A_per_fs": response.odd_velocity_A_per_fs[:, index].tolist(),
            "even_velocity_samples_A_per_fs": response.even_velocity_A_per_fs[:, index].tolist(),
            "electron_mobility_samples_cm2_per_V_s": response.field_mobility_cm2_per_V_s[:, index].tolist(),
        })
    return {
        "field_summaries": field_summaries,
        "seed_linear_responses": [asdict(value) for value in response.seed_responses],
        "seed_level_mobility_ci_cm2_per_V_s": asdict(response.mobility_ci),
        "ensemble_mean_slope_A2_per_V_fs": response.ensemble_mean_slope_A2_per_V_fs,
        "ensemble_mean_mobility_cm2_per_V_s": response.ensemble_mean_mobility_cm2_per_V_s,
        "ensemble_mean_r_squared_origin": response.ensemble_mean_r_squared_origin,
        "maximum_fractional_linearity_residual": response.maximum_fractional_linearity_residual,
        "maximum_even_to_odd_mean_ratio": response.maximum_even_to_odd_mean_ratio,
        "mobility_ci_excludes_zero": bool(response.mobility_ci.lower > 0.0 or response.mobility_ci.upper < 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--temperature-K", type=float, default=300.0)
    parser.add_argument("--gamma-u-per-fs", type=float, default=0.01)
    parser.add_argument("--gamma-v-per-fs", type=float, default=0.01)
    parser.add_argument("--field-magnitudes-mv-per-A", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--dt-fs", type=float, default=0.2)
    parser.add_argument("--final-time-fs", type=float, default=12000.0)
    parser.add_argument("--burn-in-fs", type=float, default=2000.0)
    parser.add_argument("--checkpoint-times-fs", nargs="+", type=float, default=[6000.0, 9000.0, 12000.0])
    parser.add_argument("--temperature-sample-interval-fs", type=float, default=20.0)
    parser.add_argument("--decoherence-interval-fs", type=float, default=180.0)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[20260905, 20260906, 20260907, 20260908, 20260909, 20260910, 20260911, 20260912],
    )
    parser.add_argument("--krylov-dimension", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    fields = np.asarray(args.field_magnitudes_mv_per_A, dtype=np.float64)
    if fields.size < 3 or np.any(fields <= 0.0) or np.unique(fields).size != fields.size:
        raise ValueError("TP2b requires at least three unique positive field magnitudes")
    seeds = [int(seed) for seed in args.seeds]
    if len(seeds) < 6 or len(set(seeds)) != len(seeds):
        raise ValueError("TP2b requires at least six unique independent seeds")
    checkpoints = sorted(float(value) for value in args.checkpoint_times_fs)
    if checkpoints[-1] != float(args.final_time_fs):
        raise ValueError("final_time_fs must be included as the final checkpoint")

    initial, parameters, static_metadata = _prepare_initial_state(args.size)
    bath = LangevinBath(args.temperature_K, args.gamma_u_per_fs, args.gamma_v_per_fs)

    runs: list[dict] = []
    lookup: dict[tuple[int, float, int], dict] = {}
    for seed in seeds:
        for magnitude in fields:
            for sign in (+1, -1):
                run = _run_trajectory(
                    initial,
                    parameters,
                    bath,
                    field_magnitude_mV_per_A=float(magnitude),
                    field_sign=sign,
                    seed=seed,
                    dt_fs=args.dt_fs,
                    final_time_fs=args.final_time_fs,
                    burn_in_fs=args.burn_in_fs,
                    checkpoint_times_fs=checkpoints,
                    temperature_sample_interval_fs=args.temperature_sample_interval_fs,
                    decoherence_interval_fs=args.decoherence_interval_fs,
                    krylov_dimension=args.krylov_dimension,
                )
                runs.append(run)
                lookup[(seed, float(magnitude), sign)] = run

    checkpoint_responses: list[dict] = []
    response_objects = []
    for checkpoint in checkpoints:
        key = f"{checkpoint:.6f}"
        plus = np.asarray([
            [lookup[(seed, float(field), +1)]["checkpoint_mean_velocity_x_A_per_fs"][key] for field in fields]
            for seed in seeds
        ], dtype=np.float64)
        minus = np.asarray([
            [lookup[(seed, float(field), -1)]["checkpoint_mean_velocity_x_A_per_fs"][key] for field in fields]
            for seed in seeds
        ], dtype=np.float64)
        response = analyze_paired_field_ensemble(fields, plus, minus)
        response_objects.append(response)
        checkpoint_responses.append({
            "checkpoint_time_fs": checkpoint,
            "postburn_window_fs": checkpoint - args.burn_in_fs,
            "response": _response_dict(response, fields),
        })

    final_response = response_objects[-1]
    low_mask = fields <= 1.0 + 1.0e-12
    if np.count_nonzero(low_mask) < 2:
        raise ValueError("TP2b requires at least two fields at or below 1 mV/A for low-field diagnostics")
    key_final = f"{checkpoints[-1]:.6f}"
    plus_final = np.asarray([
        [lookup[(seed, float(field), +1)]["checkpoint_mean_velocity_x_A_per_fs"][key_final] for field in fields]
        for seed in seeds
    ], dtype=np.float64)
    minus_final = np.asarray([
        [lookup[(seed, float(field), -1)]["checkpoint_mean_velocity_x_A_per_fs"][key_final] for field in fields]
        for seed in seeds
    ], dtype=np.float64)
    low_response = analyze_paired_field_ensemble(fields[low_mask], plus_final[:, low_mask], minus_final[:, low_mask])

    final_mu = float(final_response.ensemble_mean_mobility_cm2_per_V_s)
    previous_mu = float(response_objects[-2].ensemble_mean_mobility_cm2_per_V_s)
    late_change = abs(final_mu - previous_mu) / max(abs(final_mu), 1.0e-12)
    low_mu = float(low_response.ensemble_mean_mobility_cm2_per_V_s)
    low_vs_all = abs(low_mu - final_mu) / max(abs(low_mu), 1.0e-12)
    seed_mobilities = np.asarray(
        [value.electron_mobility_cm2_per_V_s for value in final_response.seed_responses],
        dtype=np.float64,
    )
    positive_seed_fraction = float(np.mean(seed_mobilities > 0.0))

    readiness_checks = {
        "mobility_95pct_ci_excludes_zero_positive": bool(final_response.mobility_ci.lower > 0.0),
        "ensemble_origin_r2_at_least_0p90": bool(final_response.ensemble_mean_r_squared_origin >= 0.90),
        "fractional_linearity_residual_at_most_0p25": bool(
            final_response.maximum_fractional_linearity_residual is not None
            and final_response.maximum_fractional_linearity_residual <= 0.25
        ),
        "even_to_odd_mean_ratio_at_most_0p50": bool(
            final_response.maximum_even_to_odd_mean_ratio is not None
            and final_response.maximum_even_to_odd_mean_ratio <= 0.50
        ),
        "late_checkpoint_mobility_change_at_most_0p25": bool(late_change <= 0.25),
        "low_field_vs_all_field_mobility_difference_at_most_0p25": bool(low_vs_all <= 0.25),
        "positive_seed_fraction_at_least_0p75": bool(positive_seed_fraction >= 0.75),
    }
    mobility_ready = all(readiness_checks.values())

    mean_temperature = float(np.mean([run["mean_lattice_temperature_K"] for run in runs]))
    max_residual = float(max(run["maximum_sampled_abs_generalized_balance_residual_eV"] for run in runs))
    max_norm = float(max(run["maximum_norm_error"] for run in runs))
    max_zero = float(max(run["maximum_zero_mode_mean"] for run in runs))
    max_work_difference = float(max(abs(run["field_work_displacement_difference_eV"]) for run in runs))
    max_power_identity = float(max(run["maximum_power_velocity_identity_error_eV_per_fs"] for run in runs))
    expected_count = len(seeds) * fields.size * 2
    numerical_checks = {
        "static_initial_state_converged": bool(static_metadata["static_converged"]),
        "trajectory_count": len(runs) == expected_count,
        "temperature": 240.0 <= mean_temperature <= 360.0,
        "full_energy_balance": max_residual < 1.0e-4,
        "electronic_norm": max_norm < 1.0e-10,
        "projected_zero_modes": max_zero < 1.0e-12,
        "event_count": all(run["idc_events"] == run["expected_idc_events"] for run in runs),
        "field_work_from_displacement": max_work_difference < 1.0e-10,
        "power_velocity_identity": max_power_identity < 1.0e-12,
    }
    numerical_pass = all(numerical_checks.values())

    payload = {
        "scope": "TP2b long-trajectory paired-field one-polaron convergence study",
        "size": args.size,
        "temperature_K": args.temperature_K,
        "gamma_u_per_fs": args.gamma_u_per_fs,
        "gamma_v_per_fs": args.gamma_v_per_fs,
        "field_magnitudes_mV_per_A": fields.tolist(),
        "dt_fs": args.dt_fs,
        "final_time_fs": args.final_time_fs,
        "burn_in_fs": args.burn_in_fs,
        "checkpoint_times_fs": checkpoints,
        "decoherence_scheme": "bm",
        "decoherence_interval_fs": args.decoherence_interval_fs,
        "decoherence_interval_material_calibrated": False,
        "field_material_calibrated": False,
        "krylov_dimension": args.krylov_dimension,
        "zero_mode_policy": "project",
        "paired_common_random_numbers": True,
        "static_initial_state": static_metadata,
        "seeds": seeds,
        "runs": runs,
        "checkpoint_responses": checkpoint_responses,
        "final_response": _response_dict(final_response, fields),
        "final_low_field_response_fields_mV_per_A": fields[low_mask].tolist(),
        "final_low_field_response": _response_dict(low_response, fields[low_mask]),
        "convergence_diagnostics": {
            "late_checkpoint_fractional_mobility_change": float(late_change),
            "low_vs_all_field_fractional_mobility_difference": float(low_vs_all),
            "positive_seed_fraction": positive_seed_fraction,
        },
        "aggregate_numerics": {
            "trajectory_count": len(runs),
            "mean_lattice_temperature_K": mean_temperature,
            "maximum_sampled_abs_generalized_balance_residual_eV": max_residual,
            "maximum_norm_error": max_norm,
            "maximum_zero_mode_mean": max_zero,
            "maximum_field_work_displacement_difference_eV": max_work_difference,
            "maximum_power_velocity_identity_error_eV_per_fs": max_power_identity,
        },
        "numerical_checks": numerical_checks,
        "numerical_pass": numerical_pass,
        "mobility_readiness_checks": readiness_checks,
        "mobility_ready": mobility_ready,
        "interpretation": (
            "TP2b is a convergence/readiness gate. A numerical PASS does not by itself establish a mobility. "
            "Only mobility_ready=True permits progression to a production mobility estimate at this control."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# TP2b paired-field convergence study",
        "",
        f"Control: {args.size}x{args.size}, T={args.temperature_K:g} K, dt={args.dt_fs:g} fs, "
        f"time={args.final_time_fs/1000:g} ps, burn={args.burn_in_fs/1000:g} ps, "
        f"{len(seeds)} seeds, IDC-BM t_d={args.decoherence_interval_fs:g} fs.",
        "",
        "## Temporal convergence",
        "",
        "| final time [ps] | post-burn window [ps] | mobility [cm2/Vs] | CI lower | CI upper | R2(origin) | max frac residual | even/odd |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in checkpoint_responses:
        r = item["response"]
        ci = r["seed_level_mobility_ci_cm2_per_V_s"]
        lines.append(
            f"| {item['checkpoint_time_fs']/1000:.1f} | {item['postburn_window_fs']/1000:.1f} | "
            f"{r['ensemble_mean_mobility_cm2_per_V_s']:.6g} | {ci['lower']:.6g} | {ci['upper']:.6g} | "
            f"{r['ensemble_mean_r_squared_origin']:.4f} | {r['maximum_fractional_linearity_residual']:.4f} | "
            f"{r['maximum_even_to_odd_mean_ratio']:.4f} |"
        )
    lines += [
        "",
        "## Numerical checks",
        "",
        *[f"- {name}: {'PASS' if value else 'FAIL'}" for name, value in numerical_checks.items()],
        "",
        "## Mobility-readiness checks",
        "",
        *[f"- {name}: {'PASS' if value else 'NOT CONVERGED'}" for name, value in readiness_checks.items()],
        "",
        f"Numerical status: {'PASS' if numerical_pass else 'FAIL'}",
        f"Mobility readiness: {'PASS' if mobility_ready else 'NOT CONVERGED'}",
        "",
        "The field and decoherence interval remain numerical controls, not calibrated material parameters.",
    ]
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    if not numerical_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
