#!/usr/bin/env python3
"""TP1 periodic transport-observable benchmark.

The benchmark reuses the TP0 finite-temperature + field + IDC control but adds
PBC-safe transport kinematics from oriented bond probability currents.  It
validates the continuity/current definitions and their exact consistency with
the already validated field-work bookkeeping.  It does not estimate mobility or
claim a steady transport regime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from holstein_peierls.dynamics.coupled import CoupledEhrenfestState
from holstein_peierls.dynamics.driven import field_dynamic_total_energy, field_power
from holstein_peierls.dynamics.driven_thermal_decoherence import (
    apply_field_idc_to_coupled_state,
)
from holstein_peierls.dynamics.ehrenfest import LatticeVelocity
from holstein_peierls.dynamics.field import UniformElectricField2D
from holstein_peierls.dynamics.langevin import LangevinBath, kinetic_temperature_K
from holstein_peierls.dynamics.thermal import (
    coupled_baoab_step,
    project_inter_molecular_zero_modes,
    thermostatted_kinetic_degrees_of_freedom,
    zero_mode_means,
)
from holstein_peierls.dynamics.thermal_decoherence import decoherence_interval_steps
from holstein_peierls.dynamics.transport_observables import (
    field_power_from_particle_velocity,
    periodic_center_of_probability,
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


def _run_seed(
    initial: CoupledEhrenfestState,
    parameters: StaticPolaronParameters,
    field: UniformElectricField2D,
    bath: LangevinBath,
    *,
    seed: int,
    dt_fs: float,
    final_time_fs: float,
    burn_in_fs: float,
    sample_interval_fs: float,
    decoherence_interval_fs: float,
    krylov_dimension: int,
) -> dict:
    steps = int(round(final_time_fs / dt_fs))
    burn_steps = int(round(burn_in_fs / dt_fs))
    sample_stride = int(round(sample_interval_fs / dt_fs))
    event_stride = decoherence_interval_steps(decoherence_interval_fs, dt_fs)
    if steps <= 0 or not np.isclose(steps * dt_fs, final_time_fs):
        raise ValueError("final_time_fs must be an integer multiple of dt_fs")
    if burn_steps < 0 or burn_steps >= steps:
        raise ValueError("burn_in_fs must lie in [0, final_time_fs)")
    if sample_stride <= 0 or not np.isclose(sample_stride * dt_fs, sample_interval_fs):
        raise ValueError("sample_interval_fs must be an integer multiple of dt_fs")

    current = CoupledEhrenfestState(
        initial.lattice.copy(),
        initial.velocity.copy(),
        initial.electronic_state.copy(),
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
    field_work_from_velocity = 0.0
    displacement_x = 0.0
    displacement_y = 0.0
    postburn_displacement_x = 0.0
    postburn_displacement_y = 0.0
    event_count = 0
    temperatures: list[float] = []
    residuals: list[float] = []
    x_resultants: list[float] = []
    y_resultants: list[float] = []
    max_power_velocity_error = 0.0
    max_norm_error = 0.0
    max_zero_mode = 0.0
    h_eval = 0
    h_apply = 0
    start = perf_counter()

    for step in range(steps):
        time_fs = step * dt_fs
        old_kinematics = transport_kinematics(
            current.lattice,
            parameters,
            current.electronic_state,
            field=field,
            time_fs=time_fs,
        )
        direct_old_power = field_power(
            current.lattice,
            parameters,
            current.electronic_state,
            field,
            time_fs,
        )
        max_power_velocity_error = max(
            max_power_velocity_error,
            abs(
                direct_old_power
                - field_power_from_particle_velocity(field, old_kinematics)
            ),
        )

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
        coherent_end = thermal.state
        new_time_fs = (step + 1) * dt_fs
        new_kinematics = transport_kinematics(
            coherent_end.lattice,
            parameters,
            coherent_end.electronic_state,
            field=field,
            time_fs=new_time_fs,
        )
        direct_new_power = field_power(
            coherent_end.lattice,
            parameters,
            coherent_end.electronic_state,
            field,
            new_time_fs,
        )
        max_power_velocity_error = max(
            max_power_velocity_error,
            abs(
                direct_new_power
                - field_power_from_particle_velocity(field, new_kinematics)
            ),
        )

        dx, dy = trapezoidal_displacement_increment(
            old_kinematics, new_kinematics, dt_fs
        )
        displacement_x += dx
        displacement_y += dy
        if step >= burn_steps:
            postburn_displacement_x += dx
            postburn_displacement_y += dy
        work_from_velocity_increment = (
            -field.ex_v_per_angstrom * dx - field.ey_v_per_angstrom * dy
        )
        field_work_from_velocity += work_from_velocity_increment

        current = coherent_end
        lattice_heat += thermal.bath_heat_eV
        field_work += thermal.field_work_eV
        h_eval += thermal.hamiltonian_evaluations
        h_apply += thermal.hamiltonian_applications

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

        if (step + 1) % sample_stride == 0 or step + 1 == steps:
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
            center = periodic_center_of_probability(
                current.electronic_state,
                parameters,
                ax_angstrom=field.ax_angstrom,
                ay_angstrom=field.ay_angstrom,
            )
            x_resultants.append(center.x_resultant)
            y_resultants.append(center.y_resultant)
            if step + 1 > burn_steps:
                temperatures.append(
                    kinetic_temperature_K(
                        current.velocity,
                        parameters,
                        degrees_of_freedom=dof,
                    )
                )

    if not temperatures or not residuals:
        raise RuntimeError("TP1 produced no post-burn-in samples")
    postburn_time_fs = final_time_fs - burn_in_fs
    return {
        "seed": int(seed),
        "steps": int(steps),
        "elapsed_seconds": float(perf_counter() - start),
        "idc_events": int(event_count),
        "expected_idc_events": int(steps // event_stride),
        "mean_lattice_temperature_K": float(np.mean(temperatures)),
        "field_work_eV": float(field_work),
        "field_work_from_velocity_eV": float(field_work_from_velocity),
        "field_work_velocity_difference_eV": float(field_work_from_velocity - field_work),
        "maximum_power_velocity_identity_error_eV_per_fs": float(
            max_power_velocity_error
        ),
        "total_displacement_x_A": float(displacement_x),
        "total_displacement_y_A": float(displacement_y),
        "postburn_displacement_x_A": float(postburn_displacement_x),
        "postburn_displacement_y_A": float(postburn_displacement_y),
        "postburn_mean_velocity_x_A_per_fs": float(
            postburn_displacement_x / postburn_time_fs
        ),
        "postburn_mean_velocity_y_A_per_fs": float(
            postburn_displacement_y / postburn_time_fs
        ),
        "mean_x_circular_resultant": float(np.mean(x_resultants)),
        "mean_y_circular_resultant": float(np.mean(y_resultants)),
        "maximum_sampled_abs_generalized_balance_residual_eV": float(
            np.max(np.abs(residuals))
        ),
        "maximum_norm_error": float(max_norm_error),
        "maximum_zero_mode_mean": float(max_zero_mode),
        "lattice_bath_heat_eV": float(lattice_heat),
        "electronic_environment_exchange_eV": float(electronic_exchange),
        "hamiltonian_evaluations": int(h_eval),
        "hamiltonian_applications": int(h_apply),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--temperature-K", type=float, default=300.0)
    parser.add_argument("--gamma-u-per-fs", type=float, default=0.01)
    parser.add_argument("--gamma-v-per-fs", type=float, default=0.01)
    parser.add_argument("--field-mv-per-A", type=float, default=2.0)
    parser.add_argument("--dt-fs", type=float, default=0.2)
    parser.add_argument("--final-time-fs", type=float, default=4000.0)
    parser.add_argument("--burn-in-fs", type=float, default=1000.0)
    parser.add_argument("--sample-interval-fs", type=float, default=10.0)
    parser.add_argument("--decoherence-interval-fs", type=float, default=180.0)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[20260905, 20260906, 20260907, 20260908],
    )
    parser.add_argument("--krylov-dimension", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    initial, parameters, static_metadata = _prepare_initial_state(args.size)
    field = UniformElectricField2D.from_millivolt_per_angstrom(
        args.field_mv_per_A,
        angle_radians=0.0,
        ax_angstrom=3.0,
        ay_angstrom=3.0,
    )
    bath = LangevinBath(
        args.temperature_K,
        args.gamma_u_per_fs,
        args.gamma_v_per_fs,
    )
    runs = [
        _run_seed(
            initial,
            parameters,
            field,
            bath,
            seed=seed,
            dt_fs=args.dt_fs,
            final_time_fs=args.final_time_fs,
            burn_in_fs=args.burn_in_fs,
            sample_interval_fs=args.sample_interval_fs,
            decoherence_interval_fs=args.decoherence_interval_fs,
            krylov_dimension=args.krylov_dimension,
        )
        for seed in args.seeds
    ]

    mean_temperature = float(np.mean([r["mean_lattice_temperature_K"] for r in runs]))
    max_work_difference = float(
        max(abs(r["field_work_velocity_difference_eV"]) for r in runs)
    )
    max_power_identity = float(
        max(r["maximum_power_velocity_identity_error_eV_per_fs"] for r in runs)
    )
    max_residual = float(
        max(r["maximum_sampled_abs_generalized_balance_residual_eV"] for r in runs)
    )
    max_norm = float(max(r["maximum_norm_error"] for r in runs))
    max_zero_mode = float(max(r["maximum_zero_mode_mean"] for r in runs))
    all_displacements = [
        value
        for run in runs
        for value in (run["total_displacement_x_A"], run["total_displacement_y_A"])
    ]
    checks = {
        "temperature": 240.0 <= mean_temperature <= 360.0,
        "field_work_from_velocity": max_work_difference < 1.0e-10,
        "instantaneous_power_velocity_identity": max_power_identity < 1.0e-12,
        "full_energy_balance": max_residual < 1.0e-4,
        "electronic_norm": max_norm < 1.0e-10,
        "projected_zero_modes": max_zero_mode < 1.0e-12,
        "event_count": all(r["idc_events"] == r["expected_idc_events"] for r in runs),
        "finite_unwrapped_displacement": all(np.isfinite(value) for value in all_displacements),
    }
    closure_pass = all(checks.values())
    payload = {
        "scope": "TP1 PBC-safe one-polaron bond-current and displacement observables",
        "size": args.size,
        "temperature_K": args.temperature_K,
        "gamma_u_per_fs": args.gamma_u_per_fs,
        "gamma_v_per_fs": args.gamma_v_per_fs,
        "field_mv_per_A": args.field_mv_per_A,
        "field_direction": "+x",
        "dt_fs": args.dt_fs,
        "final_time_fs": args.final_time_fs,
        "burn_in_fs": args.burn_in_fs,
        "sample_interval_fs": args.sample_interval_fs,
        "decoherence_scheme": "bm",
        "decoherence_interval_fs": args.decoherence_interval_fs,
        "decoherence_interval_material_calibrated": False,
        "field_material_calibrated": False,
        "krylov_dimension": args.krylov_dimension,
        "zero_mode_policy": "project",
        "static_initial_state": static_metadata,
        "runs": runs,
        "aggregate": {
            "trajectory_count": len(runs),
            "mean_lattice_temperature_K": mean_temperature,
            "maximum_field_work_velocity_difference_eV": max_work_difference,
            "maximum_power_velocity_identity_error_eV_per_fs": max_power_identity,
            "maximum_sampled_abs_generalized_balance_residual_eV": max_residual,
            "maximum_norm_error": max_norm,
            "maximum_zero_mode_mean": max_zero_mode,
            "ensemble_mean_postburn_velocity_x_A_per_fs": float(
                np.mean([r["postburn_mean_velocity_x_A_per_fs"] for r in runs])
            ),
            "ensemble_std_postburn_velocity_x_A_per_fs": float(
                np.std([r["postburn_mean_velocity_x_A_per_fs"] for r in runs])
            ),
            "ensemble_mean_postburn_velocity_y_A_per_fs": float(
                np.mean([r["postburn_mean_velocity_y_A_per_fs"] for r in runs])
            ),
            "ensemble_std_postburn_velocity_y_A_per_fs": float(
                np.std([r["postburn_mean_velocity_y_A_per_fs"] for r in runs])
            ),
        },
        "closure_checks": checks,
        "closure_pass": closure_pass,
        "interpretation": (
            "TP1 validates PBC-safe current/displacement observables and their exact "
            "consistency with field work. The finite-time drift values are diagnostics, "
            "not mobility estimates or steady-state claims."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# TP1 periodic transport-observable benchmark",
        "",
        f"Control: {args.size}x{args.size}, T={args.temperature_K:g} K, "
        f"E_x={args.field_mv_per_A:g} mV/A, dt={args.dt_fs:g} fs, "
        f"time={args.final_time_fs/1000:g} ps, IDC-BM t_d={args.decoherence_interval_fs:g} fs.",
        "",
        "| seed | dx total [A] | dy total [A] | <vx> post-burn [A/fs] | W_field [eV] | W_from_v [eV] | |difference| [eV] |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        lines.append(
            f"| {run['seed']} | {run['total_displacement_x_A']:.6e} | "
            f"{run['total_displacement_y_A']:.6e} | "
            f"{run['postburn_mean_velocity_x_A_per_fs']:.6e} | "
            f"{run['field_work_eV']:.6e} | {run['field_work_from_velocity_eV']:.6e} | "
            f"{abs(run['field_work_velocity_difference_eV']):.3e} |"
        )
    lines += ["", "## Pre-registered closure", ""]
    lines += [f"- {name}: {'PASS' if value else 'FAIL'}" for name, value in checks.items()]
    lines += [
        "",
        f"Overall: {'PASS' if closure_pass else 'FAIL'}",
        "",
        "The recorded finite-time velocities are not mobility estimates.",
    ]
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    if not closure_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
