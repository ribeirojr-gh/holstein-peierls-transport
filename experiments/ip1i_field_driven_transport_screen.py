#!/usr/bin/env python3
"""IP1i self-consistent field-driven moving-polaron screen.

This stage composes the validated deterministic D3 field dynamics, TP1
probability-current displacement, and the persistent residence-site tracker.  It
selects field/anisotropy controls suitable for a later phonon-wake experiment.
It does not infer mobility, a hopping rate, an activation energy, or a material
field threshold.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from holstein_peierls.dynamics.coupled import CoupledEhrenfestState
from holstein_peierls.dynamics.driven import coupled_field_verlet_step, field_dynamic_total_energy
from holstein_peierls.dynamics.ehrenfest import LatticeVelocity
from holstein_peierls.dynamics.field import UniformElectricField2D
from holstein_peierls.dynamics.hopping_observables import PersistentSiteTracker
from holstein_peierls.dynamics.numerical_validation import (
    extensive_energy_balance_passes,
    size_scaled_energy_balance_tolerance_eV,
)
from holstein_peierls.dynamics.thermal import project_inter_molecular_zero_modes, zero_mode_means
from holstein_peierls.dynamics.transport_observables import (
    transport_kinematics,
    trapezoidal_displacement_increment,
)
from holstein_peierls.electronic import solve_ground_state
from holstein_peierls.parameters import StaticPolaronParameters
from holstein_peierls.polaron import solve_static_polaron


PERSISTENCE_FS = 50.0
LATTICE_SPACING_A = 3.0


def _integer_stride(interval_fs: float, dt_fs: float, name: str) -> int:
    ratio = float(interval_fs) / float(dt_fs)
    value = int(round(ratio))
    if value <= 0 or not np.isclose(value, ratio, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"{name} must be a positive integer multiple of dt")
    return value


def _prepare_initial_state(size: int, anisotropy_ratio: float):
    center = (size // 2) * size + (size // 2) + 1
    base = StaticPolaronParameters(nx=size, ny=size, polaron_position=center)
    parameters = replace(base, j0y=base.j0x * float(anisotropy_ratio))
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
    initial = project_inter_molecular_zero_modes(raw)
    return initial, parameters, {
        "converged": bool(static.diagnostics.converged),
        "iterations": int(static.diagnostics.iterations),
        "total_energy_eV": float(static.total_energy),
        "formation_energy_eV": float(static.formation_energy),
        "ipr": float(static.ipr),
    }


def _run_condition(
    *,
    size: int,
    anisotropy_ratio: float,
    field_mV_per_A: float,
    dt_fs: float,
    final_time_fs: float,
    tracker_sample_interval_fs: float,
    energy_sample_interval_fs: float,
    krylov_dimension: int,
) -> dict:
    initial, parameters, static = _prepare_initial_state(size, anisotropy_ratio)
    field = UniformElectricField2D.from_millivolt_per_angstrom(
        float(field_mV_per_A),
        angle_radians=0.0,
        ax_angstrom=LATTICE_SPACING_A,
        ay_angstrom=LATTICE_SPACING_A,
    )
    steps = _integer_stride(final_time_fs, dt_fs, "final_time_fs")
    tracker_stride = _integer_stride(tracker_sample_interval_fs, dt_fs, "tracker_sample_interval_fs")
    energy_stride = _integer_stride(energy_sample_interval_fs, dt_fs, "energy_sample_interval_fs")
    persistence_samples = _integer_stride(PERSISTENCE_FS, tracker_sample_interval_fs, "persistence_fs")

    tracker = PersistentSiteTracker(
        parameters.nx,
        parameters.ny,
        persistence_samples=persistence_samples,
        minimum_max_population=0.10,
        minimum_dominance_margin=0.02,
    )
    current = CoupledEhrenfestState(
        initial.lattice.copy(),
        initial.velocity.copy(),
        initial.electronic_state.copy(),
    )
    initial_energy = field_dynamic_total_energy(
        current.lattice,
        current.velocity,
        parameters,
        current.electronic_state,
        field,
        0.0,
    ).total
    initial_norm = float(np.linalg.norm(current.electronic_state))

    displacement_x_A = 0.0
    displacement_y_A = 0.0
    field_work_eV = 0.0
    max_balance = 0.0
    max_norm_error = 0.0
    max_zero_mode = 0.0
    h_eval = 0
    h_apply = 0
    events: list[dict] = []
    start = perf_counter()

    # Initialize the residence tracker at t=0.
    p0 = np.abs(current.electronic_state) ** 2
    tracker.update(p0 / float(np.sum(p0)), 0.0)

    for step in range(steps):
        time_fs = step * dt_fs
        old_kin = transport_kinematics(
            current.lattice,
            parameters,
            current.electronic_state,
            field=field,
            time_fs=time_fs,
        )
        current, work, evaluations, applications = coupled_field_verlet_step(
            current,
            parameters,
            field,
            time_fs,
            dt_fs,
            electronic_method="cfm4_lanczos",
            krylov_dimension=krylov_dimension,
        )
        new_time_fs = (step + 1) * dt_fs
        new_kin = transport_kinematics(
            current.lattice,
            parameters,
            current.electronic_state,
            field=field,
            time_fs=new_time_fs,
        )
        dx, dy = trapezoidal_displacement_increment(old_kin, new_kin, dt_fs)
        displacement_x_A += float(dx)
        displacement_y_A += float(dy)
        field_work_eV += float(work)
        h_eval += int(evaluations)
        h_apply += int(applications)

        max_norm_error = max(
            max_norm_error,
            abs(float(np.linalg.norm(current.electronic_state)) - initial_norm),
        )
        max_zero_mode = max(
            max_zero_mode,
            max(abs(value) for value in zero_mode_means(current).values()),
        )

        if (step + 1) % tracker_stride == 0:
            populations = np.abs(current.electronic_state) ** 2
            populations /= float(np.sum(populations))
            event = tracker.update(populations, new_time_fs)
            if event is not None:
                events.append(
                    {
                        "source_site": int(event.source_site),
                        "target_site": int(event.target_site),
                        "transition_start_time_fs": float(event.transition_start_time_fs),
                        "accepted_time_fs": float(event.accepted_time_fs),
                        "dx_sites": int(event.dx_sites),
                        "dy_sites": int(event.dy_sites),
                        "is_nearest_neighbor": bool(event.is_nearest_neighbor),
                        "direction": str(event.direction),
                    }
                )

        if (step + 1) % energy_stride == 0 or step + 1 == steps:
            energy = field_dynamic_total_energy(
                current.lattice,
                current.velocity,
                parameters,
                current.electronic_state,
                field,
                new_time_fs,
            ).total
            max_balance = max(max_balance, abs((energy - initial_energy) - field_work_eV))

    elapsed = perf_counter() - start
    nearest = [event for event in events if event["is_nearest_neighbor"]]
    x_events = [event for event in nearest if abs(int(event["dx_sites"])) == 1]
    event_net_dx_sites = int(sum(int(event["dx_sites"]) for event in nearest))
    event_net_dy_sites = int(sum(int(event["dy_sites"]) for event in nearest))
    x_fraction = float(len(x_events) / len(nearest)) if nearest else 0.0
    starts = np.asarray([float(event["transition_start_time_fs"]) for event in nearest], dtype=np.float64)
    gaps = np.diff(starts) if starts.size >= 2 else np.asarray([], dtype=np.float64)
    quiet_intervals = gaps.tolist()
    if starts.size:
        quiet_intervals.append(float(final_time_fs - starts[-1]))
    longest_quiet = float(max(quiet_intervals)) if quiet_intervals else float(final_time_fs)
    median_gap = float(np.median(gaps)) if gaps.size else None
    tp1_sign = int(np.sign(displacement_x_A))
    event_sign = int(np.sign(event_net_dx_sites))
    sign_agreement = bool(tp1_sign != 0 and event_sign != 0 and tp1_sign == event_sign)
    qualified = bool(
        len(nearest) >= 2
        and abs(displacement_x_A) >= 2.0 * LATTICE_SPACING_A
        and x_fraction >= 0.70
        and sign_agreement
    )
    tolerance = size_scaled_energy_balance_tolerance_eV(parameters.n_sites)
    numerical_checks = {
        "static_relaxation_converged": bool(static["converged"]),
        "complete_requested_steps": True,
        "energy_work_balance": bool(extensive_energy_balance_passes(max_balance, parameters.n_sites)),
        "electronic_norm": bool(max_norm_error < 1.0e-10),
        "projected_zero_modes": bool(max_zero_mode < 1.0e-10),
        "finite_transport_diagnostics": bool(
            np.isfinite(displacement_x_A)
            and np.isfinite(displacement_y_A)
            and np.isfinite(max_balance)
        ),
    }
    return {
        "anisotropy_ratio": float(anisotropy_ratio),
        "field_mV_per_A": float(field_mV_per_A),
        "static": static,
        "steps": int(steps),
        "elapsed_seconds": float(elapsed),
        "field_work_eV": float(field_work_eV),
        "maximum_energy_work_residual_eV": float(max_balance),
        "energy_work_tolerance_eV": float(tolerance),
        "maximum_norm_error": float(max_norm_error),
        "maximum_zero_mode_mean": float(max_zero_mode),
        "tp1_displacement_x_A": float(displacement_x_A),
        "tp1_displacement_y_A": float(displacement_y_A),
        "tp1_mean_velocity_x_A_per_fs": float(displacement_x_A / final_time_fs),
        "persistent_events": events,
        "nearest_neighbor_event_count": int(len(nearest)),
        "x_nearest_neighbor_event_count": int(len(x_events)),
        "x_nearest_neighbor_fraction": x_fraction,
        "persistent_net_dx_sites": event_net_dx_sites,
        "persistent_net_dy_sites": event_net_dy_sites,
        "tp1_persistent_x_sign_agreement": sign_agreement,
        "first_nearest_neighbor_event": nearest[0] if nearest else None,
        "median_nn_event_gap_fs": median_gap,
        "longest_postevent_quiet_interval_fs": longest_quiet,
        "transport_qualified": qualified,
        "numerical_checks": numerical_checks,
        "numerical_pass": bool(all(numerical_checks.values())),
        "hamiltonian_evaluations": int(h_eval),
        "hamiltonian_applications": int(h_apply),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=40)
    parser.add_argument("--ratios", nargs="+", type=float, default=[1.0, 0.15])
    parser.add_argument("--fields-mv-per-A", nargs="+", type=float, default=[2.0, 5.0, 10.0])
    parser.add_argument("--dt-fs", type=float, default=0.2)
    parser.add_argument("--final-time-fs", type=float, default=5000.0)
    parser.add_argument("--tracker-sample-interval-fs", type=float, default=2.0)
    parser.add_argument("--energy-sample-interval-fs", type=float, default=10.0)
    parser.add_argument("--krylov-dimension", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    conditions: list[dict] = []
    for ratio in args.ratios:
        for field in args.fields_mv_per_A:
            conditions.append(
                _run_condition(
                    size=args.size,
                    anisotropy_ratio=ratio,
                    field_mV_per_A=field,
                    dt_fs=args.dt_fs,
                    final_time_fs=args.final_time_fs,
                    tracker_sample_interval_fs=args.tracker_sample_interval_fs,
                    energy_sample_interval_fs=args.energy_sample_interval_fs,
                    krylov_dimension=args.krylov_dimension,
                )
            )

    numerical_pass = bool(all(item["numerical_pass"] for item in conditions))
    qualified = [item for item in conditions if item["transport_qualified"]]
    payload = {
        "scope": "IP1i deterministic zero-temperature field-driven transport screen before natural-wake analysis; no mobility/rate/threshold claim",
        "size": int(args.size),
        "ratios": [float(v) for v in args.ratios],
        "fields_mV_per_A": [float(v) for v in args.fields_mv_per_A],
        "field_direction": "+x",
        "field_material_calibrated": False,
        "temperature_K": 0.0,
        "thermostat": None,
        "IDC": None,
        "dt_fs": float(args.dt_fs),
        "final_time_fs": float(args.final_time_fs),
        "tracker_sample_interval_fs": float(args.tracker_sample_interval_fs),
        "persistence_fs": PERSISTENCE_FS,
        "conditions": conditions,
        "numerical_pass": numerical_pass,
        "qualified_condition_count": int(len(qualified)),
        "interpretation_guard": {
            "transport_qualified_is_protocol_selection_not_transport_coefficient": True,
            "fields_are_numerical_controls_not_material_thresholds": True,
            "wake_analysis_deferred_until_after_screen": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# IP1i field-driven moving-polaron screen",
        "",
        f"Cell {args.size}x{args.size}; deterministic T=0; dt={args.dt_fs:g} fs; final={args.final_time_fs/1000:g} ps; field +x; no thermostat or IDC.",
        "",
        "| J0y/J0x | E [mV/A] | TP1 dx [A] | NN events | x-event fraction | event net dx [sites] | median gap [fs] | longest quiet [fs] | max |dE-W| [eV] | qualified |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in conditions:
        gap = item["median_nn_event_gap_fs"]
        lines.append(
            f"| {item['anisotropy_ratio']:.2f} | {item['field_mV_per_A']:.1f} | "
            f"{item['tp1_displacement_x_A']:.4f} | {item['nearest_neighbor_event_count']} | "
            f"{item['x_nearest_neighbor_fraction']:.3f} | {item['persistent_net_dx_sites']} | "
            f"{'NA' if gap is None else f'{gap:.1f}'} | {item['longest_postevent_quiet_interval_fs']:.1f} | "
            f"{item['maximum_energy_work_residual_eV']:.3e} | "
            f"{'YES' if item['transport_qualified'] else 'NO'} |"
        )
    lines.extend(
        [
            "",
            f"Numerical status: {'PASS' if numerical_pass else 'FAIL'}",
            f"Transport-qualified controls: {len(qualified)}",
            "",
            "Qualification selects a self-consistent moving control for the next wake experiment. It is not a mobility, rate, field-threshold, or material prediction.",
        ]
    )
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if not numerical_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
