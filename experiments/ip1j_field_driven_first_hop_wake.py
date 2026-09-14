#!/usr/bin/env python3
"""IP1j event-conditioned lattice wake around the first natural field-driven hop.

The protocol reruns only the 40x40, 10 mV/A isotropic and anisotropic controls
selected by IP1i, stores the harmonic intermolecular x-current field every 2 fs,
and analyzes the first persistent nearest-neighbor x relocation in a fixed
laboratory frame centered on that event source.

The event direction is defined as +s, irrespective of the laboratory sign. A
backward packet therefore has v_phonon * v_carrier < 0 by construction. The
analysis uses a pre-event flux baseline and stops before the next persistent
nearest-neighbor relocation. It is not a mobility, hopping-rate, field-threshold
or calibrated material-phonon calculation.
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
from holstein_peierls.dynamics.event_wake import (
    baseline_subtracted_profiles,
    event_wake_window,
    positive_energy_directionality,
)
from holstein_peierls.dynamics.field import UniformElectricField2D
from holstein_peierls.dynamics.hopping_observables import PersistentSiteTracker
from holstein_peierls.dynamics.numerical_validation import (
    extensive_energy_balance_passes,
    size_scaled_energy_balance_tolerance_eV,
)
from holstein_peierls.dynamics.phonon_recurrence import intermolecular_recurrence_scales
from holstein_peierls.dynamics.phonon_wake import (
    event_aligned_coordinates,
    intermolecular_energy_flux,
    longitudinal_profile,
)
from holstein_peierls.dynamics.thermal import project_inter_molecular_zero_modes, zero_mode_means
from holstein_peierls.dynamics.transport_observables import (
    transport_kinematics,
    trapezoidal_displacement_increment,
)
from holstein_peierls.dynamics.wavepacket_flux import (
    integrated_outward_energy_eV,
    normalized_flux_delay,
    outward_boundary_flux_series,
)
from holstein_peierls.electronic import solve_ground_state
from holstein_peierls.parameters import StaticPolaronParameters
from holstein_peierls.polaron import solve_static_polaron


PERSISTENCE_FS = 50.0
LATTICE_SPACING_A = 3.0
TRANSVERSE_HALF_WIDTH_SITES = 3
BASELINE_START_OFFSET_FS = -500.0
BASELINE_END_OFFSET_FS = -100.0
MAX_POST_FS = 1500.0
NEXT_EVENT_GUARD_FS = 50.0
LAG_MIN_FS = 300.0
LAG_MAX_FS = 1000.0


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


def _event_record(event) -> dict:
    return {
        "source_site": int(event.source_site),
        "target_site": int(event.target_site),
        "transition_start_time_fs": float(event.transition_start_time_fs),
        "accepted_time_fs": float(event.accepted_time_fs),
        "dx_sites": int(event.dx_sites),
        "dy_sites": int(event.dy_sites),
        "is_nearest_neighbor": bool(event.is_nearest_neighbor),
        "direction": str(event.direction),
    }


def _boundary_metrics(
    times_rel_fs: np.ndarray,
    profiles: np.ndarray,
    s_axis: np.ndarray,
    distance: int,
) -> dict:
    backward = outward_boundary_flux_series(profiles, s_axis, distance, side="backward")
    forward = outward_boundary_flux_series(profiles, s_axis, distance, side="forward")
    back_positive = integrated_outward_energy_eV(times_rel_fs, backward, positive_only=True)
    front_positive = integrated_outward_energy_eV(times_rel_fs, forward, positive_only=True)
    return {
        "distance_sites": int(distance),
        "backward_positive_outward_energy_eV": float(back_positive),
        "forward_positive_outward_energy_eV": float(front_positive),
        "backward_net_outward_energy_eV": float(
            integrated_outward_energy_eV(times_rel_fs, backward, positive_only=False)
        ),
        "forward_net_outward_energy_eV": float(
            integrated_outward_energy_eV(times_rel_fs, forward, positive_only=False)
        ),
        "positive_energy_directionality": positive_energy_directionality(
            back_positive, front_positive
        ),
        "backward_peak_outward_flux_eV_per_fs": float(np.max(backward)),
        "forward_peak_outward_flux_eV_per_fs": float(np.max(forward)),
    }


def _delay_metrics(
    times_rel_fs: np.ndarray,
    profiles: np.ndarray,
    s_axis: np.ndarray,
    side: str,
    vmax_sites_per_ps: float,
) -> dict:
    first = outward_boundary_flux_series(profiles, s_axis, 1, side=side)
    second = outward_boundary_flux_series(profiles, s_axis, 2, side=side)
    delay = normalized_flux_delay(
        times_rel_fs,
        first,
        second,
        lag_min_fs=LAG_MIN_FS,
        lag_max_fs=LAG_MAX_FS,
    )
    validated = bool(
        delay.correlation >= 0.80
        and delay.speed_sites_per_ps <= 1.05 * float(vmax_sites_per_ps)
    )
    return {
        "side": side,
        "from_distance_sites": 1,
        "to_distance_sites": 2,
        "lag_fs": float(delay.lag_fs),
        "speed_sites_per_ps": float(delay.speed_sites_per_ps),
        "correlation": float(delay.correlation),
        "within_harmonic_vmax": bool(
            delay.speed_sites_per_ps <= 1.05 * float(vmax_sites_per_ps)
        ),
        "high_correlation": bool(delay.correlation >= 0.80),
        "validated_packet": validated,
    }


def _run_condition(
    *,
    size: int,
    anisotropy_ratio: float,
    field_mV_per_A: float,
    dt_fs: float,
    final_time_fs: float,
    sample_interval_fs: float,
    energy_sample_interval_fs: float,
    krylov_dimension: int,
    vmax_sites_per_ps: float,
) -> tuple[dict, dict[str, np.ndarray]]:
    initial, parameters, static = _prepare_initial_state(size, anisotropy_ratio)
    field = UniformElectricField2D.from_millivolt_per_angstrom(
        float(field_mV_per_A),
        angle_radians=0.0,
        ax_angstrom=LATTICE_SPACING_A,
        ay_angstrom=LATTICE_SPACING_A,
    )
    steps = _integer_stride(final_time_fs, dt_fs, "final_time_fs")
    sample_stride = _integer_stride(sample_interval_fs, dt_fs, "sample_interval_fs")
    energy_stride = _integer_stride(energy_sample_interval_fs, dt_fs, "energy_sample_interval_fs")
    persistence_samples = _integer_stride(PERSISTENCE_FS, sample_interval_fs, "persistence_fs")

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
    sample_times: list[float] = [0.0]
    jx_frames: list[np.ndarray] = [
        intermolecular_energy_flux(current.lattice, current.velocity, parameters).jx.astype(
            np.float32, copy=True
        )
    ]

    p0 = np.abs(current.electronic_state) ** 2
    tracker.update(p0 / float(np.sum(p0)), 0.0)
    start = perf_counter()

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

        if (step + 1) % sample_stride == 0:
            populations = np.abs(current.electronic_state) ** 2
            populations /= float(np.sum(populations))
            event = tracker.update(populations, new_time_fs)
            if event is not None:
                events.append(_event_record(event))
            flux = intermolecular_energy_flux(current.lattice, current.velocity, parameters)
            sample_times.append(float(new_time_fs))
            jx_frames.append(np.asarray(flux.jx, dtype=np.float32))

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
    times = np.asarray(sample_times, dtype=np.float64)
    jx = np.stack(jx_frames, axis=0)
    nearest_x = [
        event
        for event in events
        if event["is_nearest_neighbor"]
        and abs(int(event["dx_sites"])) == 1
        and int(event["dy_sites"]) == 0
    ]

    analysis = None
    profile_payload: dict[str, np.ndarray] = {}
    if nearest_x:
        first = nearest_x[0]
        next_start = (
            float(nearest_x[1]["transition_start_time_fs"])
            if len(nearest_x) >= 2
            else None
        )
        window = event_wake_window(
            times,
            float(first["transition_start_time_fs"]),
            max_post_fs=MAX_POST_FS,
            baseline_start_offset_fs=BASELINE_START_OFFSET_FS,
            baseline_end_offset_fs=BASELINE_END_OFFSET_FS,
            next_event_time_fs=next_start,
            next_event_guard_fs=NEXT_EVENT_GUARD_FS,
        )
        coordinates = event_aligned_coordinates(
            int(first["source_site"]),
            int(first["target_site"]),
            parameters.nx,
            parameters.ny,
        )
        profiles: list[np.ndarray] = []
        s_axis = None
        direction_sign = int(first["dx_sites"])
        for frame in jx:
            axis, profile = longitudinal_profile(
                direction_sign * np.asarray(frame, dtype=np.float64),
                coordinates,
                transverse_half_width_sites=TRANSVERSE_HALF_WIDTH_SITES,
            )
            if s_axis is None:
                s_axis = axis
            profiles.append(profile)
        if s_axis is None:
            raise RuntimeError("no longitudinal flux profiles were generated")
        profiles_all = np.stack(profiles, axis=0)
        shifted = baseline_subtracted_profiles(profiles_all, window.baseline_mask)
        post_times_rel = times[window.post_mask] - float(first["transition_start_time_fs"])
        post_profiles = shifted[window.post_mask]
        boundaries = [
            _boundary_metrics(post_times_rel, post_profiles, s_axis, distance)
            for distance in (1, 2)
        ]
        backward_delay = _delay_metrics(
            post_times_rel, post_profiles, s_axis, "backward", vmax_sites_per_ps
        )
        forward_delay = _delay_metrics(
            post_times_rel, post_profiles, s_axis, "forward", vmax_sites_per_ps
        )
        d2 = next(record for record in boundaries if record["distance_sites"] == 2)
        analysis = {
            "first_persistent_x_event": first,
            "next_persistent_x_event_start_time_fs": next_start,
            "event_direction_is_positive_s": True,
            "carrier_lab_dx_sites": int(first["dx_sites"]),
            "carrier_lab_dy_sites": int(first["dy_sites"]),
            "baseline_window_fs": [
                float(window.baseline_start_fs),
                float(window.baseline_end_fs),
            ],
            "post_window_fs": [
                float(window.event_time_fs),
                float(window.post_end_fs),
            ],
            "post_window_duration_fs": float(window.post_end_fs - window.event_time_fs),
            "transverse_half_width_sites": TRANSVERSE_HALF_WIDTH_SITES,
            "boundary_metrics": boundaries,
            "backward_d1_to_d2_delay": backward_delay,
            "forward_d1_to_d2_delay": forward_delay,
            "d2_positive_energy_directionality": float(d2["positive_energy_directionality"]),
            "backward_packet_validated": bool(backward_delay["validated_packet"]),
            "forward_packet_validated": bool(forward_delay["validated_packet"]),
            "retrograde_branch_vphonon_dot_vcarrier_negative": bool(
                backward_delay["validated_packet"]
            ),
            "backward_dominates_positive_energy_at_d2": bool(
                d2["positive_energy_directionality"] > 0.0
            ),
        }
        profile_payload = {
            "time_rel_fs": np.asarray(post_times_rel, dtype=np.float64),
            "s_sites": np.asarray(s_axis, dtype=np.int64),
            "baseline_subtracted_longitudinal_flux_eV_per_fs": np.asarray(
                post_profiles, dtype=np.float64
            ),
        }

    tolerance = size_scaled_energy_balance_tolerance_eV(parameters.n_sites)
    numerical_checks = {
        "static_relaxation_converged": bool(static["converged"]),
        "complete_requested_steps": True,
        "energy_work_balance": bool(extensive_energy_balance_passes(max_balance, parameters.n_sites)),
        "electronic_norm": bool(max_norm_error < 1.0e-10),
        "projected_zero_modes": bool(max_zero_mode < 1.0e-10),
        "first_persistent_x_event_found": bool(analysis is not None),
        "finite_transport_diagnostics": bool(
            np.isfinite(displacement_x_A)
            and np.isfinite(displacement_y_A)
            and np.isfinite(max_balance)
        ),
    }
    result = {
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
        "persistent_events": events,
        "persistent_x_event_count": int(len(nearest_x)),
        "wake_analysis": analysis,
        "numerical_checks": numerical_checks,
        "numerical_pass": bool(all(numerical_checks.values())),
        "hamiltonian_evaluations": int(h_eval),
        "hamiltonian_applications": int(h_apply),
    }
    return result, profile_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=40)
    parser.add_argument("--ratios", nargs="+", type=float, default=[1.0, 0.15])
    parser.add_argument("--field-mv-per-A", type=float, default=10.0)
    parser.add_argument("--dt-fs", type=float, default=0.2)
    parser.add_argument("--final-time-fs", type=float, default=5000.0)
    parser.add_argument("--sample-interval-fs", type=float, default=2.0)
    parser.add_argument("--energy-sample-interval-fs", type=float, default=10.0)
    parser.add_argument("--krylov-dimension", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    args = parser.parse_args()

    recurrence = intermolecular_recurrence_scales(
        StaticPolaronParameters(nx=args.size, ny=args.size),
        lattice_spacing_A=LATTICE_SPACING_A,
        gamma_v_per_fs=0.0,
    )
    vmax = float(recurrence.max_group_velocity_sites_per_fs * 1000.0)

    conditions: list[dict] = []
    profile_output: dict[str, np.ndarray] = {}
    for ratio in args.ratios:
        result, profiles = _run_condition(
            size=args.size,
            anisotropy_ratio=float(ratio),
            field_mV_per_A=args.field_mv_per_A,
            dt_fs=args.dt_fs,
            final_time_fs=args.final_time_fs,
            sample_interval_fs=args.sample_interval_fs,
            energy_sample_interval_fs=args.energy_sample_interval_fs,
            krylov_dimension=args.krylov_dimension,
            vmax_sites_per_ps=vmax,
        )
        conditions.append(result)
        key = f"r{str(float(ratio)).replace('.', 'p')}"
        for name, values in profiles.items():
            profile_output[f"{key}_{name}"] = values

    numerical_checks = {
        "all_conditions_numerically_pass": bool(all(item["numerical_pass"] for item in conditions)),
        "both_conditions_have_first_x_event": bool(
            all(item["wake_analysis"] is not None for item in conditions)
        ),
        "profiles_written_for_all_conditions": bool(
            len(profile_output) == 3 * len(conditions)
        ),
    }
    payload = {
        "scope": "IP1j first-natural-hop field-driven lattice-radiation diagnostic at 10 mV/A; event-conditioned laboratory-frame flux only; no mobility/rate/threshold/material-lifetime claim",
        "size": int(args.size),
        "ratios": [float(value) for value in args.ratios],
        "field_mV_per_A": float(args.field_mv_per_A),
        "field_direction": "+x",
        "field_material_calibrated": False,
        "temperature_K": 0.0,
        "thermostat": None,
        "IDC": None,
        "dt_fs": float(args.dt_fs),
        "final_time_fs": float(args.final_time_fs),
        "sample_interval_fs": float(args.sample_interval_fs),
        "harmonic_max_group_velocity_sites_per_ps": vmax,
        "baseline_offsets_fs": [BASELINE_START_OFFSET_FS, BASELINE_END_OFFSET_FS],
        "maximum_post_event_window_fs": MAX_POST_FS,
        "next_event_guard_fs": NEXT_EVENT_GUARD_FS,
        "conditions": conditions,
        "numerical_checks": numerical_checks,
        "numerical_pass": bool(all(numerical_checks.values())),
        "interpretation_guard": {
            "event_direction_is_positive_s": True,
            "backward_validated_branch_implies_negative_vphonon_dot_vcarrier": True,
            "baseline_subtraction_does_not_uniquely_decompose_free_phonons": True,
            "continuous_field_drive_remains_present_during_wake_window": True,
            "d1_to_d2_packet_speed_is_not_unique_normal_mode_group_velocity": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(args.profiles, **profile_output)

    lines = [
        "# IP1j first natural field-driven hop wake",
        "",
        f"Cell {args.size}x{args.size}; T=0; E=+{args.field_mv_per_A:g} mV/A; dt={args.dt_fs:g} fs; final={args.final_time_fs/1000:g} ps; no thermostat or IDC.",
        f"Harmonic maximum intermolecular packet speed: {vmax:.4f} sites/ps.",
        "",
        "| J0y/J0x | TP1 dx [A] | first hop | t0 [fs] | post window [fs] | D2 directionality | back speed [sites/ps] | back corr | forward speed [sites/ps] | forward corr | retrograde branch |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in conditions:
        wake = item["wake_analysis"]
        if wake is None:
            lines.append(
                f"| {item['anisotropy_ratio']:.2f} | {item['tp1_displacement_x_A']:.4f} | NA | NA | NA | NA | NA | NA | NA | NA | NO |"
            )
            continue
        event = wake["first_persistent_x_event"]
        back = wake["backward_d1_to_d2_delay"]
        forward = wake["forward_d1_to_d2_delay"]
        lines.append(
            f"| {item['anisotropy_ratio']:.2f} | {item['tp1_displacement_x_A']:.4f} | {event['direction']} | "
            f"{event['transition_start_time_fs']:.1f} | {wake['post_window_duration_fs']:.1f} | "
            f"{wake['d2_positive_energy_directionality']:.3f} | {back['speed_sites_per_ps']:.3f} | "
            f"{back['correlation']:.3f} | {forward['speed_sites_per_ps']:.3f} | {forward['correlation']:.3f} | "
            f"{'YES' if wake['retrograde_branch_vphonon_dot_vcarrier_negative'] else 'NO'} |"
        )
    lines.extend(["", "## Numerical gates", ""])
    for name, value in numerical_checks.items():
        lines.append(f"- {name}: {'PASS' if value else 'FAIL'}")
    lines.extend(
        [
            "",
            f"Numerical status: {'PASS' if payload['numerical_pass'] else 'FAIL'}",
            "",
            "The first persistent carrier hop defines +s. Therefore a validated outward packet on the backward boundary is a directly observed lattice-radiation branch with v_phonon dot v_carrier < 0 for that event. Directionality determines whether that branch dominates the positive outward energy at d=2; existence and dominance are intentionally kept separate.",
        ]
    )
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if not payload["numerical_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
