#!/usr/bin/env python3
"""IP1m isotropic single-hop wake persistence and spatial-envelope control.

This stage reruns only the deterministic 40x40 isotropic 10 mV/A control.  The
single persistent carrier relocation defines a post-hop residence frame.  The
complete intermolecular x-current trajectory is stored and the retrograde wake
is compared with the pre-hop field-only background for more than 2 ps after the
event.

No mobility, hopping rate, activation energy, threshold field or calibrated
phonon lifetime is inferred here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import ip1l_interhop_wake_memory as base
from holstein_peierls.dynamics.event_wake import positive_energy_directionality
from holstein_peierls.dynamics.interhop_memory import (
    interval_mask,
    nonoverlapping_flux_bins,
    sliding_flux_windows,
)
from holstein_peierls.dynamics.phonon_recurrence import intermolecular_recurrence_scales
from holstein_peierls.dynamics.single_hop_memory import (
    continuation_neighbor,
    jointly_background_separated,
    ordered_envelope_fit,
    sustained_memory_gate,
)
from holstein_peierls.dynamics.wavepacket_flux import (
    integrated_outward_energy_eV,
    outward_boundary_flux_series,
)
from holstein_peierls.parameters import StaticPolaronParameters


WINDOW_WIDTH_FS = 100.0
BACKGROUND_STEP_FS = 20.0
PRE_BASELINE_START_OFFSET_FS = -700.0
PRE_BASELINE_END_OFFSET_FS = -100.0
PACKET_WINDOW_FS = 800.0
POST_END_GUARD_FS = 100.0
PERCENTILE_THRESHOLD = 95.0
LATE_MEMORY_OFFSET_FS = 1000.0
SUSTAINED_MINIMUM_PASSING_BINS = 3
SUSTAINED_MINIMUM_LATEST_OFFSET_FS = 1500.0


def _distance_records(
    *,
    times_fs: np.ndarray,
    shifted_profiles: np.ndarray,
    s_axis: np.ndarray,
    distance: int,
    event_time_fs: float,
    final_time_fs: float,
    pre_start_fs: float,
    pre_end_fs: float,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    backward = outward_boundary_flux_series(
        shifted_profiles, s_axis, int(distance), side="backward"
    )
    background = sliding_flux_windows(
        times_fs,
        backward,
        pre_start_fs,
        pre_end_fs,
        width_fs=WINDOW_WIDTH_FS,
        step_fs=BACKGROUND_STEP_FS,
    )
    energy_background = np.asarray(
        [record.positive_energy_eV for record in background], dtype=np.float64
    )
    peak_background = np.asarray(
        [record.peak_outward_flux_eV_per_fs for record in background], dtype=np.float64
    )
    post_bins = nonoverlapping_flux_bins(
        times_fs,
        backward,
        event_time_fs,
        final_time_fs - POST_END_GUARD_FS,
        width_fs=WINDOW_WIDTH_FS,
    )
    records = [
        base._window_record(
            record,
            reference_time_fs=event_time_fs,
            energy_background=energy_background,
            peak_background=peak_background,
        )
        for record in post_bins
    ]
    return records, energy_background, peak_background


def _distance_summary(distance: int, records: list[dict]) -> dict:
    passing = [record for record in records if jointly_background_separated(record)]
    first = None if not passing else min(passing, key=lambda record: float(record["center_fs"]))
    latest = None if not passing else max(float(record["center_fs"]) for record in passing)
    return {
        "distance_sites": int(distance),
        "post_bin_count": int(len(records)),
        "background_separated_bin_count": int(len(passing)),
        "first_background_separated_center_fs": (
            None if first is None else float(first["center_fs"])
        ),
        "first_background_separated_offset_fs": (
            None if first is None else float(first["center_offset_fs"])
        ),
        "latest_background_separated_center_fs": latest,
        "maximum_positive_backward_energy_eV": float(
            max((record["positive_backward_energy_eV"] for record in records), default=0.0)
        ),
        "maximum_peak_backward_flux_eV_per_fs": float(
            max((record["peak_backward_flux_eV_per_fs"] for record in records), default=0.0)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=40)
    parser.add_argument("--field-mv-per-A", dest="field_mV_per_A", type=float, default=10.0)
    parser.add_argument("--dt-fs", type=float, default=0.2)
    parser.add_argument("--final-time-fs", type=float, default=5000.0)
    parser.add_argument("--sample-interval-fs", type=float, default=2.0)
    parser.add_argument("--energy-sample-interval-fs", type=float, default=10.0)
    parser.add_argument("--krylov-dimension", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    args = parser.parse_args()

    recurrence = intermolecular_recurrence_scales(
        StaticPolaronParameters(nx=args.size, ny=args.size),
        lattice_spacing_A=base.LATTICE_SPACING_A,
        gamma_v_per_fs=0.0,
    )
    vmax = float(recurrence.max_group_velocity_sites_per_fs * 1000.0)

    raw_condition, raw_arrays = base._run(
        size=args.size,
        anisotropy_ratio=1.0,
        field_mV_per_A=args.field_mV_per_A,
        dt_fs=args.dt_fs,
        final_time_fs=args.final_time_fs,
        sample_interval_fs=args.sample_interval_fs,
        energy_sample_interval_fs=args.energy_sample_interval_fs,
        krylov_dimension=args.krylov_dimension,
        vmax_sites_per_ps=vmax,
    )

    times = np.asarray(raw_arrays["times_fs"], dtype=np.float64)
    jx = np.asarray(raw_arrays["jx_eV_per_fs"], dtype=np.float32)
    x_events = list(raw_condition["persistent_x_events"])
    analysis = None
    s_axis = np.empty(0, dtype=np.int64)
    profiles = np.empty((0, 0), dtype=np.float64)

    if len(x_events) == 1:
        event = x_events[0]
        t1 = float(event["transition_start_time_fs"])
        residence_source = int(event["target_site"])
        residence_target = continuation_neighbor(
            residence_source,
            dx_sites=int(event["dx_sites"]),
            dy_sites=int(event["dy_sites"]),
            nx=args.size,
            ny=args.size,
        )
        s_axis, profiles = base._project_profiles(
            jx,
            source_site=residence_source,
            target_site=residence_target,
            nx=args.size,
            ny=args.size,
            direction_sign=int(event["dx_sites"]),
        )

        pre_start = t1 + PRE_BASELINE_START_OFFSET_FS
        pre_end = t1 + PRE_BASELINE_END_OFFSET_FS
        pre_mask = interval_mask(times, pre_start, pre_end)
        if np.count_nonzero(pre_mask) < 2:
            raise RuntimeError("IP1m pre-hop field reference is incomplete")
        pre_profile = np.mean(profiles[pre_mask], axis=0)
        shifted = profiles - pre_profile[None, :]

        distance_records: dict[int, list[dict]] = {}
        distance_summaries: list[dict] = []
        for distance in (1, 2, 3, 4):
            records, _, _ = _distance_records(
                times_fs=times,
                shifted_profiles=shifted,
                s_axis=s_axis,
                distance=distance,
                event_time_fs=t1,
                final_time_fs=args.final_time_fs,
                pre_start_fs=pre_start,
                pre_end_fs=pre_end,
            )
            distance_records[distance] = records
            distance_summaries.append(_distance_summary(distance, records))

        d2_records = distance_records[2]
        memory = sustained_memory_gate(
            d2_records,
            first_event_time_fs=t1,
            minimum_late_offset_fs=LATE_MEMORY_OFFSET_FS,
            minimum_passing_bins=SUSTAINED_MINIMUM_PASSING_BINS,
            minimum_latest_offset_fs=SUSTAINED_MINIMUM_LATEST_OFFSET_FS,
            percentile_threshold=PERCENTILE_THRESHOLD,
        )

        packet_end = min(t1 + PACKET_WINDOW_FS, args.final_time_fs)
        packet_mask = interval_mask(times, t1, packet_end)
        first_packet = base._packet_delay(
            times[packet_mask] - t1,
            shifted[packet_mask],
            s_axis,
            side="backward",
            vmax_sites_per_ps=vmax,
        )

        arrival_map = {
            int(item["distance_sites"]): item["first_background_separated_center_fs"]
            for item in distance_summaries
        }
        envelope = ordered_envelope_fit(arrival_map)

        late_start = t1 + LATE_MEMORY_OFFSET_FS
        late_end = args.final_time_fs - POST_END_GUARD_FS
        late_mask = interval_mask(times, late_start, late_end)
        backward_d2 = outward_boundary_flux_series(shifted, s_axis, 2, side="backward")
        forward_d2 = outward_boundary_flux_series(shifted, s_axis, 2, side="forward")
        late_back_energy = integrated_outward_energy_eV(
            times[late_mask], backward_d2[late_mask], positive_only=True
        )
        late_forward_energy = integrated_outward_energy_eV(
            times[late_mask], forward_d2[late_mask], positive_only=True
        )

        finite_record_values = []
        for records in distance_records.values():
            for record in records:
                finite_record_values.extend(
                    [
                        record["positive_backward_energy_eV"],
                        record["net_backward_energy_eV"],
                        record["peak_backward_flux_eV_per_fs"],
                        record["mean_backward_flux_eV_per_fs"],
                        record["positive_energy_percentile"],
                        record["peak_flux_percentile"],
                    ]
                )
        finite_record_values.extend(
            [
                first_packet["lag_fs"],
                first_packet["speed_sites_per_ps"],
                first_packet["correlation"],
                late_back_energy,
                late_forward_energy,
            ]
        )

        analysis = {
            "event": event,
            "post_hop_residence_frame_source_site": residence_source,
            "post_hop_residence_frame_target_site": residence_target,
            "carrier_motion_is_positive_s": True,
            "retrograde_direction_is_negative_s": True,
            "pre_hop_field_reference_window_fs": [pre_start, pre_end],
            "post_hop_analysis_end_fs": float(late_end),
            "first_hop_backward_packet": first_packet,
            "d2_memory_gate": memory,
            "distance_summaries": distance_summaries,
            "spatial_envelope_fit": envelope,
            "late_positive_backward_energy_eV": float(late_back_energy),
            "late_positive_forward_energy_eV": float(late_forward_energy),
            "late_positive_energy_directionality": positive_energy_directionality(
                late_back_energy, late_forward_energy
            ),
            "all_analysis_scalars_finite": bool(np.all(np.isfinite(finite_record_values))),
        }

    expected_steps = int(round(args.final_time_fs / args.dt_fs))
    numerical_checks = {
        "static_relaxation_converged": bool(raw_condition["static"]["converged"]),
        "complete_requested_steps": bool(raw_condition["steps"] == expected_steps),
        "energy_work_balance": bool(
            raw_condition["maximum_energy_work_residual_eV"]
            <= raw_condition["energy_work_tolerance_eV"]
        ),
        "electronic_norm": bool(raw_condition["maximum_norm_error"] < 1.0e-10),
        "projected_zero_modes": bool(raw_condition["maximum_zero_mode_mean"] < 1.0e-10),
        "exactly_one_persistent_x_event": bool(len(x_events) == 1),
        "complete_ip1m_analysis": bool(analysis is not None),
        "finite_memory_and_envelope_metrics": bool(
            analysis is not None and analysis["all_analysis_scalars_finite"]
        ),
        "complete_current_trajectory_serializable": bool(
            jx.shape == (times.size, args.size, args.size) and np.all(np.isfinite(jx))
        ),
        "finite_transport_diagnostics": bool(
            np.isfinite(raw_condition["tp1_displacement_x_A"])
            and np.isfinite(raw_condition["tp1_displacement_y_A"])
            and np.isfinite(raw_condition["maximum_energy_work_residual_eV"])
        ),
    }
    numerical_pass = bool(all(numerical_checks.values()))

    physical = {
        "first_hop_backward_packet": bool(
            analysis is not None and analysis["first_hop_backward_packet"]["packet_qualified"]
        ),
        "late_memory_present": bool(
            analysis is not None and analysis["d2_memory_gate"]["late_memory_present"]
        ),
        "sustained_late_memory": bool(
            analysis is not None and analysis["d2_memory_gate"]["sustained_late_memory"]
        ),
        "no_second_persistent_hop": bool(len(x_events) == 1),
    }
    physical["single_hop_long_lived_retrograde_memory"] = bool(
        physical["first_hop_backward_packet"]
        and physical["sustained_late_memory"]
        and physical["no_second_persistent_hop"]
    )

    condition = {
        "anisotropy_ratio": 1.0,
        "field_mV_per_A": float(args.field_mV_per_A),
        "static": raw_condition["static"],
        "steps": raw_condition["steps"],
        "elapsed_seconds": raw_condition["elapsed_seconds"],
        "field_work_eV": raw_condition["field_work_eV"],
        "maximum_energy_work_residual_eV": raw_condition[
            "maximum_energy_work_residual_eV"
        ],
        "energy_work_tolerance_eV": raw_condition["energy_work_tolerance_eV"],
        "maximum_norm_error": raw_condition["maximum_norm_error"],
        "maximum_zero_mode_mean": raw_condition["maximum_zero_mode_mean"],
        "tp1_displacement_x_A": raw_condition["tp1_displacement_x_A"],
        "tp1_displacement_y_A": raw_condition["tp1_displacement_y_A"],
        "persistent_events": raw_condition["persistent_events"],
        "persistent_x_events": x_events,
        "analysis": analysis,
        "numerical_checks": numerical_checks,
        "numerical_pass": numerical_pass,
        "hamiltonian_evaluations": raw_condition["hamiltonian_evaluations"],
        "hamiltonian_applications": raw_condition["hamiltonian_applications"],
    }

    payload = {
        "scope": "IP1m natural isotropic single-hop retrograde wake persistence and spatial-envelope control at 10 mV/A",
        "size": int(args.size),
        "anisotropy_ratio": 1.0,
        "field_mV_per_A": float(args.field_mV_per_A),
        "temperature_K": 0.0,
        "thermostat": None,
        "IDC": None,
        "dt_fs": float(args.dt_fs),
        "final_time_fs": float(args.final_time_fs),
        "sample_interval_fs": float(args.sample_interval_fs),
        "harmonic_max_group_velocity_sites_per_ps": vmax,
        "condition": condition,
        "physical_decision": physical,
        "numerical_pass": numerical_pass,
        "interpretation_guard": {
            "ip1j_binary_isotropic_result_remains_unchanged": True,
            "field_remains_on_after_single_hop": True,
            "pre_hop_reference_is_same_trajectory_field_background": True,
            "no_second_persistent_hop_removes_second_event_contamination": True,
            "spatial_envelope_speed_is_not_unique_normal_mode_group_velocity": True,
            "no_transport_coefficient_or_material_lifetime_claim": True,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(
        args.trajectory,
        times_fs=times,
        jx_eV_per_fs=jx,
        residence_s_sites=np.asarray(s_axis, dtype=np.int64),
        residence_longitudinal_flux_eV_per_fs=np.asarray(profiles, dtype=np.float32),
    )

    lines = [
        "# IP1m isotropic single-hop wake persistence",
        "",
        f"Cell {args.size}x{args.size}; J0y/J0x=1.0; T=0; E=+{args.field_mV_per_A:g} mV/A; final={args.final_time_fs/1000:g} ps.",
        f"Harmonic vmax: {vmax:.4f} sites/ps.",
        "",
    ]
    if analysis is None:
        lines.append("IP1m physical analysis unavailable because exactly one persistent x hop was not found.")
    else:
        event = analysis["event"]
        packet = analysis["first_hop_backward_packet"]
        memory = analysis["d2_memory_gate"]
        lines.extend(
            [
                f"Event: {event['source_site']}->{event['target_site']} ({event['direction']}) at {event['transition_start_time_fs']:.1f} fs.",
                "",
                "## First-hop packet",
                "",
                f"- backward d1->d2 delay: {packet['lag_fs']:.1f} fs; speed={packet['speed_sites_per_ps']:.4f} sites/ps; corr={packet['correlation']:.6f}; packet={'PASS' if packet['packet_qualified'] else 'FAIL'};",
                "",
                "## Late single-hop memory",
                "",
                f"- late bins: {memory['late_bin_count']}; jointly >95th percentile: {memory['late_passing_bin_count']}; fraction={memory['late_passing_fraction']:.3f};",
                f"- latest passing center: {memory['latest_passing_center_fs']};",
                f"- late-memory present: {'YES' if memory['late_memory_present'] else 'NO'};",
                f"- sustained late-memory: {'YES' if memory['sustained_late_memory'] else 'NO'};",
                f"- late positive-energy directionality: {analysis['late_positive_energy_directionality']:.6f};",
                "",
                "## Spatial envelope",
                "",
            ]
        )
        for item in analysis["distance_summaries"]:
            lines.append(
                f"- d={item['distance_sites']}: first separated offset={item['first_background_separated_offset_fs']}; latest center={item['latest_background_separated_center_fs']}; max E+={item['maximum_positive_backward_energy_eV']:.6e} eV; max flux={item['maximum_peak_backward_flux_eV_per_fs']:.6e} eV/fs."
            )
        envelope = analysis["spatial_envelope_fit"]
        lines.append(
            f"- ordered d=1..3 envelope fit: {'YES' if envelope['available'] else 'NO'}; speed={envelope['speed_sites_per_ps']}; R2={envelope['r_squared']}."
        )
        lines.extend(
            [
                "",
                "## Physical decision",
                "",
                f"- first-hop backward packet: {'YES' if physical['first_hop_backward_packet'] else 'NO'};",
                f"- sustained late memory: {'YES' if physical['sustained_late_memory'] else 'NO'};",
                f"- no second persistent hop: {'YES' if physical['no_second_persistent_hop'] else 'NO'};",
                f"- single-hop long-lived retrograde memory: {'YES' if physical['single_hop_long_lived_retrograde_memory'] else 'NO'};",
            ]
        )
    lines.extend(["", "## Numerical gates", ""])
    for name, value in numerical_checks.items():
        lines.append(f"- {name}: {'PASS' if value else 'FAIL'}")
    lines.extend(["", f"Numerical status: {'PASS' if numerical_pass else 'FAIL'}"])
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"\nWrote {args.output}")
    print(f"Wrote {args.markdown}")
    print(f"Wrote {args.trajectory}")
    if not numerical_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
