#!/usr/bin/env python3
"""IP1l inter-hop wake memory and incremental second-hop radiation.

This experiment reruns only the deterministic 40x40 anisotropic 10 mV/A
trajectory selected by IP1i-IP1k.  It saves the complete sampled harmonic
intermolecular x-current field, measures the first-hop wake during the residence
before the second hop, and tests for renewed backward radiation at the second
hop relative to the inherited local wake state.

No transport coefficient, hopping rate, activation energy, threshold field or
material phonon lifetime is inferred here.
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
from holstein_peierls.dynamics.event_wake import positive_energy_directionality
from holstein_peierls.dynamics.field import UniformElectricField2D
from holstein_peierls.dynamics.hopping_observables import PersistentSiteTracker
from holstein_peierls.dynamics.interhop_memory import (
    empirical_percentile,
    incremental_pulse_gate,
    interval_mask,
    memory_gate,
    nonoverlapping_flux_bins,
    sliding_flux_windows,
)
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
PRE_FIRST_BASELINE_START_OFFSET_FS = -700.0
PRE_FIRST_BASELINE_END_OFFSET_FS = -100.0
SECOND_BASELINE_START_OFFSET_FS = -700.0
SECOND_BASELINE_END_OFFSET_FS = -100.0
SECOND_POST_END_OFFSET_FS = 800.0
INTERHOP_END_GUARD_FS = 100.0
WINDOW_WIDTH_FS = 100.0
BACKGROUND_STEP_FS = 20.0
LATE_MEMORY_OFFSET_FS = 1000.0
LAG_MIN_FS = 300.0
LAG_MAX_FS = 700.0
MIN_CORRELATION = 0.80
PERCENTILE_THRESHOLD = 95.0


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


def _project_profiles(
    jx_frames: np.ndarray,
    *,
    source_site: int,
    target_site: int,
    nx: int,
    ny: int,
    direction_sign: int,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = event_aligned_coordinates(source_site, target_site, nx, ny)
    profiles: list[np.ndarray] = []
    s_axis = None
    for frame in jx_frames:
        axis, profile = longitudinal_profile(
            int(direction_sign) * np.asarray(frame, dtype=np.float64),
            coordinates,
            transverse_half_width_sites=TRANSVERSE_HALF_WIDTH_SITES,
        )
        if s_axis is None:
            s_axis = axis
        profiles.append(profile)
    if s_axis is None:
        raise RuntimeError("no longitudinal profiles were generated")
    return np.asarray(s_axis, dtype=np.int64), np.stack(profiles, axis=0)


def _packet_delay(
    times_fs: np.ndarray,
    profiles: np.ndarray,
    s_axis: np.ndarray,
    *,
    side: str,
    vmax_sites_per_ps: float,
) -> dict:
    d1 = outward_boundary_flux_series(profiles, s_axis, 1, side=side)
    d2 = outward_boundary_flux_series(profiles, s_axis, 2, side=side)
    delay = normalized_flux_delay(
        times_fs,
        d1,
        d2,
        lag_min_fs=LAG_MIN_FS,
        lag_max_fs=LAG_MAX_FS,
    )
    internal = bool(LAG_MIN_FS < delay.lag_fs < LAG_MAX_FS)
    speed_ok = bool(delay.speed_sites_per_ps <= 1.05 * float(vmax_sites_per_ps))
    correlation_ok = bool(delay.correlation >= MIN_CORRELATION)
    return {
        "side": side,
        "lag_fs": float(delay.lag_fs),
        "speed_sites_per_ps": float(delay.speed_sites_per_ps),
        "correlation": float(delay.correlation),
        "lag_internal_to_search_window": internal,
        "within_harmonic_vmax": speed_ok,
        "high_correlation": correlation_ok,
        "packet_qualified": bool(internal and speed_ok and correlation_ok),
    }


def _window_record(record, *, reference_time_fs: float, energy_background, peak_background) -> dict:
    return {
        "start_fs": float(record.start_fs),
        "end_fs": float(record.end_fs),
        "center_fs": float(record.center_fs),
        "center_offset_fs": float(record.center_fs - reference_time_fs),
        "positive_backward_energy_eV": float(record.positive_energy_eV),
        "net_backward_energy_eV": float(record.net_energy_eV),
        "peak_backward_flux_eV_per_fs": float(record.peak_outward_flux_eV_per_fs),
        "mean_backward_flux_eV_per_fs": float(record.mean_outward_flux_eV_per_fs),
        "positive_energy_percentile": empirical_percentile(
            record.positive_energy_eV, energy_background
        ),
        "peak_flux_percentile": empirical_percentile(
            record.peak_outward_flux_eV_per_fs, peak_background
        ),
    }


def _run(
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

    populations = np.abs(current.electronic_state) ** 2
    tracker.update(populations / float(np.sum(populations)), 0.0)
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
    profiles = np.empty((0, 0), dtype=np.float64)
    s_axis = np.empty(0, dtype=np.int64)
    if len(nearest_x) >= 2:
        first = nearest_x[0]
        second = nearest_x[1]
        t1 = float(first["transition_start_time_fs"])
        t2 = float(second["transition_start_time_fs"])
        direction_sign = int(second["dx_sites"])
        s_axis, profiles = _project_profiles(
            jx,
            source_site=int(second["source_site"]),
            target_site=int(second["target_site"]),
            nx=parameters.nx,
            ny=parameters.ny,
            direction_sign=direction_sign,
        )

        pre_first_start = t1 + PRE_FIRST_BASELINE_START_OFFSET_FS
        pre_first_end = t1 + PRE_FIRST_BASELINE_END_OFFSET_FS
        pre_first_mask = interval_mask(times, pre_first_start, pre_first_end)
        if np.count_nonzero(pre_first_mask) < 2:
            raise RuntimeError("pre-first reference interval is incomplete")
        pre_first_profile = np.mean(profiles[pre_first_mask], axis=0)
        field_reference_shifted = profiles - pre_first_profile[None, :]
        backward_d1 = outward_boundary_flux_series(
            field_reference_shifted, s_axis, 1, side="backward"
        )
        backward_d2 = outward_boundary_flux_series(
            field_reference_shifted, s_axis, 2, side="backward"
        )
        backward_d3 = outward_boundary_flux_series(
            field_reference_shifted, s_axis, 3, side="backward"
        )
        forward_d2 = outward_boundary_flux_series(
            field_reference_shifted, s_axis, 2, side="forward"
        )

        field_background = sliding_flux_windows(
            times,
            backward_d2,
            pre_first_start,
            pre_first_end,
            width_fs=WINDOW_WIDTH_FS,
            step_fs=BACKGROUND_STEP_FS,
        )
        field_energy_background = np.asarray(
            [record.positive_energy_eV for record in field_background], dtype=np.float64
        )
        field_peak_background = np.asarray(
            [record.peak_outward_flux_eV_per_fs for record in field_background], dtype=np.float64
        )

        interhop_end = t2 - INTERHOP_END_GUARD_FS
        interhop_bins_raw = nonoverlapping_flux_bins(
            times,
            backward_d2,
            t1,
            interhop_end,
            width_fs=WINDOW_WIDTH_FS,
        )
        interhop_bins = [
            _window_record(
                record,
                reference_time_fs=t1,
                energy_background=field_energy_background,
                peak_background=field_peak_background,
            )
            for record in interhop_bins_raw
        ]
        memory = memory_gate(
            interhop_bins,
            first_event_time_fs=t1,
            minimum_late_offset_fs=LATE_MEMORY_OFFSET_FS,
            percentile_threshold=PERCENTILE_THRESHOLD,
        )

        first_packet_mask = interval_mask(times, t1, interhop_end)
        first_packet = _packet_delay(
            times[first_packet_mask] - t1,
            field_reference_shifted[first_packet_mask],
            s_axis,
            side="backward",
            vmax_sites_per_ps=vmax_sites_per_ps,
        )

        late_start = max(t1 + LATE_MEMORY_OFFSET_FS, t1)
        late_mask = interval_mask(times, late_start, interhop_end)
        late_backward_energy = integrated_outward_energy_eV(
            times[late_mask], backward_d2[late_mask], positive_only=True
        )
        late_forward_energy = integrated_outward_energy_eV(
            times[late_mask], forward_d2[late_mask], positive_only=True
        )

        second_baseline_start = t2 + SECOND_BASELINE_START_OFFSET_FS
        second_baseline_end = t2 + SECOND_BASELINE_END_OFFSET_FS
        second_baseline_mask = interval_mask(times, second_baseline_start, second_baseline_end)
        second_post_end = t2 + SECOND_POST_END_OFFSET_FS
        second_post_mask = interval_mask(times, t2, second_post_end)
        if np.count_nonzero(second_baseline_mask) < 2 or np.count_nonzero(second_post_mask) < 3:
            raise RuntimeError("second-event local baseline or post window is incomplete")
        second_baseline_profile = np.mean(profiles[second_baseline_mask], axis=0)
        second_shifted = profiles - second_baseline_profile[None, :]
        second_back_d2_full = outward_boundary_flux_series(
            second_shifted, s_axis, 2, side="backward"
        )
        second_forward_d2_full = outward_boundary_flux_series(
            second_shifted, s_axis, 2, side="forward"
        )
        second_post_times_rel = times[second_post_mask] - t2
        second_post_profiles = second_shifted[second_post_mask]
        second_back = second_back_d2_full[second_post_mask]
        second_forward = second_forward_d2_full[second_post_mask]
        second_back_energy = integrated_outward_energy_eV(
            second_post_times_rel, second_back, positive_only=True
        )
        second_forward_energy = integrated_outward_energy_eV(
            second_post_times_rel, second_forward, positive_only=True
        )
        second_back_peak = float(np.max(second_back))
        second_forward_peak = float(np.max(second_forward))
        second_backward_packet = _packet_delay(
            second_post_times_rel,
            second_post_profiles,
            s_axis,
            side="backward",
            vmax_sites_per_ps=vmax_sites_per_ps,
        )
        second_forward_packet = _packet_delay(
            second_post_times_rel,
            second_post_profiles,
            s_axis,
            side="forward",
            vmax_sites_per_ps=vmax_sites_per_ps,
        )

        local_windows = sliding_flux_windows(
            times,
            second_back_d2_full,
            second_baseline_start,
            second_baseline_end,
            width_fs=WINDOW_WIDTH_FS,
            step_fs=BACKGROUND_STEP_FS,
        )
        local_peak_background = np.asarray(
            [record.peak_outward_flux_eV_per_fs for record in local_windows], dtype=np.float64
        )
        second_peak_percentile = empirical_percentile(second_back_peak, local_peak_background)
        local_max_peak = float(np.max(local_peak_background))
        second_incremental = incremental_pulse_gate(
            packet_qualified=bool(second_backward_packet["packet_qualified"]),
            peak_percentile=second_peak_percentile,
            positive_excess_energy_eV=second_back_energy,
            percentile_threshold=PERCENTILE_THRESHOLD,
        )

        finite_values = [
            late_backward_energy,
            late_forward_energy,
            second_back_energy,
            second_forward_energy,
            second_back_peak,
            second_forward_peak,
            second_peak_percentile,
            local_max_peak,
            first_packet["lag_fs"],
            first_packet["speed_sites_per_ps"],
            first_packet["correlation"],
            second_backward_packet["lag_fs"],
            second_backward_packet["speed_sites_per_ps"],
            second_backward_packet["correlation"],
        ]
        analysis = {
            "first_event": first,
            "second_event": second,
            "event_chain_continuous": bool(int(first["target_site"]) == int(second["source_site"])),
            "event_separation_fs": float(t2 - t1),
            "common_frame_source_site": int(second["source_site"]),
            "common_frame_target_site": int(second["target_site"]),
            "carrier_motion_is_positive_s": True,
            "retrograde_direction_is_negative_s": True,
            "pre_first_field_reference_window_fs": [pre_first_start, pre_first_end],
            "field_background_window_count": int(len(field_background)),
            "interhop_analysis_end_fs": float(interhop_end),
            "interhop_bins": interhop_bins,
            "memory_gate": memory,
            "first_hop_backward_packet_in_B_residence_frame": first_packet,
            "late_interhop_positive_backward_energy_eV": float(late_backward_energy),
            "late_interhop_positive_forward_energy_eV": float(late_forward_energy),
            "late_interhop_directionality": positive_energy_directionality(
                late_backward_energy, late_forward_energy
            ),
            "second_event_local_baseline_window_fs": [
                second_baseline_start,
                second_baseline_end,
            ],
            "second_event_post_window_fs": [t2, second_post_end],
            "second_event_backward_positive_excess_energy_eV": float(second_back_energy),
            "second_event_forward_positive_excess_energy_eV": float(second_forward_energy),
            "second_event_positive_energy_directionality": positive_energy_directionality(
                second_back_energy, second_forward_energy
            ),
            "second_event_backward_peak_excess_flux_eV_per_fs": second_back_peak,
            "second_event_forward_peak_excess_flux_eV_per_fs": second_forward_peak,
            "second_event_local_pre_peak_window_count": int(len(local_windows)),
            "second_event_backward_peak_percentile_vs_local_pre": float(second_peak_percentile),
            "second_event_backward_peak_ratio_to_local_pre_max": float(
                second_back_peak / max(local_max_peak, 1.0e-30)
            ),
            "second_event_backward_packet": second_backward_packet,
            "second_event_forward_packet": second_forward_packet,
            "incremental_second_hop_gate_pass": bool(second_incremental),
            "all_analysis_scalars_finite": bool(np.all(np.isfinite(finite_values))),
            "diagnostic_backward_d3_peak_eV_per_fs": float(np.max(backward_d3[late_mask])),
        }

    tolerance = size_scaled_energy_balance_tolerance_eV(parameters.n_sites)
    numerical_checks = {
        "static_relaxation_converged": bool(static["converged"]),
        "complete_requested_steps": True,
        "energy_work_balance": bool(extensive_energy_balance_passes(max_balance, parameters.n_sites)),
        "electronic_norm": bool(max_norm_error < 1.0e-10),
        "projected_zero_modes": bool(max_zero_mode < 1.0e-10),
        "exactly_two_persistent_x_events": bool(len(nearest_x) == 2),
        "complete_ip1l_analysis": bool(analysis is not None),
        "continuous_two_hop_chain": bool(analysis is not None and analysis["event_chain_continuous"]),
        "finite_memory_and_incremental_metrics": bool(
            analysis is not None and analysis["all_analysis_scalars_finite"]
        ),
        "complete_current_trajectory_serializable": bool(
            jx.shape == (times.size, parameters.nx, parameters.ny)
            and np.all(np.isfinite(jx))
        ),
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
        "persistent_x_events": nearest_x,
        "analysis": analysis,
        "numerical_checks": numerical_checks,
        "numerical_pass": bool(all(numerical_checks.values())),
        "hamiltonian_evaluations": int(h_eval),
        "hamiltonian_applications": int(h_apply),
    }
    arrays = {
        "times_fs": times,
        "jx_eV_per_fs": np.asarray(jx, dtype=np.float32),
        "common_s_sites": np.asarray(s_axis, dtype=np.int64),
        "common_longitudinal_flux_eV_per_fs": np.asarray(profiles, dtype=np.float32),
    }
    return result, arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=40)
    parser.add_argument("--anisotropy-ratio", type=float, default=0.15)
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
        lattice_spacing_A=LATTICE_SPACING_A,
        gamma_v_per_fs=0.0,
    )
    vmax = float(recurrence.max_group_velocity_sites_per_fs * 1000.0)

    condition, arrays = _run(
        size=args.size,
        anisotropy_ratio=args.anisotropy_ratio,
        field_mV_per_A=args.field_mV_per_A,
        dt_fs=args.dt_fs,
        final_time_fs=args.final_time_fs,
        sample_interval_fs=args.sample_interval_fs,
        energy_sample_interval_fs=args.energy_sample_interval_fs,
        krylov_dimension=args.krylov_dimension,
        vmax_sites_per_ps=vmax,
    )
    analysis = condition["analysis"]
    physical = {
        "late_interhop_memory": bool(
            analysis is not None and analysis["memory_gate"]["late_memory_gate_pass"]
        ),
        "first_hop_backward_packet": bool(
            analysis is not None
            and analysis["first_hop_backward_packet_in_B_residence_frame"]["packet_qualified"]
        ),
        "incremental_second_hop_retrograde_pulse": bool(
            analysis is not None and analysis["incremental_second_hop_gate_pass"]
        ),
    }
    physical["history_dependent_wake_plus_renewed_radiation"] = bool(
        physical["late_interhop_memory"]
        and physical["first_hop_backward_packet"]
        and physical["incremental_second_hop_retrograde_pulse"]
    )

    payload = {
        "scope": "IP1l inter-hop retrograde wake memory and incremental second-hop radiation in the deterministic 40x40 anisotropic 10 mV/A control",
        "size": int(args.size),
        "anisotropy_ratio": float(args.anisotropy_ratio),
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
        "numerical_pass": bool(condition["numerical_pass"]),
        "interpretation_guard": {
            "ip1k_replication_result_remains_negative": True,
            "pre_first_reference_is_same_trajectory_field_background": True,
            "pre_second_reference_contains_inherited_wake_memory": True,
            "packet_speed_is_not_unique_normal_mode_group_velocity": True,
            "no_transport_coefficient_or_material_lifetime_claim": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(args.trajectory, **arrays)

    lines = [
        "# IP1l inter-hop wake memory and incremental second-hop radiation",
        "",
        f"Cell {args.size}x{args.size}; J0y/J0x={args.anisotropy_ratio:g}; T=0; E=+{args.field_mV_per_A:g} mV/A; final={args.final_time_fs/1000:g} ps.",
        f"Harmonic vmax: {vmax:.4f} sites/ps.",
        "",
    ]
    if analysis is None:
        lines.append("IP1l analysis unavailable because fewer than two complete persistent x hops were found.")
    else:
        first = analysis["first_event"]
        second = analysis["second_event"]
        memory = analysis["memory_gate"]
        packet1 = analysis["first_hop_backward_packet_in_B_residence_frame"]
        packet2 = analysis["second_event_backward_packet"]
        lines.extend(
            [
                f"Events: {first['source_site']}->{first['target_site']} at {first['transition_start_time_fs']:.1f} fs; {second['source_site']}->{second['target_site']} at {second['transition_start_time_fs']:.1f} fs; separation={analysis['event_separation_fs']:.1f} fs.",
                "",
                "## First-hop wake memory",
                "",
                f"- d1->d2 backward delay: {packet1['lag_fs']:.1f} fs; speed={packet1['speed_sites_per_ps']:.4f} sites/ps; corr={packet1['correlation']:.6f}; packet={'PASS' if packet1['packet_qualified'] else 'FAIL'};",
                f"- late bins: {memory['late_bin_count']}; jointly >95th percentile: {memory['late_passing_bin_count']}; fraction={memory['late_passing_fraction']:.3f};",
                f"- latest passing bin center: {memory['latest_passing_center_fs']};",
                f"- late-memory gate: {'PASS' if memory['late_memory_gate_pass'] else 'FAIL'};",
                f"- late inter-hop directionality: {analysis['late_interhop_directionality']:.6f}.",
                "",
                "## Incremental second-hop pulse",
                "",
                f"- backward positive excess energy: {analysis['second_event_backward_positive_excess_energy_eV']:.6e} eV;",
                f"- backward peak excess flux: {analysis['second_event_backward_peak_excess_flux_eV_per_fs']:.6e} eV/fs;",
                f"- peak percentile vs local pre-second fluctuations: {analysis['second_event_backward_peak_percentile_vs_local_pre']:.1f};",
                f"- peak / local-pre maximum: {analysis['second_event_backward_peak_ratio_to_local_pre_max']:.3f};",
                f"- d1->d2 backward delay: {packet2['lag_fs']:.1f} fs; speed={packet2['speed_sites_per_ps']:.4f} sites/ps; corr={packet2['correlation']:.6f}; packet={'PASS' if packet2['packet_qualified'] else 'FAIL'};",
                f"- incremental second-hop gate: {'PASS' if analysis['incremental_second_hop_gate_pass'] else 'FAIL'}.",
                "",
                "## Physical decision",
                "",
                f"- late inter-hop memory: {'YES' if physical['late_interhop_memory'] else 'NO'};",
                f"- first-hop propagating backward packet: {'YES' if physical['first_hop_backward_packet'] else 'NO'};",
                f"- renewed second-hop retrograde pulse: {'YES' if physical['incremental_second_hop_retrograde_pulse'] else 'NO'};",
                f"- history-dependent wake + renewed radiation: {'YES' if physical['history_dependent_wake_plus_renewed_radiation'] else 'NO'};",
            ]
        )
    lines.extend(["", "## Numerical gates", ""])
    for name, value in condition["numerical_checks"].items():
        lines.append(f"- {name}: {'PASS' if value else 'FAIL'}")
    lines.extend(["", f"Numerical status: {'PASS' if condition['numerical_pass'] else 'FAIL'}"])
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"\nWrote {args.output}")
    print(f"Wrote {args.markdown}")
    print(f"Wrote {args.trajectory}")
    if not condition["numerical_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
