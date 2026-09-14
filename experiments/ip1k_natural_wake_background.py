#!/usr/bin/env python3
"""IP1k replication and matched-background audit for natural hop wakes.

The experiment reruns the two 40x40, 10 mV/A deterministic controls selected
by IP1i/IP1j. Every complete persistent nearest-neighbor x relocation is
compared with event-free pseudo-event windows from the same trajectory.

The aim is to decide whether the weak IP1j retrograde branch is reproducibly
associated with carrier relocation rather than continuous field-driven lattice
oscillation. No mobility, hopping-rate, activation-energy or material phonon
parameter is extracted here.
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
from holstein_peierls.dynamics.event_background import (
    background_separated,
    complete_real_event_times,
    empirical_percentile,
    matched_window,
    packet_gate,
    pseudo_event_times,
)
from holstein_peierls.dynamics.event_wake import (
    baseline_subtracted_profiles,
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
BASELINE_START_OFFSET_FS = -700.0
BASELINE_END_OFFSET_FS = -100.0
POST_END_OFFSET_FS = 800.0
PSEUDO_CADENCE_FS = 100.0
LAG_MIN_FS = 300.0
LAG_MAX_FS = 700.0
MIN_CORRELATION = 0.80
BACKGROUND_PERCENTILE_THRESHOLD = 95.0


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


def _delay_metrics(
    times_rel_fs: np.ndarray,
    post_profiles: np.ndarray,
    s_axis: np.ndarray,
    *,
    side: str,
    vmax_sites_per_ps: float,
) -> dict:
    first = outward_boundary_flux_series(post_profiles, s_axis, 1, side=side)
    second = outward_boundary_flux_series(post_profiles, s_axis, 2, side=side)
    delay = normalized_flux_delay(
        times_rel_fs,
        first,
        second,
        lag_min_fs=LAG_MIN_FS,
        lag_max_fs=LAG_MAX_FS,
    )
    qualified = packet_gate(
        lag_fs=delay.lag_fs,
        correlation=delay.correlation,
        speed_sites_per_ps=delay.speed_sites_per_ps,
        vmax_sites_per_ps=vmax_sites_per_ps,
        lag_min_fs=LAG_MIN_FS,
        lag_max_fs=LAG_MAX_FS,
        minimum_correlation=MIN_CORRELATION,
    )
    return {
        "side": side,
        "lag_fs": float(delay.lag_fs),
        "speed_sites_per_ps": float(delay.speed_sites_per_ps),
        "correlation": float(delay.correlation),
        "lag_internal_to_search_window": bool(LAG_MIN_FS < delay.lag_fs < LAG_MAX_FS),
        "within_harmonic_vmax": bool(
            delay.speed_sites_per_ps <= 1.05 * float(vmax_sites_per_ps)
        ),
        "packet_qualified": bool(qualified),
    }


def _window_metrics(
    times_fs: np.ndarray,
    profiles_all: np.ndarray,
    s_axis: np.ndarray,
    center_time_fs: float,
    vmax_sites_per_ps: float,
) -> tuple[dict, dict[str, np.ndarray]]:
    window = matched_window(
        times_fs,
        center_time_fs,
        baseline_start_offset_fs=BASELINE_START_OFFSET_FS,
        baseline_end_offset_fs=BASELINE_END_OFFSET_FS,
        post_end_offset_fs=POST_END_OFFSET_FS,
    )
    shifted = baseline_subtracted_profiles(profiles_all, window.baseline_mask)
    post_times = times_fs[window.post_mask] - float(center_time_fs)
    post_profiles = shifted[window.post_mask]
    backward_d2 = outward_boundary_flux_series(post_profiles, s_axis, 2, side="backward")
    forward_d2 = outward_boundary_flux_series(post_profiles, s_axis, 2, side="forward")
    back_energy = integrated_outward_energy_eV(post_times, backward_d2, positive_only=True)
    front_energy = integrated_outward_energy_eV(post_times, forward_d2, positive_only=True)
    backward_delay = _delay_metrics(
        post_times, post_profiles, s_axis, side="backward", vmax_sites_per_ps=vmax_sites_per_ps
    )
    forward_delay = _delay_metrics(
        post_times, post_profiles, s_axis, side="forward", vmax_sites_per_ps=vmax_sites_per_ps
    )
    metrics = {
        "center_time_fs": float(center_time_fs),
        "baseline_window_fs": [float(window.baseline_start_fs), float(window.baseline_end_fs)],
        "post_window_fs": [float(center_time_fs), float(window.post_end_fs)],
        "backward_positive_outward_energy_eV": float(back_energy),
        "forward_positive_outward_energy_eV": float(front_energy),
        "d2_positive_energy_directionality": positive_energy_directionality(back_energy, front_energy),
        "backward_peak_outward_flux_eV_per_fs": float(np.max(backward_d2)),
        "forward_peak_outward_flux_eV_per_fs": float(np.max(forward_d2)),
        "backward_d1_to_d2_delay": backward_delay,
        "forward_d1_to_d2_delay": forward_delay,
    }
    arrays = {
        "time_rel_fs": np.asarray(post_times, dtype=np.float64),
        "s_sites": np.asarray(s_axis, dtype=np.int64),
        "baseline_subtracted_longitudinal_flux_eV_per_fs": np.asarray(
            post_profiles, dtype=np.float64
        ),
    }
    return metrics, arrays


def _metrics_finite(record: dict) -> bool:
    scalar_keys = (
        "backward_positive_outward_energy_eV",
        "forward_positive_outward_energy_eV",
        "d2_positive_energy_directionality",
        "backward_peak_outward_flux_eV_per_fs",
        "forward_peak_outward_flux_eV_per_fs",
    )
    if not all(np.isfinite(float(record[key])) for key in scalar_keys):
        return False
    for side in ("backward_d1_to_d2_delay", "forward_d1_to_d2_delay"):
        delay = record[side]
        if not all(
            np.isfinite(float(delay[key]))
            for key in ("lag_fs", "speed_sites_per_ps", "correlation")
        ):
            return False
    return True


def _analyze_real_event(
    *,
    event: dict,
    times_fs: np.ndarray,
    jx_frames: np.ndarray,
    parameters: StaticPolaronParameters,
    pseudo_times_fs: np.ndarray,
    vmax_sites_per_ps: float,
) -> tuple[dict, dict[str, np.ndarray]]:
    direction_sign = int(event["dx_sites"])
    s_axis, profiles = _project_profiles(
        jx_frames,
        source_site=int(event["source_site"]),
        target_site=int(event["target_site"]),
        nx=parameters.nx,
        ny=parameters.ny,
        direction_sign=direction_sign,
    )
    real_metrics, real_arrays = _window_metrics(
        times_fs,
        profiles,
        s_axis,
        float(event["transition_start_time_fs"]),
        vmax_sites_per_ps,
    )
    pseudo_metrics: list[dict] = []
    for center in pseudo_times_fs:
        record, _ = _window_metrics(
            times_fs,
            profiles,
            s_axis,
            float(center),
            vmax_sites_per_ps,
        )
        pseudo_metrics.append(record)

    back_energy_background = np.asarray(
        [item["backward_positive_outward_energy_eV"] for item in pseudo_metrics], dtype=np.float64
    )
    back_peak_background = np.asarray(
        [item["backward_peak_outward_flux_eV_per_fs"] for item in pseudo_metrics], dtype=np.float64
    )
    energy_percentile = empirical_percentile(
        real_metrics["backward_positive_outward_energy_eV"], back_energy_background
    )
    peak_percentile = empirical_percentile(
        real_metrics["backward_peak_outward_flux_eV_per_fs"], back_peak_background
    )
    packet_qualified = bool(real_metrics["backward_d1_to_d2_delay"]["packet_qualified"])
    separated = background_separated(
        packet_qualified=packet_qualified,
        energy_percentile=energy_percentile,
        peak_percentile=peak_percentile,
        threshold_percentile=BACKGROUND_PERCENTILE_THRESHOLD,
    )
    return {
        "event": event,
        "real_metrics": real_metrics,
        "pseudo_event_count": int(len(pseudo_metrics)),
        "pseudo_event_times_fs": [float(value) for value in pseudo_times_fs],
        "pseudo_metrics": pseudo_metrics,
        "backward_positive_energy_percentile": float(energy_percentile),
        "backward_peak_flux_percentile": float(peak_percentile),
        "backward_packet_qualified": packet_qualified,
        "background_separated": bool(separated),
        "all_metrics_finite": bool(
            _metrics_finite(real_metrics) and all(_metrics_finite(item) for item in pseudo_metrics)
        ),
    }, real_arrays


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
    x_event_times = np.asarray(
        [event["transition_start_time_fs"] for event in nearest_x], dtype=np.float64
    )
    complete_times = complete_real_event_times(
        times,
        x_event_times,
        baseline_start_offset_fs=BASELINE_START_OFFSET_FS,
        post_end_offset_fs=POST_END_OFFSET_FS,
    )
    pseudo_times = pseudo_event_times(
        times,
        x_event_times,
        cadence_fs=PSEUDO_CADENCE_FS,
        baseline_start_offset_fs=BASELINE_START_OFFSET_FS,
        post_end_offset_fs=POST_END_OFFSET_FS,
    )
    complete_events = [
        event
        for event in nearest_x
        if np.any(
            np.isclose(
                complete_times,
                float(event["transition_start_time_fs"]),
                rtol=0.0,
                atol=1.0e-8,
            )
        )
    ]

    analyses: list[dict] = []
    profile_payload: dict[str, np.ndarray] = {}
    for index, event in enumerate(complete_events):
        analysis, arrays = _analyze_real_event(
            event=event,
            times_fs=times,
            jx_frames=jx,
            parameters=parameters,
            pseudo_times_fs=pseudo_times,
            vmax_sites_per_ps=vmax_sites_per_ps,
        )
        analyses.append(analysis)
        for name, values in arrays.items():
            profile_payload[f"event{index}_{name}"] = values

    tolerance = size_scaled_energy_balance_tolerance_eV(parameters.n_sites)
    finite_analysis = bool(
        analyses
        and all(item["all_metrics_finite"] for item in analyses)
        and np.all(np.isfinite(pseudo_times))
    )
    numerical_checks = {
        "static_relaxation_converged": bool(static["converged"]),
        "complete_requested_steps": True,
        "energy_work_balance": bool(extensive_energy_balance_passes(max_balance, parameters.n_sites)),
        "electronic_norm": bool(max_norm_error < 1.0e-10),
        "projected_zero_modes": bool(max_zero_mode < 1.0e-10),
        "at_least_one_complete_real_event": bool(len(complete_events) >= 1),
        "at_least_five_pseudo_events": bool(pseudo_times.size >= 5),
        "finite_event_and_background_metrics": finite_analysis,
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
        "complete_real_event_count": int(len(complete_events)),
        "pseudo_event_count": int(pseudo_times.size),
        "complete_event_analyses": analyses,
        "background_separated_event_count": int(
            sum(bool(item["background_separated"]) for item in analyses)
        ),
        "numerical_checks": numerical_checks,
        "numerical_pass": bool(all(numerical_checks.values())),
        "hamiltonian_evaluations": int(h_eval),
        "hamiltonian_applications": int(h_apply),
    }
    return result, profile_payload


def _replication_summary(conditions: list[dict]) -> dict:
    anisotropic = next(
        (item for item in conditions if np.isclose(item["anisotropy_ratio"], 0.15)), None
    )
    if anisotropic is None:
        return {
            "anisotropic_condition_present": False,
            "anisotropic_two_complete_events": False,
            "anisotropic_two_background_separated_events": False,
            "replicated_event_associated_retrograde_evidence": False,
        }
    two_complete = anisotropic["complete_real_event_count"] >= 2
    two_separated = anisotropic["background_separated_event_count"] >= 2
    return {
        "anisotropic_condition_present": True,
        "anisotropic_two_complete_events": bool(two_complete),
        "anisotropic_two_background_separated_events": bool(two_separated),
        "replicated_event_associated_retrograde_evidence": bool(two_complete and two_separated),
    }


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
        "profiles_written_for_every_complete_real_event": bool(
            len(profile_output)
            == 3 * sum(item["complete_real_event_count"] for item in conditions)
        ),
    }
    replication = _replication_summary(conditions)
    payload = {
        "scope": "IP1k natural-hop wake replication against matched event-free same-trajectory background; no transport coefficient or material phonon parameter claim",
        "size": int(args.size),
        "ratios": [float(value) for value in args.ratios],
        "field_mV_per_A": float(args.field_mV_per_A),
        "temperature_K": 0.0,
        "thermostat": None,
        "IDC": None,
        "dt_fs": float(args.dt_fs),
        "final_time_fs": float(args.final_time_fs),
        "sample_interval_fs": float(args.sample_interval_fs),
        "harmonic_max_group_velocity_sites_per_ps": vmax,
        "baseline_offsets_fs": [BASELINE_START_OFFSET_FS, BASELINE_END_OFFSET_FS],
        "post_end_offset_fs": POST_END_OFFSET_FS,
        "pseudo_event_cadence_fs": PSEUDO_CADENCE_FS,
        "delay_search_window_fs": [LAG_MIN_FS, LAG_MAX_FS],
        "minimum_delay_correlation": MIN_CORRELATION,
        "background_percentile_threshold": BACKGROUND_PERCENTILE_THRESHOLD,
        "conditions": conditions,
        "replication_decision": replication,
        "numerical_checks": numerical_checks,
        "numerical_pass": bool(all(numerical_checks.values())),
        "interpretation_guard": {
            "pseudo_windows_are_correlated_not_independent_samples": True,
            "percentiles_are_robustness_discriminators_not_formal_p_values": True,
            "field_remains_on_in_real_and_pseudo_windows": True,
            "d1_to_d2_delay_is_not_unique_normal_mode_group_velocity": True,
            "weak_retrograde_and_forward_dominated_total_radiation_can_coexist": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(args.profiles, **profile_output)

    lines = [
        "# IP1k natural wake replication and matched background",
        "",
        f"40x40-equivalent requested cell: {args.size}x{args.size}; T=0; E=+{args.field_mV_per_A:g} mV/A; dt={args.dt_fs:g} fs; final={args.final_time_fs/1000:g} ps.",
        f"Harmonic vmax: {vmax:.4f} sites/ps. Delay gate: {LAG_MIN_FS:g}-{LAG_MAX_FS:g} fs, correlation >= {MIN_CORRELATION:.2f}.",
        "",
        "| J0y/J0x | complete real events | pseudo windows | background-separated real events | replicated? |",
        "|---:|---:|---:|---:|---:|",
    ]
    for condition in conditions:
        replicated = bool(
            np.isclose(condition["anisotropy_ratio"], 0.15)
            and condition["complete_real_event_count"] >= 2
            and condition["background_separated_event_count"] >= 2
        )
        lines.append(
            f"| {condition['anisotropy_ratio']:.2f} | {condition['complete_real_event_count']} | "
            f"{condition['pseudo_event_count']} | {condition['background_separated_event_count']} | "
            f"{'YES' if replicated else 'NO'} |"
        )
        for index, analysis in enumerate(condition["complete_event_analyses"], start=1):
            event = analysis["event"]
            metrics = analysis["real_metrics"]
            back = metrics["backward_d1_to_d2_delay"]
            lines.extend(
                [
                    "",
                    f"### J0y/J0x={condition['anisotropy_ratio']:.2f}, real event {index}",
                    "",
                    f"- event: {event['source_site']} -> {event['target_site']} ({event['direction']}), t0={event['transition_start_time_fs']:.1f} fs;",
                    f"- backward d2 positive energy: {metrics['backward_positive_outward_energy_eV']:.6e} eV (pseudo percentile {analysis['backward_positive_energy_percentile']:.1f});",
                    f"- backward d2 peak flux: {metrics['backward_peak_outward_flux_eV_per_fs']:.6e} eV/fs (pseudo percentile {analysis['backward_peak_flux_percentile']:.1f});",
                    f"- backward delay: {back['lag_fs']:.1f} fs, {back['speed_sites_per_ps']:.4f} sites/ps, corr={back['correlation']:.6f}, packet={'PASS' if analysis['backward_packet_qualified'] else 'FAIL'};",
                    f"- background separated: {'YES' if analysis['background_separated'] else 'NO'}.",
                ]
            )
    lines.extend(
        [
            "",
            "## Replication decision",
            "",
            f"- anisotropic two complete real events: {'YES' if replication['anisotropic_two_complete_events'] else 'NO'};",
            f"- anisotropic two background-separated events: {'YES' if replication['anisotropic_two_background_separated_events'] else 'NO'};",
            f"- replicated event-associated retrograde evidence: {'YES' if replication['replicated_event_associated_retrograde_evidence'] else 'NO'};",
            "",
            "Physical replication is intentionally not a numerical acceptance gate.",
            "",
            "## Numerical gates",
            "",
        ]
    )
    for name, value in numerical_checks.items():
        lines.append(f"- {name}: {'PASS' if value else 'FAIL'}")
    lines.append("")
    lines.append(f"Numerical status: {'PASS' if payload['numerical_pass'] else 'FAIL'}")
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"\nWrote {args.output}")
    print(f"Wrote {args.markdown}")
    print(f"Wrote {args.profiles}")
    if not payload["numerical_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
