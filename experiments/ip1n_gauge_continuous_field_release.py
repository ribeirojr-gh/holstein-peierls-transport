#!/usr/bin/env python3
"""IP1n gauge-continuous field-release audit of isotropic post-hop memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

import ip1l_interhop_wake_memory as base
from holstein_peierls.dynamics.coupled import CoupledEhrenfestState
from holstein_peierls.dynamics.driven import (
    coupled_field_verlet_step,
    field_dynamic_total_energy,
)
from holstein_peierls.dynamics.field import (
    UniformElectricField2D,
    build_sparse_field_hamiltonian,
)
from holstein_peierls.dynamics.field_release import (
    HeldPeierlsPhase2D,
    empirical_percentile,
    sliding_amplitude_windows,
    sustained_pattern_gate,
    trailing_amplitude_series,
    trailing_outward_matrix,
)
from holstein_peierls.dynamics.hopping_observables import PersistentSiteTracker
from holstein_peierls.dynamics.interhop_memory import interval_mask
from holstein_peierls.dynamics.numerical_validation import (
    extensive_energy_balance_passes,
    size_scaled_energy_balance_tolerance_eV,
)
from holstein_peierls.dynamics.phonon_recurrence import intermolecular_recurrence_scales
from holstein_peierls.dynamics.phonon_wake import intermolecular_energy_flux
from holstein_peierls.dynamics.single_hop_memory import continuation_neighbor
from holstein_peierls.dynamics.thermal import zero_mode_means
from holstein_peierls.dynamics.wavepacket_flux import (
    integrated_outward_energy_eV,
    outward_boundary_flux_series,
)
from holstein_peierls.parameters import StaticPolaronParameters


PERSISTENCE_FS = 50.0
LATTICE_SPACING_A = 3.0
TRANSVERSE_HALF_WIDTH_SITES = 3
PRE_BASELINE_START_OFFSET_FS = -700.0
PRE_BASELINE_END_OFFSET_FS = -100.0
WINDOW_WIDTH_FS = 100.0
BACKGROUND_STEP_FS = 20.0
PERCENTILE_THRESHOLD = 95.0
LATE_OFFSET_FS = 1000.0
MINIMUM_PASSING_BINS = 3
MINIMUM_LATEST_OFFSET_FS = 1500.0
PRIMARY_DISTANCES = (1, 2, 3, 4)


def _copy_state(state: CoupledEhrenfestState) -> CoupledEhrenfestState:
    return CoupledEhrenfestState(
        state.lattice.copy(),
        state.velocity.copy(),
        np.asarray(state.electronic_state, dtype=np.complex128).copy(),
    )


def _states_identical(a: CoupledEhrenfestState, b: CoupledEhrenfestState) -> bool:
    return bool(
        np.array_equal(a.lattice.u, b.lattice.u)
        and np.array_equal(a.lattice.vx, b.lattice.vx)
        and np.array_equal(a.lattice.vy, b.lattice.vy)
        and np.array_equal(a.velocity.u, b.velocity.u)
        and np.array_equal(a.velocity.vx, b.velocity.vx)
        and np.array_equal(a.velocity.vy, b.velocity.vy)
        and np.array_equal(a.electronic_state, b.electronic_state)
    )


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


def _branch_tracker(nx: int, ny: int, sample_interval_fs: float) -> PersistentSiteTracker:
    persistence_samples = base._integer_stride(PERSISTENCE_FS, sample_interval_fs, "persistence_fs")
    return PersistentSiteTracker(
        nx,
        ny,
        persistence_samples=persistence_samples,
        minimum_max_population=0.10,
        minimum_dominance_margin=0.02,
    )


def _run_branch(
    *,
    label: str,
    initial: CoupledEhrenfestState,
    parameters: StaticPolaronParameters,
    field,
    switch_time_fs: float,
    continuation_fs: float,
    dt_fs: float,
    sample_interval_fs: float,
    energy_sample_interval_fs: float,
    krylov_dimension: int,
    zero_external_work_expected: bool,
) -> tuple[dict, dict[str, np.ndarray]]:
    steps = base._integer_stride(continuation_fs, dt_fs, "continuation_fs")
    sample_stride = base._integer_stride(sample_interval_fs, dt_fs, "sample_interval_fs")
    energy_stride = base._integer_stride(
        energy_sample_interval_fs, dt_fs, "energy_sample_interval_fs"
    )
    current = _copy_state(initial)
    initial_energy = field_dynamic_total_energy(
        current.lattice,
        current.velocity,
        parameters,
        current.electronic_state,
        field,
        switch_time_fs,
    ).total
    initial_norm = float(np.linalg.norm(current.electronic_state))
    tracker = _branch_tracker(parameters.nx, parameters.ny, sample_interval_fs)
    p0 = np.abs(current.electronic_state) ** 2
    tracker.update(p0 / float(np.sum(p0)), switch_time_fs)

    times = [float(switch_time_fs)]
    jx_frames = [
        intermolecular_energy_flux(current.lattice, current.velocity, parameters).jx.astype(
            np.float32, copy=True
        )
    ]
    branch_events: list[dict] = []
    accumulated_work = 0.0
    max_balance = 0.0
    max_norm_error = 0.0
    max_zero_mode = 0.0
    h_eval = 0
    h_apply = 0
    start = perf_counter()

    for step in range(steps):
        time_fs = switch_time_fs + step * dt_fs
        current, work, evaluations, applications = coupled_field_verlet_step(
            current,
            parameters,
            field,
            time_fs,
            dt_fs,
            electronic_method="cfm4_lanczos",
            krylov_dimension=krylov_dimension,
        )
        new_time = time_fs + dt_fs
        accumulated_work += float(work)
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
            event = tracker.update(populations, new_time)
            if event is not None:
                branch_events.append(_event_record(event))
            flux = intermolecular_energy_flux(current.lattice, current.velocity, parameters)
            times.append(float(new_time))
            jx_frames.append(np.asarray(flux.jx, dtype=np.float32))

        if (step + 1) % energy_stride == 0 or step + 1 == steps:
            energy = field_dynamic_total_energy(
                current.lattice,
                current.velocity,
                parameters,
                current.electronic_state,
                field,
                new_time,
            ).total
            if zero_external_work_expected:
                residual = abs(energy - initial_energy)
            else:
                residual = abs((energy - initial_energy) - accumulated_work)
            max_balance = max(max_balance, float(residual))

    elapsed = perf_counter() - start
    result = {
        "label": label,
        "steps": int(steps),
        "elapsed_seconds": float(elapsed),
        "accumulated_external_work_eV": float(accumulated_work),
        "maximum_energy_balance_residual_eV": float(max_balance),
        "maximum_norm_error": float(max_norm_error),
        "maximum_zero_mode_mean": float(max_zero_mode),
        "persistent_events_after_switch": branch_events,
        "hamiltonian_evaluations": int(h_eval),
        "hamiltonian_applications": int(h_apply),
    }
    arrays = {
        "times_fs": np.asarray(times, dtype=np.float64),
        "jx_eV_per_fs": np.stack(jx_frames, axis=0),
    }
    return result, arrays


def _amplitude_records(
    *,
    times_fs: np.ndarray,
    amplitude: np.ndarray,
    switch_time_fs: float,
    background_rms: np.ndarray,
    background_peak: np.ndarray,
) -> list[dict]:
    windows = sliding_amplitude_windows(
        times_fs,
        amplitude,
        switch_time_fs,
        float(times_fs[-1]),
        width_fs=WINDOW_WIDTH_FS,
        step_fs=WINDOW_WIDTH_FS,
    )
    return [
        {
            "start_fs": float(record.start_fs),
            "end_fs": float(record.end_fs),
            "center_fs": float(record.center_fs),
            "center_offset_fs": float(record.center_fs - switch_time_fs),
            "rms_amplitude_eV_per_fs": float(record.rms_amplitude_eV_per_fs),
            "peak_amplitude_eV_per_fs": float(record.peak_amplitude_eV_per_fs),
            "rms_percentile": empirical_percentile(
                record.rms_amplitude_eV_per_fs, background_rms
            ),
            "peak_percentile": empirical_percentile(
                record.peak_amplitude_eV_per_fs, background_peak
            ),
        }
        for record in windows
    ]


def _directional_summary(
    *,
    times_fs: np.ndarray,
    shifted_profiles: np.ndarray,
    s_axis: np.ndarray,
) -> dict:
    by_distance = []
    for distance in PRIMARY_DISTANCES:
        values = outward_boundary_flux_series(
            shifted_profiles, s_axis, distance, side="backward"
        )
        by_distance.append(
            {
                "distance_sites": int(distance),
                "positive_fraction": float(np.mean(values > 0.0)),
                "positive_outward_energy_eV": float(
                    integrated_outward_energy_eV(times_fs, values, positive_only=True)
                ),
                "net_outward_energy_eV": float(
                    integrated_outward_energy_eV(times_fs, values, positive_only=False)
                ),
                "mean_outward_flux_eV_per_fs": float(np.mean(values)),
                "peak_outward_flux_eV_per_fs": float(np.max(values)),
            }
        )
    return {"by_distance": by_distance}


def _analyze_branch(
    *,
    branch: dict,
    arrays: dict[str, np.ndarray],
    pre_profile: np.ndarray,
    s_axis: np.ndarray,
    switch_time_fs: float,
    background_rms: np.ndarray,
    background_peak: np.ndarray,
    vmax_sites_per_ps: float,
) -> tuple[dict, dict[str, np.ndarray]]:
    profiles = arrays["profiles"]
    shifted = profiles - pre_profile[None, :]
    trailing = trailing_outward_matrix(shifted, s_axis, distances_sites=PRIMARY_DISTANCES)
    amplitude = trailing_amplitude_series(trailing)
    records = _amplitude_records(
        times_fs=arrays["times_fs"],
        amplitude=amplitude,
        switch_time_fs=switch_time_fs,
        background_rms=background_rms,
        background_peak=background_peak,
    )
    gate = sustained_pattern_gate(
        records,
        switch_time_fs=switch_time_fs,
        percentile_threshold=PERCENTILE_THRESHOLD,
        late_offset_fs=LATE_OFFSET_FS,
        minimum_passing_bins=MINIMUM_PASSING_BINS,
        minimum_latest_offset_fs=MINIMUM_LATEST_OFFSET_FS,
    )
    early_end = min(float(arrays["times_fs"][-1]), switch_time_fs + 800.0)
    early_mask = interval_mask(arrays["times_fs"], switch_time_fs, early_end)
    d1d2 = base._packet_delay(
        arrays["times_fs"][early_mask] - switch_time_fs,
        shifted[early_mask],
        s_axis,
        side="backward",
        vmax_sites_per_ps=vmax_sites_per_ps,
    )
    # Shift the s axis by one boundary so the common helper can evaluate d3->d4
    # as an adjacent-boundary delay without changing the validated correlation code.
    d3 = outward_boundary_flux_series(shifted[early_mask], s_axis, 3, side="backward")
    d4 = outward_boundary_flux_series(shifted[early_mask], s_axis, 4, side="backward")
    from holstein_peierls.dynamics.wavepacket_flux import normalized_flux_delay
    delay34 = normalized_flux_delay(
        arrays["times_fs"][early_mask] - switch_time_fs,
        d3,
        d4,
        lag_min_fs=300.0,
        lag_max_fs=700.0,
    )
    d34 = {
        "lag_fs": float(delay34.lag_fs),
        "speed_sites_per_ps": float(delay34.speed_sites_per_ps),
        "correlation": float(delay34.correlation),
        "both_boundaries_majority_outward": bool(
            np.mean(d3 > 0.0) > 0.5 and np.mean(d4 > 0.0) > 0.5
        ),
    }
    result = {
        **branch,
        "amplitude_windows": records,
        "sustained_pattern_gate": gate,
        "directional_summary": _directional_summary(
            times_fs=arrays["times_fs"], shifted_profiles=shifted, s_axis=s_axis
        ),
        "early_d1_to_d2_delay": d1d2,
        "early_d3_to_d4_delay": d34,
        "mean_trailing_amplitude_eV_per_fs": float(np.mean(amplitude)),
        "late_mean_trailing_amplitude_eV_per_fs": float(
            np.mean(amplitude[arrays["times_fs"] >= switch_time_fs + LATE_OFFSET_FS])
        ),
    }
    out_arrays = {
        "times_fs": arrays["times_fs"],
        "shifted_profiles_eV_per_fs": shifted,
        "trailing_outward_eV_per_fs": trailing,
        "trailing_amplitude_eV_per_fs": amplitude,
    }
    return result, out_arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=40)
    parser.add_argument("--field-mv-per-A", dest="field_mV_per_A", type=float, default=10.0)
    parser.add_argument("--dt-fs", type=float, default=0.2)
    parser.add_argument("--search-time-fs", type=float, default=4000.0)
    parser.add_argument("--continuation-fs", type=float, default=2000.0)
    parser.add_argument("--sample-interval-fs", type=float, default=2.0)
    parser.add_argument("--energy-sample-interval-fs", type=float, default=10.0)
    parser.add_argument("--krylov-dimension", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    args = parser.parse_args()

    initial, parameters, static = base._prepare_initial_state(args.size, 1.0)
    field = UniformElectricField2D.from_millivolt_per_angstrom(
        args.field_mV_per_A,
        angle_radians=0.0,
        ax_angstrom=LATTICE_SPACING_A,
        ay_angstrom=LATTICE_SPACING_A,
    )
    search_steps = base._integer_stride(args.search_time_fs, args.dt_fs, "search_time_fs")
    sample_stride = base._integer_stride(
        args.sample_interval_fs, args.dt_fs, "sample_interval_fs"
    )
    tracker = _branch_tracker(parameters.nx, parameters.ny, args.sample_interval_fs)
    current = _copy_state(initial)
    p0 = np.abs(current.electronic_state) ** 2
    tracker.update(p0 / float(np.sum(p0)), 0.0)
    pre_times = [0.0]
    pre_jx = [
        intermolecular_energy_flux(current.lattice, current.velocity, parameters).jx.astype(
            np.float32, copy=True
        )
    ]
    pre_events: list[dict] = []
    selected_event = None
    selected_state = None
    start = perf_counter()

    for step in range(search_steps):
        time_fs = step * args.dt_fs
        current, _, _, _ = coupled_field_verlet_step(
            current,
            parameters,
            field,
            time_fs,
            args.dt_fs,
            electronic_method="cfm4_lanczos",
            krylov_dimension=args.krylov_dimension,
        )
        new_time = time_fs + args.dt_fs
        if (step + 1) % sample_stride == 0:
            populations = np.abs(current.electronic_state) ** 2
            populations /= float(np.sum(populations))
            event = tracker.update(populations, new_time)
            if event is not None:
                record = _event_record(event)
                pre_events.append(record)
                if (
                    selected_event is None
                    and record["is_nearest_neighbor"]
                    and abs(record["dx_sites"]) == 1
                    and record["dy_sites"] == 0
                ):
                    selected_event = record
                    selected_state = _copy_state(current)
            flux = intermolecular_energy_flux(current.lattice, current.velocity, parameters)
            pre_times.append(float(new_time))
            pre_jx.append(np.asarray(flux.jx, dtype=np.float32))
            if selected_event is not None:
                break

    search_elapsed = perf_counter() - start
    pre_times_arr = np.asarray(pre_times, dtype=np.float64)
    pre_jx_arr = np.stack(pre_jx, axis=0)
    analysis_available = selected_event is not None and selected_state is not None
    branch_payload = None
    serialized: dict[str, np.ndarray] = {
        "pre_times_fs": pre_times_arr,
        "pre_jx_eV_per_fs": pre_jx_arr,
    }
    tolerance = size_scaled_energy_balance_tolerance_eV(parameters.n_sites)

    numerical_checks = {
        "static_relaxation_converged": bool(static["converged"]),
        "natural_x_event_found": bool(analysis_available),
    }

    if analysis_available:
        t0 = float(selected_event["transition_start_time_fs"])
        ts = float(selected_event["accepted_time_fs"])
        if not np.isclose(pre_times_arr[-1], ts, rtol=0.0, atol=1.0e-10):
            raise RuntimeError("branch state was not captured at accepted event sample")
        source = int(selected_event["target_site"])
        target = continuation_neighbor(
            source,
            dx_sites=int(selected_event["dx_sites"]),
            dy_sites=int(selected_event["dy_sites"]),
            nx=parameters.nx,
            ny=parameters.ny,
        )
        s_axis, pre_profiles = base._project_profiles(
            pre_jx_arr,
            source_site=source,
            target_site=target,
            nx=parameters.nx,
            ny=parameters.ny,
            direction_sign=int(selected_event["dx_sites"]),
        )
        pre_start = t0 + PRE_BASELINE_START_OFFSET_FS
        pre_end = t0 + PRE_BASELINE_END_OFFSET_FS
        pre_mask = interval_mask(pre_times_arr, pre_start, pre_end)
        if np.count_nonzero(pre_mask) < 2:
            raise RuntimeError("pre-hop baseline is incomplete")
        pre_profile = np.mean(pre_profiles[pre_mask], axis=0)
        pre_shifted = pre_profiles - pre_profile[None, :]
        pre_trailing = trailing_outward_matrix(
            pre_shifted, s_axis, distances_sites=PRIMARY_DISTANCES
        )
        pre_amplitude = trailing_amplitude_series(pre_trailing)
        background_windows = sliding_amplitude_windows(
            pre_times_arr,
            pre_amplitude,
            pre_start,
            pre_end,
            width_fs=WINDOW_WIDTH_FS,
            step_fs=BACKGROUND_STEP_FS,
        )
        background_rms = np.asarray(
            [record.rms_amplitude_eV_per_fs for record in background_windows], dtype=np.float64
        )
        background_peak = np.asarray(
            [record.peak_amplitude_eV_per_fs for record in background_windows], dtype=np.float64
        )

        held = HeldPeierlsPhase2D.from_driving_field(field, ts)
        phi_drive = np.asarray(field.phases(ts), dtype=np.float64)
        phi_held = np.asarray(held.phases(ts), dtype=np.float64)
        phase_continuity_error = float(np.max(np.abs(phi_drive - phi_held)))
        h_drive = build_sparse_field_hamiltonian(
            selected_state.lattice, parameters, field, ts
        )
        h_held = build_sparse_field_hamiltonian(
            selected_state.lattice, parameters, held, ts
        )
        h_continuity_error = float(np.max(np.abs((h_drive - h_held).data))) if (h_drive - h_held).nnz else 0.0

        on_initial = _copy_state(selected_state)
        off_initial = _copy_state(selected_state)
        identical_branch_states = _states_identical(on_initial, off_initial)
        on_result, on_arrays = _run_branch(
            label="field_on",
            initial=on_initial,
            parameters=parameters,
            field=field,
            switch_time_fs=ts,
            continuation_fs=args.continuation_fs,
            dt_fs=args.dt_fs,
            sample_interval_fs=args.sample_interval_fs,
            energy_sample_interval_fs=args.energy_sample_interval_fs,
            krylov_dimension=args.krylov_dimension,
            zero_external_work_expected=False,
        )
        off_result, off_arrays = _run_branch(
            label="released_held_phase",
            initial=off_initial,
            parameters=parameters,
            field=held,
            switch_time_fs=ts,
            continuation_fs=args.continuation_fs,
            dt_fs=args.dt_fs,
            sample_interval_fs=args.sample_interval_fs,
            energy_sample_interval_fs=args.energy_sample_interval_fs,
            krylov_dimension=args.krylov_dimension,
            zero_external_work_expected=True,
        )
        for arrays in (on_arrays, off_arrays):
            _, arrays["profiles"] = base._project_profiles(
                arrays["jx_eV_per_fs"],
                source_site=source,
                target_site=target,
                nx=parameters.nx,
                ny=parameters.ny,
                direction_sign=int(selected_event["dx_sites"]),
            )

        recurrence = intermolecular_recurrence_scales(
            parameters, lattice_spacing_A=LATTICE_SPACING_A, gamma_v_per_fs=0.0
        )
        vmax = float(recurrence.max_group_velocity_sites_per_fs * 1000.0)
        on_analysis, on_serial = _analyze_branch(
            branch=on_result,
            arrays=on_arrays,
            pre_profile=pre_profile,
            s_axis=s_axis,
            switch_time_fs=ts,
            background_rms=background_rms,
            background_peak=background_peak,
            vmax_sites_per_ps=vmax,
        )
        off_analysis, off_serial = _analyze_branch(
            branch=off_result,
            arrays=off_arrays,
            pre_profile=pre_profile,
            s_axis=s_axis,
            switch_time_fs=ts,
            background_rms=background_rms,
            background_peak=background_peak,
            vmax_sites_per_ps=vmax,
        )

        late_on = on_analysis["late_mean_trailing_amplitude_eV_per_fs"]
        late_off = off_analysis["late_mean_trailing_amplitude_eV_per_fs"]
        branch_payload = {
            "event": selected_event,
            "switch_time_fs": ts,
            "post_hop_frame_source_site": source,
            "post_hop_frame_target_site": target,
            "phase_at_switch_radians": phi_drive.tolist(),
            "phase_continuity_error": phase_continuity_error,
            "hamiltonian_continuity_max_abs_eV": h_continuity_error,
            "held_phase_rates_per_fs": list(held.phase_rates_per_fs()),
            "pre_hop_reference_window_fs": [pre_start, pre_end],
            "background_window_count": int(len(background_windows)),
            "field_on": on_analysis,
            "released": off_analysis,
            "late_released_to_on_amplitude_ratio": float(late_off / max(late_on, 1.0e-30)),
        }
        released_memory = bool(
            off_analysis["sustained_pattern_gate"]["sustained_pattern_memory"]
        )
        on_memory = bool(on_analysis["sustained_pattern_gate"]["sustained_pattern_memory"])
        physical = {
            "field_on_sustained_pattern_memory": on_memory,
            "released_sustained_pattern_memory": released_memory,
            "autonomous_stored_lattice_memory": released_memory,
            "continued_field_required_for_late_pattern": bool(on_memory and not released_memory),
        }

        numerical_checks.update(
            {
                "selected_event_is_first_persistent_event": bool(
                    pre_events and pre_events[0] == selected_event
                ),
                "phase_continuity": bool(phase_continuity_error < 1.0e-14),
                "hamiltonian_continuity": bool(h_continuity_error < 1.0e-13),
                "held_phase_rates_zero": bool(
                    max(abs(value) for value in held.phase_rates_per_fs()) < 1.0e-15
                ),
                "branch_states_identical_at_switch": bool(identical_branch_states),
                "released_external_work_zero": bool(
                    abs(off_result["accumulated_external_work_eV"]) < 1.0e-12
                ),
                "released_energy_conservation": bool(
                    extensive_energy_balance_passes(
                        off_result["maximum_energy_balance_residual_eV"], parameters.n_sites
                    )
                ),
                "field_on_energy_work_balance": bool(
                    extensive_energy_balance_passes(
                        on_result["maximum_energy_balance_residual_eV"], parameters.n_sites
                    )
                ),
                "field_on_electronic_norm": bool(on_result["maximum_norm_error"] < 1.0e-10),
                "released_electronic_norm": bool(off_result["maximum_norm_error"] < 1.0e-10),
                "field_on_projected_zero_modes": bool(
                    on_result["maximum_zero_mode_mean"] < 1.0e-10
                ),
                "released_projected_zero_modes": bool(
                    off_result["maximum_zero_mode_mean"] < 1.0e-10
                ),
                "finite_branch_diagnostics": bool(
                    np.isfinite(late_on)
                    and np.isfinite(late_off)
                    and np.isfinite(on_result["maximum_energy_balance_residual_eV"])
                    and np.isfinite(off_result["maximum_energy_balance_residual_eV"])
                ),
                "branch_current_trajectories_serializable": bool(
                    np.all(np.isfinite(on_arrays["jx_eV_per_fs"]))
                    and np.all(np.isfinite(off_arrays["jx_eV_per_fs"]))
                ),
            }
        )
        serialized.update(
            {
                "s_sites": np.asarray(s_axis, dtype=np.int64),
                "pre_longitudinal_flux_eV_per_fs": np.asarray(pre_profiles, dtype=np.float32),
                "field_on_times_fs": on_serial["times_fs"],
                "field_on_shifted_profiles_eV_per_fs": np.asarray(
                    on_serial["shifted_profiles_eV_per_fs"], dtype=np.float32
                ),
                "field_on_trailing_outward_eV_per_fs": np.asarray(
                    on_serial["trailing_outward_eV_per_fs"], dtype=np.float32
                ),
                "field_on_trailing_amplitude_eV_per_fs": on_serial[
                    "trailing_amplitude_eV_per_fs"
                ],
                "released_times_fs": off_serial["times_fs"],
                "released_shifted_profiles_eV_per_fs": np.asarray(
                    off_serial["shifted_profiles_eV_per_fs"], dtype=np.float32
                ),
                "released_trailing_outward_eV_per_fs": np.asarray(
                    off_serial["trailing_outward_eV_per_fs"], dtype=np.float32
                ),
                "released_trailing_amplitude_eV_per_fs": off_serial[
                    "trailing_amplitude_eV_per_fs"
                ],
            }
        )
    else:
        physical = {
            "field_on_sustained_pattern_memory": False,
            "released_sustained_pattern_memory": False,
            "autonomous_stored_lattice_memory": False,
            "continued_field_required_for_late_pattern": False,
        }

    numerical_pass = bool(all(numerical_checks.values()))
    payload = {
        "scope": "IP1n gauge-continuous field-release audit of isotropic post-hop phase-structured lattice-current memory",
        "size": int(args.size),
        "anisotropy_ratio": 1.0,
        "initial_field_mV_per_A": float(args.field_mV_per_A),
        "temperature_K": 0.0,
        "thermostat": None,
        "IDC": None,
        "dt_fs": float(args.dt_fs),
        "continuation_fs": float(args.continuation_fs),
        "search_elapsed_seconds": float(search_elapsed),
        "static": static,
        "pre_switch_persistent_events": pre_events,
        "analysis": branch_payload,
        "physical_decision": physical,
        "energy_balance_tolerance_eV": float(tolerance),
        "numerical_checks": numerical_checks,
        "numerical_pass": numerical_pass,
        "interpretation_guard": {
            "constant_vector_potential_means_zero_post_release_electric_field": True,
            "alternating_boundary_signs_are_not_monotonic_outward_transport": True,
            "trailing_rms_is_pattern_strength_not_directional_energy_flux": True,
            "no_transport_coefficient_or_material_lifetime_claim": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(args.trajectory, **serialized)

    lines = [
        "# IP1n gauge-continuous field-release audit",
        "",
        f"Cell {args.size}x{args.size}; isotropic; initial E=+{args.field_mV_per_A:g} mV/A; T=0; continuation={args.continuation_fs/1000:g} ps.",
        "",
    ]
    if branch_payload is None:
        lines.append("No qualifying natural x event was found before the search cutoff.")
    else:
        event = branch_payload["event"]
        on = branch_payload["field_on"]
        off = branch_payload["released"]
        lines.extend(
            [
                f"Event {event['source_site']}->{event['target_site']} ({event['direction']}): start={event['transition_start_time_fs']:.1f} fs; accepted/switch={event['accepted_time_fs']:.1f} fs.",
                f"Phase continuity error={branch_payload['phase_continuity_error']:.3e}; Hamiltonian continuity error={branch_payload['hamiltonian_continuity_max_abs_eV']:.3e} eV.",
                "",
                "## Field-on continuation",
                "",
                f"- late passing bins: {on['sustained_pattern_gate']['late_passing_bin_count']}/{on['sustained_pattern_gate']['late_bin_count']};",
                f"- latest passing center: {on['sustained_pattern_gate']['latest_passing_center_fs']};",
                f"- sustained pattern memory: {'YES' if on['sustained_pattern_gate']['sustained_pattern_memory'] else 'NO'};",
                f"- incremental energy-work residual: {on['maximum_energy_balance_residual_eV']:.6e} eV;",
                "",
                "## Gauge-continuous released continuation",
                "",
                f"- accumulated external work: {off['accumulated_external_work_eV']:.6e} eV;",
                f"- maximum matter-energy drift: {off['maximum_energy_balance_residual_eV']:.6e} eV;",
                f"- late passing bins: {off['sustained_pattern_gate']['late_passing_bin_count']}/{off['sustained_pattern_gate']['late_bin_count']};",
                f"- latest passing center: {off['sustained_pattern_gate']['latest_passing_center_fs']};",
                f"- sustained pattern memory: {'YES' if off['sustained_pattern_gate']['sustained_pattern_memory'] else 'NO'};",
                f"- released/on late amplitude ratio: {branch_payload['late_released_to_on_amplitude_ratio']:.6f};",
                "",
                "## Physical decision",
                "",
                f"- autonomous stored lattice memory: {'YES' if physical['autonomous_stored_lattice_memory'] else 'NO'};",
                f"- continued field required for late pattern: {'YES' if physical['continued_field_required_for_late_pattern'] else 'NO'};",
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
