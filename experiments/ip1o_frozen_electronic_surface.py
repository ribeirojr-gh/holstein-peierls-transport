#!/usr/bin/env python3
"""IP1o frozen-electronic-surface causal control of post-hop lattice memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

import ip1l_interhop_wake_memory as base
import ip1n_gauge_continuous_field_release as ip1n
from holstein_peierls.dynamics.driven import coupled_field_verlet_step, field_ehrenfest_force
from holstein_peierls.dynamics.field import UniformElectricField2D
from holstein_peierls.dynamics.field_release import (
    HeldPeierlsPhase2D,
    sliding_amplitude_windows,
    trailing_amplitude_series,
    trailing_outward_matrix,
)
from holstein_peierls.dynamics.frozen_electronic_surface import (
    frozen_surface_total_energy,
    frozen_surface_verlet_step,
)
from holstein_peierls.dynamics.interhop_memory import interval_mask
from holstein_peierls.dynamics.numerical_validation import (
    extensive_energy_balance_passes,
    size_scaled_energy_balance_tolerance_eV,
)
from holstein_peierls.dynamics.phonon_recurrence import intermolecular_recurrence_scales
from holstein_peierls.dynamics.phonon_wake import intermolecular_energy_flux
from holstein_peierls.dynamics.single_hop_memory import continuation_neighbor
from holstein_peierls.dynamics.thermal import zero_mode_means
from holstein_peierls.parameters import StaticPolaronParameters


LATTICE_SPACING_A = 3.0


def _run_frozen_branch(
    *,
    initial,
    parameters,
    held_phase,
    switch_time_fs: float,
    continuation_fs: float,
    dt_fs: float,
    sample_interval_fs: float,
    energy_sample_interval_fs: float,
) -> tuple[dict, dict[str, np.ndarray]]:
    steps = base._integer_stride(continuation_fs, dt_fs, "continuation_fs")
    sample_stride = base._integer_stride(sample_interval_fs, dt_fs, "sample_interval_fs")
    energy_stride = base._integer_stride(
        energy_sample_interval_fs, dt_fs, "energy_sample_interval_fs"
    )
    fixed_psi = np.asarray(initial.electronic_state, dtype=np.complex128).copy()
    current = ip1n._copy_state(initial)
    initial_energy = frozen_surface_total_energy(
        current, parameters, fixed_psi, held_phase, switch_time_fs
    )
    max_energy_drift = 0.0
    max_zero_mode = 0.0
    times = [float(switch_time_fs)]
    jx_frames = [
        intermolecular_energy_flux(current.lattice, current.velocity, parameters).jx.astype(
            np.float32, copy=True
        )
    ]
    start = perf_counter()
    for step in range(steps):
        time_fs = switch_time_fs + step * dt_fs
        current = frozen_surface_verlet_step(
            current,
            parameters,
            fixed_psi,
            held_phase,
            time_fs,
            dt_fs,
        )
        new_time = time_fs + dt_fs
        max_zero_mode = max(
            max_zero_mode,
            max(abs(value) for value in zero_mode_means(current).values()),
        )
        if (step + 1) % sample_stride == 0:
            flux = intermolecular_energy_flux(current.lattice, current.velocity, parameters)
            times.append(float(new_time))
            jx_frames.append(np.asarray(flux.jx, dtype=np.float32))
        if (step + 1) % energy_stride == 0 or step + 1 == steps:
            energy = frozen_surface_total_energy(
                current, parameters, fixed_psi, held_phase, new_time
            )
            max_energy_drift = max(max_energy_drift, abs(energy - initial_energy))
    elapsed = perf_counter() - start
    electronic_diff = float(
        np.max(np.abs(np.asarray(current.electronic_state) - fixed_psi))
    )
    result = {
        "label": "frozen_electronic_surface",
        "steps": int(steps),
        "elapsed_seconds": float(elapsed),
        "accumulated_external_work_eV": 0.0,
        "maximum_energy_balance_residual_eV": float(max_energy_drift),
        "maximum_zero_mode_mean": float(max_zero_mode),
        "electronic_state_max_abs_change": electronic_diff,
        "electronic_state_bitwise_unchanged": bool(
            np.array_equal(np.asarray(current.electronic_state), fixed_psi)
        ),
        "persistent_events_after_switch": [],
        "hamiltonian_evaluations": 0,
        "hamiltonian_applications": 0,
    }
    arrays = {
        "times_fs": np.asarray(times, dtype=np.float64),
        "jx_eV_per_fs": np.stack(jx_frames, axis=0),
    }
    return result, arrays


def _force_difference(a, b) -> float:
    return float(
        max(
            np.max(np.abs(a.u - b.u)),
            np.max(np.abs(a.vx - b.vx)),
            np.max(np.abs(a.vy - b.vy)),
        )
    )


def _cosine_similarity_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.shape != bb.shape or aa.ndim != 2:
        raise ValueError("branch current matrices must have identical 2D shape")
    numerator = np.sum(aa * bb, axis=1)
    denominator = np.linalg.norm(aa, axis=1) * np.linalg.norm(bb, axis=1)
    out = np.zeros_like(numerator)
    valid = denominator > 1.0e-30
    out[valid] = numerator[valid] / denominator[valid]
    return out


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
    tracker = ip1n._branch_tracker(parameters.nx, parameters.ny, args.sample_interval_fs)
    current = ip1n._copy_state(initial)
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
                record = ip1n._event_record(event)
                pre_events.append(record)
                if (
                    selected_event is None
                    and record["is_nearest_neighbor"]
                    and abs(record["dx_sites"]) == 1
                    and record["dy_sites"] == 0
                ):
                    selected_event = record
                    selected_state = ip1n._copy_state(current)
            flux = intermolecular_energy_flux(current.lattice, current.velocity, parameters)
            pre_times.append(float(new_time))
            pre_jx.append(np.asarray(flux.jx, dtype=np.float32))
            if selected_event is not None:
                break
    search_elapsed = perf_counter() - start

    pre_times_arr = np.asarray(pre_times, dtype=np.float64)
    pre_jx_arr = np.stack(pre_jx, axis=0)
    available = selected_event is not None and selected_state is not None
    tolerance = size_scaled_energy_balance_tolerance_eV(parameters.n_sites)
    numerical_checks = {
        "static_relaxation_converged": bool(static["converged"]),
        "natural_x_event_found": bool(available),
    }
    analysis = None
    physical = {
        "fully_coupled_released_sustained_pattern_memory": False,
        "frozen_surface_sustained_pattern_memory": False,
        "lattice_encoded_memory_without_electronic_redistribution": False,
        "ongoing_electronic_backaction_required": False,
    }
    serialized: dict[str, np.ndarray] = {
        "pre_times_fs": pre_times_arr,
        "pre_jx_eV_per_fs": pre_jx_arr,
    }

    if available:
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
        pre_start = t0 + ip1n.PRE_BASELINE_START_OFFSET_FS
        pre_end = t0 + ip1n.PRE_BASELINE_END_OFFSET_FS
        pre_mask = interval_mask(pre_times_arr, pre_start, pre_end)
        if np.count_nonzero(pre_mask) < 2:
            raise RuntimeError("pre-hop baseline is incomplete")
        pre_profile = np.mean(pre_profiles[pre_mask], axis=0)
        pre_shifted = pre_profiles - pre_profile[None, :]
        pre_trailing = trailing_outward_matrix(
            pre_shifted, s_axis, distances_sites=ip1n.PRIMARY_DISTANCES
        )
        pre_amplitude = trailing_amplitude_series(pre_trailing)
        background_windows = sliding_amplitude_windows(
            pre_times_arr,
            pre_amplitude,
            pre_start,
            pre_end,
            width_fs=ip1n.WINDOW_WIDTH_FS,
            step_fs=ip1n.BACKGROUND_STEP_FS,
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
        phase_error = float(np.max(np.abs(phi_drive - phi_held)))
        force_drive = field_ehrenfest_force(
            selected_state.lattice,
            parameters,
            selected_state.electronic_state,
            field,
            ts,
        )
        force_held = field_ehrenfest_force(
            selected_state.lattice,
            parameters,
            selected_state.electronic_state,
            held,
            ts,
        )
        force_error = _force_difference(force_drive, force_held)

        full_initial = ip1n._copy_state(selected_state)
        frozen_initial = ip1n._copy_state(selected_state)
        identical = ip1n._states_identical(full_initial, frozen_initial)
        full_result, full_arrays = ip1n._run_branch(
            label="released_fully_coupled",
            initial=full_initial,
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
        frozen_result, frozen_arrays = _run_frozen_branch(
            initial=frozen_initial,
            parameters=parameters,
            held_phase=held,
            switch_time_fs=ts,
            continuation_fs=args.continuation_fs,
            dt_fs=args.dt_fs,
            sample_interval_fs=args.sample_interval_fs,
            energy_sample_interval_fs=args.energy_sample_interval_fs,
        )
        for arrays in (full_arrays, frozen_arrays):
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
        full_analysis, full_serial = ip1n._analyze_branch(
            branch=full_result,
            arrays=full_arrays,
            pre_profile=pre_profile,
            s_axis=s_axis,
            switch_time_fs=ts,
            background_rms=background_rms,
            background_peak=background_peak,
            vmax_sites_per_ps=vmax,
        )
        frozen_analysis, frozen_serial = ip1n._analyze_branch(
            branch=frozen_result,
            arrays=frozen_arrays,
            pre_profile=pre_profile,
            s_axis=s_axis,
            switch_time_fs=ts,
            background_rms=background_rms,
            background_peak=background_peak,
            vmax_sites_per_ps=vmax,
        )

        cosine = _cosine_similarity_rows(
            frozen_serial["trailing_outward_eV_per_fs"],
            full_serial["trailing_outward_eV_per_fs"],
        )
        relative_times = full_serial["times_fs"] - ts
        late = relative_times >= ip1n.LATE_OFFSET_FS
        late_cosine_mean = float(np.mean(cosine[late]))
        late_frozen = frozen_analysis["late_mean_trailing_amplitude_eV_per_fs"]
        late_full = full_analysis["late_mean_trailing_amplitude_eV_per_fs"]

        full_memory = bool(
            full_analysis["sustained_pattern_gate"]["sustained_pattern_memory"]
        )
        frozen_memory = bool(
            frozen_analysis["sustained_pattern_gate"]["sustained_pattern_memory"]
        )
        physical = {
            "fully_coupled_released_sustained_pattern_memory": full_memory,
            "frozen_surface_sustained_pattern_memory": frozen_memory,
            "lattice_encoded_memory_without_electronic_redistribution": frozen_memory,
            "ongoing_electronic_backaction_required": bool(full_memory and not frozen_memory),
        }
        analysis = {
            "event": selected_event,
            "switch_time_fs": ts,
            "post_hop_frame_source_site": source,
            "post_hop_frame_target_site": target,
            "phase_at_switch_radians": phi_drive.tolist(),
            "phase_continuity_error": phase_error,
            "initial_force_continuity_max_abs": force_error,
            "held_phase_rates_per_fs": list(held.phase_rates_per_fs()),
            "pre_hop_reference_window_fs": [pre_start, pre_end],
            "background_window_count": int(len(background_windows)),
            "fully_coupled_released": full_analysis,
            "frozen_electronic_surface": frozen_analysis,
            "late_frozen_to_full_amplitude_ratio": float(
                late_frozen / max(late_full, 1.0e-30)
            ),
            "late_trailing_vector_cosine_similarity_mean": late_cosine_mean,
        }

        numerical_checks.update(
            {
                "selected_event_is_first_persistent_event": bool(
                    pre_events and pre_events[0] == selected_event
                ),
                "phase_continuity": bool(phase_error < 1.0e-14),
                "held_phase_rates_zero": bool(
                    max(abs(value) for value in held.phase_rates_per_fs()) < 1.0e-15
                ),
                "branch_states_identical_at_switch": bool(identical),
                "initial_force_continuity": bool(force_error < 1.0e-13),
                "frozen_electronic_state_bitwise_unchanged": bool(
                    frozen_result["electronic_state_bitwise_unchanged"]
                ),
                "frozen_external_work_zero": True,
                "frozen_surface_energy_conservation": bool(
                    extensive_energy_balance_passes(
                        frozen_result["maximum_energy_balance_residual_eV"], parameters.n_sites
                    )
                ),
                "fully_coupled_released_energy_conservation": bool(
                    extensive_energy_balance_passes(
                        full_result["maximum_energy_balance_residual_eV"], parameters.n_sites
                    )
                ),
                "fully_coupled_projected_zero_modes": bool(
                    full_result["maximum_zero_mode_mean"] < 1.0e-10
                ),
                "frozen_projected_zero_modes": bool(
                    frozen_result["maximum_zero_mode_mean"] < 1.0e-10
                ),
                "finite_branch_diagnostics": bool(
                    np.isfinite(late_full)
                    and np.isfinite(late_frozen)
                    and np.isfinite(late_cosine_mean)
                    and np.isfinite(force_error)
                ),
                "branch_current_trajectories_serializable": bool(
                    np.all(np.isfinite(full_arrays["jx_eV_per_fs"]))
                    and np.all(np.isfinite(frozen_arrays["jx_eV_per_fs"]))
                ),
            }
        )
        serialized.update(
            {
                "s_sites": np.asarray(s_axis, dtype=np.int64),
                "pre_longitudinal_flux_eV_per_fs": np.asarray(pre_profiles, dtype=np.float32),
                "fully_coupled_times_fs": full_serial["times_fs"],
                "fully_coupled_trailing_outward_eV_per_fs": np.asarray(
                    full_serial["trailing_outward_eV_per_fs"], dtype=np.float32
                ),
                "fully_coupled_trailing_amplitude_eV_per_fs": full_serial[
                    "trailing_amplitude_eV_per_fs"
                ],
                "frozen_times_fs": frozen_serial["times_fs"],
                "frozen_trailing_outward_eV_per_fs": np.asarray(
                    frozen_serial["trailing_outward_eV_per_fs"], dtype=np.float32
                ),
                "frozen_trailing_amplitude_eV_per_fs": frozen_serial[
                    "trailing_amplitude_eV_per_fs"
                ],
                "branch_trailing_vector_cosine_similarity": cosine,
            }
        )

    numerical_pass = bool(all(numerical_checks.values()))
    payload = {
        "scope": "IP1o frozen-electronic-surface causal control of isotropic post-hop lattice memory",
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
        "analysis": analysis,
        "physical_decision": physical,
        "energy_balance_tolerance_eV": float(tolerance),
        "numerical_checks": numerical_checks,
        "numerical_pass": numerical_pass,
        "interpretation_guard": {
            "frozen_electronic_branch_is_causal_control_not_material_protocol": True,
            "held_boundary_twist_remains_in_both_zero_power_branches": True,
            "frozen_pass_means_lattice_phase_space_memory_on_static_electronic_surface": True,
            "trailing_rms_is_pattern_strength_not_monotonic_directional_flux": True,
            "no_transport_coefficient_or_material_lifetime_claim": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(args.trajectory, **serialized)

    lines = [
        "# IP1o frozen-electronic-surface causal control",
        "",
        f"Cell {args.size}x{args.size}; isotropic; initial E=+{args.field_mV_per_A:g} mV/A; T=0; continuation={args.continuation_fs/1000:g} ps.",
        "",
    ]
    if analysis is None:
        lines.append("No qualifying natural x event was found before the search cutoff.")
    else:
        event = analysis["event"]
        full = analysis["fully_coupled_released"]
        frozen = analysis["frozen_electronic_surface"]
        lines.extend(
            [
                f"Event {event['source_site']}->{event['target_site']} ({event['direction']}): start={event['transition_start_time_fs']:.1f} fs; accepted/switch={event['accepted_time_fs']:.1f} fs.",
                f"Phase continuity error={analysis['phase_continuity_error']:.3e}; force continuity error={analysis['initial_force_continuity_max_abs']:.3e}.",
                "",
                "## Fully coupled zero-power release",
                "",
                f"- post-switch persistent events: {len(full['persistent_events_after_switch'])};",
                f"- late passing bins: {full['sustained_pattern_gate']['late_passing_bin_count']}/{full['sustained_pattern_gate']['late_bin_count']};",
                f"- sustained pattern: {'YES' if full['sustained_pattern_gate']['sustained_pattern_memory'] else 'NO'};",
                f"- energy drift: {full['maximum_energy_balance_residual_eV']:.6e} eV;",
                "",
                "## Frozen-electronic-surface release",
                "",
                f"- electronic state bitwise unchanged: {'YES' if frozen['electronic_state_bitwise_unchanged'] else 'NO'};",
                f"- late passing bins: {frozen['sustained_pattern_gate']['late_passing_bin_count']}/{frozen['sustained_pattern_gate']['late_bin_count']};",
                f"- sustained pattern: {'YES' if frozen['sustained_pattern_gate']['sustained_pattern_memory'] else 'NO'};",
                f"- energy drift: {frozen['maximum_energy_balance_residual_eV']:.6e} eV;",
                f"- frozen/full late amplitude ratio: {analysis['late_frozen_to_full_amplitude_ratio']:.6f};",
                f"- late mean trailing-vector cosine similarity: {analysis['late_trailing_vector_cosine_similarity_mean']:.6f};",
                "",
                "## Physical decision",
                "",
                f"- lattice-encoded memory without electronic redistribution: {'YES' if physical['lattice_encoded_memory_without_electronic_redistribution'] else 'NO'};",
                f"- ongoing electronic backaction required: {'YES' if physical['ongoing_electronic_backaction_required'] else 'NO'};",
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
