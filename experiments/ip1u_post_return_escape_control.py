#!/usr/bin/env python3
"""IP1u post-return energy-preserving Peierls escape-stability control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

import ip1l_interhop_wake_memory as base
import ip1n_gauge_continuous_field_release as ip1n
import ip1s_direction_reversal_causal_control as ip1s
from holstein_peierls.dynamics.direction_reversal import (
    direction_reversal_fourier_errors,
    reverse_non_special_vx_velocity,
    vx_velocity_kinetic_energy_eV,
)
from holstein_peierls.dynamics.driven import (
    coupled_field_verlet_step,
    field_dynamic_total_energy,
)
from holstein_peierls.dynamics.field import UniformElectricField2D
from holstein_peierls.dynamics.field_release import (
    HeldPeierlsPhase2D,
    trailing_amplitude_series,
    trailing_outward_matrix,
)
from holstein_peierls.dynamics.mode_memory import (
    frozen_surface_equilibrium,
    traveling_vx_energy_split,
)
from holstein_peierls.dynamics.numerical_validation import (
    extensive_energy_balance_passes,
    size_scaled_energy_balance_tolerance_eV,
)
from holstein_peierls.dynamics.phonon_wake import intermolecular_energy_flux
from holstein_peierls.dynamics.post_return_control import (
    classify_post_return_escape,
    first_x_event_in_window,
)
from holstein_peierls.dynamics.single_hop_memory import continuation_neighbor
from holstein_peierls.dynamics.thermal import zero_mode_means


LATTICE_SPACING_A = 3.0
ESCAPE_WINDOW_FS = 1500.0
DIVERGENCE_WINDOW_FS = 2000.0
EVENT_TIME_DIFFERENCE_FS = 100.0
L1_DIVERGENCE_THRESHOLD = 0.25
L1_EARLY_THRESHOLD = 0.10
PRIMARY_DISTANCES = (1, 2, 3, 4)


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


def _propagate_one_step(state, parameters, held, time_fs, dt_fs, krylov_dimension):
    return coupled_field_verlet_step(
        state,
        parameters,
        held,
        time_fs,
        dt_fs,
        electronic_method="cfm4_lanczos",
        krylov_dimension=krylov_dimension,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=40)
    parser.add_argument("--field-mv-per-A", dest="field_mV_per_A", type=float, default=10.0)
    parser.add_argument("--dt-fs", type=float, default=0.2)
    parser.add_argument("--search-time-fs", type=float, default=4000.0)
    parser.add_argument("--common-search-time-fs", type=float, default=1500.0)
    parser.add_argument("--continuation-fs", type=float, default=2000.0)
    parser.add_argument("--sample-interval-fs", type=float, default=2.0)
    parser.add_argument("--current-sample-interval-fs", type=float, default=10.0)
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
    sample_stride = base._integer_stride(args.sample_interval_fs, args.dt_fs, "sample_interval_fs")
    current_stride = base._integer_stride(
        args.current_sample_interval_fs, args.dt_fs, "current_sample_interval_fs"
    )
    energy_stride = base._integer_stride(
        args.energy_sample_interval_fs, args.dt_fs, "energy_sample_interval_fs"
    )

    # Stage 1: reproduce the first natural driven x hop.
    search_steps = base._integer_stride(args.search_time_fs, args.dt_fs, "search_time_fs")
    tracker = ip1n._branch_tracker(parameters.nx, parameters.ny, args.sample_interval_fs)
    current = ip1n._copy_state(initial)
    p = np.abs(current.electronic_state) ** 2
    tracker.update(p / float(np.sum(p)), 0.0)
    drive_initial_energy = field_dynamic_total_energy(
        current.lattice, current.velocity, parameters, current.electronic_state, field, 0.0
    ).total
    drive_work = 0.0
    max_drive_balance = 0.0
    pre_events: list[dict] = []
    first_event = None
    first_state = None
    search_start = perf_counter()
    for step in range(search_steps):
        time_fs = step * args.dt_fs
        current, work, _, _ = coupled_field_verlet_step(
            current,
            parameters,
            field,
            time_fs,
            args.dt_fs,
            electronic_method="cfm4_lanczos",
            krylov_dimension=args.krylov_dimension,
        )
        new_time = time_fs + args.dt_fs
        drive_work += float(work)
        if (step + 1) % energy_stride == 0:
            energy = field_dynamic_total_energy(
                current.lattice, current.velocity, parameters, current.electronic_state, field, new_time
            ).total
            max_drive_balance = max(
                max_drive_balance, abs((energy - drive_initial_energy) - drive_work)
            )
        if (step + 1) % sample_stride == 0:
            p = np.abs(current.electronic_state) ** 2
            p /= float(np.sum(p))
            event = tracker.update(p, new_time)
            if event is not None:
                record = _event_record(event)
                pre_events.append(record)
                if (
                    record["is_nearest_neighbor"]
                    and abs(record["dx_sites"]) == 1
                    and record["dy_sites"] == 0
                ):
                    first_event = record
                    first_state = ip1n._copy_state(current)
                    break
    search_elapsed = perf_counter() - search_start

    tolerance = size_scaled_energy_balance_tolerance_eV(parameters.n_sites)
    numerical_checks = {
        "static_relaxation_converged": bool(static["converged"]),
        "first_natural_x_event_found": bool(first_event is not None and first_state is not None),
        "pre_switch_energy_work_balance": bool(
            extensive_energy_balance_passes(max_drive_balance, parameters.n_sites)
        ),
    }
    analysis = None
    serialized: dict[str, np.ndarray] = {}

    if first_event is not None and first_state is not None:
        ts = float(first_event["accepted_time_fs"])
        held = HeldPeierlsPhase2D.from_driving_field(field, ts)

        # Stage 2: one common released trajectory through the recrossing and return.
        common = ip1n._copy_state(first_state)
        common_initial_energy = field_dynamic_total_energy(
            common.lattice, common.velocity, parameters, common.electronic_state, held, ts
        ).total
        common_work = 0.0
        common_max_energy_drift = 0.0
        common_max_norm_error = 0.0
        common_max_zero_mode = 0.0
        common_norm = float(np.linalg.norm(common.electronic_state))
        common_tracker = ip1n._branch_tracker(
            parameters.nx, parameters.ny, args.sample_interval_fs
        )
        p = np.abs(common.electronic_state) ** 2
        common_tracker.update(p / float(np.sum(p)), ts)
        common_events: list[dict] = []
        recrossing_event = None
        recrossing_state = None
        return_event = None
        return_state = None
        common_steps = base._integer_stride(
            args.common_search_time_fs, args.dt_fs, "common_search_time_fs"
        )
        common_start = perf_counter()
        for step in range(common_steps):
            time_fs = ts + step * args.dt_fs
            common, work, _, _ = _propagate_one_step(
                common, parameters, held, time_fs, args.dt_fs, args.krylov_dimension
            )
            new_time = time_fs + args.dt_fs
            common_work += float(work)
            common_max_norm_error = max(
                common_max_norm_error,
                abs(float(np.linalg.norm(common.electronic_state)) - common_norm),
            )
            common_max_zero_mode = max(
                common_max_zero_mode,
                max(abs(value) for value in zero_mode_means(common).values()),
            )
            if (step + 1) % energy_stride == 0:
                energy = field_dynamic_total_energy(
                    common.lattice,
                    common.velocity,
                    parameters,
                    common.electronic_state,
                    held,
                    new_time,
                ).total
                common_max_energy_drift = max(
                    common_max_energy_drift, abs(energy - common_initial_energy)
                )
            if (step + 1) % sample_stride == 0:
                p = np.abs(common.electronic_state) ** 2
                p /= float(np.sum(p))
                event = common_tracker.update(p, new_time)
                if event is not None:
                    record = _event_record(event)
                    common_events.append(record)
                    if (
                        recrossing_event is None
                        and record["is_nearest_neighbor"]
                        and record["source_site"] == first_event["target_site"]
                        and record["target_site"] == first_event["source_site"]
                        and record["dx_sites"] == -first_event["dx_sites"]
                        and record["dy_sites"] == 0
                    ):
                        recrossing_event = record
                        recrossing_state = ip1n._copy_state(common)
                    elif (
                        recrossing_event is not None
                        and return_event is None
                        and record["is_nearest_neighbor"]
                        and record["source_site"] == first_event["source_site"]
                        and record["target_site"] == first_event["target_site"]
                        and record["dx_sites"] == first_event["dx_sites"]
                        and record["dy_sites"] == 0
                    ):
                        return_event = record
                        return_state = ip1n._copy_state(common)
                        break
        common_elapsed = perf_counter() - common_start
        numerical_checks.update(
            {
                "common_direct_recrossing_found": bool(
                    recrossing_event is not None and recrossing_state is not None
                ),
                "common_direct_return_found": bool(
                    return_event is not None and return_state is not None
                ),
                "common_external_work_zero": bool(abs(common_work) < 1.0e-12),
                "common_energy_conservation": bool(
                    extensive_energy_balance_passes(common_max_energy_drift, parameters.n_sites)
                ),
                "common_electronic_norm": bool(common_max_norm_error < 1.0e-10),
                "common_projected_zero_modes": bool(common_max_zero_mode < 1.0e-10),
            }
        )

        if return_event is not None and return_state is not None:
            tr = float(return_event["accepted_time_fs"])
            native = ip1n._copy_state(return_state)
            reversed_vx = reverse_non_special_vx_velocity(return_state.velocity.vx, parameters)
            reversed_state = ip1s._copy_with_vx_velocity(return_state, reversed_vx)

            coordinates_unchanged = bool(
                np.array_equal(native.lattice.u, reversed_state.lattice.u)
                and np.array_equal(native.lattice.vx, reversed_state.lattice.vx)
                and np.array_equal(native.lattice.vy, reversed_state.lattice.vy)
            )
            electronic_unchanged = bool(
                np.array_equal(native.electronic_state, reversed_state.electronic_state)
            )
            other_velocities_unchanged = bool(
                np.array_equal(native.velocity.u, reversed_state.velocity.u)
                and np.array_equal(native.velocity.vy, reversed_state.velocity.vy)
            )
            fourier_errors = direction_reversal_fourier_errors(
                native.velocity.vx, reversed_state.velocity.vx, parameters
            )
            kinetic_error = abs(
                vx_velocity_kinetic_energy_eV(native.velocity.vx, parameters)
                - vx_velocity_kinetic_energy_eV(reversed_state.velocity.vx, parameters)
            )
            energy_native = field_dynamic_total_energy(
                native.lattice, native.velocity, parameters, native.electronic_state, held, tr
            ).total
            energy_reversed = field_dynamic_total_energy(
                reversed_state.lattice,
                reversed_state.velocity,
                parameters,
                reversed_state.electronic_state,
                held,
                tr,
            ).total
            intervention_energy_error = abs(energy_reversed - energy_native)

            equilibrium = frozen_surface_equilibrium(parameters, native.electronic_state, held)
            split_before = traveling_vx_energy_split(
                native.lattice,
                native.velocity,
                equilibrium,
                parameters,
                carrier_dx_sites=int(return_event["dx_sites"]),
            )
            split_after = traveling_vx_energy_split(
                reversed_state.lattice,
                reversed_state.velocity,
                equilibrium,
                parameters,
                carrier_dx_sites=int(return_event["dx_sites"]),
            )
            swap_error = max(
                abs(split_after["retrograde_eV"] - split_before["comoving_eV"]),
                abs(split_after["comoving_eV"] - split_before["retrograde_eV"]),
                abs(
                    split_after["nondirectional_special_eV"]
                    - split_before["nondirectional_special_eV"]
                ),
            )
            intervention_checks = {
                "coordinates_unchanged": coordinates_unchanged,
                "electronic_state_unchanged": electronic_unchanged,
                "u_vy_velocities_unchanged": other_velocities_unchanged,
                "special_vx_velocity_sectors_unchanged": bool(
                    fourier_errors["special_sector_unchanged_max_abs_A_per_fs"] < 1.0e-15
                ),
                "non_special_vx_velocity_sectors_reversed": bool(
                    fourier_errors["non_special_sector_sign_reversal_max_abs_A_per_fs"] < 1.0e-15
                ),
                "vx_kinetic_energy_preserved": bool(kinetic_error < 1.0e-12),
                "matter_energy_preserved": bool(intervention_energy_error < 1.0e-12),
                "retrograde_comoving_energies_exchanged": bool(swap_error < 1.0e-12),
            }

            states = {"native": native, "reversed": reversed_state}
            trackers = {
                key: ip1n._branch_tracker(parameters.nx, parameters.ny, args.sample_interval_fs)
                for key in states
            }
            initial_energies = {"native": energy_native, "reversed": energy_reversed}
            initial_norms = {
                key: float(np.linalg.norm(state.electronic_state)) for key, state in states.items()
            }
            works = {"native": 0.0, "reversed": 0.0}
            max_energy_drift = {"native": 0.0, "reversed": 0.0}
            max_norm_error = {"native": 0.0, "reversed": 0.0}
            max_zero_mode = {"native": 0.0, "reversed": 0.0}
            events = {"native": [], "reversed": []}
            for key, state in states.items():
                p = np.abs(state.electronic_state) ** 2
                trackers[key].update(p / float(np.sum(p)), tr)

            branch_site = int(return_event["target_site"])
            previous_site = int(return_event["source_site"])
            sample_times = [tr]
            obs_native = ip1s._population_observables(states["native"], branch_site, previous_site)
            obs_reversed = ip1s._population_observables(states["reversed"], branch_site, previous_site)
            site_branch_native = [obs_native["site_a_population"]]
            site_branch_reversed = [obs_reversed["site_a_population"]]
            site_previous_native = [obs_native["site_b_population"]]
            site_previous_reversed = [obs_reversed["site_b_population"]]
            max_site_native = [obs_native["maximum_site"]]
            max_site_reversed = [obs_reversed["maximum_site"]]
            ipr_native = [obs_native["ipr"]]
            ipr_reversed = [obs_reversed["ipr"]]
            l1_series = [
                float(np.sum(np.abs(obs_native["population"] - obs_reversed["population"])))
            ]

            frame_target = continuation_neighbor(
                branch_site,
                dx_sites=int(return_event["dx_sites"]),
                dy_sites=int(return_event["dy_sites"]),
                nx=parameters.nx,
                ny=parameters.ny,
            )
            current_times = [tr]
            current_jx = {
                key: [
                    np.asarray(
                        intermolecular_energy_flux(state.lattice, state.velocity, parameters).jx,
                        dtype=np.float32,
                    )
                ]
                for key, state in states.items()
            }

            continuation_steps = base._integer_stride(
                args.continuation_fs, args.dt_fs, "continuation_fs"
            )
            branch_start = perf_counter()
            for step in range(continuation_steps):
                time_fs = tr + step * args.dt_fs
                new_time = time_fs + args.dt_fs
                for key in ("native", "reversed"):
                    state, work, _, _ = _propagate_one_step(
                        states[key], parameters, held, time_fs, args.dt_fs, args.krylov_dimension
                    )
                    states[key] = state
                    works[key] += float(work)
                    max_norm_error[key] = max(
                        max_norm_error[key],
                        abs(float(np.linalg.norm(state.electronic_state)) - initial_norms[key]),
                    )
                    max_zero_mode[key] = max(
                        max_zero_mode[key],
                        max(abs(value) for value in zero_mode_means(state).values()),
                    )
                if (step + 1) % energy_stride == 0 or step + 1 == continuation_steps:
                    for key, state in states.items():
                        energy = field_dynamic_total_energy(
                            state.lattice,
                            state.velocity,
                            parameters,
                            state.electronic_state,
                            held,
                            new_time,
                        ).total
                        max_energy_drift[key] = max(
                            max_energy_drift[key], abs(energy - initial_energies[key])
                        )
                if (step + 1) % sample_stride == 0:
                    sample_times.append(float(new_time))
                    observations = {}
                    for key, state in states.items():
                        obs = ip1s._population_observables(state, branch_site, previous_site)
                        observations[key] = obs
                        event = trackers[key].update(obs["population"], new_time)
                        if event is not None:
                            events[key].append(_event_record(event))
                    site_branch_native.append(observations["native"]["site_a_population"])
                    site_branch_reversed.append(observations["reversed"]["site_a_population"])
                    site_previous_native.append(observations["native"]["site_b_population"])
                    site_previous_reversed.append(observations["reversed"]["site_b_population"])
                    max_site_native.append(observations["native"]["maximum_site"])
                    max_site_reversed.append(observations["reversed"]["maximum_site"])
                    ipr_native.append(observations["native"]["ipr"])
                    ipr_reversed.append(observations["reversed"]["ipr"])
                    l1_series.append(
                        float(
                            np.sum(
                                np.abs(
                                    observations["native"]["population"]
                                    - observations["reversed"]["population"]
                                )
                            )
                        )
                    )
                if (step + 1) % current_stride == 0:
                    current_times.append(float(new_time))
                    for key, state in states.items():
                        current_jx[key].append(
                            np.asarray(
                                intermolecular_energy_flux(
                                    state.lattice, state.velocity, parameters
                                ).jx,
                                dtype=np.float32,
                            )
                        )
            branch_elapsed = perf_counter() - branch_start

            sample_times_arr = np.asarray(sample_times, dtype=np.float64)
            l1_arr = np.asarray(l1_series, dtype=np.float64)
            first_l1_010 = None
            first_l1_025 = None
            for threshold, name in (
                (L1_EARLY_THRESHOLD, "010"),
                (L1_DIVERGENCE_THRESHOLD, "025"),
            ):
                indices = np.flatnonzero(l1_arr >= threshold)
                value = None if indices.size == 0 else float(sample_times_arr[int(indices[0])] - tr)
                if name == "010":
                    first_l1_010 = value
                else:
                    first_l1_025 = value

            native_first = first_x_event_in_window(
                events["native"], branch_time_fs=tr, window_fs=ESCAPE_WINDOW_FS
            )
            reversed_first = first_x_event_in_window(
                events["reversed"], branch_time_fs=tr, window_fs=ESCAPE_WINDOW_FS
            )
            commitment = classify_post_return_escape(
                native_first,
                reversed_first,
                returned_site=branch_site,
                previous_site=previous_site,
                time_difference_fs=EVENT_TIME_DIFFERENCE_FS,
            )
            divergence_mask = sample_times_arr <= tr + DIVERGENCE_WINDOW_FS + 1.0e-10
            max_l1_first_2ps = float(np.max(l1_arr[divergence_mask]))
            electronic_diverged = bool(max_l1_first_2ps >= L1_DIVERGENCE_THRESHOLD)
            causal_pass = bool(commitment["reescape_status_changed"] and electronic_diverged)

            current_times_arr = np.asarray(current_times, dtype=np.float64)
            trailing = {}
            amplitudes = {}
            s_axis = None
            for key in ("native", "reversed"):
                jx_arr = np.stack(current_jx[key], axis=0)
                s_axis_local, profiles = base._project_profiles(
                    jx_arr,
                    source_site=branch_site,
                    target_site=frame_target,
                    nx=parameters.nx,
                    ny=parameters.ny,
                    direction_sign=int(return_event["dx_sites"]),
                )
                s_axis = s_axis_local
                trailing[key] = trailing_outward_matrix(
                    profiles, s_axis_local, distances_sites=PRIMARY_DISTANCES
                )
                amplitudes[key] = trailing_amplitude_series(trailing[key])
            cosine = ip1s._cosine_rows(trailing["native"], trailing["reversed"])
            early_current_mask = current_times_arr <= tr + 500.0 + 1.0e-10
            late_current_mask = current_times_arr >= tr + 1000.0 - 1.0e-10

            numerical_checks.update(intervention_checks)
            numerical_checks.update(
                {
                    "first_event_is_first_persistent_event": bool(
                        pre_events and pre_events[0] == first_event
                    ),
                    "common_recrossing_is_first_post_release_x_event": bool(
                        common_events and common_events[0] == recrossing_event
                    ),
                    "common_return_is_second_post_release_x_event": bool(
                        len(common_events) >= 2 and common_events[1] == return_event
                    ),
                    "held_phase_rates_zero": bool(
                        max(abs(value) for value in held.phase_rates_per_fs()) < 1.0e-15
                    ),
                    "native_complete_steps": True,
                    "reversed_complete_steps": True,
                    "native_external_work_zero": bool(abs(works["native"]) < 1.0e-12),
                    "reversed_external_work_zero": bool(abs(works["reversed"]) < 1.0e-12),
                    "native_energy_conservation": bool(
                        extensive_energy_balance_passes(max_energy_drift["native"], parameters.n_sites)
                    ),
                    "reversed_energy_conservation": bool(
                        extensive_energy_balance_passes(max_energy_drift["reversed"], parameters.n_sites)
                    ),
                    "native_electronic_norm": bool(max_norm_error["native"] < 1.0e-10),
                    "reversed_electronic_norm": bool(max_norm_error["reversed"] < 1.0e-10),
                    "native_projected_zero_modes": bool(max_zero_mode["native"] < 1.0e-10),
                    "reversed_projected_zero_modes": bool(max_zero_mode["reversed"] < 1.0e-10),
                    "finite_population_and_current_diagnostics": bool(
                        np.all(np.isfinite(l1_arr))
                        and np.all(np.isfinite(trailing["native"]))
                        and np.all(np.isfinite(trailing["reversed"]))
                    ),
                }
            )

            physical = {
                "reescape_status_changed": bool(commitment["reescape_status_changed"]),
                "electronic_state_diverged": electronic_diverged,
                "directional_memory_controls_post_return_escape": causal_pass,
            }
            analysis = {
                "first_natural_event": first_event,
                "first_release_time_fs": ts,
                "common_post_release_events": common_events,
                "common_recrossing_event": recrossing_event,
                "common_return_event": return_event,
                "branch_time_fs": tr,
                "branch_site": branch_site,
                "previous_site": previous_site,
                "frame_target_site": frame_target,
                "common_segment": {
                    "elapsed_seconds": float(common_elapsed),
                    "accumulated_external_work_eV": float(common_work),
                    "maximum_energy_drift_eV": float(common_max_energy_drift),
                    "maximum_norm_error": float(common_max_norm_error),
                    "maximum_zero_mode_mean": float(common_max_zero_mode),
                },
                "intervention": {
                    "fourier_errors": fourier_errors,
                    "vx_kinetic_energy_error_eV": float(kinetic_error),
                    "matter_energy_error_eV": float(intervention_energy_error),
                    "traveling_energy_swap_error_eV": float(swap_error),
                    "traveling_split_before": split_before,
                    "traveling_split_after": split_after,
                    "checks": intervention_checks,
                },
                "native": {
                    "events": events["native"],
                    "first_x_event_within_escape_window": native_first,
                    "accumulated_external_work_eV": float(works["native"]),
                    "maximum_energy_drift_eV": float(max_energy_drift["native"]),
                    "maximum_norm_error": float(max_norm_error["native"]),
                    "maximum_zero_mode_mean": float(max_zero_mode["native"]),
                    "late_mean_trailing_amplitude_eV_per_fs": float(
                        np.mean(amplitudes["native"][late_current_mask])
                    ),
                },
                "reversed": {
                    "events": events["reversed"],
                    "first_x_event_within_escape_window": reversed_first,
                    "accumulated_external_work_eV": float(works["reversed"]),
                    "maximum_energy_drift_eV": float(max_energy_drift["reversed"]),
                    "maximum_norm_error": float(max_norm_error["reversed"]),
                    "maximum_zero_mode_mean": float(max_zero_mode["reversed"]),
                    "late_mean_trailing_amplitude_eV_per_fs": float(
                        np.mean(amplitudes["reversed"][late_current_mask])
                    ),
                },
                "escape_classification": commitment,
                "population_divergence": {
                    "maximum_l1_first_2ps": max_l1_first_2ps,
                    "first_l1_ge_0p10_offset_fs": first_l1_010,
                    "first_l1_ge_0p25_offset_fs": first_l1_025,
                },
                "trailing_current_comparison": {
                    "early_0_to_0p5ps_cosine_mean": float(np.mean(cosine[early_current_mask])),
                    "late_ge_1ps_cosine_mean": float(np.mean(cosine[late_current_mask])),
                },
                "branch_elapsed_seconds": float(branch_elapsed),
            }

            serialized = {
                "sample_times_fs": sample_times_arr,
                "population_l1": l1_arr,
                "branch_site_population_native": np.asarray(site_branch_native, dtype=np.float64),
                "branch_site_population_reversed": np.asarray(site_branch_reversed, dtype=np.float64),
                "previous_site_population_native": np.asarray(site_previous_native, dtype=np.float64),
                "previous_site_population_reversed": np.asarray(site_previous_reversed, dtype=np.float64),
                "max_site_native": np.asarray(max_site_native, dtype=np.int64),
                "max_site_reversed": np.asarray(max_site_reversed, dtype=np.int64),
                "ipr_native": np.asarray(ipr_native, dtype=np.float64),
                "ipr_reversed": np.asarray(ipr_reversed, dtype=np.float64),
                "current_times_fs": current_times_arr,
                "s_sites": np.asarray(s_axis, dtype=np.int64),
                "native_trailing_outward_eV_per_fs": np.asarray(trailing["native"], dtype=np.float32),
                "reversed_trailing_outward_eV_per_fs": np.asarray(trailing["reversed"], dtype=np.float32),
                "native_trailing_amplitude_eV_per_fs": np.asarray(amplitudes["native"], dtype=np.float64),
                "reversed_trailing_amplitude_eV_per_fs": np.asarray(amplitudes["reversed"], dtype=np.float64),
                "trailing_vector_cosine_similarity": np.asarray(cosine, dtype=np.float64),
            }
        else:
            physical = {
                "reescape_status_changed": False,
                "electronic_state_diverged": False,
                "directional_memory_controls_post_return_escape": False,
            }
    else:
        physical = {
            "reescape_status_changed": False,
            "electronic_state_diverged": False,
            "directional_memory_controls_post_return_escape": False,
        }

    numerical_pass = bool(all(numerical_checks.values()))
    payload = {
        "scope": "IP1u post-return energy-preserving Peierls escape-stability control",
        "size": int(args.size),
        "anisotropy_ratio": 1.0,
        "initial_field_mV_per_A": float(args.field_mV_per_A),
        "temperature_K": 0.0,
        "thermostat": None,
        "IDC": None,
        "dt_fs": float(args.dt_fs),
        "escape_window_fs": ESCAPE_WINDOW_FS,
        "divergence_window_fs": DIVERGENCE_WINDOW_FS,
        "search_elapsed_seconds": float(search_elapsed),
        "static": static,
        "analysis": analysis,
        "physical_decision": physical,
        "energy_balance_tolerance_eV": float(tolerance),
        "numerical_checks": numerical_checks,
        "numerical_pass": numerical_pass,
        "interpretation_guard": {
            "counterfactual_is_not_experimental_pulse_protocol": True,
            "positive_result_is_deterministic_sensitivity_not_hopping_probability": True,
            "ip1s_ip1t_primary_results_remain_formally_negative": True,
            "no_transport_coefficient_or_material_lifetime_claim": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(args.trajectory, **serialized)

    lines = [
        "# IP1u post-return escape-stability control",
        "",
        f"Numerical status: {'PASS' if numerical_pass else 'FAIL'}",
        "",
    ]
    if analysis is not None:
        first = analysis["first_natural_event"]
        recross = analysis["common_recrossing_event"]
        returned = analysis["common_return_event"]
        lines.extend(
            [
                f"First natural event: {first['source_site']}->{first['target_site']} {first['direction']}, accepted {first['accepted_time_fs']:.1f} fs.",
                f"Common recrossing: {recross['source_site']}->{recross['target_site']} {recross['direction']}, accepted {recross['accepted_time_fs']:.1f} fs.",
                f"Common return branch point: {returned['source_site']}->{returned['target_site']} {returned['direction']}, accepted {returned['accepted_time_fs']:.1f} fs.",
                "",
                f"Post-return escape status changed: {'YES' if physical['reescape_status_changed'] else 'NO'}",
                f"Electronic state diverged: {'YES' if physical['electronic_state_diverged'] else 'NO'}",
                f"Primary causal classification: {'PASS' if physical['directional_memory_controls_post_return_escape'] else 'FAIL'}",
                "",
            ]
        )
    lines.extend(["## Numerical gates", ""])
    for name, value in numerical_checks.items():
        lines.append(f"- {name}: {'PASS' if value else 'FAIL'}")
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"\nWrote {args.output}")
    print(f"Wrote {args.markdown}")
    print(f"Wrote {args.trajectory}")
    if not numerical_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
