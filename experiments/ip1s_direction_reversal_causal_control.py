#!/usr/bin/env python3
"""IP1s energy-preserving x-Peierls direction-reversal causal control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

import ip1l_interhop_wake_memory as base
import ip1n_gauge_continuous_field_release as ip1n
from holstein_peierls.dynamics.coupled import CoupledEhrenfestState
from holstein_peierls.dynamics.direction_reversal import (
    direction_reversal_fourier_errors,
    reverse_non_special_vx_velocity,
    vx_velocity_kinetic_energy_eV,
)
from holstein_peierls.dynamics.driven import (
    coupled_field_verlet_step,
    field_dynamic_total_energy,
)
from holstein_peierls.dynamics.ehrenfest import LatticeVelocity
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
from holstein_peierls.dynamics.single_hop_memory import continuation_neighbor
from holstein_peierls.dynamics.thermal import zero_mode_means


LATTICE_SPACING_A = 3.0
FIRST_DECISION_WINDOW_FS = 2000.0
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


def _copy_with_vx_velocity(state: CoupledEhrenfestState, vx_velocity: np.ndarray) -> CoupledEhrenfestState:
    return CoupledEhrenfestState(
        state.lattice.copy(),
        LatticeVelocity(
            state.velocity.u.copy(),
            np.asarray(vx_velocity, dtype=np.float64).copy(),
            state.velocity.vy.copy(),
        ),
        np.asarray(state.electronic_state, dtype=np.complex128).copy(),
    )


def _population_observables(state: CoupledEhrenfestState, site_a: int, site_b: int) -> dict:
    p = np.abs(np.asarray(state.electronic_state, dtype=np.complex128)) ** 2
    p /= float(np.sum(p))
    return {
        "population": p,
        "site_a_population": float(p[int(site_a)]),
        "site_b_population": float(p[int(site_b)]),
        "maximum_site": int(np.argmax(p)),
        "maximum_population": float(np.max(p)),
        "ipr": float(np.sum(p * p)),
    }


def _first_x_event(events: list[dict], switch_time_fs: float, window_fs: float) -> dict | None:
    cutoff = float(switch_time_fs) + float(window_fs)
    candidates = [
        event for event in events
        if event["is_nearest_neighbor"]
        and abs(int(event["dx_sites"])) == 1
        and int(event["dy_sites"]) == 0
        and float(event["transition_start_time_fs"]) <= cutoff
    ]
    return None if not candidates else min(candidates, key=lambda item: item["transition_start_time_fs"])


def _cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    numerator = np.sum(aa * bb, axis=1)
    denominator = np.linalg.norm(aa, axis=1) * np.linalg.norm(bb, axis=1)
    out = np.zeros(aa.shape[0], dtype=np.float64)
    good = denominator > 1.0e-30
    out[good] = numerator[good] / denominator[good]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=40)
    parser.add_argument("--field-mv-per-A", dest="field_mV_per_A", type=float, default=10.0)
    parser.add_argument("--dt-fs", type=float, default=0.2)
    parser.add_argument("--search-time-fs", type=float, default=4000.0)
    parser.add_argument("--continuation-fs", type=float, default=5000.0)
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
    search_steps = base._integer_stride(args.search_time_fs, args.dt_fs, "search_time_fs")
    sample_stride = base._integer_stride(args.sample_interval_fs, args.dt_fs, "sample_interval_fs")
    energy_stride = base._integer_stride(
        args.energy_sample_interval_fs, args.dt_fs, "energy_sample_interval_fs"
    )
    current_stride = base._integer_stride(
        args.current_sample_interval_fs, args.dt_fs, "current_sample_interval_fs"
    )

    tracker = ip1n._branch_tracker(parameters.nx, parameters.ny, args.sample_interval_fs)
    current = ip1n._copy_state(initial)
    p0 = np.abs(current.electronic_state) ** 2
    tracker.update(p0 / float(np.sum(p0)), 0.0)
    pre_events: list[dict] = []
    selected_event = None
    selected_state = None
    drive_initial_energy = field_dynamic_total_energy(
        current.lattice, current.velocity, parameters, current.electronic_state, field, 0.0
    ).total
    drive_work = 0.0
    max_drive_balance = 0.0
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
                current.lattice,
                current.velocity,
                parameters,
                current.electronic_state,
                field,
                new_time,
            ).total
            max_drive_balance = max(
                max_drive_balance,
                abs((energy - drive_initial_energy) - drive_work),
            )
        if (step + 1) % sample_stride == 0:
            p = np.abs(current.electronic_state) ** 2
            p /= float(np.sum(p))
            event = tracker.update(p, new_time)
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
                    selected_state = ip1n._copy_state(current)
                    break
    search_elapsed = perf_counter() - search_start

    tolerance = size_scaled_energy_balance_tolerance_eV(parameters.n_sites)
    numerical_checks = {
        "static_relaxation_converged": bool(static["converged"]),
        "natural_x_event_found": bool(selected_event is not None and selected_state is not None),
        "pre_switch_energy_work_balance": bool(
            extensive_energy_balance_passes(max_drive_balance, parameters.n_sites)
        ),
    }
    analysis = None
    serialized: dict[str, np.ndarray] = {}

    if selected_event is not None and selected_state is not None:
        ts = float(selected_event["accepted_time_fs"])
        held = HeldPeierlsPhase2D.from_driving_field(field, ts)
        native = ip1n._copy_state(selected_state)
        reversed_vx = reverse_non_special_vx_velocity(selected_state.velocity.vx, parameters)
        reversed_state = _copy_with_vx_velocity(selected_state, reversed_vx)

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
        kinetic_before = vx_velocity_kinetic_energy_eV(native.velocity.vx, parameters)
        kinetic_after = vx_velocity_kinetic_energy_eV(reversed_state.velocity.vx, parameters)
        kinetic_error = abs(kinetic_after - kinetic_before)
        energy_native = field_dynamic_total_energy(
            native.lattice, native.velocity, parameters, native.electronic_state, held, ts
        ).total
        energy_reversed = field_dynamic_total_energy(
            reversed_state.lattice,
            reversed_state.velocity,
            parameters,
            reversed_state.electronic_state,
            held,
            ts,
        ).total
        intervention_energy_error = abs(energy_reversed - energy_native)

        equilibrium = frozen_surface_equilibrium(
            parameters, native.electronic_state, held
        )
        split_before = traveling_vx_energy_split(
            native.lattice,
            native.velocity,
            equilibrium,
            parameters,
            carrier_dx_sites=int(selected_event["dx_sites"]),
        )
        split_after = traveling_vx_energy_split(
            reversed_state.lattice,
            reversed_state.velocity,
            equilibrium,
            parameters,
            carrier_dx_sites=int(selected_event["dx_sites"]),
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

        trackers = {
            "native": ip1n._branch_tracker(parameters.nx, parameters.ny, args.sample_interval_fs),
            "reversed": ip1n._branch_tracker(parameters.nx, parameters.ny, args.sample_interval_fs),
        }
        states = {"native": native, "reversed": reversed_state}
        initial_energies = {"native": energy_native, "reversed": energy_reversed}
        works = {"native": 0.0, "reversed": 0.0}
        max_energy_drift = {"native": 0.0, "reversed": 0.0}
        max_norm_error = {"native": 0.0, "reversed": 0.0}
        max_zero_mode = {"native": 0.0, "reversed": 0.0}
        initial_norms = {
            key: float(np.linalg.norm(state.electronic_state)) for key, state in states.items()
        }
        events = {"native": [], "reversed": []}
        for key, state in states.items():
            p = np.abs(state.electronic_state) ** 2
            trackers[key].update(p / float(np.sum(p)), ts)

        site_a = int(selected_event["source_site"])
        site_b = int(selected_event["target_site"])
        sample_times = [ts]
        native_obs = _population_observables(states["native"], site_a, site_b)
        reversed_obs = _population_observables(states["reversed"], site_a, site_b)
        site_a_native = [native_obs["site_a_population"]]
        site_a_reversed = [reversed_obs["site_a_population"]]
        site_b_native = [native_obs["site_b_population"]]
        site_b_reversed = [reversed_obs["site_b_population"]]
        max_site_native = [native_obs["maximum_site"]]
        max_site_reversed = [reversed_obs["maximum_site"]]
        ipr_native = [native_obs["ipr"]]
        ipr_reversed = [reversed_obs["ipr"]]
        l1_series = [float(np.sum(np.abs(native_obs["population"] - reversed_obs["population"])))]
        current_times = [ts]
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
            time_fs = ts + step * args.dt_fs
            new_time = time_fs + args.dt_fs
            for key in ("native", "reversed"):
                state, work, _, _ = coupled_field_verlet_step(
                    states[key],
                    parameters,
                    held,
                    time_fs,
                    args.dt_fs,
                    electronic_method="cfm4_lanczos",
                    krylov_dimension=args.krylov_dimension,
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
                    obs = _population_observables(state, site_a, site_b)
                    observations[key] = obs
                    event = trackers[key].update(obs["population"], new_time)
                    if event is not None:
                        events[key].append(_event_record(event))
                site_a_native.append(observations["native"]["site_a_population"])
                site_a_reversed.append(observations["reversed"]["site_a_population"])
                site_b_native.append(observations["native"]["site_b_population"])
                site_b_reversed.append(observations["reversed"]["site_b_population"])
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
            value = None if indices.size == 0 else float(sample_times_arr[int(indices[0])] - ts)
            if name == "010":
                first_l1_010 = value
            else:
                first_l1_025 = value

        native_first = _first_x_event(events["native"], ts, FIRST_DECISION_WINDOW_FS)
        reversed_first = _first_x_event(events["reversed"], ts, FIRST_DECISION_WINDOW_FS)
        if (native_first is None) != (reversed_first is None):
            event_sequence_changed = True
        elif native_first is None and reversed_first is None:
            event_sequence_changed = False
        else:
            assert native_first is not None and reversed_first is not None
            event_sequence_changed = bool(
                native_first["direction"] != reversed_first["direction"]
                or abs(
                    float(native_first["transition_start_time_fs"])
                    - float(reversed_first["transition_start_time_fs"])
                ) >= EVENT_TIME_DIFFERENCE_FS
            )
        decision_mask = sample_times_arr <= ts + FIRST_DECISION_WINDOW_FS + 1.0e-10
        max_l1_first_2ps = float(np.max(l1_arr[decision_mask]))
        electronic_diverged = bool(max_l1_first_2ps >= L1_DIVERGENCE_THRESHOLD)
        causal_pass = bool(event_sequence_changed and electronic_diverged)

        source = site_b
        target = continuation_neighbor(
            source,
            dx_sites=int(selected_event["dx_sites"]),
            dy_sites=int(selected_event["dy_sites"]),
            nx=parameters.nx,
            ny=parameters.ny,
        )
        current_times_arr = np.asarray(current_times, dtype=np.float64)
        trailing = {}
        amplitudes = {}
        for key in ("native", "reversed"):
            jx_arr = np.stack(current_jx[key], axis=0)
            s_axis, profiles = base._project_profiles(
                jx_arr,
                source_site=source,
                target_site=target,
                nx=parameters.nx,
                ny=parameters.ny,
                direction_sign=int(selected_event["dx_sites"]),
            )
            trailing[key] = trailing_outward_matrix(
                profiles, s_axis, distances_sites=PRIMARY_DISTANCES
            )
            amplitudes[key] = trailing_amplitude_series(trailing[key])
        cosine = _cosine_rows(trailing["native"], trailing["reversed"])
        early_current_mask = current_times_arr <= ts + 1000.0 + 1.0e-10
        late_current_mask = current_times_arr >= ts + 1000.0 - 1.0e-10

        numerical_checks.update(intervention_checks)
        numerical_checks.update(
            {
                "selected_event_is_first_persistent_event": bool(
                    pre_events and pre_events[0] == selected_event
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
            "event_sequence_changed": event_sequence_changed,
            "electronic_state_diverged": electronic_diverged,
            "directional_memory_changes_electronic_trajectory": causal_pass,
        }
        analysis = {
            "event": selected_event,
            "switch_time_fs": ts,
            "post_hop_frame_source_site": source,
            "post_hop_frame_target_site": target,
            "intervention_fourier_errors": fourier_errors,
            "vx_kinetic_energy_error_eV": kinetic_error,
            "matter_energy_intervention_error_eV": intervention_energy_error,
            "traveling_energy_swap_error_eV": swap_error,
            "traveling_vx_before": split_before,
            "traveling_vx_after": split_after,
            "native_events_after_switch": events["native"],
            "reversed_events_after_switch": events["reversed"],
            "native_first_x_event_within_2ps": native_first,
            "reversed_first_x_event_within_2ps": reversed_first,
            "maximum_population_l1_first_2ps": max_l1_first_2ps,
            "first_l1_ge_0p10_offset_fs": first_l1_010,
            "first_l1_ge_0p25_offset_fs": first_l1_025,
            "native_accumulated_external_work_eV": float(works["native"]),
            "reversed_accumulated_external_work_eV": float(works["reversed"]),
            "native_maximum_energy_drift_eV": float(max_energy_drift["native"]),
            "reversed_maximum_energy_drift_eV": float(max_energy_drift["reversed"]),
            "native_maximum_norm_error": float(max_norm_error["native"]),
            "reversed_maximum_norm_error": float(max_norm_error["reversed"]),
            "native_maximum_zero_mode_mean": float(max_zero_mode["native"]),
            "reversed_maximum_zero_mode_mean": float(max_zero_mode["reversed"]),
            "branch_elapsed_seconds": float(branch_elapsed),
            "early_trailing_vector_cosine_mean": float(np.mean(cosine[early_current_mask])),
            "late_trailing_vector_cosine_mean": float(np.mean(cosine[late_current_mask])),
            "native_late_mean_trailing_amplitude_eV_per_fs": float(
                np.mean(amplitudes["native"][late_current_mask])
            ),
            "reversed_late_mean_trailing_amplitude_eV_per_fs": float(
                np.mean(amplitudes["reversed"][late_current_mask])
            ),
        }

        serialized = {
            "sample_times_fs": sample_times_arr,
            "population_l1_distance": l1_arr,
            "site_a_native_population": np.asarray(site_a_native, dtype=np.float64),
            "site_a_reversed_population": np.asarray(site_a_reversed, dtype=np.float64),
            "site_b_native_population": np.asarray(site_b_native, dtype=np.float64),
            "site_b_reversed_population": np.asarray(site_b_reversed, dtype=np.float64),
            "max_site_native": np.asarray(max_site_native, dtype=np.int64),
            "max_site_reversed": np.asarray(max_site_reversed, dtype=np.int64),
            "ipr_native": np.asarray(ipr_native, dtype=np.float64),
            "ipr_reversed": np.asarray(ipr_reversed, dtype=np.float64),
            "current_times_fs": current_times_arr,
            "native_trailing_d1_d4_eV_per_fs": trailing["native"],
            "reversed_trailing_d1_d4_eV_per_fs": trailing["reversed"],
            "trailing_vector_cosine_similarity": cosine,
        }
    else:
        physical = {
            "event_sequence_changed": False,
            "electronic_state_diverged": False,
            "directional_memory_changes_electronic_trajectory": False,
        }

    numerical_pass = bool(all(numerical_checks.values()))
    payload = {
        "scope": "IP1s energy-preserving x-Peierls traveling-direction reversal causal control",
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
            "direction_reversal_is_counterfactual_not_experimental_protocol": True,
            "total_energy_and_vx_modal_spectrum_preserved_at_intervention": True,
            "deterministic_trajectory_sensitivity_is_not_hopping_rate_or_probability": True,
            "held_boundary_twist_is_common_to_both_branches": True,
            "no_transport_coefficient_or_activation_claim": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if serialized:
        np.savez_compressed(args.trajectory, **serialized)

    lines = [
        "# IP1s energy-preserving Peierls direction-reversal control",
        "",
        f"Cell {args.size}x{args.size}; isotropic; initial field +{args.field_mV_per_A:g} mV/A; released continuation {args.continuation_fs/1000:g} ps.",
        "",
    ]
    if analysis is None:
        lines.append("No qualifying natural x event was found before the search cutoff.")
    else:
        lines.extend(
            [
                f"Event {selected_event['source_site']}->{selected_event['target_site']} ({selected_event['direction']}), start={selected_event['transition_start_time_fs']:.1f} fs, switch={selected_event['accepted_time_fs']:.1f} fs.",
                "",
                "## Intervention",
                "",
                f"- vx kinetic energy error: {analysis['vx_kinetic_energy_error_eV']:.6e} eV;",
                f"- total matter-energy error: {analysis['matter_energy_intervention_error_eV']:.6e} eV;",
                f"- retrograde/comoving energy-swap error: {analysis['traveling_energy_swap_error_eV']:.6e} eV;",
                "",
                "## First 2 ps electronic response",
                "",
                f"- native first x event: {analysis['native_first_x_event_within_2ps']};",
                f"- reversed first x event: {analysis['reversed_first_x_event_within_2ps']};",
                f"- maximum population L1 distance: {analysis['maximum_population_l1_first_2ps']:.6f};",
                f"- first L1 >=0.10: {analysis['first_l1_ge_0p10_offset_fs']} fs;",
                f"- first L1 >=0.25: {analysis['first_l1_ge_0p25_offset_fs']} fs;",
                "",
                "## Physical decision",
                "",
                f"- event sequence changed: {'YES' if physical['event_sequence_changed'] else 'NO'};",
                f"- electronic state diverged: {'YES' if physical['electronic_state_diverged'] else 'NO'};",
                f"- directional memory changes electronic trajectory: {'YES' if physical['directional_memory_changes_electronic_trajectory'] else 'NO'};",
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
    if serialized:
        print(f"Wrote {args.trajectory}")
    if not numerical_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
