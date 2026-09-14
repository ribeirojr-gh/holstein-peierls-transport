#!/usr/bin/env python3
"""IP1f carrier-centered phonon-wake characterization.

This stage measures the lattice-energy wake around persistent electronic
nearest-neighbour relocations. It is explicitly designed to test the user's
observed phenomenon that a propagating polaron emits an intermolecular phonon
packet whose energy propagates opposite to the carrier direction.

The dynamics remain zero-field finite-temperature HP dynamics. Every accepted
nearest-neighbour event is aligned so that the carrier moves toward +s. The
analysis then asks whether event-conditioned lattice energy and, more directly,
the harmonic Peierls energy current propagate toward -s.

Finite-T local energy includes thermal background and the bound polaron. IP1f
therefore subtracts a pre-event baseline, excludes the immediate polaron core,
and treats fitted centroid velocities as descriptive wake velocities, not as a
normal-mode-resolved material phonon group velocity.

No hopping rate, activation energy, diffusion coefficient, or mobility is
inferred here.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from holstein_peierls.dynamics.coupled import CoupledEhrenfestState
from holstein_peierls.dynamics.ehrenfest import dynamic_total_energy
from holstein_peierls.dynamics.electronic_decoherence import apply_instantaneous_decoherence
from holstein_peierls.dynamics.hopping_observables import PersistentHopEvent, PersistentSiteTracker
from holstein_peierls.dynamics.langevin import LangevinBath, kinetic_temperature_K
from holstein_peierls.dynamics.numerical_validation import (
    extensive_energy_balance_passes,
    size_scaled_energy_balance_tolerance_eV,
)
from holstein_peierls.dynamics.phonon_recurrence import intermolecular_recurrence_scales
from holstein_peierls.dynamics.phonon_wake import (
    backward_excess_centroid_sites,
    event_aligned_coordinates,
    fit_group_velocity_sites_per_fs,
    intermolecular_energy_flux,
    local_lattice_energy_density,
    longitudinal_flux_field,
    longitudinal_profile,
    wake_window_metrics,
)
from holstein_peierls.dynamics.thermal import (
    coupled_baoab_step,
    thermostatted_kinetic_degrees_of_freedom,
    zero_mode_means,
)
from holstein_peierls.dynamics.thermal_decoherence import decoherence_interval_steps

from ip1c_event_conditioned_dressed_hopping import (  # noqa: E402
    _integer_stride,
    _prepare_initial_state,
)


PRIMARY_PERSISTENCE_FS = 50.0
PRE_EVENT_FS = 200.0
POST_EVENT_FS = 2000.0
BASELINE_START_FS = -200.0
BASELINE_END_FS = -80.0
TRANSVERSE_HALF_WIDTH_SITES = 5
CORE_HALF_WIDTH_SITES = 1
GROUP_FIT_START_FS = 200.0
GROUP_FIT_END_FS = 1800.0
IDC_CLEAN_HALF_WINDOW_FS = 100.0
WINDOWS_FS = (
    (0.0, 500.0),
    (500.0, 1000.0),
    (1000.0, 2000.0),
)


def _window_label(start: float, end: float) -> str:
    return f"{int(start)}:{int(end)}fs"


def _mean_or_none(values) -> float | None:
    data = np.asarray(list(values), dtype=np.float64)
    data = data[np.isfinite(data)]
    return float(np.mean(data)) if data.size else None


def _median_or_none(values) -> float | None:
    data = np.asarray(list(values), dtype=np.float64)
    data = data[np.isfinite(data)]
    return float(np.median(data)) if data.size else None


def _profile_stack(
    maps: np.ndarray,
    coordinates,
    frame_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    profiles: list[np.ndarray] = []
    s_axis: np.ndarray | None = None
    for index in frame_indices:
        s, profile = longitudinal_profile(
            maps[int(index)],
            coordinates,
            transverse_half_width_sites=TRANSVERSE_HALF_WIDTH_SITES,
        )
        if s_axis is None:
            s_axis = s
        profiles.append(profile)
    if s_axis is None or not profiles:
        raise ValueError("empty event profile request")
    return np.asarray(s_axis, dtype=np.int64), np.asarray(profiles, dtype=np.float64)


def _event_wake_record(
    event: PersistentHopEvent,
    *,
    times_fs: np.ndarray,
    energy_u: np.ndarray,
    energy_vx: np.ndarray,
    energy_vy: np.ndarray,
    flux_jx: np.ndarray,
    flux_jy: np.ndarray,
    idc_times_fs: np.ndarray,
    nx: int,
    ny: int,
    previous_gap_fs: float | None,
    next_gap_fs: float | None,
) -> tuple[dict, dict[str, np.ndarray]] | None:
    if not event.is_nearest_neighbor:
        return None
    t0 = float(event.transition_start_time_fs)
    rel_all = np.asarray(times_fs - t0, dtype=np.float64)
    complete = (
        np.min(rel_all) <= -PRE_EVENT_FS + 1.0e-9
        and np.max(rel_all) >= POST_EVENT_FS - 1.0e-9
    )
    if not complete:
        return None
    selected = np.flatnonzero(
        (rel_all >= -PRE_EVENT_FS - 1.0e-9)
        & (rel_all <= POST_EVENT_FS + 1.0e-9)
    )
    rel = rel_all[selected]
    coordinates = event_aligned_coordinates(
        event.source_site,
        event.target_site,
        nx,
        ny,
    )

    long_maps = energy_vx if event.dx_sites != 0 else energy_vy
    transverse_maps = energy_vy if event.dx_sites != 0 else energy_vx
    s_axis, u_profile = _profile_stack(energy_u, coordinates, selected)
    _, long_profile = _profile_stack(long_maps, coordinates, selected)
    _, transverse_profile = _profile_stack(transverse_maps, coordinates, selected)
    inter_profile = long_profile + transverse_profile

    flux_profiles: list[np.ndarray] = []
    for index in selected:
        from holstein_peierls.dynamics.phonon_wake import IntermolecularEnergyFlux

        projected = longitudinal_flux_field(
            IntermolecularEnergyFlux(
                np.asarray(flux_jx[int(index)], dtype=np.float64),
                np.asarray(flux_jy[int(index)], dtype=np.float64),
            ),
            coordinates,
        )
        _, profile = longitudinal_profile(
            projected,
            coordinates,
            transverse_half_width_sites=TRANSVERSE_HALF_WIDTH_SITES,
        )
        flux_profiles.append(profile)
    flux_profile = np.asarray(flux_profiles, dtype=np.float64)

    baseline = (rel >= BASELINE_START_FS) & (rel <= BASELINE_END_FS)
    if not np.any(baseline):
        return None
    u_excess = u_profile - np.mean(u_profile[baseline], axis=0)
    long_excess = long_profile - np.mean(long_profile[baseline], axis=0)
    transverse_excess = transverse_profile - np.mean(transverse_profile[baseline], axis=0)
    inter_excess = inter_profile - np.mean(inter_profile[baseline], axis=0)
    flux_excess = flux_profile - np.mean(flux_profile[baseline], axis=0)

    windows: dict[str, dict] = {}
    for start, end in WINDOWS_FS:
        mask = (rel >= start) & (rel < end + 1.0e-9)
        if not np.any(mask):
            continue
        averaged_inter = np.mean(inter_excess[mask], axis=0)
        averaged_flux = np.mean(flux_excess[mask], axis=0)
        metrics = wake_window_metrics(
            averaged_inter,
            averaged_flux,
            s_axis,
            core_half_width_sites=CORE_HALF_WIDTH_SITES,
        )
        long_average = np.mean(long_excess[mask], axis=0)
        transverse_average = np.mean(transverse_excess[mask], axis=0)
        u_average = np.mean(u_excess[mask], axis=0)
        outer = np.abs(s_axis) > CORE_HALF_WIDTH_SITES
        long_positive = float(np.sum(np.clip(long_average[outer], 0.0, None)))
        transverse_positive = float(
            np.sum(np.clip(transverse_average[outer], 0.0, None))
        )
        u_positive = float(np.sum(np.clip(u_average[outer], 0.0, None)))
        mode_denominator = long_positive + transverse_positive
        transverse_fraction = (
            None
            if mode_denominator <= 1.0e-20
            else float(transverse_positive / mode_denominator)
        )
        windows[_window_label(start, end)] = {
            **asdict(metrics),
            "longitudinal_positive_excess_eV": long_positive,
            "transverse_positive_excess_eV": transverse_positive,
            "intramolecular_positive_excess_eV": u_positive,
            "transverse_inter_fraction": transverse_fraction,
        }

    fit_mask = (rel >= GROUP_FIT_START_FS) & (rel <= GROUP_FIT_END_FS)
    fit_times: list[float] = []
    fit_centers: list[float] = []
    for time, profile in zip(rel[fit_mask], inter_excess[fit_mask], strict=True):
        center = backward_excess_centroid_sites(
            profile,
            s_axis,
            core_half_width_sites=CORE_HALF_WIDTH_SITES,
        )
        if center is not None:
            fit_times.append(float(time))
            fit_centers.append(float(center))
    group_fit = fit_group_velocity_sites_per_fs(
        np.asarray(fit_times, dtype=np.float64),
        np.asarray(fit_centers, dtype=np.float64),
        minimum_points=20,
    )

    nearest_idc = None
    if idc_times_fs.size:
        nearest_idc = float(np.min(np.abs(idc_times_fs - t0)))
    idc_clean = bool(nearest_idc is None or nearest_idc > IDC_CLEAN_HALF_WINDOW_FS)
    isolated_1000 = bool(
        (previous_gap_fs is None or previous_gap_fs > 1000.0)
        and (next_gap_fs is None or next_gap_fs > 1000.0)
    )
    record = {
        "source_site": int(event.source_site),
        "target_site": int(event.target_site),
        "direction": event.direction,
        "transition_start_time_fs": t0,
        "accepted_time_fs": float(event.accepted_time_fs),
        "previous_nn_event_gap_fs": previous_gap_fs,
        "next_nn_event_gap_fs": next_gap_fs,
        "isolated_1000fs": isolated_1000,
        "nearest_idc_distance_fs": nearest_idc,
        "idc_clean_100fs": idc_clean,
        "windows": windows,
        "backward_centroid_group_fit": group_fit,
    }
    profiles = {
        "relative_time_fs": rel,
        "s_sites": s_axis.astype(np.float64),
        "u_excess": u_excess,
        "longitudinal_excess": long_excess,
        "transverse_excess": transverse_excess,
        "intermolecular_excess": inter_excess,
        "longitudinal_flux_excess": flux_excess,
    }
    return record, profiles


def _aggregate_events(events: list[dict], pooled_profiles: dict[str, np.ndarray] | None) -> dict:
    result = {
        "complete_event_count": int(len(events)),
        "idc_clean_event_count": int(sum(e["idc_clean_100fs"] for e in events)),
        "isolated_1000fs_event_count": int(sum(e["isolated_1000fs"] for e in events)),
        "direction_counts": {
            direction: int(sum(e["direction"] == direction for e in events))
            for direction in ("+x", "-x", "+y", "-y")
        },
    }
    if not events:
        return result

    window_summary: dict[str, dict] = {}
    for start, end in WINDOWS_FS:
        label = _window_label(start, end)
        available = [e["windows"][label] for e in events if label in e["windows"]]
        if not available:
            continue
        fields = (
            "signed_asymmetry",
            "backward_positive_excess_eV",
            "forward_positive_excess_eV",
            "backward_outward_flux_eV_per_fs",
            "forward_outward_flux_eV_per_fs",
            "flux_bias",
            "longitudinal_positive_excess_eV",
            "transverse_positive_excess_eV",
            "intramolecular_positive_excess_eV",
            "transverse_inter_fraction",
        )
        window_summary[label] = {
            field: _mean_or_none(
                item[field] for item in available if item.get(field) is not None
            )
            for field in fields
        }
        window_summary[label]["retrograde_flux_positive_fraction"] = float(
            np.mean([item["backward_outward_flux_eV_per_fs"] > 0.0 for item in available])
        )
        window_summary[label]["retrograde_flux_dominant_fraction"] = float(
            np.mean(
                [
                    item["backward_outward_flux_eV_per_fs"]
                    > item["forward_outward_flux_eV_per_fs"]
                    for item in available
                ]
            )
        )
    result["windows"] = window_summary

    valid_fits = [
        e["backward_centroid_group_fit"]
        for e in events
        if e["backward_centroid_group_fit"]["velocity_sites_per_ps"] is not None
    ]
    result["event_group_velocity"] = {
        "valid_fit_count": int(len(valid_fits)),
        "mean_sites_per_ps": _mean_or_none(
            fit["velocity_sites_per_ps"] for fit in valid_fits
        ),
        "median_sites_per_ps": _median_or_none(
            fit["velocity_sites_per_ps"] for fit in valid_fits
        ),
        "negative_fraction": (
            None
            if not valid_fits
            else float(
                np.mean([fit["velocity_sites_per_ps"] < 0.0 for fit in valid_fits])
            )
        ),
        "median_r_squared": _median_or_none(fit["r_squared"] for fit in valid_fits),
    }

    if pooled_profiles is not None:
        rel = pooled_profiles["relative_time_fs"]
        s_axis = pooled_profiles["s_sites"].astype(np.int64)
        inter = pooled_profiles["intermolecular_excess"]
        fit_mask = (rel >= GROUP_FIT_START_FS) & (rel <= GROUP_FIT_END_FS)
        centers = []
        fit_times = []
        for time, profile in zip(rel[fit_mask], inter[fit_mask], strict=True):
            center = backward_excess_centroid_sites(
                profile,
                s_axis,
                core_half_width_sites=CORE_HALF_WIDTH_SITES,
            )
            if center is not None:
                fit_times.append(float(time))
                centers.append(float(center))
        result["pooled_group_velocity"] = fit_group_velocity_sites_per_fs(
            np.asarray(fit_times, dtype=np.float64),
            np.asarray(centers, dtype=np.float64),
            minimum_points=20,
        )
    return result


def _run_trajectory(
    initial: CoupledEhrenfestState,
    parameters,
    *,
    temperature_K: float,
    gamma_u_per_fs: float,
    gamma_v_per_fs: float,
    dt_fs: float,
    final_time_fs: float,
    burn_in_fs: float,
    sample_interval_fs: float,
    decoherence_interval_fs: float,
    lattice_seed: int,
    decoherence_seed: int,
    krylov_dimension: int,
) -> tuple[dict, dict[str, np.ndarray] | None]:
    steps = _integer_stride(final_time_fs, dt_fs, "final_time_fs")
    sample_stride = _integer_stride(sample_interval_fs, dt_fs, "sample_interval_fs")
    persistence_samples = _integer_stride(
        PRIMARY_PERSISTENCE_FS, sample_interval_fs, "primary persistence"
    )
    event_stride = decoherence_interval_steps(decoherence_interval_fs, dt_fs)
    expected_samples = int(
        sum(
            1
            for step in range(steps)
            if (step + 1) % sample_stride == 0
            and (step + 1) * dt_fs >= burn_in_fs - 1.0e-9
        )
    )
    shape = (parameters.ny, parameters.nx)
    times = np.empty(expected_samples, dtype=np.float64)
    energy_u = np.empty((expected_samples, *shape), dtype=np.float32)
    energy_vx = np.empty((expected_samples, *shape), dtype=np.float32)
    energy_vy = np.empty((expected_samples, *shape), dtype=np.float32)
    flux_jx = np.empty((expected_samples, *shape), dtype=np.float32)
    flux_jy = np.empty((expected_samples, *shape), dtype=np.float32)

    tracker = PersistentSiteTracker(
        parameters.nx,
        parameters.ny,
        persistence_samples=persistence_samples,
        minimum_max_population=0.10,
        minimum_dominance_margin=0.02,
    )
    current = CoupledEhrenfestState(
        initial.lattice.copy(), initial.velocity.copy(), initial.electronic_state.copy()
    )
    bath = LangevinBath(float(temperature_K), float(gamma_u_per_fs), float(gamma_v_per_fs))
    lattice_rng = np.random.default_rng(int(lattice_seed))
    decoherence_rng = np.random.default_rng(int(decoherence_seed))
    dof = thermostatted_kinetic_degrees_of_freedom(parameters, "project")

    initial_energy = dynamic_total_energy(
        current.lattice, current.velocity, parameters, current.electronic_state
    ).total
    initial_norm = float(np.linalg.norm(current.electronic_state))
    lattice_heat = 0.0
    electronic_exchange = 0.0
    max_norm_error = 0.0
    max_balance_residual = 0.0
    max_zero_mode = 0.0
    idc_events = 0
    idc_times: list[float] = []
    temperatures: list[float] = []
    accepted_events: list[PersistentHopEvent] = []
    sample_index = 0

    start_clock = perf_counter()
    for step in range(steps):
        time_fs = float(step * dt_fs)
        thermal = coupled_baoab_step(
            current,
            parameters,
            bath,
            lattice_rng,
            time_fs,
            dt_fs,
            zero_mode_policy="project",
            electronic_method="cfm4_lanczos",
            krylov_dimension=int(krylov_dimension),
        )
        current = thermal.state
        lattice_heat += float(thermal.bath_heat_eV)
        max_norm_error = max(
            max_norm_error,
            abs(float(np.linalg.norm(current.electronic_state)) - initial_norm),
        )
        end_time_fs = float((step + 1) * dt_fs)
        if (step + 1) % event_stride == 0:
            collapse = apply_instantaneous_decoherence(
                current.lattice,
                parameters,
                current.electronic_state,
                bath.temperature_K,
                "bm",
                decoherence_rng,
            )
            electronic_exchange += float(collapse.electronic_environment_exchange_eV)
            current = CoupledEhrenfestState(
                current.lattice.copy(), current.velocity.copy(), collapse.electronic_state.copy()
            )
            idc_events += 1
            idc_times.append(end_time_fs)
            matter = dynamic_total_energy(
                current.lattice, current.velocity, parameters, current.electronic_state
            ).total
            residual = matter - initial_energy - lattice_heat - electronic_exchange
            max_balance_residual = max(max_balance_residual, abs(float(residual)))

        if (step + 1) % sample_stride != 0 or end_time_fs < burn_in_fs - 1.0e-9:
            continue
        density = local_lattice_energy_density(current.lattice, current.velocity, parameters)
        flux = intermolecular_energy_flux(current.lattice, current.velocity, parameters)
        times[sample_index] = end_time_fs
        energy_u[sample_index] = density.u.astype(np.float32)
        energy_vx[sample_index] = density.vx.astype(np.float32)
        energy_vy[sample_index] = density.vy.astype(np.float32)
        flux_jx[sample_index] = flux.jx.astype(np.float32)
        flux_jy[sample_index] = flux.jy.astype(np.float32)
        temperatures.append(
            kinetic_temperature_K(current.velocity, parameters, degrees_of_freedom=dof)
        )

        population = np.abs(np.asarray(current.electronic_state)) ** 2
        population = np.asarray(population / np.sum(population), dtype=np.float64)
        event = tracker.update(population, end_time_fs)
        if event is not None and event.is_nearest_neighbor:
            accepted_events.append(event)
        means = zero_mode_means(current)
        max_zero_mode = max(max_zero_mode, max(abs(float(value)) for value in means.values()))
        sample_index += 1

    if sample_index != expected_samples:
        raise RuntimeError("unexpected number of diagnostic samples")
    final_energy = dynamic_total_energy(
        current.lattice, current.velocity, parameters, current.electronic_state
    ).total
    final_residual = final_energy - initial_energy - lattice_heat - electronic_exchange
    max_balance_residual = max(max_balance_residual, abs(float(final_residual)))

    event_records: list[dict] = []
    pooled_sum: dict[str, np.ndarray] | None = None
    pooled_count = 0
    starts = np.asarray([e.transition_start_time_fs for e in accepted_events], dtype=np.float64)
    idc_array = np.asarray(idc_times, dtype=np.float64)
    for index, event in enumerate(accepted_events):
        previous_gap = None if index == 0 else float(starts[index] - starts[index - 1])
        next_gap = None if index + 1 == len(starts) else float(starts[index + 1] - starts[index])
        analyzed = _event_wake_record(
            event,
            times_fs=times,
            energy_u=energy_u,
            energy_vx=energy_vx,
            energy_vy=energy_vy,
            flux_jx=flux_jx,
            flux_jy=flux_jy,
            idc_times_fs=idc_array,
            nx=parameters.nx,
            ny=parameters.ny,
            previous_gap_fs=previous_gap,
            next_gap_fs=next_gap,
        )
        if analyzed is None:
            continue
        record, profiles = analyzed
        event_records.append(record)
        if pooled_sum is None:
            pooled_sum = {key: np.asarray(value, dtype=np.float64).copy() for key, value in profiles.items()}
        else:
            for key in (
                "u_excess",
                "longitudinal_excess",
                "transverse_excess",
                "intermolecular_excess",
                "longitudinal_flux_excess",
            ):
                pooled_sum[key] += profiles[key]
        pooled_count += 1

    pooled_profiles = None
    if pooled_sum is not None and pooled_count > 0:
        pooled_profiles = pooled_sum
        for key in (
            "u_excess",
            "longitudinal_excess",
            "transverse_excess",
            "intermolecular_excess",
            "longitudinal_flux_excess",
        ):
            pooled_profiles[key] = pooled_profiles[key] / float(pooled_count)

    elapsed = perf_counter() - start_clock
    record = {
        "temperature_K": float(temperature_K),
        "gamma_u_per_fs": float(gamma_u_per_fs),
        "gamma_v_per_fs": float(gamma_v_per_fs),
        "lattice_seed": int(lattice_seed),
        "decoherence_seed": int(decoherence_seed),
        "steps": int(steps),
        "elapsed_seconds": float(elapsed),
        "idc_events": int(idc_events),
        "sample_count_post_burn": int(sample_index),
        "temperature_mean_K": float(np.mean(temperatures)),
        "electronic_nearest_neighbor_event_count": int(len(accepted_events)),
        "complete_wake_event_count": int(len(event_records)),
        "event_records": event_records,
        "maximum_generalized_energy_residual_eV": float(max_balance_residual),
        "final_generalized_energy_residual_eV": float(final_residual),
        "maximum_electronic_norm_error": float(max_norm_error),
        "maximum_zero_mode_magnitude": float(max_zero_mode),
    }
    record["aggregate"] = _aggregate_events(event_records, pooled_profiles)
    return record, pooled_profiles


def _combine_profiles(profile_records: list[tuple[dict[str, np.ndarray] | None, int]]) -> dict[str, np.ndarray] | None:
    valid = [(profile, count) for profile, count in profile_records if profile is not None and count > 0]
    if not valid:
        return None
    total_count = int(sum(count for _, count in valid))
    first = valid[0][0]
    assert first is not None
    combined = {
        "relative_time_fs": first["relative_time_fs"].copy(),
        "s_sites": first["s_sites"].copy(),
    }
    for key in (
        "u_excess",
        "longitudinal_excess",
        "transverse_excess",
        "intermolecular_excess",
        "longitudinal_flux_excess",
    ):
        accumulator = np.zeros_like(first[key], dtype=np.float64)
        for profile, count in valid:
            assert profile is not None
            accumulator += profile[key] * float(count)
        combined[key] = accumulator / float(total_count)
    return combined


def _condition_aggregate(trajectories: list[dict], pooled_profiles: dict[str, np.ndarray] | None) -> dict:
    events = [event for trajectory in trajectories for event in trajectory["event_records"]]
    result = _aggregate_events(events, pooled_profiles)
    result.update(
        {
            "trajectory_count": int(len(trajectories)),
            "temperature_mean_K": float(np.mean([t["temperature_mean_K"] for t in trajectories])),
            "electronic_nearest_neighbor_event_count": int(
                sum(t["electronic_nearest_neighbor_event_count"] for t in trajectories)
            ),
            "maximum_generalized_energy_residual_eV": float(
                max(t["maximum_generalized_energy_residual_eV"] for t in trajectories)
            ),
            "maximum_electronic_norm_error": float(
                max(t["maximum_electronic_norm_error"] for t in trajectories)
            ),
            "maximum_zero_mode_magnitude": float(
                max(t["maximum_zero_mode_magnitude"] for t in trajectories)
            ),
            "seed_summaries": [
                {
                    "seed": int(t["lattice_seed"]),
                    "complete_wake_event_count": int(t["complete_wake_event_count"]),
                    "aggregate": t["aggregate"],
                }
                for t in trajectories
            ],
        }
    )
    return result


def _evaluate_numerical_checks(payload: dict) -> dict[str, bool]:
    trajectories = [t for c in payload["conditions"] for t in c["trajectories"]]
    expected_idc = int(np.floor(payload["final_time_fs"] / payload["decoherence_interval_fs"] + 1.0e-12))
    requested = len(payload["conditions"]) * len(payload["lattice_seeds"])
    temperature_ok = all(
        abs(float(t["temperature_mean_K"]) - float(t["temperature_K"]))
        <= max(15.0, 0.10 * float(t["temperature_K"]))
        for t in trajectories
    )
    tolerance = size_scaled_energy_balance_tolerance_eV(payload["size"] ** 2)
    balance_ok = all(
        extensive_energy_balance_passes(
            float(t["maximum_generalized_energy_residual_eV"]),
            payload["size"] ** 2,
        )
        for t in trajectories
    )
    finite = True
    for condition in payload["conditions"]:
        aggregate = condition["aggregate"]
        pooled = aggregate.get("pooled_group_velocity")
        if pooled is not None and pooled["velocity_sites_per_ps"] is not None:
            finite = finite and np.isfinite(float(pooled["velocity_sites_per_ps"]))
        for window in aggregate.get("windows", {}).values():
            for value in window.values():
                if value is not None:
                    finite = finite and np.isfinite(float(value))
    return {
        "all_static_relaxations_converged": bool(all(payload["static"][key]["converged"] for key in payload["static"])),
        "all_requested_trajectories_completed": bool(len(trajectories) == requested),
        "exact_idc_event_count": bool(all(int(t["idc_events"]) == expected_idc for t in trajectories)),
        "lattice_temperature": bool(temperature_ok),
        "size_scaled_zero_field_generalized_energy_balance": bool(balance_ok),
        "electronic_norm": bool(max(t["maximum_electronic_norm_error"] for t in trajectories) < 1.0e-10),
        "projected_zero_modes": bool(max(t["maximum_zero_mode_magnitude"] for t in trajectories) < 1.0e-12),
        "stationary_carrier_recurrence_guard": bool(payload["final_time_fs"] < payload["recurrence"]["earliest_stationary_carrier_wrap_fs"]),
        "finite_wake_diagnostics": bool(finite),
        "energy_tolerance_recorded": bool(np.isclose(payload["energy_balance_tolerance_eV"], tolerance)),
    }


def _markdown(payload: dict) -> str:
    lines = [
        "# IP1f carrier-centered phonon-wake characterization",
        "",
        f"Cell {payload['size']}x{payload['size']}; T={payload['temperature_K']:.0f} K; final={payload['final_time_fs']/1000:.1f} ps; burn={payload['burn_in_fs']/1000:.1f} ps; event window=[-{PRE_EVENT_FS:.0f}, +{POST_EVENT_FS:.0f}] fs.",
        "",
        "All carrier events are aligned so that the electronic relocation points toward +s. Negative fitted wake velocity is therefore retrograde relative to the carrier event direction.",
        "",
        "| gamma_v [fs^-1] | J0y/J0x | complete events | IDC-clean | wake asym 0:500 | retro flux bias 0:500 | retro flux + frac | pooled v_wake [sites/ps] | R2 | transverse fraction 0:500 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in payload["conditions"]:
        aggregate = condition["aggregate"]
        early = aggregate.get("windows", {}).get("0:500fs", {})
        pooled = aggregate.get("pooled_group_velocity", {})
        def fmt(value, digits=3):
            return "nan" if value is None else f"{float(value):.{digits}f}"
        lines.append(
            "| "
            f"{condition['gamma_v_per_fs']:.3f} | {condition['anisotropy_ratio']:.2f} | "
            f"{aggregate['complete_event_count']} | {aggregate['idc_clean_event_count']} | "
            f"{fmt(early.get('signed_asymmetry'))} | {fmt(early.get('flux_bias'))} | "
            f"{fmt(early.get('retrograde_flux_positive_fraction'))} | "
            f"{fmt(pooled.get('velocity_sites_per_ps'))} | {fmt(pooled.get('r_squared'))} | "
            f"{fmt(early.get('transverse_inter_fraction'))} |"
        )
    lines.extend(["", "## Numerical gates", ""])
    for name, passed in payload["numerical_checks"].items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    lines.extend(
        [
            "",
            f"Numerical status: {'PASS' if payload['numerical_pass'] else 'FAIL'}",
            "",
            "Wake asymmetry, energy-current direction, centroid velocity and mode fractions are physical diagnostics, never numerical PASS gates. In the strongly damped gamma_v=0.01 control, a fitted centroid velocity must not be over-interpreted as a sharp phonon quasiparticle group velocity.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=40)
    parser.add_argument("--temperature-K", type=float, default=300.0)
    parser.add_argument("--ratios", type=float, nargs="+", default=[1.0, 0.15])
    parser.add_argument("--gamma-v-values", type=float, nargs="+", default=[0.01, 0.002])
    parser.add_argument("--gamma-u-per-fs", type=float, default=0.01)
    parser.add_argument("--dt-fs", type=float, default=0.2)
    parser.add_argument("--final-time-fs", type=float, default=15000.0)
    parser.add_argument("--burn-in-fs", type=float, default=5000.0)
    parser.add_argument("--sample-interval-fs", type=float, default=2.0)
    parser.add_argument("--decoherence-interval-fs", type=float, default=180.0)
    parser.add_argument("--lattice-seeds", type=int, nargs="+", default=[20260905, 20260906])
    parser.add_argument("--krylov-dimension", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    args = parser.parse_args()

    if args.final_time_fs <= args.burn_in_fs + PRE_EVENT_FS + POST_EVENT_FS:
        raise ValueError("trajectory is too short for the requested wake event window")

    prepared: dict[float, tuple] = {}
    static: dict[str, dict] = {}
    for ratio in args.ratios:
        initial, parameters, metadata = _prepare_initial_state(args.size, ratio)
        prepared[float(ratio)] = (initial, parameters)
        static[str(float(ratio))] = metadata

    recurrence = intermolecular_recurrence_scales(
        prepared[float(args.ratios[0])][1],
        lattice_spacing_A=3.0,
        gamma_v_per_fs=min(float(v) for v in args.gamma_v_values),
    )
    if args.final_time_fs >= recurrence.earliest_stationary_carrier_wrap_fs:
        raise ValueError("planned IP1f trajectory reaches the stationary-carrier PBC wrap estimate")

    conditions: list[dict] = []
    npz_payload: dict[str, np.ndarray] = {}
    for gamma_v in args.gamma_v_values:
        for ratio in args.ratios:
            initial, parameters = prepared[float(ratio)]
            trajectories: list[dict] = []
            profile_records: list[tuple[dict[str, np.ndarray] | None, int]] = []
            for seed in args.lattice_seeds:
                trajectory, profiles = _run_trajectory(
                    initial,
                    parameters,
                    temperature_K=args.temperature_K,
                    gamma_u_per_fs=args.gamma_u_per_fs,
                    gamma_v_per_fs=float(gamma_v),
                    dt_fs=args.dt_fs,
                    final_time_fs=args.final_time_fs,
                    burn_in_fs=args.burn_in_fs,
                    sample_interval_fs=args.sample_interval_fs,
                    decoherence_interval_fs=args.decoherence_interval_fs,
                    lattice_seed=int(seed),
                    decoherence_seed=int(seed) + 100000,
                    krylov_dimension=args.krylov_dimension,
                )
                trajectories.append(trajectory)
                profile_records.append((profiles, trajectory["complete_wake_event_count"]))
            pooled_profiles = _combine_profiles(profile_records)
            aggregate = _condition_aggregate(trajectories, pooled_profiles)
            condition = {
                "gamma_v_per_fs": float(gamma_v),
                "anisotropy_ratio": float(ratio),
                "temperature_K": float(args.temperature_K),
                "trajectories": trajectories,
                "aggregate": aggregate,
            }
            conditions.append(condition)
            if pooled_profiles is not None:
                key = f"g{str(gamma_v).replace('.', 'p')}_r{str(ratio).replace('.', 'p')}"
                for name, value in pooled_profiles.items():
                    npz_payload[f"{key}_{name}"] = np.asarray(value)

    payload = {
        "scope": "IP1f carrier-centered event-conditioned phonon-wake characterization; no mobility/diffusion/activation/rate claim",
        "size": int(args.size),
        "temperature_K": float(args.temperature_K),
        "dt_fs": float(args.dt_fs),
        "final_time_fs": float(args.final_time_fs),
        "burn_in_fs": float(args.burn_in_fs),
        "sample_interval_fs": float(args.sample_interval_fs),
        "primary_persistence_fs": PRIMARY_PERSISTENCE_FS,
        "pre_event_fs": PRE_EVENT_FS,
        "post_event_fs": POST_EVENT_FS,
        "baseline_window_fs": [BASELINE_START_FS, BASELINE_END_FS],
        "transverse_half_width_sites": TRANSVERSE_HALF_WIDTH_SITES,
        "core_half_width_sites": CORE_HALF_WIDTH_SITES,
        "group_fit_window_fs": [GROUP_FIT_START_FS, GROUP_FIT_END_FS],
        "gamma_u_per_fs": float(args.gamma_u_per_fs),
        "gamma_v_values_per_fs": [float(value) for value in args.gamma_v_values],
        "decoherence_interval_fs": float(args.decoherence_interval_fs),
        "decoherence_interval_is_material_calibrated": False,
        "lattice_seeds": [int(seed) for seed in args.lattice_seeds],
        "static": static,
        "recurrence": asdict(recurrence),
        "energy_balance_tolerance_eV": size_scaled_energy_balance_tolerance_eV(args.size * args.size),
        "profiles_npz": str(args.profiles),
        "conditions": conditions,
        "physical_interpretation_is_manual": True,
    }
    payload["numerical_checks"] = _evaluate_numerical_checks(payload)
    payload["numerical_pass"] = bool(all(payload["numerical_checks"].values()))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.markdown.write_text(_markdown(payload), encoding="utf-8")
    np.savez_compressed(args.profiles, **npz_payload)
    print(_markdown(payload), end="")
    if not payload["numerical_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
