#!/usr/bin/env python3
"""IP1d matched-counterfactual structural precursor validation.

The dynamics are intentionally identical to IP1c.  Every persistent electronic
nearest-neighbour relocation is compared with the three alternative neighbour
directions available at the same source and same time.  This removes reliance
on a naive 0.25 symmetry baseline and adds seed-level screening uncertainty.

No hopping rate, activation energy, diffusion coefficient, or mobility is
inferred here.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import stats

from holstein_peierls.dynamics.coupled import CoupledEhrenfestState
from holstein_peierls.dynamics.ehrenfest import dynamic_total_energy
from holstein_peierls.dynamics.electronic_decoherence import apply_instantaneous_decoherence
from holstein_peierls.dynamics.event_conditioned_hopping import PeriodicTemplateCorrelationField
from holstein_peierls.dynamics.hopping_observables import (
    PersistentHopEvent,
    PersistentSiteTracker,
    local_bond_transfer_magnitudes,
)
from holstein_peierls.dynamics.langevin import LangevinBath, kinetic_temperature_K
from holstein_peierls.dynamics.matched_precursors import (
    DIRECTIONS,
    compare_true_direction,
    nearest_negative_to_nonnegative_crossing,
    nearest_neighbor_targets,
)
from holstein_peierls.dynamics.thermal import (
    coupled_baoab_step,
    project_inter_molecular_zero_modes,
    thermostatted_kinetic_degrees_of_freedom,
    zero_mode_means,
)
from holstein_peierls.dynamics.thermal_decoherence import decoherence_interval_steps
from holstein_peierls.dynamics.transport_observables import (
    transport_kinematics,
    trapezoidal_displacement_increment,
)

# Reuse the already validated IP1c setup helpers so the frozen dynamics do not
# silently diverge between the two mechanistic experiments.
from ip1c_event_conditioned_dressed_hopping import (  # noqa: E402
    _integer_stride,
    _nearest_frame_index,
    _prepare_initial_state,
)


PRIMARY_PERSISTENCE_FS = 50.0
EVENT_WINDOW_FS = 500.0
QUIET_EDGE_FS = 100.0
CURRENT_WINDOWS_FS = (100.0, 250.0, 500.0)
PREHOP_BINS_FS = (
    (-500.0, -400.0),
    (-400.0, -300.0),
    (-300.0, -200.0),
    (-200.0, -150.0),
    (-150.0, -100.0),
    (-100.0, -80.0),
    (-80.0, -60.0),
    (-60.0, -40.0),
    (-40.0, -20.0),
    (-20.0, 0.000001),
)


def _mean_or_none(values) -> float | None:
    data = np.asarray(list(values), dtype=np.float64)
    data = data[np.isfinite(data)]
    return float(np.mean(data)) if data.size else None


def _median_or_none(values) -> float | None:
    data = np.asarray(list(values), dtype=np.float64)
    data = data[np.isfinite(data)]
    return float(np.median(data)) if data.size else None


def _student_t_summary(values) -> dict:
    data = np.asarray(list(values), dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return {"n": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    mean = float(np.mean(data))
    if data.size == 1:
        return {"n": 1, "mean": mean, "ci95_low": None, "ci95_high": None}
    sem = float(np.std(data, ddof=1) / np.sqrt(data.size))
    if sem == 0.0:
        low = high = mean
    else:
        low, high = stats.t.interval(0.95, data.size - 1, loc=mean, scale=sem)
    return {
        "n": int(data.size),
        "mean": mean,
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def _amplitude_series(maps: np.ndarray, field: PeriodicTemplateCorrelationField, site: int) -> np.ndarray:
    iy, ix = field.shift_index_for_center(site)
    return np.asarray(maps[:, iy, ix], dtype=np.float64)


def _directional_q_metrics(
    source_total: np.ndarray,
    target_total: np.ndarray,
    rel: np.ndarray,
) -> dict:
    denominator = np.abs(source_total) + np.abs(target_total)
    q = np.divide(
        target_total - source_total,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 1.0e-15,
    )
    pre = (rel >= -EVENT_WINDOW_FS) & (rel <= -QUIET_EDGE_FS)
    post = (rel >= QUIET_EDGE_FS) & (rel <= EVENT_WINDOW_FS)
    local_pre = (rel >= -100.0) & (rel <= -20.0)
    local_post = (rel >= 20.0) & (rel <= 100.0)
    if not np.any(pre) or not np.any(post):
        raise ValueError("event window does not contain required pre/post regions")
    q_pre = float(np.mean(q[pre]))
    q_post = float(np.mean(q[post]))
    nearest = nearest_negative_to_nonnegative_crossing(rel, q, 0.0)
    local_pre_mean = float(np.mean(q[local_pre])) if np.any(local_pre) else None
    local_post_mean = float(np.mean(q[local_post])) if np.any(local_post) else None
    near_reversal = bool(
        local_pre_mean is not None
        and local_post_mean is not None
        and local_pre_mean < 0.0
        and local_post_mean > 0.0
    )
    start_index = int(np.argmin(np.abs(rel)))
    return {
        "q_pre_mean": q_pre,
        "q_post_mean": q_post,
        "q_delta": float(q_post - q_pre),
        "template_signal_pre_mean": float(np.mean(denominator[pre])),
        "template_signal_post_mean": float(np.mean(denominator[post])),
        "template_signal_at_event": float(denominator[start_index]),
        "nearest_crossing_lag_fs": nearest,
        "near_event_sign_reversal": near_reversal,
        "local_pre_mean": local_pre_mean,
        "local_post_mean": local_post_mean,
    }


def _component_change(
    component_maps: np.ndarray,
    mask: np.ndarray,
    rel: np.ndarray,
    field: PeriodicTemplateCorrelationField,
    source_site: int,
    target_site: int,
) -> float:
    source = _amplitude_series(component_maps[mask], field, source_site)
    target = _amplitude_series(component_maps[mask], field, target_site)
    difference = target - source
    pre = (rel >= -EVENT_WINDOW_FS) & (rel <= -QUIET_EDGE_FS)
    post = (rel >= QUIET_EDGE_FS) & (rel <= EVENT_WINDOW_FS)
    return float(np.mean(difference[post]) - np.mean(difference[pre]))


def _event_record(
    event: PersistentHopEvent,
    *,
    field: PeriodicTemplateCorrelationField,
    nx: int,
    ny: int,
    times: np.ndarray,
    total_maps: np.ndarray,
    u_maps: np.ndarray,
    x_maps: np.ndarray,
    y_maps: np.ndarray,
    cumulative_x_A: np.ndarray,
    cumulative_y_A: np.ndarray,
    metric_sites: np.ndarray,
    bond_values: np.ndarray,
    after_idc: np.ndarray,
) -> dict | None:
    if not event.is_nearest_neighbor:
        return None
    start = float(event.transition_start_time_fs)
    if start - EVENT_WINDOW_FS < float(times[0]) - 1.0e-9:
        return None
    if start + EVENT_WINDOW_FS > float(times[-1]) + 1.0e-9:
        return None

    mask = (times >= start - EVENT_WINDOW_FS - 1.0e-9) & (times <= start + EVENT_WINDOW_FS + 1.0e-9)
    rel = np.asarray(times[mask] - start, dtype=np.float64)
    if rel.size < 3:
        return None

    source_total = _amplitude_series(total_maps[mask], field, event.source_site)
    targets = nearest_neighbor_targets(event.source_site, nx, ny)
    true_target_matches = bool(targets[event.direction] == int(event.target_site))

    directional_lattice: dict[str, dict] = {}
    q_delta_values: dict[str, float] = {}
    for direction in DIRECTIONS:
        target_site = targets[direction]
        target_total = _amplitude_series(total_maps[mask], field, target_site)
        metrics = _directional_q_metrics(source_total, target_total, rel)
        directional_lattice[direction] = {
            "target_site": int(target_site),
            **metrics,
        }
        q_delta_values[direction] = float(metrics["q_delta"])
    q_comparison = compare_true_direction(q_delta_values, event.direction)

    true_target = int(event.target_site)
    du = _component_change(u_maps, mask, rel, field, event.source_site, true_target)
    dx = _component_change(x_maps, mask, rel, field, event.source_site, true_target)
    dy = _component_change(y_maps, mask, rel, field, event.source_site, true_target)

    current_windows: dict[str, dict] = {}
    for half_window in CURRENT_WINDOWS_FS:
        left = _nearest_frame_index(times, start - half_window)
        right = _nearest_frame_index(times, start + half_window)
        displacement_x = float(cumulative_x_A[right] - cumulative_x_A[left])
        displacement_y = float(cumulative_y_A[right] - cumulative_y_A[left])
        values = {
            "+x": displacement_x,
            "-x": -displacement_x,
            "+y": displacement_y,
            "-y": -displacement_y,
        }
        comparison = compare_true_direction(values, event.direction)
        current_windows[f"{int(half_window)}fs"] = {
            "vector_x_A": displacement_x,
            "vector_y_A": displacement_y,
            "true_parallel_A": comparison.true_value,
            "counterfactual_parallel_mean_A": comparison.counterfactual_mean,
            "matched_advantage_A": comparison.advantage,
            "true_rank": comparison.true_rank,
            "true_is_maximum": comparison.true_is_maximum,
        }

    prehop_bins: dict[str, dict] = {}
    for left, right in PREHOP_BINS_FS:
        selected = (times >= start + left - 1.0e-9) & (times < start + right - 1.0e-9)
        selected &= metric_sites == int(event.source_site)
        indices = np.flatnonzero(selected)
        true_values: list[float] = []
        counterfactual_means: list[float] = []
        advantages: list[float] = []
        ranks: list[int] = []
        true_max: list[bool] = []
        counterfactual_top_fraction: list[float] = []
        for index in indices:
            values = {direction: float(bond_values[index, j]) for j, direction in enumerate(DIRECTIONS)}
            comparison = compare_true_direction(values, event.direction)
            true_values.append(comparison.true_value)
            counterfactual_means.append(comparison.counterfactual_mean)
            advantages.append(comparison.advantage)
            ranks.append(comparison.true_rank)
            true_max.append(comparison.true_is_maximum)
            maximum = max(values.values())
            tolerance = 1.0e-12 * max(1.0, abs(maximum))
            other_top = [
                abs(values[d] - maximum) <= tolerance
                for d in DIRECTIONS
                if d != event.direction
            ]
            counterfactual_top_fraction.append(float(np.mean(other_top)))
        label = f"{int(round(left))}:{int(round(min(right, 0.0)))}fs"
        prehop_bins[label] = {
            "sample_count": int(len(indices)),
            "true_bond_mean_eV": _mean_or_none(true_values),
            "counterfactual_bond_mean_eV": _mean_or_none(counterfactual_means),
            "matched_advantage_mean_eV": _mean_or_none(advantages),
            "true_rank_mean": _mean_or_none(ranks),
            "true_top1_fraction": _mean_or_none(true_max),
            "counterfactual_top1_fraction_mean": _mean_or_none(counterfactual_top_fraction),
        }

    start_index = _nearest_frame_index(times, start)
    true_lattice = directional_lattice[event.direction]
    output = asdict(event)
    output.update(
        {
            "candidate_direction_count": 4,
            "true_target_matches_event": true_target_matches,
            "idc_associated": bool(abs(float(times[start_index]) - start) <= 1.0e-9 and after_idc[start_index]),
            "directional_lattice": directional_lattice,
            "q_lattice_true_delta": q_comparison.true_value,
            "q_lattice_counterfactual_mean_delta": q_comparison.counterfactual_mean,
            "q_lattice_matched_advantage": q_comparison.advantage,
            "q_lattice_true_rank": q_comparison.true_rank,
            "q_lattice_true_is_maximum": q_comparison.true_is_maximum,
            "q_lattice_true_nearest_crossing_lag_fs": true_lattice["nearest_crossing_lag_fs"],
            "q_lattice_true_near_event_sign_reversal": true_lattice["near_event_sign_reversal"],
            "q_lattice_true_template_signal_at_event": true_lattice["template_signal_at_event"],
            "u_target_minus_source_change": du,
            "bond_x_target_minus_source_change": dx,
            "bond_y_target_minus_source_change": dy,
            "longitudinal_bond_change": dx if event.dx_sites != 0 else dy,
            "transverse_bond_change": dy if event.dx_sites != 0 else dx,
            "current_windows": current_windows,
            "prehop_matched_bins": prehop_bins,
        }
    )
    return output


def _run_trajectory(
    initial: CoupledEhrenfestState,
    parameters,
    *,
    static_center_site: int,
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
    lattice_spacing_A: float,
) -> dict:
    steps = _integer_stride(final_time_fs, dt_fs, "final_time_fs")
    burn_steps = int(round(burn_in_fs / dt_fs))
    if burn_steps < 0 or burn_steps >= steps or not np.isclose(burn_steps * dt_fs, burn_in_fs):
        raise ValueError("burn_in_fs must be an integer multiple of dt_fs within the run")
    sample_stride = _integer_stride(sample_interval_fs, dt_fs, "sample_interval_fs")
    persistence_samples = _integer_stride(PRIMARY_PERSISTENCE_FS, sample_interval_fs, "persistence")
    event_stride = decoherence_interval_steps(decoherence_interval_fs, dt_fs)

    tracker = PersistentSiteTracker(
        parameters.nx,
        parameters.ny,
        persistence_samples=persistence_samples,
        minimum_max_population=0.10,
        minimum_dominance_margin=0.02,
    )
    correlation = PeriodicTemplateCorrelationField(initial.lattice, parameters, static_center_site)
    self_maps = correlation.maps(initial.lattice)
    self_amplitude = correlation.amplitude_at_center(self_maps, static_center_site)[0]

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
    cumulative_x = 0.0
    cumulative_y = 0.0

    times: list[float] = []
    total_maps: list[np.ndarray] = []
    u_maps: list[np.ndarray] = []
    x_maps: list[np.ndarray] = []
    y_maps: list[np.ndarray] = []
    cumulative_x_samples: list[float] = []
    cumulative_y_samples: list[float] = []
    metric_sites: list[int] = []
    bond_values: list[list[float]] = []
    after_idc_samples: list[bool] = []
    temperatures: list[float] = []
    accepted_events: list[PersistentHopEvent] = []

    start_clock = perf_counter()
    for step in range(steps):
        time_fs = float(step * dt_fs)
        old_kinematics = transport_kinematics(
            current.lattice,
            parameters,
            current.electronic_state,
            ax_angstrom=lattice_spacing_A,
            ay_angstrom=lattice_spacing_A,
        )
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
        new_kinematics = transport_kinematics(
            current.lattice,
            parameters,
            current.electronic_state,
            ax_angstrom=lattice_spacing_A,
            ay_angstrom=lattice_spacing_A,
        )
        increment_x, increment_y = trapezoidal_displacement_increment(
            old_kinematics, new_kinematics, dt_fs
        )
        cumulative_x += float(increment_x)
        cumulative_y += float(increment_y)
        lattice_heat += float(thermal.bath_heat_eV)
        max_norm_error = max(
            max_norm_error,
            abs(float(np.linalg.norm(current.electronic_state)) - initial_norm),
        )

        end_time_fs = float((step + 1) * dt_fs)
        did_idc = False
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
            did_idc = True
            matter = dynamic_total_energy(
                current.lattice, current.velocity, parameters, current.electronic_state
            ).total
            residual = matter - initial_energy - lattice_heat - electronic_exchange
            max_balance_residual = max(max_balance_residual, abs(float(residual)))

        if (step + 1) % sample_stride != 0 or end_time_fs < burn_in_fs:
            continue

        raw_population = np.abs(np.asarray(current.electronic_state)) ** 2
        population = np.asarray(raw_population / float(np.sum(raw_population)), dtype=np.float64)
        dominant_site = int(np.argmax(population))
        metric_site = dominant_site if tracker.current_site is None else int(tracker.current_site)
        bonds = local_bond_transfer_magnitudes(current.lattice, parameters, metric_site)
        maps = correlation.maps(current.lattice)

        times.append(end_time_fs)
        total_maps.append(np.asarray(maps.total, dtype=np.float32))
        u_maps.append(np.asarray(maps.u, dtype=np.float32))
        x_maps.append(np.asarray(maps.bond_x, dtype=np.float32))
        y_maps.append(np.asarray(maps.bond_y, dtype=np.float32))
        cumulative_x_samples.append(cumulative_x)
        cumulative_y_samples.append(cumulative_y)
        metric_sites.append(metric_site)
        bond_values.append([float(bonds[key]) for key in DIRECTIONS])
        after_idc_samples.append(bool(did_idc))
        temperatures.append(
            kinetic_temperature_K(current.velocity, parameters, degrees_of_freedom=dof)
        )

        event = tracker.update(population, end_time_fs)
        if event is not None and event.is_nearest_neighbor:
            accepted_events.append(event)

        means = zero_mode_means(current)
        max_zero_mode = max(max_zero_mode, max(abs(float(value)) for value in means.values()))

    final_energy = dynamic_total_energy(
        current.lattice, current.velocity, parameters, current.electronic_state
    ).total
    final_residual = final_energy - initial_energy - lattice_heat - electronic_exchange
    max_balance_residual = max(max_balance_residual, abs(float(final_residual)))

    time_array = np.asarray(times, dtype=np.float64)
    total_array = np.stack(total_maps, axis=0)
    u_array = np.stack(u_maps, axis=0)
    x_array = np.stack(x_maps, axis=0)
    y_array = np.stack(y_maps, axis=0)
    cumulative_x_array = np.asarray(cumulative_x_samples, dtype=np.float64)
    cumulative_y_array = np.asarray(cumulative_y_samples, dtype=np.float64)
    metric_array = np.asarray(metric_sites, dtype=np.int64)
    bond_array = np.asarray(bond_values, dtype=np.float64)
    idc_array = np.asarray(after_idc_samples, dtype=bool)

    event_records: list[dict] = []
    for event in accepted_events:
        record = _event_record(
            event,
            field=correlation,
            nx=parameters.nx,
            ny=parameters.ny,
            times=time_array,
            total_maps=total_array,
            u_maps=u_array,
            x_maps=x_array,
            y_maps=y_array,
            cumulative_x_A=cumulative_x_array,
            cumulative_y_A=cumulative_y_array,
            metric_sites=metric_array,
            bond_values=bond_array,
            after_idc=idc_array,
        )
        if record is not None:
            event_records.append(record)

    elapsed = perf_counter() - start_clock
    return {
        "temperature_K": float(temperature_K),
        "lattice_seed": int(lattice_seed),
        "decoherence_seed": int(decoherence_seed),
        "steps": int(steps),
        "elapsed_seconds": float(elapsed),
        "idc_events": int(idc_events),
        "sample_count_post_burn": int(len(times)),
        "temperature_mean_K": float(np.mean(temperatures)),
        "static_template_self_amplitude": float(self_amplitude),
        "electronic_nearest_neighbor_event_count": int(len(accepted_events)),
        "complete_window_event_count": int(len(event_records)),
        "event_records": event_records,
        "accumulated_lattice_bath_heat_eV": float(lattice_heat),
        "accumulated_electronic_environment_exchange_eV": float(electronic_exchange),
        "final_generalized_energy_residual_eV": float(final_residual),
        "maximum_generalized_energy_residual_eV": float(max_balance_residual),
        "maximum_electronic_norm_error": float(max_norm_error),
        "maximum_zero_mode_magnitude": float(max_zero_mode),
    }


def _trajectory_mechanistic_summary(record: dict) -> dict:
    events = record["event_records"]
    early = "-100:-80fs"
    far = "-500:-400fs"
    if not events:
        return {
            "seed": int(record["lattice_seed"]),
            "event_count": 0,
            "q_advantage_mean": None,
            "q_advantage_positive_fraction": None,
            "q_true_maximum_fraction": None,
            "current100_true_maximum_fraction": None,
            "bond_advantage_early_mean_eV": None,
            "bond_true_top1_early_mean": None,
            "bond_advantage_far_mean_eV": None,
        }
    return {
        "seed": int(record["lattice_seed"]),
        "event_count": int(len(events)),
        "q_advantage_mean": float(np.mean([e["q_lattice_matched_advantage"] for e in events])),
        "q_advantage_positive_fraction": float(np.mean([e["q_lattice_matched_advantage"] > 0.0 for e in events])),
        "q_true_maximum_fraction": float(np.mean([e["q_lattice_true_is_maximum"] for e in events])),
        "current100_true_maximum_fraction": float(
            np.mean([e["current_windows"]["100fs"]["true_is_maximum"] for e in events])
        ),
        "bond_advantage_early_mean_eV": _mean_or_none(
            e["prehop_matched_bins"][early]["matched_advantage_mean_eV"]
            for e in events
            if e["prehop_matched_bins"][early]["matched_advantage_mean_eV"] is not None
        ),
        "bond_true_top1_early_mean": _mean_or_none(
            e["prehop_matched_bins"][early]["true_top1_fraction"]
            for e in events
            if e["prehop_matched_bins"][early]["true_top1_fraction"] is not None
        ),
        "bond_advantage_far_mean_eV": _mean_or_none(
            e["prehop_matched_bins"][far]["matched_advantage_mean_eV"]
            for e in events
            if e["prehop_matched_bins"][far]["matched_advantage_mean_eV"] is not None
        ),
    }


def _aggregate_condition(records: list[dict]) -> dict:
    events = [event for record in records for event in record["event_records"]]
    result = {
        "trajectory_count": int(len(records)),
        "temperature_mean_K": float(np.mean([r["temperature_mean_K"] for r in records])),
        "electronic_nearest_neighbor_event_count": int(
            sum(r["electronic_nearest_neighbor_event_count"] for r in records)
        ),
        "complete_window_event_count": int(len(events)),
        "maximum_generalized_energy_residual_eV": float(
            max(r["maximum_generalized_energy_residual_eV"] for r in records)
        ),
        "maximum_electronic_norm_error": float(max(r["maximum_electronic_norm_error"] for r in records)),
        "maximum_zero_mode_magnitude": float(max(r["maximum_zero_mode_magnitude"] for r in records)),
    }
    if not events:
        return result

    result.update(
        {
            "q_true_delta_mean": _mean_or_none(e["q_lattice_true_delta"] for e in events),
            "q_counterfactual_delta_mean": _mean_or_none(
                e["q_lattice_counterfactual_mean_delta"] for e in events
            ),
            "q_matched_advantage_mean": _mean_or_none(e["q_lattice_matched_advantage"] for e in events),
            "q_matched_advantage_median": _median_or_none(e["q_lattice_matched_advantage"] for e in events),
            "q_matched_advantage_positive_fraction": float(
                np.mean([e["q_lattice_matched_advantage"] > 0.0 for e in events])
            ),
            "q_true_maximum_fraction": float(np.mean([e["q_lattice_true_is_maximum"] for e in events])),
            "q_true_near_event_sign_reversal_fraction": float(
                np.mean([e["q_lattice_true_near_event_sign_reversal"] for e in events])
            ),
            "q_true_nearest_crossing_lag_median_fs": _median_or_none(
                e["q_lattice_true_nearest_crossing_lag_fs"]
                for e in events
                if e["q_lattice_true_nearest_crossing_lag_fs"] is not None
            ),
            "template_signal_at_event_mean": _mean_or_none(
                e["q_lattice_true_template_signal_at_event"] for e in events
            ),
            "idc_associated_fraction": float(np.mean([e["idc_associated"] for e in events])),
            "u_target_minus_source_change_mean": _mean_or_none(
                e["u_target_minus_source_change"] for e in events
            ),
            "longitudinal_bond_change_mean": _mean_or_none(
                e["longitudinal_bond_change"] for e in events
            ),
            "transverse_bond_change_mean": _mean_or_none(
                e["transverse_bond_change"] for e in events
            ),
        }
    )

    current: dict[str, dict] = {}
    for half_window in CURRENT_WINDOWS_FS:
        key = f"{int(half_window)}fs"
        current[key] = {
            "true_parallel_mean_A": _mean_or_none(
                e["current_windows"][key]["true_parallel_A"] for e in events
            ),
            "true_parallel_median_A": _median_or_none(
                e["current_windows"][key]["true_parallel_A"] for e in events
            ),
            "true_parallel_positive_fraction": float(
                np.mean([e["current_windows"][key]["true_parallel_A"] > 0.0 for e in events])
            ),
            "matched_advantage_mean_A": _mean_or_none(
                e["current_windows"][key]["matched_advantage_A"] for e in events
            ),
            "true_maximum_fraction": float(
                np.mean([e["current_windows"][key]["true_is_maximum"] for e in events])
            ),
        }
    result["current_windows"] = current

    bins: dict[str, dict] = {}
    for label in events[0]["prehop_matched_bins"]:
        available = [e["prehop_matched_bins"][label] for e in events if e["prehop_matched_bins"][label]["sample_count"] > 0]
        sample_count = int(sum(item["sample_count"] for item in available))
        if sample_count == 0:
            bins[label] = {"sample_count": 0}
            continue
        def weighted(key: str) -> float | None:
            values = [(item[key], item["sample_count"]) for item in available if item[key] is not None]
            if not values:
                return None
            return float(sum(float(value) * count for value, count in values) / sum(count for _, count in values))
        bins[label] = {
            "sample_count": sample_count,
            "true_bond_mean_eV": weighted("true_bond_mean_eV"),
            "counterfactual_bond_mean_eV": weighted("counterfactual_bond_mean_eV"),
            "matched_advantage_mean_eV": weighted("matched_advantage_mean_eV"),
            "true_rank_mean": weighted("true_rank_mean"),
            "true_top1_fraction": weighted("true_top1_fraction"),
            "counterfactual_top1_fraction_mean": weighted("counterfactual_top1_fraction_mean"),
        }
    result["prehop_matched_bins"] = bins

    seed_summaries = [_trajectory_mechanistic_summary(record) for record in records]
    result["seed_summaries"] = seed_summaries
    result["seed_level_screening"] = {
        "q_advantage_mean": _student_t_summary(
            item["q_advantage_mean"] for item in seed_summaries if item["q_advantage_mean"] is not None
        ),
        "q_advantage_positive_fraction": _student_t_summary(
            item["q_advantage_positive_fraction"]
            for item in seed_summaries
            if item["q_advantage_positive_fraction"] is not None
        ),
        "q_true_maximum_fraction": _student_t_summary(
            item["q_true_maximum_fraction"]
            for item in seed_summaries
            if item["q_true_maximum_fraction"] is not None
        ),
        "current100_true_maximum_fraction": _student_t_summary(
            item["current100_true_maximum_fraction"]
            for item in seed_summaries
            if item["current100_true_maximum_fraction"] is not None
        ),
        "bond_advantage_early_mean_eV": _student_t_summary(
            item["bond_advantage_early_mean_eV"]
            for item in seed_summaries
            if item["bond_advantage_early_mean_eV"] is not None
        ),
        "bond_true_top1_early_mean": _student_t_summary(
            item["bond_true_top1_early_mean"]
            for item in seed_summaries
            if item["bond_true_top1_early_mean"] is not None
        ),
        "bond_advantage_far_mean_eV": _student_t_summary(
            item["bond_advantage_far_mean_eV"]
            for item in seed_summaries
            if item["bond_advantage_far_mean_eV"] is not None
        ),
    }
    return result


def _evaluate_numerical_checks(payload: dict) -> dict[str, bool]:
    records = [r for condition in payload["conditions"] for r in condition["trajectories"]]
    events = [e for record in records for e in record["event_records"]]
    expected_idc = int(np.floor(payload["final_time_fs"] / payload["decoherence_interval_fs"] + 1.0e-12))
    requested = len(payload["conditions"]) * len(payload["lattice_seeds"])
    temperature_ok = all(
        abs(float(r["temperature_mean_K"]) - float(r["temperature_K"]))
        <= max(15.0, 0.10 * float(r["temperature_K"]))
        for r in records
    )
    finite = True
    candidate_ok = True
    for event in events:
        candidate_ok = candidate_ok and int(event["candidate_direction_count"]) == 4
        candidate_ok = candidate_ok and bool(event["true_target_matches_event"])
        finite = finite and np.isfinite(float(event["q_lattice_matched_advantage"]))
        finite = finite and np.isfinite(float(event["q_lattice_true_template_signal_at_event"]))
        for window in event["current_windows"].values():
            finite = finite and all(
                np.isfinite(float(window[key]))
                for key in (
                    "vector_x_A",
                    "vector_y_A",
                    "true_parallel_A",
                    "counterfactual_parallel_mean_A",
                    "matched_advantage_A",
                )
            )
    return {
        "all_static_relaxations_converged": bool(all(v["converged"] for v in payload["static"].values())),
        "all_requested_trajectories_completed": bool(len(records) == requested),
        "exact_idc_event_count": bool(all(int(r["idc_events"]) == expected_idc for r in records)),
        "lattice_temperature": bool(temperature_ok),
        "zero_field_generalized_energy_balance": bool(
            max(float(r["maximum_generalized_energy_residual_eV"]) for r in records) < 5.0e-5
        ),
        "electronic_norm": bool(max(float(r["maximum_electronic_norm_error"]) for r in records) < 1.0e-10),
        "projected_zero_modes": bool(max(float(r["maximum_zero_mode_magnitude"]) for r in records) < 1.0e-12),
        "static_template_self_correlation": bool(
            all(abs(float(r["static_template_self_amplitude"]) - 1.0) < 1.0e-10 for r in records)
        ),
        "finite_matched_diagnostics": bool(finite),
        "exact_four_candidate_neighbours": bool(candidate_ok),
    }


def _markdown(payload: dict) -> str:
    lines = [
        "# IP1d matched counterfactual precursor validation",
        "",
        f"20x20 zero-field control; dt={payload['dt_fs']} fs; final={payload['final_time_fs']/1000:.1f} ps; burn={payload['burn_in_fs']/1000:.1f} ps; electronic persistence={PRIMARY_PERSISTENCE_FS:.0f} fs; IDC-BM td={payload['decoherence_interval_fs']} fs.",
        "",
        "| J0y/J0x | T [K] | complete events | <Delta q true-cf> | q advantage + | q true max | current true max +/-100 fs | bond advantage -100:-80 [meV] | bond true top1 -100:-80 | bond advantage -500:-400 [meV] |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in payload["conditions"]:
        a = condition["aggregate"]
        early = a.get("prehop_matched_bins", {}).get("-100:-80fs", {})
        far = a.get("prehop_matched_bins", {}).get("-500:-400fs", {})
        lines.append(
            "| "
            f"{condition['anisotropy_ratio']:.2f} | {condition['temperature_K']:.0f} | "
            f"{a['complete_window_event_count']} | {a.get('q_matched_advantage_mean', float('nan')):.4f} | "
            f"{a.get('q_matched_advantage_positive_fraction', float('nan')):.3f} | "
            f"{a.get('q_true_maximum_fraction', float('nan')):.3f} | "
            f"{a.get('current_windows', {}).get('100fs', {}).get('true_maximum_fraction', float('nan')):.3f} | "
            f"{1000.0 * early.get('matched_advantage_mean_eV', float('nan')):.3f} | "
            f"{early.get('true_top1_fraction', float('nan')):.3f} | "
            f"{1000.0 * far.get('matched_advantage_mean_eV', float('nan')):.3f} |"
        )
    lines.extend(["", "## Numerical gates", ""])
    for name, passed in payload["numerical_checks"].items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    lines.extend(
        [
            "",
            f"Numerical status: {'PASS' if payload['numerical_pass'] else 'FAIL'}",
            "",
            "Matched advantages, crossing lags and seed-level intervals are physical diagnostics and are not numerical PASS gates.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--temperatures-K", type=float, nargs="+", default=[100.0, 300.0, 500.0])
    parser.add_argument("--isotropic-ratio", type=float, default=1.0)
    parser.add_argument("--reference-ratio", type=float, default=0.15)
    parser.add_argument("--reference-temperature-K", type=float, default=300.0)
    parser.add_argument("--gamma-u-per-fs", type=float, default=0.01)
    parser.add_argument("--gamma-v-per-fs", type=float, default=0.01)
    parser.add_argument("--dt-fs", type=float, default=0.2)
    parser.add_argument("--final-time-fs", type=float, default=20000.0)
    parser.add_argument("--burn-in-fs", type=float, default=2000.0)
    parser.add_argument("--sample-interval-fs", type=float, default=2.0)
    parser.add_argument("--decoherence-interval-fs", type=float, default=180.0)
    parser.add_argument("--lattice-spacing-A", type=float, default=3.0)
    parser.add_argument(
        "--lattice-seeds",
        type=int,
        nargs="+",
        default=[20260905, 20260906, 20260907, 20260908],
    )
    parser.add_argument("--krylov-dimension", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    conditions = [
        (float(args.isotropic_ratio), float(temperature))
        for temperature in args.temperatures_K
    ]
    reference = (float(args.reference_ratio), float(args.reference_temperature_K))
    if reference not in conditions:
        conditions.append(reference)

    prepared: dict[float, tuple] = {}
    static: dict[str, dict] = {}
    for ratio, _ in conditions:
        if ratio in prepared:
            continue
        initial, parameters, metadata = _prepare_initial_state(args.size, ratio)
        prepared[ratio] = (initial, parameters)
        static[str(ratio)] = metadata

    output_conditions: list[dict] = []
    for ratio, temperature in conditions:
        initial, parameters = prepared[ratio]
        trajectories: list[dict] = []
        for seed in args.lattice_seeds:
            trajectories.append(
                _run_trajectory(
                    initial,
                    parameters,
                    static_center_site=int(static[str(ratio)]["maximum_population_site"]),
                    temperature_K=temperature,
                    gamma_u_per_fs=args.gamma_u_per_fs,
                    gamma_v_per_fs=args.gamma_v_per_fs,
                    dt_fs=args.dt_fs,
                    final_time_fs=args.final_time_fs,
                    burn_in_fs=args.burn_in_fs,
                    sample_interval_fs=args.sample_interval_fs,
                    decoherence_interval_fs=args.decoherence_interval_fs,
                    lattice_seed=int(seed),
                    decoherence_seed=int(seed) + 100000,
                    krylov_dimension=args.krylov_dimension,
                    lattice_spacing_A=args.lattice_spacing_A,
                )
            )
        output_conditions.append(
            {
                "anisotropy_ratio": ratio,
                "temperature_K": temperature,
                "trajectories": trajectories,
                "aggregate": _aggregate_condition(trajectories),
            }
        )

    payload = {
        "scope": "IP1d zero-field matched counterfactual precursor validation; no activation/mobility/diffusion claim",
        "size": int(args.size),
        "dt_fs": float(args.dt_fs),
        "final_time_fs": float(args.final_time_fs),
        "burn_in_fs": float(args.burn_in_fs),
        "sample_interval_fs": float(args.sample_interval_fs),
        "primary_persistence_fs": float(PRIMARY_PERSISTENCE_FS),
        "event_window_fs": float(EVENT_WINDOW_FS),
        "quiet_edge_fs": float(QUIET_EDGE_FS),
        "current_windows_fs": list(CURRENT_WINDOWS_FS),
        "prehop_bins_fs": [list(item) for item in PREHOP_BINS_FS],
        "lattice_spacing_A": float(args.lattice_spacing_A),
        "gamma_u_per_fs": float(args.gamma_u_per_fs),
        "gamma_v_per_fs": float(args.gamma_v_per_fs),
        "scheme": "bm",
        "decoherence_interval_fs": float(args.decoherence_interval_fs),
        "decoherence_interval_is_material_calibrated": False,
        "lattice_seeds": [int(seed) for seed in args.lattice_seeds],
        "static": static,
        "conditions": output_conditions,
        "physical_interpretation_is_manual": True,
    }
    payload["numerical_checks"] = _evaluate_numerical_checks(payload)
    payload["numerical_pass"] = bool(all(payload["numerical_checks"].values()))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.markdown.write_text(_markdown(payload), encoding="utf-8")
    print(_markdown(payload), end="")
    if not payload["numerical_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
