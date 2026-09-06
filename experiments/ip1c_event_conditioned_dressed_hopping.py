#!/usr/bin/env python3
"""IP1c event-conditioned dressed-polaron translation at zero field.

Each persistent 50 fs electronic nearest-neighbour residence change is used as
an anchor.  Instead of requiring an independently discretized lattice hop, the
full lattice-distortion template correlation is evaluated continuously at the
electronic source and target sites.  TP1 probability current is integrated as a
third, independent PBC-safe translation diagnostic.

This experiment does not infer an activation energy, diffusion coefficient, or
mobility.  Event-count based kinetics remain deferred until the dressed-event
definition is physically validated.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from holstein_peierls.dynamics.coupled import CoupledEhrenfestState
from holstein_peierls.dynamics.dressed_hopping import prospective_bond_rank
from holstein_peierls.dynamics.ehrenfest import LatticeVelocity, dynamic_total_energy
from holstein_peierls.dynamics.electronic_decoherence import apply_instantaneous_decoherence
from holstein_peierls.dynamics.event_conditioned_hopping import (
    PeriodicTemplateCorrelationField,
    first_negative_to_nonnegative_crossing,
    project_displacement_on_hop,
)
from holstein_peierls.dynamics.hopping_observables import (
    PersistentHopEvent,
    PersistentSiteTracker,
    local_bond_transfer_magnitudes,
)
from holstein_peierls.dynamics.langevin import LangevinBath, kinetic_temperature_K
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
from holstein_peierls.electronic import solve_ground_state
from holstein_peierls.parameters import StaticPolaronParameters
from holstein_peierls.polaron import solve_static_polaron


PRIMARY_PERSISTENCE_FS = 50.0
EVENT_WINDOW_FS = 500.0
QUIET_EDGE_FS = 100.0
CURRENT_WINDOWS_FS = (100.0, 250.0, 500.0)
PREHOP_BIN_EDGES_FS = (-100.0, -80.0, -60.0, -40.0, -20.0, 0.000001)
DIRECTIONS = ("+x", "-x", "+y", "-y")


def _integer_stride(interval_fs: float, base_fs: float, name: str) -> int:
    ratio = float(interval_fs) / float(base_fs)
    value = int(round(ratio))
    if value <= 0 or not np.isclose(value, ratio, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"{name} must be a positive integer multiple of the base interval")
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
    state = CoupledEhrenfestState(
        static.state.copy(),
        LatticeVelocity.zeros(size, size),
        np.asarray(ground.wavefunction, dtype=np.complex128),
    )
    state = project_inter_molecular_zero_modes(state)
    density = np.asarray(static.charge_density, dtype=np.float64).reshape(-1, order="C")
    metadata = {
        "converged": bool(static.diagnostics.converged),
        "iterations": int(static.diagnostics.iterations),
        "total_energy_eV": float(static.total_energy),
        "formation_energy_eV": float(static.formation_energy),
        "ipr": float(static.ipr),
        "participation_number": float(1.0 / static.ipr),
        "maximum_population": float(np.max(density)),
        "maximum_population_site": int(np.argmax(density)),
    }
    return state, parameters, metadata


def _direction_index(direction: str) -> int:
    if direction not in DIRECTIONS:
        raise ValueError("nearest-neighbour direction required")
    return DIRECTIONS.index(direction)


def _amplitude_series(
    maps: np.ndarray,
    field: PeriodicTemplateCorrelationField,
    site: int,
) -> np.ndarray:
    iy, ix = field.shift_index_for_center(site)
    return np.asarray(maps[:, iy, ix], dtype=np.float64)


def _nearest_frame_index(times: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(times - float(target))))


def _mean_or_none(values) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else None


def _median_or_none(values) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else None


def _event_record(
    event: PersistentHopEvent,
    *,
    field: PeriodicTemplateCorrelationField,
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
    target_total = _amplitude_series(total_maps[mask], field, event.target_site)
    denominator = np.abs(source_total) + np.abs(target_total)
    q = np.divide(
        target_total - source_total,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 1.0e-15,
    )

    def component_difference(component_maps: np.ndarray) -> np.ndarray:
        source = _amplitude_series(component_maps[mask], field, event.source_site)
        target = _amplitude_series(component_maps[mask], field, event.target_site)
        return np.asarray(target - source, dtype=np.float64)

    du = component_difference(u_maps)
    dx = component_difference(x_maps)
    dy = component_difference(y_maps)
    pre = (rel >= -EVENT_WINDOW_FS) & (rel <= -QUIET_EDGE_FS)
    post = (rel >= QUIET_EDGE_FS) & (rel <= EVENT_WINDOW_FS)
    if not np.any(pre) or not np.any(post):
        return None

    q_pre = float(np.mean(q[pre]))
    q_post = float(np.mean(q[post]))
    du_delta = float(np.mean(du[post]) - np.mean(du[pre]))
    dx_delta = float(np.mean(dx[post]) - np.mean(dx[pre]))
    dy_delta = float(np.mean(dy[post]) - np.mean(dy[pre]))
    crossing_abs = first_negative_to_nonnegative_crossing(times[mask], q)
    crossing_lag = None if crossing_abs is None else float(crossing_abs - start)

    current_windows: dict[str, dict] = {}
    for half_window in CURRENT_WINDOWS_FS:
        left = _nearest_frame_index(times, start - half_window)
        right = _nearest_frame_index(times, start + half_window)
        displacement_x = float(cumulative_x_A[right] - cumulative_x_A[left])
        displacement_y = float(cumulative_y_A[right] - cumulative_y_A[left])
        parallel, perpendicular = project_displacement_on_hop(
            displacement_x,
            displacement_y,
            event.dx_sites,
            event.dy_sites,
        )
        current_windows[f"{int(half_window)}fs"] = {
            "parallel_A": parallel,
            "perpendicular_A": perpendicular,
            "vector_x_A": displacement_x,
            "vector_y_A": displacement_y,
        }

    prehop_bins: dict[str, dict] = {}
    direction_index = _direction_index(event.direction)
    for left, right in zip(PREHOP_BIN_EDGES_FS[:-1], PREHOP_BIN_EDGES_FS[1:]):
        # Last bin includes t=0; all others are half-open.
        selected = (times >= start + left - 1.0e-9) & (times < start + right - 1.0e-9)
        selected &= metric_sites == int(event.source_site)
        indices = np.flatnonzero(selected)
        ranks: list[int] = []
        for index in indices:
            values = {key: float(bond_values[index, j]) for j, key in enumerate(DIRECTIONS)}
            ranks.append(prospective_bond_rank(values, event.direction))
        label = f"{int(round(left))}:{int(round(min(right, 0.0)))}fs"
        prehop_bins[label] = {
            "sample_count": int(len(ranks)),
            "strongest_count": int(sum(rank == 1 for rank in ranks)),
            "rank_sum": int(sum(ranks)),
            "strongest_fraction": float(np.mean(np.asarray(ranks) == 1)) if ranks else None,
            "mean_rank": float(np.mean(ranks)) if ranks else None,
        }

    start_index = _nearest_frame_index(times, start)
    output = asdict(event)
    output.update(
        {
            "idc_associated": bool(abs(float(times[start_index]) - start) <= 1.0e-9 and after_idc[start_index]),
            "q_lattice_pre_mean": q_pre,
            "q_lattice_post_mean": q_post,
            "q_lattice_delta": float(q_post - q_pre),
            "lattice_source_to_target_sign_reversal": bool(q_pre < 0.0 and q_post > 0.0),
            "lattice_crossing_lag_fs": crossing_lag,
            "u_target_minus_source_change": du_delta,
            "bond_x_target_minus_source_change": dx_delta,
            "bond_y_target_minus_source_change": dy_delta,
            "longitudinal_bond_change": dx_delta if event.dx_sites != 0 else dy_delta,
            "transverse_bond_change": dy_delta if event.dx_sites != 0 else dx_delta,
            "current_windows": current_windows,
            "prehop_future_bond_bins": prehop_bins,
            "future_direction_index": int(direction_index),
        }
    )
    return output


def _run_trajectory(
    initial: CoupledEhrenfestState,
    parameters: StaticPolaronParameters,
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
    correlation = PeriodicTemplateCorrelationField(
        initial.lattice,
        parameters,
        static_center_site,
    )
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
            old_kinematics,
            new_kinematics,
            dt_fs,
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
    sign_reversal = [bool(r["lattice_source_to_target_sign_reversal"]) for r in event_records]
    crossing_lags = [
        float(r["lattice_crossing_lag_fs"])
        for r in event_records
        if r["lattice_crossing_lag_fs"] is not None
    ]
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
        "lattice_sign_reversal_count": int(sum(sign_reversal)),
        "lattice_sign_reversal_fraction": float(np.mean(sign_reversal)) if sign_reversal else 0.0,
        "lattice_crossing_lag_median_fs": _median_or_none(crossing_lags),
        "event_records": event_records,
        "accumulated_lattice_bath_heat_eV": float(lattice_heat),
        "accumulated_electronic_environment_exchange_eV": float(electronic_exchange),
        "final_generalized_energy_residual_eV": float(final_residual),
        "maximum_generalized_energy_residual_eV": float(max_balance_residual),
        "maximum_electronic_norm_error": float(max_norm_error),
        "maximum_zero_mode_magnitude": float(max_zero_mode),
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
        result.update(
            {
                "lattice_sign_reversal_fraction": 0.0,
                "q_lattice_delta_mean": None,
                "lattice_crossing_lag_median_fs": None,
                "idc_associated_fraction": 0.0,
            }
        )
        return result

    sign = np.asarray([bool(e["lattice_source_to_target_sign_reversal"]) for e in events])
    result["lattice_sign_reversal_fraction"] = float(np.mean(sign))
    result["q_lattice_delta_mean"] = _mean_or_none(e["q_lattice_delta"] for e in events)
    result["q_lattice_delta_median"] = _median_or_none(e["q_lattice_delta"] for e in events)
    result["lattice_crossing_lag_median_fs"] = _median_or_none(
        e["lattice_crossing_lag_fs"] for e in events if e["lattice_crossing_lag_fs"] is not None
    )
    result["idc_associated_fraction"] = float(np.mean([bool(e["idc_associated"]) for e in events]))
    result["u_target_minus_source_change_mean"] = _mean_or_none(
        e["u_target_minus_source_change"] for e in events
    )
    result["longitudinal_bond_change_mean"] = _mean_or_none(
        e["longitudinal_bond_change"] for e in events
    )
    result["transverse_bond_change_mean"] = _mean_or_none(
        e["transverse_bond_change"] for e in events
    )

    current: dict[str, dict] = {}
    for half_window in CURRENT_WINDOWS_FS:
        key = f"{int(half_window)}fs"
        parallel = np.asarray([float(e["current_windows"][key]["parallel_A"]) for e in events])
        perpendicular = np.asarray([float(e["current_windows"][key]["perpendicular_A"]) for e in events])
        current[key] = {
            "parallel_mean_A": float(np.mean(parallel)),
            "parallel_median_A": float(np.median(parallel)),
            "parallel_positive_fraction": float(np.mean(parallel > 0.0)),
            "perpendicular_abs_median_A": float(np.median(np.abs(perpendicular))),
            "sign_reversal_and_positive_parallel_fraction": float(np.mean(sign & (parallel > 0.0))),
        }
    result["current_windows"] = current

    bins: dict[str, dict] = {}
    labels = list(events[0]["prehop_future_bond_bins"].keys())
    for label in labels:
        sample_count = int(sum(e["prehop_future_bond_bins"][label]["sample_count"] for e in events))
        strongest_count = int(sum(e["prehop_future_bond_bins"][label]["strongest_count"] for e in events))
        rank_sum = int(sum(e["prehop_future_bond_bins"][label]["rank_sum"] for e in events))
        bins[label] = {
            "sample_count": sample_count,
            "future_bond_strongest_fraction": float(strongest_count / sample_count) if sample_count else None,
            "future_bond_mean_rank": float(rank_sum / sample_count) if sample_count else None,
        }
    result["prehop_future_bond_bins"] = bins
    return result


def _evaluate_numerical_checks(payload: dict) -> dict[str, bool]:
    records = [r for c in payload["conditions"] for r in c["trajectories"]]
    expected_events = int(np.floor(payload["final_time_fs"] / payload["decoherence_interval_fs"] + 1.0e-12))
    requested = len(payload["conditions"]) * len(payload["lattice_seeds"])
    temperature_ok = all(
        abs(float(r["temperature_mean_K"]) - float(r["temperature_K"]))
        <= max(15.0, 0.10 * float(r["temperature_K"]))
        for r in records
    )
    finite_events = True
    for record in records:
        for event in record["event_records"]:
            finite_events = finite_events and np.isfinite(float(event["q_lattice_delta"]))
            for item in event["current_windows"].values():
                finite_events = finite_events and all(np.isfinite(float(v)) for v in item.values())
    return {
        "all_static_relaxations_converged": bool(all(v["converged"] for v in payload["static"].values())),
        "all_requested_trajectories_completed": bool(len(records) == requested),
        "exact_idc_event_count": bool(all(int(r["idc_events"]) == expected_events for r in records)),
        "lattice_temperature": bool(temperature_ok),
        "zero_field_generalized_energy_balance": bool(
            max(float(r["maximum_generalized_energy_residual_eV"]) for r in records) < 5.0e-5
        ),
        "electronic_norm": bool(max(float(r["maximum_electronic_norm_error"]) for r in records) < 1.0e-10),
        "projected_zero_modes": bool(max(float(r["maximum_zero_mode_magnitude"]) for r in records) < 1.0e-12),
        "static_template_self_correlation": bool(
            all(abs(float(r["static_template_self_amplitude"]) - 1.0) < 1.0e-10 for r in records)
        ),
        "finite_event_conditioned_diagnostics": bool(finite_events),
    }


def _markdown(payload: dict) -> str:
    lines = [
        "# IP1c event-conditioned dressed-polaron translation",
        "",
        f"20x20 zero-field control; dt={payload['dt_fs']} fs; final={payload['final_time_fs']/1000:.1f} ps; burn={payload['burn_in_fs']/1000:.1f} ps; electronic persistence={PRIMARY_PERSISTENCE_FS:.0f} fs; IDC-BM td={payload['decoherence_interval_fs']} fs.",
        "",
        "| J0y/J0x | T [K] | e-NN | complete windows | lattice source->target reversal | <Delta q_L> | current + fraction (250 fs) | median current parallel (250 fs) [A] | future bond strongest -100:-80 fs | future bond strongest -20:0 fs |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in payload["conditions"]:
        a = condition["aggregate"]
        current = a.get("current_windows", {}).get("250fs", {})
        bins = a.get("prehop_future_bond_bins", {})
        early = bins.get("-100:-80fs", {}).get("future_bond_strongest_fraction")
        late = bins.get("-20:0fs", {}).get("future_bond_strongest_fraction")
        lines.append(
            f"| {condition['anisotropy_ratio']:.2f} | {condition['temperature_K']:.0f} | {a['electronic_nearest_neighbor_event_count']} | {a['complete_window_event_count']} | {a.get('lattice_sign_reversal_fraction', 0.0):.3f} | {a.get('q_lattice_delta_mean') if a.get('q_lattice_delta_mean') is not None else float('nan'):.4f} | {current.get('parallel_positive_fraction', float('nan')):.3f} | {current.get('parallel_median_A', float('nan')):.4f} | {early if early is not None else float('nan'):.3f} | {late if late is not None else float('nan'):.3f} |"
        )
    lines.extend(["", "## Numerical gates", ""])
    for name, passed in payload["numerical_checks"].items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    lines.extend(
        [
            "",
            f"Numerical status: {'PASS' if payload['numerical_pass'] else 'FAIL'}",
            "",
            "Lattice sign reversal, current-displacement agreement, component response, and lag-resolved future-bond predictability are physical diagnostics and are not numerical PASS gates.",
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
        "--lattice-seeds", type=int, nargs="+", default=[20260905, 20260906, 20260907, 20260908]
    )
    parser.add_argument("--decoherence-seed-offset", type=int, default=3000000)
    parser.add_argument("--krylov-dimension", type=int, default=6)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    conditions = [(float(args.isotropic_ratio), float(t)) for t in args.temperatures_K]
    conditions.append((float(args.reference_ratio), float(args.reference_temperature_K)))
    unique_ratios = sorted({ratio for ratio, _ in conditions})
    prepared = {}
    static = {}
    for ratio in unique_ratios:
        initial, parameters, metadata = _prepare_initial_state(args.size, ratio)
        prepared[ratio] = (initial, parameters)
        static[str(ratio)] = metadata

    condition_payloads = []
    for ratio, temperature in conditions:
        initial, parameters = prepared[ratio]
        trajectories = []
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
                    lattice_seed=seed,
                    decoherence_seed=args.decoherence_seed_offset + seed,
                    krylov_dimension=args.krylov_dimension,
                    lattice_spacing_A=args.lattice_spacing_A,
                )
            )
        condition_payloads.append(
            {
                "anisotropy_ratio": ratio,
                "temperature_K": temperature,
                "trajectories": trajectories,
                "aggregate": _aggregate_condition(trajectories),
            }
        )

    payload = {
        "scope": "IP1c zero-field event-conditioned dressed-polaron translation; continuous source-target lattice correlation, component response, lag-resolved future-bond rank, and TP1 current displacement; no activation/mobility/diffusion claim",
        "size": args.size,
        "dt_fs": args.dt_fs,
        "final_time_fs": args.final_time_fs,
        "burn_in_fs": args.burn_in_fs,
        "sample_interval_fs": args.sample_interval_fs,
        "primary_persistence_fs": PRIMARY_PERSISTENCE_FS,
        "event_window_fs": EVENT_WINDOW_FS,
        "quiet_edge_fs": QUIET_EDGE_FS,
        "current_windows_fs": list(CURRENT_WINDOWS_FS),
        "prehop_bin_edges_fs": list(PREHOP_BIN_EDGES_FS),
        "lattice_spacing_A": args.lattice_spacing_A,
        "gamma_u_per_fs": args.gamma_u_per_fs,
        "gamma_v_per_fs": args.gamma_v_per_fs,
        "scheme": "bm",
        "decoherence_interval_fs": args.decoherence_interval_fs,
        "decoherence_interval_is_material_calibrated": False,
        "lattice_seeds": args.lattice_seeds,
        "static": static,
        "conditions": condition_payloads,
        "physical_interpretation_is_manual": True,
    }
    payload["numerical_checks"] = _evaluate_numerical_checks(payload)
    payload["numerical_pass"] = bool(all(payload["numerical_checks"].values()))
    text = _markdown(payload)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(text, encoding="utf-8")
    if not payload["numerical_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
