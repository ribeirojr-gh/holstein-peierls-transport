#!/usr/bin/env python3
"""IP1a zero-field thermal hopping and transient-anisotropy screening.

Persistent residence-site changes are detected from the propagated electronic
population while the validated BAOAB + CF4-Lanczos + IDC-BM dynamics evolves at
zero field.  Short dominant-site flicker is rejected by a persistence filter.

This experiment is a mechanistic screen.  It does not infer mobility, diffusion
or an activation energy, and zero observed hops is a valid physical outcome.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from holstein_peierls.dynamics.coupled import CoupledEhrenfestState
from holstein_peierls.dynamics.ehrenfest import LatticeVelocity, dynamic_total_energy
from holstein_peierls.dynamics.electronic_decoherence import apply_instantaneous_decoherence
from holstein_peierls.dynamics.hopping_observables import (
    PersistentSiteTracker,
    directional_transfer_bias,
    local_bond_transfer_magnitudes,
    local_transfer_anisotropy,
)
from holstein_peierls.dynamics.langevin import LangevinBath, kinetic_temperature_K
from holstein_peierls.dynamics.thermal import (
    coupled_baoab_step,
    project_inter_molecular_zero_modes,
    thermostatted_kinetic_degrees_of_freedom,
    zero_mode_means,
)
from holstein_peierls.dynamics.thermal_decoherence import decoherence_interval_steps
from holstein_peierls.electronic import solve_ground_state
from holstein_peierls.parameters import StaticPolaronParameters
from holstein_peierls.polaron import solve_static_polaron


def _integer_stride(interval_fs: float, dt_fs: float, name: str) -> int:
    ratio = float(interval_fs) / float(dt_fs)
    value = int(round(ratio))
    if value <= 0 or not np.isclose(value, ratio, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"{name} must be a positive integer multiple of dt_fs")
    return value


def _prepare_initial_state(
    size: int,
    anisotropy_ratio: float,
) -> tuple[CoupledEhrenfestState, StaticPolaronParameters, dict]:
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
    return state, parameters, {
        "converged": bool(static.diagnostics.converged),
        "iterations": int(static.diagnostics.iterations),
        "total_energy_eV": float(static.total_energy),
        "formation_energy_eV": float(static.formation_energy),
        "ipr": float(static.ipr),
        "participation_number": float(1.0 / static.ipr),
        "maximum_population": float(np.max(density)),
        "maximum_population_site": int(np.argmax(density)),
    }


def _finite_or_none(values: list[float], function=np.mean) -> float | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return None if array.size == 0 else float(function(array))


def _event_from_history(event, history, prehop_window_fs: float) -> dict:
    start = float(event.transition_start_time_fs)
    selected = [
        item
        for item in history
        if start - prehop_window_fs - 1.0e-9 <= item["time_fs"] <= start + 1.0e-9
    ]
    anisotropy = [float(item["local_anisotropy"]) for item in selected]
    directional: list[float] = []
    if event.is_nearest_neighbor:
        directional = [
            directional_transfer_bias(item["bond_magnitudes_eV"], event.direction)
            for item in selected
        ]
    start_matches = [
        item for item in history if abs(float(item["time_fs"]) - start) <= 1.0e-9
    ]
    return {
        "source_site": int(event.source_site),
        "target_site": int(event.target_site),
        "transition_start_time_fs": start,
        "accepted_time_fs": float(event.accepted_time_fs),
        "dx_sites": int(event.dx_sites),
        "dy_sites": int(event.dy_sites),
        "is_nearest_neighbor": bool(event.is_nearest_neighbor),
        "direction": str(event.direction),
        "idc_associated": bool(start_matches and start_matches[-1]["after_idc"]),
        "prehop_sample_count": int(len(selected)),
        "prehop_local_anisotropy_mean": _finite_or_none(anisotropy),
        "prehop_local_anisotropy_max": _finite_or_none(anisotropy, np.max),
        "prehop_directional_bias_mean": _finite_or_none(directional),
        "prehop_directional_bias_max": _finite_or_none(directional, np.max),
    }


def _run_trajectory(
    initial: CoupledEhrenfestState,
    parameters: StaticPolaronParameters,
    *,
    temperature_K: float,
    gamma_u_per_fs: float,
    gamma_v_per_fs: float,
    dt_fs: float,
    final_time_fs: float,
    burn_in_fs: float,
    sample_interval_fs: float,
    persistence_fs: float,
    prehop_window_fs: float,
    decoherence_interval_fs: float,
    lattice_seed: int,
    decoherence_seed: int,
    krylov_dimension: int,
) -> dict:
    steps = _integer_stride(final_time_fs, dt_fs, "final_time_fs")
    burn_steps = int(round(burn_in_fs / dt_fs))
    if burn_steps < 0 or burn_steps >= steps or not np.isclose(burn_steps * dt_fs, burn_in_fs):
        raise ValueError("burn_in_fs must be an integer multiple of dt_fs within the run")
    sample_stride = _integer_stride(sample_interval_fs, dt_fs, "sample_interval_fs")
    persistence_samples = _integer_stride(persistence_fs, sample_interval_fs, "persistence_fs")
    event_stride = decoherence_interval_steps(decoherence_interval_fs, dt_fs)

    current = CoupledEhrenfestState(
        initial.lattice.copy(),
        initial.velocity.copy(),
        np.asarray(initial.electronic_state, dtype=np.complex128).copy(),
    )
    bath = LangevinBath(
        temperature_K=float(temperature_K),
        gamma_u_per_fs=float(gamma_u_per_fs),
        gamma_v_per_fs=float(gamma_v_per_fs),
    )
    lattice_rng = np.random.default_rng(int(lattice_seed))
    decoherence_rng = np.random.default_rng(int(decoherence_seed))
    dof = thermostatted_kinetic_degrees_of_freedom(parameters, "project")
    tracker = PersistentSiteTracker(
        parameters.nx,
        parameters.ny,
        persistence_samples=persistence_samples,
        minimum_max_population=0.10,
        minimum_dominance_margin=0.02,
    )

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
    idc_dominant_changes = 0

    temperatures: list[float] = []
    anisotropies: list[float] = []
    participations: list[float] = []
    maximum_populations: list[float] = []
    events: list[dict] = []
    direction_counts: Counter[str] = Counter()
    net_dx = 0
    net_dy = 0
    last_residence_start_fs = float(burn_in_fs)
    completed_residences_fs: list[float] = []

    history_count = int(np.ceil((prehop_window_fs + persistence_fs) / sample_interval_fs)) + 8
    history = deque(maxlen=max(16, history_count))

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
        did_idc = False
        if (step + 1) % event_stride == 0:
            pre_density = np.abs(np.asarray(current.electronic_state)) ** 2
            pre_site = int(np.argmax(pre_density))
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
                current.lattice.copy(),
                current.velocity.copy(),
                collapse.electronic_state.copy(),
            )
            idc_events += 1
            did_idc = True
            post_site = int(np.argmax(np.abs(np.asarray(current.electronic_state)) ** 2))
            if end_time_fs >= burn_in_fs and post_site != pre_site:
                idc_dominant_changes += 1
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
        anisotropy = local_transfer_anisotropy(bonds)
        ipr = float(np.sum(population * population))
        participation = float(1.0 / ipr)
        temperature = kinetic_temperature_K(
            current.velocity,
            parameters,
            degrees_of_freedom=dof,
        )
        means = zero_mode_means(current)
        max_zero_mode = max(max_zero_mode, max(abs(float(value)) for value in means.values()))

        entry = {
            "time_fs": end_time_fs,
            "after_idc": bool(did_idc),
            "local_anisotropy": float(anisotropy),
            "bond_magnitudes_eV": bonds,
            "participation_number": participation,
            "maximum_population": float(np.max(population)),
        }
        history.append(entry)
        temperatures.append(float(temperature))
        anisotropies.append(float(anisotropy))
        participations.append(participation)
        maximum_populations.append(float(np.max(population)))

        accepted = tracker.update(population, end_time_fs)
        if accepted is not None:
            record = _event_from_history(accepted, list(history), prehop_window_fs)
            record["residence_time_before_transition_fs"] = float(
                accepted.transition_start_time_fs - last_residence_start_fs
            )
            completed_residences_fs.append(record["residence_time_before_transition_fs"])
            last_residence_start_fs = float(accepted.transition_start_time_fs)
            events.append(record)
            net_dx += int(accepted.dx_sites)
            net_dy += int(accepted.dy_sites)
            if accepted.is_nearest_neighbor:
                direction_counts[str(accepted.direction)] += 1

    final_energy = dynamic_total_energy(
        current.lattice, current.velocity, parameters, current.electronic_state
    ).total
    final_residual = final_energy - initial_energy - lattice_heat - electronic_exchange
    max_balance_residual = max(max_balance_residual, abs(float(final_residual)))
    elapsed = perf_counter() - start_clock

    nearest = [event for event in events if event["is_nearest_neighbor"]]
    nonlocal_events = [event for event in events if not event["is_nearest_neighbor"]]
    final_censored_residence_fs = float(final_time_fs - last_residence_start_fs)
    post_burn_duration_ps = float((final_time_fs - burn_in_fs) / 1000.0)
    return {
        "temperature_K": float(temperature_K),
        "lattice_seed": int(lattice_seed),
        "decoherence_seed": int(decoherence_seed),
        "steps": int(steps),
        "elapsed_seconds": float(elapsed),
        "idc_events": int(idc_events),
        "idc_dominant_site_changes_post_burn": int(idc_dominant_changes),
        "sample_count_post_burn": int(len(temperatures)),
        "temperature_mean_K": float(np.mean(temperatures)),
        "temperature_std_K": float(np.std(temperatures, ddof=1)) if len(temperatures) > 1 else 0.0,
        "baseline_local_anisotropy_mean": float(np.mean(anisotropies)),
        "baseline_local_anisotropy_std": float(np.std(anisotropies, ddof=1)) if len(anisotropies) > 1 else 0.0,
        "baseline_local_anisotropy_p95": float(np.quantile(anisotropies, 0.95)),
        "participation_number_mean": float(np.mean(participations)),
        "participation_number_p95": float(np.quantile(participations, 0.95)),
        "maximum_population_mean": float(np.mean(maximum_populations)),
        "persistent_transition_count": int(len(events)),
        "nearest_neighbor_hop_count": int(len(nearest)),
        "nonlocal_transition_count": int(len(nonlocal_events)),
        "nearest_neighbor_hop_rate_per_ps": float(len(nearest) / post_burn_duration_ps),
        "direction_counts": {key: int(direction_counts.get(key, 0)) for key in ("+x", "-x", "+y", "-y")},
        "idc_associated_persistent_transition_count": int(sum(bool(event["idc_associated"]) for event in events)),
        "net_accepted_dx_sites": int(net_dx),
        "net_accepted_dy_sites": int(net_dy),
        "completed_residence_times_fs": completed_residences_fs,
        "final_censored_residence_time_fs": final_censored_residence_fs,
        "events": events,
        "accumulated_lattice_bath_heat_eV": float(lattice_heat),
        "accumulated_electronic_environment_exchange_eV": float(electronic_exchange),
        "final_generalized_energy_residual_eV": float(final_residual),
        "maximum_generalized_energy_residual_eV": float(max_balance_residual),
        "maximum_electronic_norm_error": float(max_norm_error),
        "maximum_zero_mode_magnitude": float(max_zero_mode),
    }


def _aggregate_condition(records: list[dict], final_time_fs: float, burn_in_fs: float) -> dict:
    duration_ps = (float(final_time_fs) - float(burn_in_fs)) / 1000.0
    events = [event for record in records for event in record["events"]]
    nearest = [event for event in events if event["is_nearest_neighbor"]]
    pre_a = [float(event["prehop_local_anisotropy_mean"]) for event in nearest if event["prehop_local_anisotropy_mean"] is not None]
    pre_b = [float(event["prehop_directional_bias_mean"]) for event in nearest if event["prehop_directional_bias_mean"] is not None]
    baseline = np.asarray([float(record["baseline_local_anisotropy_mean"]) for record in records])
    direction = Counter()
    for record in records:
        direction.update(record["direction_counts"])
    total_events = len(events)
    return {
        "trajectory_count": int(len(records)),
        "temperature_mean_K": float(np.mean([record["temperature_mean_K"] for record in records])),
        "temperature_std_across_trajectories_K": float(np.std([record["temperature_mean_K"] for record in records], ddof=1)) if len(records) > 1 else 0.0,
        "persistent_transition_count": int(total_events),
        "nearest_neighbor_hop_count": int(len(nearest)),
        "nonlocal_transition_count": int(total_events - len(nearest)),
        "fraction_trajectories_with_nearest_neighbor_hop": float(np.mean([record["nearest_neighbor_hop_count"] > 0 for record in records])),
        "pooled_nearest_neighbor_hop_rate_per_ps": float(len(nearest) / (len(records) * duration_ps)),
        "trajectory_hop_rate_mean_per_ps": float(np.mean([record["nearest_neighbor_hop_rate_per_ps"] for record in records])),
        "trajectory_hop_rate_std_per_ps": float(np.std([record["nearest_neighbor_hop_rate_per_ps"] for record in records], ddof=1)) if len(records) > 1 else 0.0,
        "direction_counts": {key: int(direction.get(key, 0)) for key in ("+x", "-x", "+y", "-y")},
        "idc_associated_persistent_fraction": None if total_events == 0 else float(sum(bool(event["idc_associated"]) for event in events) / total_events),
        "baseline_local_anisotropy_mean": float(np.mean(baseline)),
        "baseline_local_anisotropy_p95_mean": float(np.mean([record["baseline_local_anisotropy_p95"] for record in records])),
        "prehop_local_anisotropy_mean": _finite_or_none(pre_a),
        "prehop_minus_baseline_anisotropy": None if not pre_a else float(np.mean(pre_a) - np.mean(baseline)),
        "prehop_directional_bias_mean": _finite_or_none(pre_b),
        "participation_number_mean": float(np.mean([record["participation_number_mean"] for record in records])),
        "maximum_generalized_energy_residual_eV": float(max(record["maximum_generalized_energy_residual_eV"] for record in records)),
        "maximum_electronic_norm_error": float(max(record["maximum_electronic_norm_error"] for record in records)),
        "maximum_zero_mode_magnitude": float(max(record["maximum_zero_mode_magnitude"] for record in records)),
    }


def _markdown(payload: dict) -> str:
    lines = [
        "# IP1a zero-field thermal hopping screen",
        "",
        f"Control: {payload['size']}x{payload['size']}, dt={payload['dt_fs']} fs, final={payload['final_time_fs']/1000.0:.1f} ps, burn={payload['burn_in_fs']/1000.0:.1f} ps, IDC-BM td={payload['decoherence_interval_fs']} fs.",
        "",
        "| J0y/J0x | T target [K] | <T> [K] | NN hops | traj with hop | rate [ps^-1] | IDC-associated | baseline A | pre-hop A | pre-hop directional bias |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in payload["conditions"]:
        a = condition["aggregate"]
        def fmt(value, spec=".4f"):
            return "n/a" if value is None else format(float(value), spec)
        lines.append(
            f"| {condition['anisotropy_ratio']:.2f} | {condition['temperature_K']:.0f} | {a['temperature_mean_K']:.2f} | "
            f"{a['nearest_neighbor_hop_count']} | {a['fraction_trajectories_with_nearest_neighbor_hop']:.2f} | "
            f"{a['pooled_nearest_neighbor_hop_rate_per_ps']:.5f} | {fmt(a['idc_associated_persistent_fraction'], '.3f')} | "
            f"{a['baseline_local_anisotropy_mean']:.4f} | {fmt(a['prehop_local_anisotropy_mean'])} | {fmt(a['prehop_directional_bias_mean'])} |"
        )
    lines += ["", "## Numerical gates", ""]
    lines += [f"- {name}: {'PASS' if value else 'FAIL'}" for name, value in payload["numerical_checks"].items()]
    lines += [
        "",
        f"Numerical status: {'PASS' if payload['numerical_pass'] else 'FAIL'}",
        "",
        "Hop-count, temperature trend, anisotropic/isotropic ordering, IDC association and transient-anisotropy enhancement are physical diagnostics, not PASS gates.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--temperatures-K", type=float, nargs="+", default=[100.0, 200.0, 300.0, 400.0, 500.0])
    parser.add_argument("--isotropic-ratio", type=float, default=1.0)
    parser.add_argument("--reference-ratio", type=float, default=0.15)
    parser.add_argument("--reference-temperature-K", type=float, default=300.0)
    parser.add_argument("--gamma-u-per-fs", type=float, default=0.01)
    parser.add_argument("--gamma-v-per-fs", type=float, default=0.01)
    parser.add_argument("--dt-fs", type=float, default=0.2)
    parser.add_argument("--final-time-fs", type=float, default=20000.0)
    parser.add_argument("--burn-in-fs", type=float, default=2000.0)
    parser.add_argument("--sample-interval-fs", type=float, default=2.0)
    parser.add_argument("--persistence-fs", type=float, default=20.0)
    parser.add_argument("--prehop-window-fs", type=float, default=100.0)
    parser.add_argument("--decoherence-interval-fs", type=float, default=180.0)
    parser.add_argument("--lattice-seeds", type=int, nargs="+", default=[20260905, 20260906, 20260907, 20260908])
    parser.add_argument("--decoherence-seed-offset", type=int, default=6000000)
    parser.add_argument("--krylov-dimension", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    if args.size < 4:
        raise ValueError("size must be at least 4")
    temperatures = [float(value) for value in args.temperatures_K]
    if any(not np.isfinite(value) or value <= 0.0 for value in temperatures):
        raise ValueError("screening temperatures must be positive and finite")

    condition_specs = [(float(args.isotropic_ratio), temperature) for temperature in temperatures]
    reference = (float(args.reference_ratio), float(args.reference_temperature_K))
    if reference not in condition_specs:
        condition_specs.append(reference)

    prepared = {}
    static_metadata = {}
    for ratio in sorted(set(ratio for ratio, _ in condition_specs)):
        initial, parameters, metadata = _prepare_initial_state(args.size, ratio)
        prepared[ratio] = (initial, parameters)
        static_metadata[str(ratio)] = metadata

    conditions = []
    for ratio, temperature in condition_specs:
        initial, parameters = prepared[ratio]
        records = []
        for seed in args.lattice_seeds:
            records.append(
                _run_trajectory(
                    initial,
                    parameters,
                    temperature_K=temperature,
                    gamma_u_per_fs=args.gamma_u_per_fs,
                    gamma_v_per_fs=args.gamma_v_per_fs,
                    dt_fs=args.dt_fs,
                    final_time_fs=args.final_time_fs,
                    burn_in_fs=args.burn_in_fs,
                    sample_interval_fs=args.sample_interval_fs,
                    persistence_fs=args.persistence_fs,
                    prehop_window_fs=args.prehop_window_fs,
                    decoherence_interval_fs=args.decoherence_interval_fs,
                    lattice_seed=seed,
                    decoherence_seed=args.decoherence_seed_offset + seed,
                    krylov_dimension=args.krylov_dimension,
                )
            )
        conditions.append(
            {
                "anisotropy_ratio": ratio,
                "temperature_K": temperature,
                "trajectories": records,
                "aggregate": _aggregate_condition(records, args.final_time_fs, args.burn_in_fs),
            }
        )

    all_records = [record for condition in conditions for record in condition["trajectories"]]
    expected_events = int(np.floor((args.final_time_fs / args.dt_fs) / decoherence_interval_steps(args.decoherence_interval_fs, args.dt_fs)))
    static_ok = all(bool(item["converged"]) for item in static_metadata.values())
    trajectories_complete = len(all_records) == len(condition_specs) * len(args.lattice_seeds)
    exact_events = all(int(record["idc_events"]) == expected_events for record in all_records)
    temperature_ok = all(
        0.70 * float(condition["temperature_K"]) <= float(condition["aggregate"]["temperature_mean_K"]) <= 1.30 * float(condition["temperature_K"])
        for condition in conditions
    )
    energy_ok = max(float(record["maximum_generalized_energy_residual_eV"]) for record in all_records) < 1.0e-4
    norm_ok = max(float(record["maximum_electronic_norm_error"]) for record in all_records) < 1.0e-10
    zero_ok = max(float(record["maximum_zero_mode_magnitude"]) for record in all_records) < 1.0e-12
    diagnostics_ok = True
    for record in all_records:
        diagnostics_ok &= int(record["persistent_transition_count"]) == int(record["nearest_neighbor_hop_count"]) + int(record["nonlocal_transition_count"])
        diagnostics_ok &= sum(int(value) for value in record["direction_counts"].values()) == int(record["nearest_neighbor_hop_count"])
        diagnostics_ok &= all(np.isfinite(float(record[key])) for key in (
            "temperature_mean_K",
            "baseline_local_anisotropy_mean",
            "participation_number_mean",
            "maximum_population_mean",
            "nearest_neighbor_hop_rate_per_ps",
        ))

    checks = {
        "all_static_relaxations_converged": bool(static_ok),
        "all_requested_trajectories_completed": bool(trajectories_complete),
        "exact_idc_event_count": bool(exact_events),
        "lattice_temperature": bool(temperature_ok),
        "zero_field_generalized_energy_balance": bool(energy_ok),
        "electronic_norm": bool(norm_ok),
        "projected_zero_modes": bool(zero_ok),
        "finite_consistent_hopping_diagnostics": bool(diagnostics_ok),
    }
    payload = {
        "scope": "IP1a zero-field finite-temperature persistent hopping and transient local transfer-anisotropy screening; no mobility/diffusion/activation claim",
        "size": int(args.size),
        "dt_fs": float(args.dt_fs),
        "final_time_fs": float(args.final_time_fs),
        "burn_in_fs": float(args.burn_in_fs),
        "sample_interval_fs": float(args.sample_interval_fs),
        "persistence_fs": float(args.persistence_fs),
        "prehop_window_fs": float(args.prehop_window_fs),
        "gamma_u_per_fs": float(args.gamma_u_per_fs),
        "gamma_v_per_fs": float(args.gamma_v_per_fs),
        "scheme": "bm",
        "decoherence_interval_fs": float(args.decoherence_interval_fs),
        "decoherence_interval_is_material_calibrated": False,
        "lattice_seeds": [int(value) for value in args.lattice_seeds],
        "static": static_metadata,
        "conditions": conditions,
        "numerical_checks": checks,
        "numerical_pass": bool(all(checks.values())),
        "physical_interpretation_is_manual": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.markdown.write_text(_markdown(payload), encoding="utf-8")
    if not payload["numerical_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
