#!/usr/bin/env python3
"""IP1b dressed-polaron hopping validation at zero electric field.

Electronic residence-site changes and translations of the full lattice-distortion
template are tracked independently at 20, 50 and 100 fs persistence.  The
experiment is a mechanistic validation of the IP1a event definition; it does not
infer a diffusion coefficient, mobility, or activation energy.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import asdict, replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from holstein_peierls.dynamics.coupled import CoupledEhrenfestState
from holstein_peierls.dynamics.dressed_hopping import (
    PeriodicDistortionTemplateMatcher,
    PersistentTemplateTracker,
    match_transition_events,
    prospective_bond_rank,
)
from holstein_peierls.dynamics.ehrenfest import LatticeVelocity, dynamic_total_energy
from holstein_peierls.dynamics.electronic_decoherence import apply_instantaneous_decoherence
from holstein_peierls.dynamics.hopping_observables import (
    PersistentHopEvent,
    PersistentSiteTracker,
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


PERSISTENCE_WINDOWS_FS = (20.0, 50.0, 100.0)
MATCH_LAG_WINDOWS_FS = (100.0, 250.0, 500.0, 1000.0)
PRIMARY_PERSISTENCE_FS = 50.0


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


def _nearest(events: list[PersistentHopEvent]) -> list[PersistentHopEvent]:
    return [event for event in events if event.is_nearest_neighbor]


def _serialize_event(event: PersistentHopEvent) -> dict:
    return asdict(event)


def _event_prehop_record(event: PersistentHopEvent, history: list[dict], window_fs: float) -> dict:
    start = float(event.transition_start_time_fs)
    selected = [
        item
        for item in history
        if start - window_fs - 1.0e-9 <= float(item["time_fs"]) <= start + 1.0e-9
    ]
    ranks: list[int] = []
    anisotropies: list[float] = []
    if event.is_nearest_neighbor:
        for item in selected:
            ranks.append(prospective_bond_rank(item["bond_magnitudes_eV"], event.direction))
            anisotropies.append(float(item["local_anisotropy"]))
    start_matches = [item for item in history if abs(float(item["time_fs"]) - start) <= 1.0e-9]
    output = _serialize_event(event)
    output.update(
        {
            "idc_associated": bool(start_matches and start_matches[-1]["after_idc"]),
            "prehop_sample_count": int(len(ranks)),
            "prehop_future_bond_strongest_fraction": (
                float(np.mean(np.asarray(ranks) == 1)) if ranks else None
            ),
            "prehop_future_bond_mean_rank": float(np.mean(ranks)) if ranks else None,
            "prehop_local_anisotropy_mean": float(np.mean(anisotropies)) if anisotropies else None,
            "template_center_at_transition_start": (
                int(start_matches[-1]["template_center_site"]) if start_matches else None
            ),
            "template_amplitude_at_transition_start": (
                float(start_matches[-1]["template_amplitude"]) if start_matches else None
            ),
            "template_gap_at_transition_start": (
                float(start_matches[-1]["template_relative_gap"]) if start_matches else None
            ),
        }
    )
    return output


def _tracker_summary(
    electronic_events: list[PersistentHopEvent],
    lattice_events: list[PersistentHopEvent],
) -> dict:
    electronic_nn = _nearest(electronic_events)
    lattice_nn = _nearest(lattice_events)
    matches: dict[str, dict] = {}
    for window in MATCH_LAG_WINDOWS_FS:
        paired = match_transition_events(
            electronic_nn,
            lattice_nn,
            maximum_abs_lag_fs=window,
        )
        lags = np.asarray([item.lag_fs for item in paired], dtype=np.float64)
        matches[f"{int(window)}fs"] = {
            "matched_count": int(len(paired)),
            "matched_fraction_of_electronic_nn": (
                float(len(paired) / len(electronic_nn)) if electronic_nn else 0.0
            ),
            "lag_mean_fs": float(np.mean(lags)) if lags.size else None,
            "lag_median_fs": float(np.median(lags)) if lags.size else None,
            "lag_abs_mean_fs": float(np.mean(np.abs(lags))) if lags.size else None,
        }
    return {
        "electronic_persistent_transition_count": int(len(electronic_events)),
        "electronic_nearest_neighbor_count": int(len(electronic_nn)),
        "lattice_persistent_transition_count": int(len(lattice_events)),
        "lattice_nearest_neighbor_count": int(len(lattice_nn)),
        "match_windows": matches,
    }


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
    prehop_window_fs: float,
    decoherence_interval_fs: float,
    lattice_seed: int,
    decoherence_seed: int,
    krylov_dimension: int,
    minimum_template_amplitude: float,
    minimum_template_relative_gap: float,
) -> dict:
    steps = _integer_stride(final_time_fs, dt_fs, "final_time_fs")
    burn_steps = int(round(burn_in_fs / dt_fs))
    if burn_steps < 0 or burn_steps >= steps or not np.isclose(burn_steps * dt_fs, burn_in_fs):
        raise ValueError("burn_in_fs must be an integer multiple of dt_fs within the run")
    sample_stride = _integer_stride(sample_interval_fs, dt_fs, "sample_interval_fs")
    event_stride = decoherence_interval_steps(decoherence_interval_fs, dt_fs)

    electronic_trackers: dict[int, PersistentSiteTracker] = {}
    lattice_trackers: dict[int, PersistentTemplateTracker] = {}
    electronic_events: dict[int, list[PersistentHopEvent]] = {}
    lattice_events: dict[int, list[PersistentHopEvent]] = {}
    for persistence_fs in PERSISTENCE_WINDOWS_FS:
        samples = _integer_stride(persistence_fs, sample_interval_fs, "persistence_fs")
        key = int(round(persistence_fs))
        electronic_trackers[key] = PersistentSiteTracker(
            parameters.nx,
            parameters.ny,
            persistence_samples=samples,
            minimum_max_population=0.10,
            minimum_dominance_margin=0.02,
        )
        lattice_trackers[key] = PersistentTemplateTracker(
            parameters.nx,
            parameters.ny,
            persistence_samples=samples,
            minimum_amplitude=minimum_template_amplitude,
            minimum_relative_gap=minimum_template_relative_gap,
        )
        electronic_events[key] = []
        lattice_events[key] = []

    matcher = PeriodicDistortionTemplateMatcher(
        initial.lattice,
        parameters,
        static_center_site,
    )
    static_match = matcher.match(initial.lattice)
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

    temperatures: list[float] = []
    template_amplitudes: list[float] = []
    template_gaps: list[float] = []
    template_confident: list[bool] = []
    template_center_equal_electronic_dominant: list[bool] = []
    primary_event_records: list[dict] = []
    history_count = int(np.ceil((prehop_window_fs + max(PERSISTENCE_WINDOWS_FS)) / sample_interval_fs)) + 8
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
        match = matcher.match(current.lattice)
        confident = bool(
            match.amplitude >= minimum_template_amplitude
            and match.relative_gap >= minimum_template_relative_gap
        )
        primary_tracker = electronic_trackers[int(PRIMARY_PERSISTENCE_FS)]
        metric_site = dominant_site if primary_tracker.current_site is None else int(primary_tracker.current_site)
        bonds = local_bond_transfer_magnitudes(current.lattice, parameters, metric_site)
        local_anisotropy = local_transfer_anisotropy(bonds)
        history.append(
            {
                "time_fs": end_time_fs,
                "after_idc": bool(did_idc),
                "bond_magnitudes_eV": bonds,
                "local_anisotropy": float(local_anisotropy),
                "template_center_site": int(match.center_site),
                "template_amplitude": float(match.amplitude),
                "template_relative_gap": float(match.relative_gap),
            }
        )

        for key, tracker in electronic_trackers.items():
            event = tracker.update(population, end_time_fs)
            if event is not None:
                electronic_events[key].append(event)
                if key == int(PRIMARY_PERSISTENCE_FS):
                    primary_event_records.append(
                        _event_prehop_record(event, list(history), prehop_window_fs)
                    )
        for key, tracker in lattice_trackers.items():
            event = tracker.update(match, end_time_fs)
            if event is not None:
                lattice_events[key].append(event)

        temperature = kinetic_temperature_K(
            current.velocity,
            parameters,
            degrees_of_freedom=dof,
        )
        means = zero_mode_means(current)
        max_zero_mode = max(max_zero_mode, max(abs(float(value)) for value in means.values()))
        temperatures.append(float(temperature))
        template_amplitudes.append(float(match.amplitude))
        template_gaps.append(float(match.relative_gap))
        template_confident.append(confident)
        template_center_equal_electronic_dominant.append(match.center_site == dominant_site)

    final_energy = dynamic_total_energy(
        current.lattice, current.velocity, parameters, current.electronic_state
    ).total
    final_residual = final_energy - initial_energy - lattice_heat - electronic_exchange
    max_balance_residual = max(max_balance_residual, abs(float(final_residual)))
    elapsed = perf_counter() - start_clock

    tracker_summaries = {
        f"{key}fs": _tracker_summary(electronic_events[key], lattice_events[key])
        for key in sorted(electronic_trackers)
    }
    primary_nn_records = [record for record in primary_event_records if record["is_nearest_neighbor"]]
    strongest = [
        float(record["prehop_future_bond_strongest_fraction"])
        for record in primary_nn_records
        if record["prehop_future_bond_strongest_fraction"] is not None
    ]
    mean_ranks = [
        float(record["prehop_future_bond_mean_rank"])
        for record in primary_nn_records
        if record["prehop_future_bond_mean_rank"] is not None
    ]
    anisotropy = [
        float(record["prehop_local_anisotropy_mean"])
        for record in primary_nn_records
        if record["prehop_local_anisotropy_mean"] is not None
    ]
    idc_associated = [bool(record["idc_associated"]) for record in primary_event_records]
    direction_counts = Counter(
        str(record["direction"]) for record in primary_nn_records
    )

    return {
        "temperature_K": float(temperature_K),
        "lattice_seed": int(lattice_seed),
        "decoherence_seed": int(decoherence_seed),
        "steps": int(steps),
        "elapsed_seconds": float(elapsed),
        "idc_events": int(idc_events),
        "sample_count_post_burn": int(len(temperatures)),
        "temperature_mean_K": float(np.mean(temperatures)),
        "temperature_std_K": float(np.std(temperatures, ddof=1)) if len(temperatures) > 1 else 0.0,
        "static_template_self_match_center_site": int(static_match.center_site),
        "static_template_self_match_amplitude": float(static_match.amplitude),
        "static_template_self_match_relative_gap": float(static_match.relative_gap),
        "template_confident_sample_fraction": float(np.mean(template_confident)),
        "template_amplitude_mean": float(np.mean(template_amplitudes)),
        "template_amplitude_p05": float(np.quantile(template_amplitudes, 0.05)),
        "template_relative_gap_mean": float(np.mean(template_gaps)),
        "template_relative_gap_p05": float(np.quantile(template_gaps, 0.05)),
        "instantaneous_template_electronic_center_agreement_fraction": float(
            np.mean(template_center_equal_electronic_dominant)
        ),
        "tracker_summaries": tracker_summaries,
        "primary_persistence_fs": float(PRIMARY_PERSISTENCE_FS),
        "primary_nearest_neighbor_direction_counts": {
            key: int(direction_counts.get(key, 0)) for key in ("+x", "-x", "+y", "-y")
        },
        "primary_idc_associated_persistent_fraction": (
            float(np.mean(idc_associated)) if idc_associated else 0.0
        ),
        "primary_prehop_future_bond_strongest_fraction_mean": (
            float(np.mean(strongest)) if strongest else None
        ),
        "primary_prehop_future_bond_mean_rank_mean": (
            float(np.mean(mean_ranks)) if mean_ranks else None
        ),
        "primary_prehop_local_anisotropy_mean": (
            float(np.mean(anisotropy)) if anisotropy else None
        ),
        "primary_event_records": primary_event_records,
        "accumulated_lattice_bath_heat_eV": float(lattice_heat),
        "accumulated_electronic_environment_exchange_eV": float(electronic_exchange),
        "final_generalized_energy_residual_eV": float(final_residual),
        "maximum_generalized_energy_residual_eV": float(max_balance_residual),
        "maximum_electronic_norm_error": float(max_norm_error),
        "maximum_zero_mode_magnitude": float(max_zero_mode),
    }


def _aggregate_condition(records: list[dict]) -> dict:
    primary_key = f"{int(PRIMARY_PERSISTENCE_FS)}fs"
    result: dict = {
        "trajectory_count": int(len(records)),
        "temperature_mean_K": float(np.mean([r["temperature_mean_K"] for r in records])),
        "template_confident_sample_fraction": float(
            np.mean([r["template_confident_sample_fraction"] for r in records])
        ),
        "template_amplitude_mean": float(np.mean([r["template_amplitude_mean"] for r in records])),
        "template_relative_gap_mean": float(np.mean([r["template_relative_gap_mean"] for r in records])),
        "instantaneous_template_electronic_center_agreement_fraction": float(
            np.mean([r["instantaneous_template_electronic_center_agreement_fraction"] for r in records])
        ),
        "maximum_generalized_energy_residual_eV": float(
            max(r["maximum_generalized_energy_residual_eV"] for r in records)
        ),
        "maximum_electronic_norm_error": float(max(r["maximum_electronic_norm_error"] for r in records)),
        "maximum_zero_mode_magnitude": float(max(r["maximum_zero_mode_magnitude"] for r in records)),
    }
    persistence: dict[str, dict] = {}
    for persistence_fs in PERSISTENCE_WINDOWS_FS:
        key = f"{int(persistence_fs)}fs"
        summaries = [r["tracker_summaries"][key] for r in records]
        item = {
            "electronic_nearest_neighbor_count": int(
                sum(s["electronic_nearest_neighbor_count"] for s in summaries)
            ),
            "lattice_nearest_neighbor_count": int(
                sum(s["lattice_nearest_neighbor_count"] for s in summaries)
            ),
        }
        for lag_window in MATCH_LAG_WINDOWS_FS:
            lag_key = f"{int(lag_window)}fs"
            matched = int(sum(s["match_windows"][lag_key]["matched_count"] for s in summaries))
            electronic = int(sum(s["electronic_nearest_neighbor_count"] for s in summaries))
            lags: list[float] = []
            for s in summaries:
                value = s["match_windows"][lag_key]["lag_mean_fs"]
                count = int(s["match_windows"][lag_key]["matched_count"])
                if value is not None and count > 0:
                    lags.extend([float(value)] * count)
            item[f"matched_{lag_key}_count"] = matched
            item[f"matched_{lag_key}_fraction"] = float(matched / electronic) if electronic else 0.0
            item[f"matched_{lag_key}_lag_mean_fs_approx"] = float(np.mean(lags)) if lags else None
        persistence[key] = item
    result["persistence"] = persistence

    strongest = [
        r["primary_prehop_future_bond_strongest_fraction_mean"]
        for r in records
        if r["primary_prehop_future_bond_strongest_fraction_mean"] is not None
    ]
    ranks = [
        r["primary_prehop_future_bond_mean_rank_mean"]
        for r in records
        if r["primary_prehop_future_bond_mean_rank_mean"] is not None
    ]
    result["primary_prehop_future_bond_strongest_fraction_mean"] = (
        float(np.mean(strongest)) if strongest else None
    )
    result["primary_prehop_future_bond_mean_rank_mean"] = float(np.mean(ranks)) if ranks else None
    result["primary_idc_associated_persistent_fraction_mean"] = float(
        np.mean([r["primary_idc_associated_persistent_fraction"] for r in records])
    )
    return result


def _evaluate_numerical_checks(payload: dict) -> dict[str, bool]:
    records = [r for condition in payload["conditions"] for r in condition["trajectories"]]
    expected_events = int(np.floor(payload["final_time_fs"] / payload["decoherence_interval_fs"] + 1.0e-12))
    static_ok = all(bool(item["converged"]) for item in payload["static"].values())
    requested = len(payload["conditions"]) * len(payload["lattice_seeds"])
    temperature_ok = all(
        abs(float(r["temperature_mean_K"]) - float(r["temperature_K"]))
        <= max(15.0, 0.10 * float(r["temperature_K"]))
        for r in records
    )
    self_match_ok = all(
        int(r["static_template_self_match_center_site"])
        == int(payload["static"][str(payload["condition_lookup"][str(index)]["anisotropy_ratio"])]["maximum_population_site"])
        and abs(float(r["static_template_self_match_amplitude"]) - 1.0) <= 1.0e-10
        for index, r in enumerate(records)
    ) if False else True
    # Self-match is checked per condition below without relying on record ordering.
    self_match_ok = True
    for condition in payload["conditions"]:
        static_center = int(payload["static"][str(condition["anisotropy_ratio"])]["maximum_population_site"])
        for record in condition["trajectories"]:
            self_match_ok = self_match_ok and (
                int(record["static_template_self_match_center_site"]) == static_center
                and abs(float(record["static_template_self_match_amplitude"]) - 1.0) <= 1.0e-10
            )
    finite_diagnostics = all(
        np.isfinite(float(r["template_confident_sample_fraction"]))
        and np.isfinite(float(r["template_amplitude_mean"]))
        and np.isfinite(float(r["template_relative_gap_mean"]))
        for r in records
    )
    return {
        "all_static_relaxations_converged": bool(static_ok),
        "all_requested_trajectories_completed": bool(len(records) == requested),
        "exact_idc_event_count": bool(all(int(r["idc_events"]) == expected_events for r in records)),
        "lattice_temperature": bool(temperature_ok),
        "zero_field_generalized_energy_balance": bool(
            max(float(r["maximum_generalized_energy_residual_eV"]) for r in records) < 5.0e-5
        ),
        "electronic_norm": bool(max(float(r["maximum_electronic_norm_error"]) for r in records) < 1.0e-10),
        "projected_zero_modes": bool(max(float(r["maximum_zero_mode_magnitude"]) for r in records) < 1.0e-12),
        "static_template_self_match": bool(self_match_ok),
        "finite_dressed_hopping_diagnostics": bool(finite_diagnostics),
    }


def _markdown(payload: dict) -> str:
    lines = [
        "# IP1b dressed-polaron hopping validation",
        "",
        f"20x20 zero-field control; dt={payload['dt_fs']} fs; final={payload['final_time_fs']/1000:.1f} ps; burn={payload['burn_in_fs']/1000:.1f} ps; IDC-BM td={payload['decoherence_interval_fs']} fs.",
        "",
        "| J0y/J0x | T [K] | e-NN 20 fs | e-NN 50 fs | e-NN 100 fs | lattice-NN 50 fs | dressed <=500 fs | template/e-center agree | future bond strongest | mean future-bond rank |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in payload["conditions"]:
        a = condition["aggregate"]
        p20 = a["persistence"]["20fs"]
        p50 = a["persistence"]["50fs"]
        p100 = a["persistence"]["100fs"]
        strongest = a["primary_prehop_future_bond_strongest_fraction_mean"]
        rank = a["primary_prehop_future_bond_mean_rank_mean"]
        lines.append(
            "| {ratio:.2f} | {temp:.0f} | {e20} | {e50} | {e100} | {l50} | {dress:.3f} | {agree:.3f} | {strong} | {rank} |".format(
                ratio=condition["anisotropy_ratio"],
                temp=condition["temperature_K"],
                e20=p20["electronic_nearest_neighbor_count"],
                e50=p50["electronic_nearest_neighbor_count"],
                e100=p100["electronic_nearest_neighbor_count"],
                l50=p50["lattice_nearest_neighbor_count"],
                dress=p50["matched_500fs_fraction"],
                agree=a["instantaneous_template_electronic_center_agreement_fraction"],
                strong="n/a" if strongest is None else f"{strongest:.3f}",
                rank="n/a" if rank is None else f"{rank:.3f}",
            )
        )
    lines.extend(["", "## Numerical gates", ""])
    for name, passed in payload["numerical_checks"].items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    lines.extend(
        [
            "",
            f"Numerical status: {'PASS' if payload['numerical_pass'] else 'FAIL'}",
            "",
            "Dressed-event counts, lag distributions, persistence sensitivity, and future-bond predictability are physical diagnostics and are not numerical PASS gates.",
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
    parser.add_argument("--prehop-window-fs", type=float, default=100.0)
    parser.add_argument("--decoherence-interval-fs", type=float, default=180.0)
    parser.add_argument("--minimum-template-amplitude", type=float, default=0.05)
    parser.add_argument("--minimum-template-relative-gap", type=float, default=0.01)
    parser.add_argument(
        "--lattice-seeds",
        type=int,
        nargs="+",
        default=[20260905, 20260906, 20260907, 20260908],
    )
    parser.add_argument("--decoherence-seed-offset", type=int, default=3000000)
    parser.add_argument("--krylov-dimension", type=int, default=6)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    condition_specs = [(args.isotropic_ratio, temperature) for temperature in args.temperatures_K]
    condition_specs.append((args.reference_ratio, args.reference_temperature_K))
    unique_ratios = sorted(set(float(ratio) for ratio, _ in condition_specs))
    prepared = {}
    static = {}
    for ratio in unique_ratios:
        initial, parameters, metadata = _prepare_initial_state(args.size, ratio)
        prepared[ratio] = (initial, parameters)
        static[str(ratio)] = metadata

    conditions = []
    for ratio, temperature in condition_specs:
        initial, parameters = prepared[float(ratio)]
        trajectories = []
        for seed in args.lattice_seeds:
            trajectories.append(
                _run_trajectory(
                    initial,
                    parameters,
                    static_center_site=int(static[str(float(ratio))]["maximum_population_site"]),
                    temperature_K=float(temperature),
                    gamma_u_per_fs=args.gamma_u_per_fs,
                    gamma_v_per_fs=args.gamma_v_per_fs,
                    dt_fs=args.dt_fs,
                    final_time_fs=args.final_time_fs,
                    burn_in_fs=args.burn_in_fs,
                    sample_interval_fs=args.sample_interval_fs,
                    prehop_window_fs=args.prehop_window_fs,
                    decoherence_interval_fs=args.decoherence_interval_fs,
                    lattice_seed=int(seed),
                    decoherence_seed=int(args.decoherence_seed_offset + seed),
                    krylov_dimension=args.krylov_dimension,
                    minimum_template_amplitude=args.minimum_template_amplitude,
                    minimum_template_relative_gap=args.minimum_template_relative_gap,
                )
            )
        conditions.append(
            {
                "anisotropy_ratio": float(ratio),
                "temperature_K": float(temperature),
                "trajectories": trajectories,
                "aggregate": _aggregate_condition(trajectories),
            }
        )

    payload = {
        "scope": "IP1b zero-field dressed-polaron hopping validation; independent electronic and lattice-distortion centers with persistence sensitivity and directional predictability; no mobility/diffusion/activation claim",
        "size": int(args.size),
        "dt_fs": float(args.dt_fs),
        "final_time_fs": float(args.final_time_fs),
        "burn_in_fs": float(args.burn_in_fs),
        "sample_interval_fs": float(args.sample_interval_fs),
        "prehop_window_fs": float(args.prehop_window_fs),
        "persistence_windows_fs": list(PERSISTENCE_WINDOWS_FS),
        "primary_persistence_fs": float(PRIMARY_PERSISTENCE_FS),
        "match_lag_windows_fs": list(MATCH_LAG_WINDOWS_FS),
        "minimum_template_amplitude": float(args.minimum_template_amplitude),
        "minimum_template_relative_gap": float(args.minimum_template_relative_gap),
        "gamma_u_per_fs": float(args.gamma_u_per_fs),
        "gamma_v_per_fs": float(args.gamma_v_per_fs),
        "scheme": "bm",
        "decoherence_interval_fs": float(args.decoherence_interval_fs),
        "decoherence_interval_is_material_calibrated": False,
        "lattice_seeds": [int(seed) for seed in args.lattice_seeds],
        "static": static,
        "conditions": conditions,
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
