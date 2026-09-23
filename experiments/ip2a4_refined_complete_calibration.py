#!/usr/bin/env python3
"""IP2a-4 pre-intervention event-generation calibration.

This stage never creates native/reversed post-event branches.  It uses only
field-free stability and field-driven first-event generation diagnostics to
select (or reject) one preregistered perturbation energy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

import ip1l_interhop_wake_memory as base
import ip1n_gauge_continuous_field_release as ip1n
from holstein_peierls.dynamics.coupled import CoupledEhrenfestState, coupled_verlet_step
from holstein_peierls.dynamics.driven import (
    coupled_field_verlet_step,
    field_dynamic_total_energy,
)
from holstein_peierls.dynamics.ehrenfest import dynamic_total_energy
from holstein_peierls.dynamics.ensemble_calibration import select_production_energy
from holstein_peierls.dynamics.ensemble_preparation import peierls_velocity_member
from holstein_peierls.dynamics.field import UniformElectricField2D
from holstein_peierls.dynamics.numerical_validation import extensive_energy_balance_passes
from holstein_peierls.dynamics.thermal import zero_mode_means


LATTICE_SPACING_A = 3.0
PILOT_MEMBER_IDS = (0, 4, 8, 12, 16, 20, 24, 28)
FIELD_FREE_CONTROL_FS = 200.0
EARLY_EVENT_CUTOFF_FS = 200.0
PRE_EVENT_BALANCE_LIMIT_EV = 2.0e-6
NORM_LIMIT = 1.0e-10
ZERO_MODE_LIMIT = 1.0e-10
REQUIRED_VALID_X = 6


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


def _prepared_state(initial, velocity) -> CoupledEhrenfestState:
    return CoupledEhrenfestState(
        initial.lattice.copy(),
        velocity.copy(),
        np.asarray(initial.electronic_state, dtype=np.complex128).copy(),
    )


def _field_free_control(
    prepared: CoupledEhrenfestState,
    parameters,
    *,
    dt_fs: float,
    sample_interval_fs: float,
    energy_interval_fs: float,
    krylov_dimension: int,
) -> dict:
    steps = base._integer_stride(FIELD_FREE_CONTROL_FS, dt_fs, "field_free_control_fs")
    sample_stride = base._integer_stride(sample_interval_fs, dt_fs, "sample_interval_fs")
    energy_stride = base._integer_stride(energy_interval_fs, dt_fs, "energy_interval_fs")
    current = ip1n._copy_state(prepared)
    initial_energy = dynamic_total_energy(
        current.lattice, current.velocity, parameters, current.electronic_state
    ).total
    initial_norm = float(np.linalg.norm(current.electronic_state))
    tracker = ip1n._branch_tracker(parameters.nx, parameters.ny, sample_interval_fs)
    p = np.abs(current.electronic_state) ** 2
    tracker.update(p / float(np.sum(p)), 0.0)
    first_event = None
    max_energy_drift = 0.0
    max_norm_error = 0.0
    max_zero_mode = 0.0

    for step in range(steps):
        current, _, _ = coupled_verlet_step(
            current,
            parameters,
            dt_fs,
            electronic_method="cfm4_lanczos",
            krylov_dimension=krylov_dimension,
        )
        new_time = (step + 1) * dt_fs
        if (step + 1) % energy_stride == 0 or step + 1 == steps:
            energy = dynamic_total_energy(
                current.lattice, current.velocity, parameters, current.electronic_state
            ).total
            max_energy_drift = max(max_energy_drift, abs(energy - initial_energy))
            max_norm_error = max(
                max_norm_error,
                abs(float(np.linalg.norm(current.electronic_state)) - initial_norm),
            )
            max_zero_mode = max(
                max_zero_mode,
                max(abs(value) for value in zero_mode_means(current).values()),
            )
        if (step + 1) % sample_stride == 0:
            p = np.abs(current.electronic_state) ** 2
            p /= float(np.sum(p))
            event = tracker.update(p, new_time)
            if event is not None:
                first_event = _event_record(event)
                break

    numerical = bool(
        max_energy_drift <= PRE_EVENT_BALANCE_LIMIT_EV
        and extensive_energy_balance_passes(max_energy_drift, parameters.n_sites)
        and max_norm_error < NORM_LIMIT
        and max_zero_mode < ZERO_MODE_LIMIT
    )
    return {
        "first_persistent_event": first_event,
        "clean_no_persistent_relocation": first_event is None,
        "maximum_energy_drift_eV": float(max_energy_drift),
        "maximum_norm_error": float(max_norm_error),
        "maximum_zero_mode_mean": float(max_zero_mode),
        "numerically_stable": numerical,
    }


def _driven_first_event(
    prepared: CoupledEhrenfestState,
    parameters,
    field,
    *,
    dt_fs: float,
    search_time_fs: float,
    sample_interval_fs: float,
    energy_interval_fs: float,
    krylov_dimension: int,
) -> dict:
    steps = base._integer_stride(search_time_fs, dt_fs, "search_time_fs")
    sample_stride = base._integer_stride(sample_interval_fs, dt_fs, "sample_interval_fs")
    energy_stride = base._integer_stride(energy_interval_fs, dt_fs, "energy_interval_fs")
    current = ip1n._copy_state(prepared)
    initial_energy = field_dynamic_total_energy(
        current.lattice,
        current.velocity,
        parameters,
        current.electronic_state,
        field,
        0.0,
    ).total
    initial_norm = float(np.linalg.norm(current.electronic_state))
    tracker = ip1n._branch_tracker(parameters.nx, parameters.ny, sample_interval_fs)
    p = np.abs(current.electronic_state) ** 2
    tracker.update(p / float(np.sum(p)), 0.0)

    work = 0.0
    first_event = None
    max_balance = 0.0
    max_norm_error = 0.0
    max_zero_mode = 0.0

    for step in range(steps):
        time_fs = step * dt_fs
        current, step_work, _, _ = coupled_field_verlet_step(
            current,
            parameters,
            field,
            time_fs,
            dt_fs,
            electronic_method="cfm4_lanczos",
            krylov_dimension=krylov_dimension,
        )
        new_time = time_fs + dt_fs
        work += float(step_work)
        if (step + 1) % energy_stride == 0 or step + 1 == steps:
            energy = field_dynamic_total_energy(
                current.lattice,
                current.velocity,
                parameters,
                current.electronic_state,
                field,
                new_time,
            ).total
            max_balance = max(max_balance, abs((energy - initial_energy) - work))
            max_norm_error = max(
                max_norm_error,
                abs(float(np.linalg.norm(current.electronic_state)) - initial_norm),
            )
            max_zero_mode = max(
                max_zero_mode,
                max(abs(value) for value in zero_mode_means(current).values()),
            )
        if (step + 1) % sample_stride == 0:
            p = np.abs(current.electronic_state) ** 2
            p /= float(np.sum(p))
            event = tracker.update(p, new_time)
            if event is not None:
                first_event = _event_record(event)
                break

    numerical = bool(
        max_balance <= PRE_EVENT_BALANCE_LIMIT_EV
        and extensive_energy_balance_passes(max_balance, parameters.n_sites)
        and max_norm_error < NORM_LIMIT
        and max_zero_mode < ZERO_MODE_LIMIT
    )
    early = bool(
        first_event is not None
        and float(first_event["transition_start_time_fs"]) < EARLY_EVENT_CUTOFF_FS
    )
    first_is_x = bool(
        first_event is not None
        and first_event["is_nearest_neighbor"]
        and abs(int(first_event["dx_sites"])) == 1
        and int(first_event["dy_sites"]) == 0
    )
    return {
        "first_persistent_event": first_event,
        "early_driven_event": early,
        "first_event_is_nearest_neighbor_x": first_is_x,
        "maximum_energy_work_residual_eV": float(max_balance),
        "maximum_norm_error": float(max_norm_error),
        "maximum_zero_mode_mean": float(max_zero_mode),
        "accumulated_external_work_eV": float(work),
        "numerically_stable": numerical,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=40)
    parser.add_argument("--field-mv-per-A", dest="field_mV_per_A", type=float, default=10.0)
    parser.add_argument("--dt-fs", type=float, default=0.1)
    parser.add_argument("--search-time-fs", type=float, default=4000.0)
    parser.add_argument("--sample-interval-fs", type=float, default=2.0)
    parser.add_argument("--energy-sample-interval-fs", type=float, default=10.0)
    parser.add_argument("--krylov-dimension", type=int, default=6)
    parser.add_argument(
        "--candidate-energies-eV",
        type=float,
        nargs="+",
        default=(1.0e-5, 3.0e-5, 1.0e-4),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    args = parser.parse_args()

    if args.size != 40 or args.field_mV_per_A != 10.0:
        raise ValueError("IP2a-4 lattice size and field are fixed at 40 and +10 mV/A x")
    if abs(args.dt_fs - 0.1) > 1.0e-12:
        raise ValueError("IP2a-4 preregisters dt_fs=0.10 fs")
    if abs(args.sample_interval_fs - 2.0) > 1.0e-12:
        raise ValueError("IP2a-4 event sampling is fixed at 2 fs")
    if abs(args.energy_sample_interval_fs - 10.0) > 1.0e-12:
        raise ValueError("IP2a-4 energy sampling is fixed at 10 fs")
    if args.krylov_dimension != 6:
        raise ValueError("IP2a-4 preregisters Krylov dimension 6")

    if tuple(float(x) for x in args.candidate_energies_eV) != (1.0e-5, 3.0e-5, 1.0e-4):
        raise ValueError("candidate energies must match the IP2a-4 preregistration")
    if abs(args.search_time_fs - 4000.0) > 1.0e-12:
        raise ValueError("search_time_fs must remain 4000 fs for IP2a-4")

    initial, parameters, static = base._prepare_initial_state(args.size, 1.0)
    field = UniformElectricField2D.from_millivolt_per_angstrom(
        args.field_mV_per_A,
        angle_radians=0.0,
        ax_angstrom=LATTICE_SPACING_A,
        ay_angstrom=LATTICE_SPACING_A,
    )

    candidate_records = []
    event_start = np.full((3, len(PILOT_MEMBER_IDS)), np.nan, dtype=np.float64)
    event_accept = np.full_like(event_start, np.nan)
    event_dx = np.zeros_like(event_start, dtype=np.int64)
    event_dy = np.zeros_like(event_start, dtype=np.int64)
    driven_residual = np.full_like(event_start, np.nan)
    free_drift = np.full_like(event_start, np.nan)
    elapsed_start = perf_counter()

    for ie, energy in enumerate(args.candidate_energies_eV):
        trials = []
        for ip, member_id in enumerate(PILOT_MEMBER_IDS):
            member = peierls_velocity_member(
                parameters,
                member_id,
                ensemble_size=32,
                target_energy_eV=float(energy),
                vx_energy_fraction=0.5,
                max_mode_index=4,
            )
            prepared = _prepared_state(initial, member.velocity)
            free = _field_free_control(
                prepared,
                parameters,
                dt_fs=args.dt_fs,
                sample_interval_fs=args.sample_interval_fs,
                energy_interval_fs=args.energy_sample_interval_fs,
                krylov_dimension=args.krylov_dimension,
            )
            driven = _driven_first_event(
                prepared,
                parameters,
                field,
                dt_fs=args.dt_fs,
                search_time_fs=args.search_time_fs,
                sample_interval_fs=args.sample_interval_fs,
                energy_interval_fs=args.energy_sample_interval_fs,
                krylov_dimension=args.krylov_dimension,
            )
            event = driven["first_persistent_event"]
            if event is not None:
                event_start[ie, ip] = event["transition_start_time_fs"]
                event_accept[ie, ip] = event["accepted_time_fs"]
                event_dx[ie, ip] = event["dx_sites"]
                event_dy[ie, ip] = event["dy_sites"]
            driven_residual[ie, ip] = driven["maximum_energy_work_residual_eV"]
            free_drift[ie, ip] = free["maximum_energy_drift_eV"]

            stable = bool(free["numerically_stable"] and driven["numerically_stable"])
            valid_x = bool(
                stable
                and free["clean_no_persistent_relocation"]
                and not driven["early_driven_event"]
                and driven["first_event_is_nearest_neighbor_x"]
            )
            trials.append(
                {
                    "member_id": int(member_id),
                    "pair_id": int(member.pair_id),
                    "pair_sign": int(member.pair_sign),
                    "field_free": free,
                    "driven": driven,
                    "field_free_clean": bool(free["clean_no_persistent_relocation"]),
                    "early_driven_event": bool(driven["early_driven_event"]),
                    "numerically_stable": stable,
                    "valid_first_x_event": valid_x,
                }
            )
        candidate_records.append(
            {
                "candidate_energy_eV": float(energy),
                "trials": trials,
            }
        )

    selection = select_production_energy(
        candidate_records,
        pilot_count=len(PILOT_MEMBER_IDS),
        required_valid_x=REQUIRED_VALID_X,
    )
    elapsed = perf_counter() - elapsed_start

    checks = {
        "static_relaxation_converged": bool(static["converged"]),
        "pilot_member_ids_match_preregistration": tuple(PILOT_MEMBER_IDS)
        == (0, 4, 8, 12, 16, 20, 24, 28),
        "candidate_energies_match_preregistration": tuple(
            float(x) for x in args.candidate_energies_eV
        )
        == (1.0e-5, 3.0e-5, 1.0e-4),
        "field_free_control_is_200fs": FIELD_FREE_CONTROL_FS == 200.0,
        "early_event_cutoff_is_200fs": EARLY_EVENT_CUTOFF_FS == 200.0,
        "search_horizon_is_4000fs": args.search_time_fs == 4000.0,
        "pre_event_balance_limit_is_2e-6eV": PRE_EVENT_BALANCE_LIMIT_EV == 2.0e-6,
        "refined_dt_is_preregistered_0p1_fs": abs(args.dt_fs - 0.1) < 1.0e-12,
        "event_sample_2fs_energy_sample_10fs": (
            abs(args.sample_interval_fs - 2.0) < 1.0e-12
            and abs(args.energy_sample_interval_fs - 10.0) < 1.0e-12
        ),
        "no_native_reversed_post_event_branches_created": True,
        "all_trials_serializable_and_finite": bool(
            np.all(np.isfinite(driven_residual)) and np.all(np.isfinite(free_drift))
        ),
    }
    numerical_pass = bool(all(checks.values()))

    payload = {
        "scope": "IP2a-4 refined-dt complete pre-intervention calibration",
        "refined_dt_fs": float(args.dt_fs),
        "prior_ip2a2_calibration_remains_failed": True,
        "source_ip2a3_numerical_refinement_pass": True,
        "size": int(args.size),
        "pilot_member_ids": list(PILOT_MEMBER_IDS),
        "candidate_energies_eV": [float(x) for x in args.candidate_energies_eV],
        "field_free_control_fs": FIELD_FREE_CONTROL_FS,
        "early_event_cutoff_fs": EARLY_EVENT_CUTOFF_FS,
        "search_horizon_fs": float(args.search_time_fs),
        "pre_event_balance_limit_eV": PRE_EVENT_BALANCE_LIMIT_EV,
        "required_valid_x_trials": REQUIRED_VALID_X,
        "candidate_records": candidate_records,
        "selection": selection,
        "elapsed_seconds": float(elapsed),
        "numerical_checks": checks,
        "numerical_pass": numerical_pass,
        "interpretation_guard": {
            "no_post_event_counterfactual_outcomes_inspected": True,
            "selection_uses_only_pre_intervention_feasibility": True,
            "deterministic_pilot_is_not_random_sample": True,
            "no_hopping_probability_rate_or_mobility_inferred": True,
            "ip2a2_calibration_remains_failed": True,
            "ip2a3_diagnostic_does_not_select_energy": True,
            "production_energy_requires_successful_full_ip2a4_selection": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(
        args.arrays,
        event_start_time_fs=event_start,
        event_accepted_time_fs=event_accept,
        event_dx_sites=event_dx,
        event_dy_sites=event_dy,
        driven_max_energy_work_residual_eV=driven_residual,
        field_free_max_energy_drift_eV=free_drift,
        candidate_energies_eV=np.asarray(args.candidate_energies_eV, dtype=np.float64),
        pilot_member_ids=np.asarray(PILOT_MEMBER_IDS, dtype=np.int64),
    )

    lines = [
        "# IP2a-4 pre-intervention event-generation calibration",
        "",
        f"Numerical audit status: {'PASS' if numerical_pass else 'FAIL'}",
        f"Production-energy selection: {'PASS' if selection['selection_succeeded'] else 'NO QUALIFYING ENERGY'}",
        f"Selected energy: {selection['selected_energy_eV']}",
        "",
        "No native/reversed post-event branches were created.",
        "",
        "## Candidate feasibility",
        "",
    ]
    for item in selection["candidate_evaluations"]:
        lines.append(
            f"- {item['candidate_energy_eV']:.1e} eV: "
            f"valid-x={item['valid_first_x_event_count']}/8, "
            f"stable={item['all_trials_numerically_stable']}, "
            f"field-free-clean={item['all_field_free_controls_clean']}, "
            f"no-early={item['no_early_driven_events']}, "
            f"qualifies={item['qualifies']}"
        )
    lines.extend(["", "## Construction gates", ""])
    for name, value in checks.items():
        lines.append(f"- {name}: {'PASS' if value else 'FAIL'}")
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"\nWrote {args.output}")
    print(f"Wrote {args.markdown}")
    print(f"Wrote {args.arrays}")
    if not numerical_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
