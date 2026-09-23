#!/usr/bin/env python3
"""IP2a-3: pre-event field-work time-step convergence, without paired branches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

import ip1l_interhop_wake_memory as base
import ip2a2_event_generation_calibration as pilot
from holstein_peierls.dynamics.energy_work_refinement import (
    CALIBRATION_BOUND_EV,
    DT_GRID_FS,
    REFERENCE_LABELS,
    assess_timestep_refinement,
)
from holstein_peierls.dynamics.ensemble_preparation import peierls_velocity_member
from holstein_peierls.dynamics.field import UniformElectricField2D


REFERENCE_IP2A2_MEMBER0_DT02_RESIDUAL_EV = 2.2239577499405354e-6


def _trial(initial, parameters, field, *, reference: str, dt_fs: float) -> dict:
    if reference == "unperturbed":
        prepared = pilot._prepared_state(initial, initial.velocity)
        added_energy = 0.0
    elif reference == "member_0_1e-5_eV":
        member = peierls_velocity_member(
            parameters,
            0,
            ensemble_size=32,
            target_energy_eV=1.0e-5,
            vx_energy_fraction=0.5,
            max_mode_index=4,
        )
        prepared = pilot._prepared_state(initial, member.velocity)
        added_energy = float(member.target_energy_eV)
    else:
        raise ValueError("unregistered reference")

    t_start = perf_counter()
    result = pilot._driven_first_event(
        prepared,
        parameters,
        field,
        dt_fs=float(dt_fs),
        search_time_fs=4000.0,
        sample_interval_fs=2.0,
        energy_interval_fs=10.0,
        krylov_dimension=6,
    )
    elapsed = perf_counter() - t_start
    return {
        "reference": reference,
        "dt_fs": float(dt_fs),
        "added_peierls_energy_eV": added_energy,
        "first_persistent_event": result["first_persistent_event"],
        "maximum_energy_work_residual_eV": float(
            result["maximum_energy_work_residual_eV"]
        ),
        "maximum_norm_error": float(result["maximum_norm_error"]),
        "maximum_zero_mode_mean": float(result["maximum_zero_mode_mean"]),
        "accumulated_external_work_eV": float(
            result["accumulated_external_work_eV"]
        ),
        "meets_original_balance_bound": bool(
            result["maximum_energy_work_residual_eV"] <= CALIBRATION_BOUND_EV
        ),
        "original_ip2a2_stability_gate": bool(result["numerically_stable"]),
        "elapsed_seconds": float(elapsed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=40)
    parser.add_argument("--field-mv-per-A", dest="field_mV_per_A", type=float, default=10.0)
    parser.add_argument("--dt-grid-fs", nargs="+", type=float, default=DT_GRID_FS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    args = parser.parse_args()

    if args.size != 40 or args.field_mV_per_A != 10.0 or tuple(args.dt_grid_fs) != DT_GRID_FS:
        raise ValueError("IP2a-3 physical model and dt grid are fixed by preregistration")

    initial, parameters, static = base._prepare_initial_state(40, 1.0)
    field = UniformElectricField2D.from_millivolt_per_angstrom(
        10.0,
        angle_radians=0.0,
        ax_angstrom=3.0,
        ay_angstrom=3.0,
    )
    records = []
    for reference in REFERENCE_LABELS:
        for dt in DT_GRID_FS:
            record = _trial(initial, parameters, field, reference=reference, dt_fs=dt)
            records.append(record)
            event = record["first_persistent_event"]
            print(
                f"{reference} dt={dt:g} fs: balance="
                f"{record['maximum_energy_work_residual_eV']:.9e} eV "
                f"event={None if event is None else event['direction']} "
                f"accepted={None if event is None else event['accepted_time_fs']} fs "
                f"runtime={record['elapsed_seconds']:.2f} s",
                flush=True,
            )

    refinement = assess_timestep_refinement(records)
    reference_dt02 = next(
        item for item in records
        if item["reference"] == "member_0_1e-5_eV" and item["dt_fs"] == 0.2
    )
    reproduced_ip2a2_absolute_error = abs(
        reference_dt02["maximum_energy_work_residual_eV"]
        - REFERENCE_IP2A2_MEMBER0_DT02_RESIDUAL_EV
    )
    checks = {
        "static_relaxation_converged": bool(static["converged"]),
        "all_six_preregistered_trials_completed": len(records) == 6,
        "every_first_event_is_persistent_nearest_neighbor_x": all(
            item["first_persistent_event"] is not None
            and item["first_persistent_event"]["is_nearest_neighbor"]
            and abs(item["first_persistent_event"]["dx_sites"]) == 1
            and item["first_persistent_event"]["dy_sites"] == 0
            for item in records
        ),
        "norms_controlled": all(
            item["maximum_norm_error"] < 1.0e-10 for item in records
        ),
        "zero_modes_controlled": all(
            item["maximum_zero_mode_mean"] < 1.0e-10 for item in records
        ),
        "all_diagnostics_finite": all(
            np.all(
                np.isfinite(
                    [
                        item["maximum_energy_work_residual_eV"],
                        item["maximum_norm_error"],
                        item["maximum_zero_mode_mean"],
                        item["accumulated_external_work_eV"],
                        item["elapsed_seconds"],
                    ]
                )
            )
            for item in records
        ),
        "ip2a2_dt02_member0_reproduced": bool(
            reproduced_ip2a2_absolute_error <= 2.0e-8
        ),
        "no_post_event_native_or_reversed_branches": True,
        "original_calibration_threshold_unchanged": CALIBRATION_BOUND_EV == 2.0e-6,
    }
    numerical_pass = bool(all(checks.values()))

    payload = {
        "scope": "IP2a-3 pre-event field-work time-step convergence",
        "protocol": {
            "size": 40,
            "field_mV_per_A": 10.0,
            "dt_grid_fs": list(DT_GRID_FS),
            "references": list(REFERENCE_LABELS),
            "sample_interval_fs": 2.0,
            "energy_interval_fs": 10.0,
            "maximum_search_time_fs": 4000.0,
            "krylov_dimension": 6,
            "original_balance_bound_eV": CALIBRATION_BOUND_EV,
        },
        "static_converged": bool(static["converged"]),
        "reference_ip2a2_dt02_residual_eV": REFERENCE_IP2A2_MEMBER0_DT02_RESIDUAL_EV,
        "ip2a2_reproduction_absolute_error_eV": reproduced_ip2a2_absolute_error,
        "records": records,
        "diagnostic_decision": refinement,
        "numerical_checks": checks,
        "numerical_pass": numerical_pass,
        "interpretation_guard": {
            "ip2a2_calibration_remains_failed": True,
            "does_not_select_a_production_peierls_energy": True,
            "no_post_event_counterfactuals_computed": True,
            "candidate_dt_requires_new_full_calibration_before_ip2b": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(
        args.arrays,
        dt_fs=np.asarray([item["dt_fs"] for item in records], dtype=np.float64),
        maximum_energy_work_residual_eV=np.asarray(
            [item["maximum_energy_work_residual_eV"] for item in records],
            dtype=np.float64,
        ),
        accepted_time_fs=np.asarray(
            [
                np.nan if item["first_persistent_event"] is None
                else item["first_persistent_event"]["accepted_time_fs"]
                for item in records
            ],
            dtype=np.float64,
        ),
    )
    lines = [
        "# IP2a-3 pre-event work-balance refinement",
        "",
        f"Execution numerical status: {'PASS' if numerical_pass else 'FAIL'}",
        f"Independent refinement diagnostic: {'PASS' if refinement['numerical_refinement_succeeded'] else 'FAIL'}",
        f"Candidate refined dt (not a production selection): {refinement['candidate_refined_dt_fs']}",
        "",
        "IP2a-2 remains formally unqualified; no production Peierls energy is selected.",
        "",
        "## Per-reference residuals",
        "",
    ]
    for row in refinement["reference_diagnostics"]:
        lines.append(
            f"- {row['reference']}: residuals at dt=0.2/0.1/0.05 fs = "
            f"{row['residuals_eV_by_decreasing_dt']}; "
            f"reduction factors={row['reduction_factors']}; "
            f"effective orders={row['effective_log2_orders']}."
        )
    lines.extend(["", "## Execution integrity", ""])
    for name, value in checks.items():
        lines.append(f"- {name}: {'PASS' if value else 'FAIL'}")
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"checks": checks, "refinement": refinement}, indent=2))
    print(f"Artifacts: {args.output}, {args.markdown}, {args.arrays}")
    if not numerical_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
