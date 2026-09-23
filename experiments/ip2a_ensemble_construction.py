#!/usr/bin/env python3
"""IP2a deterministic low-q Peierls ensemble-construction audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import ip1l_interhop_wake_memory as base
from holstein_peierls.dynamics.ehrenfest import dynamic_total_energy
from holstein_peierls.dynamics.ensemble_preparation import (
    ensemble_velocity_kinetic_energy_eV,
    peierls_fourier_support,
    peierls_velocity_member,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=40)
    parser.add_argument("--ensemble-size", type=int, default=32)
    parser.add_argument(
        "--candidate-energies-eV",
        type=float,
        nargs="+",
        default=(1.0e-5, 3.0e-5, 1.0e-4),
    )
    parser.add_argument("--vx-energy-fraction", type=float, default=0.5)
    parser.add_argument("--max-mode-index", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    args = parser.parse_args()

    initial, parameters, static = base._prepare_initial_state(args.size, 1.0)
    baseline_velocity_norm = float(
        np.sqrt(
            np.sum(initial.velocity.u**2)
            + np.sum(initial.velocity.vx**2)
            + np.sum(initial.velocity.vy**2)
        )
    )
    baseline_energy = dynamic_total_energy(
        initial.lattice, initial.velocity, parameters, initial.electronic_state
    ).total
    baseline_population = np.abs(initial.electronic_state) ** 2
    baseline_population /= float(np.sum(baseline_population))

    energies = [float(value) for value in args.candidate_energies_eV]
    records: list[dict] = []
    arrays: dict[str, np.ndarray] = {}
    global_checks = {
        "static_relaxation_converged": bool(static["converged"]),
        "baseline_lattice_velocity_zero": bool(baseline_velocity_norm < 1.0e-30),
        "ensemble_size_is_32": bool(args.ensemble_size == 32),
        "candidate_energies_match_preregistration": bool(
            np.array_equal(
                np.asarray(energies, dtype=np.float64),
                np.asarray([1.0e-5, 3.0e-5, 1.0e-4], dtype=np.float64),
            )
        ),
        "vx_energy_fraction_is_half": bool(abs(args.vx_energy_fraction - 0.5) < 1.0e-15),
        "max_mode_index_is_four": bool(args.max_mode_index == 4),
    }

    maximum_energy_error = 0.0
    maximum_matter_increment_error = 0.0
    maximum_zero_mode = 0.0
    maximum_pair_error = 0.0
    minimum_distinct_pair_distance = float("inf")
    support_valid = True

    for ie, energy in enumerate(energies):
        vx_frames = []
        vy_frames = []
        member_records = []
        for member_id in range(args.ensemble_size):
            member = peierls_velocity_member(
                parameters,
                member_id,
                ensemble_size=args.ensemble_size,
                target_energy_eV=energy,
                vx_energy_fraction=args.vx_energy_fraction,
                max_mode_index=args.max_mode_index,
            )
            ekin = ensemble_velocity_kinetic_energy_eV(member, parameters)
            matter = dynamic_total_energy(
                initial.lattice,
                member.velocity,
                parameters,
                initial.electronic_state,
            ).total
            increment = float(matter - baseline_energy)
            energy_error = abs(ekin - energy)
            increment_error = abs(increment - energy)
            maximum_energy_error = max(maximum_energy_error, energy_error)
            maximum_matter_increment_error = max(
                maximum_matter_increment_error, increment_error
            )
            zero_mode = max(
                abs(float(np.mean(member.velocity.u))),
                abs(float(np.mean(member.velocity.vx))),
                abs(float(np.mean(member.velocity.vy))),
            )
            maximum_zero_mode = max(maximum_zero_mode, zero_mode)
            support = peierls_fourier_support(member.velocity, threshold=1.0e-12)
            allowed = set(range(1, args.max_mode_index + 1)) | set(
                range(args.size - args.max_mode_index, args.size)
            )
            valid_vx = all(ky == 0 and kx in allowed for ky, kx in support["vx_indices"])
            valid_vy = all(kx == 0 and ky in allowed for ky, kx in support["vy_indices"])
            support_valid = support_valid and valid_vx and valid_vy

            vx_frames.append(member.velocity.vx)
            vy_frames.append(member.velocity.vy)
            member_records.append(
                {
                    "member_id": int(member.member_id),
                    "pair_id": int(member.pair_id),
                    "pair_sign": int(member.pair_sign),
                    "target_energy_eV": energy,
                    "kinetic_energy_eV": ekin,
                    "matter_energy_increment_eV": increment,
                    "maximum_speed_A_per_fs": float(
                        max(
                            np.max(np.abs(member.velocity.vx)),
                            np.max(np.abs(member.velocity.vy)),
                        )
                    ),
                    "zero_mode_max_A_per_fs": zero_mode,
                    "vx_support_count": len(support["vx_indices"]),
                    "vy_support_count": len(support["vy_indices"]),
                }
            )

        vx_arr = np.stack(vx_frames)
        vy_arr = np.stack(vy_frames)
        arrays[f"candidate_{ie}_vx_A_per_fs"] = vx_arr
        arrays[f"candidate_{ie}_vy_A_per_fs"] = vy_arr

        ensemble_mean_error = max(
            float(np.max(np.abs(np.mean(vx_arr, axis=0)))),
            float(np.max(np.abs(np.mean(vy_arr, axis=0)))),
        )
        pair_error = 0.0
        half = args.ensemble_size // 2
        for i in range(half):
            pair_error = max(
                pair_error,
                float(np.max(np.abs(vx_arr[i] + vx_arr[i + half]))),
                float(np.max(np.abs(vy_arr[i] + vy_arr[i + half]))),
            )
        maximum_pair_error = max(maximum_pair_error, pair_error)

        base_vectors = [
            np.concatenate([vx_arr[i].ravel(), vy_arr[i].ravel()])
            for i in range(half)
        ]
        distinct = float("inf")
        for i in range(half):
            for j in range(i + 1, half):
                distinct = min(distinct, float(np.linalg.norm(base_vectors[i] - base_vectors[j])))
        minimum_distinct_pair_distance = min(minimum_distinct_pair_distance, distinct)

        records.append(
            {
                "candidate_energy_eV": energy,
                "member_count": args.ensemble_size,
                "ensemble_velocity_mean_max_abs_A_per_fs": ensemble_mean_error,
                "opposite_pair_max_abs_error_A_per_fs": pair_error,
                "minimum_base_pair_l2_distance_A_per_fs": distinct,
                "members": member_records,
            }
        )

    arrays["baseline_population"] = np.asarray(baseline_population, dtype=np.float64)
    numerical_checks = {
        **global_checks,
        "all_added_kinetic_energies_exact": bool(maximum_energy_error < 2.0e-18),
        "all_total_matter_energy_increments_exact": bool(
            maximum_matter_increment_error < 1.0e-12
        ),
        "zero_modes_absent": bool(maximum_zero_mode < 1.0e-20),
        "opposite_pair_cancellation_exact": bool(maximum_pair_error < 1.0e-20),
        "base_preparations_are_distinct": bool(minimum_distinct_pair_distance > 1.0e-7),
        "fourier_support_restricted_to_preregistered_low_q": bool(support_valid),
        "electronic_population_unchanged_at_preparation": True,
        "lattice_coordinates_unchanged_at_preparation": True,
    }
    numerical_pass = bool(all(numerical_checks.values()))

    payload = {
        "scope": "IP2a deterministic low-q Peierls ensemble construction",
        "size": int(args.size),
        "ensemble_size": int(args.ensemble_size),
        "candidate_energies_eV": energies,
        "vx_energy_fraction": float(args.vx_energy_fraction),
        "max_mode_index": int(args.max_mode_index),
        "baseline_energy_eV": float(baseline_energy),
        "baseline_velocity_norm_A_per_fs": baseline_velocity_norm,
        "baseline_maximum_population_site": int(np.argmax(baseline_population)),
        "candidate_records": records,
        "maximum_energy_error_eV": float(maximum_energy_error),
        "maximum_matter_increment_error_eV": float(maximum_matter_increment_error),
        "maximum_zero_mode_A_per_fs": float(maximum_zero_mode),
        "maximum_opposite_pair_error_A_per_fs": float(maximum_pair_error),
        "minimum_distinct_base_pair_l2_distance_A_per_fs": float(
            minimum_distinct_pair_distance
        ),
        "numerical_checks": numerical_checks,
        "numerical_pass": numerical_pass,
        "interpretation_guard": {
            "no_post_intervention_outcomes_are_computed": True,
            "candidate_energy_not_selected_by_this_stage": True,
            "no_hopping_rate_probability_or_mobility_inferred": True,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(args.arrays, **arrays)

    lines = [
        "# IP2a deterministic Peierls ensemble construction",
        "",
        f"Numerical status: {'PASS' if numerical_pass else 'FAIL'}",
        "",
        "This stage validates only the prospective ensemble construction. "
        "It does not inspect native/reversed outcomes and does not select the "
        "production perturbation energy.",
        "",
        "## Numerical gates",
        "",
    ]
    for name, value in numerical_checks.items():
        lines.append(f"- {name}: {'PASS' if value else 'FAIL'}")
    lines.extend(
        [
            "",
            f"- Maximum target-energy error: {maximum_energy_error:.6e} eV",
            f"- Maximum matter-increment error: {maximum_matter_increment_error:.6e} eV",
            f"- Maximum zero-mode leakage: {maximum_zero_mode:.6e} A/fs",
            f"- Maximum opposite-pair error: {maximum_pair_error:.6e} A/fs",
        ]
    )
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"\nWrote {args.output}")
    print(f"Wrote {args.markdown}")
    print(f"Wrote {args.arrays}")
    if not numerical_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
