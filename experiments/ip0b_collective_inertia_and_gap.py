#!/usr/bin/env python3
"""IP0b collective-inertia and midpoint-gap scan versus transfer anisotropy.

IP0a found no growth of the easy-direction frozen translation barrier toward
isotropy.  This benchmark therefore tests whether translation becomes
dynamically heavy and/or electronically near-degenerate even when the potential
profile is shallow.

The reported mass metric is a path-specific lattice reaction-coordinate inertia,
not a complete polaron band effective mass.  Frequencies are harmonic scales of
the chosen frozen path, not hopping prefactors.  Midpoint gaps are adiabatic
low-state splittings, not nonadiabatic rates.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from holstein_peierls.parameters import StaticPolaronParameters
from holstein_peierls.polaron import solve_static_polaron
from holstein_peierls.translation_barrier import (
    frozen_translation_profile,
    translate_lattice_state,
)
from holstein_peierls.translation_inertia import (
    collective_angular_frequency_per_fs,
    mass_weighted_translation_metric,
    midpoint_low_state_spectrum,
    quadratic_endpoint_curvature,
)


def _frequency_payload(curvature_eV: float, metric_eV_fs2: float) -> dict:
    if not np.isfinite(curvature_eV) or curvature_eV <= 0.0:
        return {
            "angular_frequency_per_fs": None,
            "frequency_THz": None,
            "period_fs": None,
        }
    omega = collective_angular_frequency_per_fs(curvature_eV, metric_eV_fs2)
    frequency_thz = omega / (2.0 * np.pi) * 1000.0
    period_fs = 2.0 * np.pi / omega
    return {
        "angular_frequency_per_fs": float(omega),
        "frequency_THz": float(frequency_thz),
        "period_fs": float(period_fs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument(
        "--anisotropy-ratios",
        nargs="+",
        type=float,
        default=[0.15, 0.30, 0.50, 0.70, 1.00],
    )
    parser.add_argument("--image-count", type=int, default=21)
    parser.add_argument("--fit-points", type=int, default=5)
    parser.add_argument("--solver", choices=["dense_lowest", "sparse"], default="sparse")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    ratios = [float(value) for value in args.anisotropy_ratios]
    if not ratios or any(not np.isfinite(value) or value <= 0.0 for value in ratios):
        raise ValueError("anisotropy ratios must be positive finite values")
    if len(set(ratios)) != len(ratios):
        raise ValueError("anisotropy ratios must be unique")
    if args.image_count < 7 or args.image_count % 2 == 0:
        raise ValueError("image_count must be an odd integer >= 7")

    center_zero = (args.size // 2) * args.size + (args.size // 2)
    base = StaticPolaronParameters(
        nx=args.size,
        ny=args.size,
        polaron_position=center_zero + 1,
    )

    cases: list[dict] = []
    for ratio in ratios:
        parameters = replace(base, j0y=base.j0x * ratio)
        static = solve_static_polaron(
            parameters,
            solver=args.solver,
            gradient_mode="optimized",
            legacy_convergence=False,
        )
        charge = np.asarray(static.charge_density, dtype=np.float64).reshape(-1, order="C")
        source_site = int(np.argmax(charge))
        directions: dict[str, dict] = {}

        for direction in ("+x", "+y"):
            translated = translate_lattice_state(static.state, direction)
            metric = mass_weighted_translation_metric(static.state, translated, parameters)
            profile = frozen_translation_profile(
                static.state,
                parameters,
                direction=direction,
                image_count=args.image_count,
                solver=args.solver,
                source_site=source_site,
            )
            fractions = np.asarray([image.fraction for image in profile.images], dtype=np.float64)
            energies = np.asarray([image.energy.total for image in profile.images], dtype=np.float64)
            curvature_start = quadratic_endpoint_curvature(
                fractions,
                energies,
                side="start",
                fit_points=args.fit_points,
            )
            curvature_end = quadratic_endpoint_curvature(
                fractions,
                energies,
                side="end",
                fit_points=args.fit_points,
            )
            curvature_mean = 0.5 * (curvature_start + curvature_end)
            midpoint = midpoint_low_state_spectrum(
                static.state,
                parameters,
                direction,
                source_site=source_site,
                level_count=4,
            )
            barrier_image = profile.images[profile.barrier_image_index]
            directions[direction] = {
                "barrier_eV": float(profile.barrier_eV),
                "barrier_fraction": float(barrier_image.fraction),
                "mass_metric": asdict(metric)
                | {
                    "intermolecular_eV_fs2": metric.intermolecular_eV_fs2,
                    "total_eV_fs2": metric.total_eV_fs2,
                },
                "mass_metric_component_fractions": {
                    "intramolecular": float(metric.intramolecular_eV_fs2 / metric.total_eV_fs2),
                    "vx": float(metric.vx_eV_fs2 / metric.total_eV_fs2),
                    "vy": float(metric.vy_eV_fs2 / metric.total_eV_fs2),
                },
                "endpoint_curvature_start_eV": float(curvature_start),
                "endpoint_curvature_end_eV": float(curvature_end),
                "endpoint_curvature_mean_eV": float(curvature_mean),
                "collective_harmonic_scale": _frequency_payload(
                    curvature_mean,
                    metric.total_eV_fs2,
                ),
                "midpoint_spectrum": asdict(midpoint),
                "sparse_midpoint_source_population": float(barrier_image.source_population),
                "sparse_midpoint_target_population": float(barrier_image.target_population),
                "endpoint_energy_mismatch_eV": float(profile.endpoint_energy_mismatch_eV),
                "maximum_decomposition_error_eV": float(profile.maximum_decomposition_error_eV),
                "maximum_norm_error": float(max(image.norm_error for image in profile.images)),
            }

        cases.append(
            {
                "anisotropy_ratio_j0y_over_j0x": ratio,
                "j0x_eV": float(parameters.j0x),
                "j0y_eV": float(parameters.j0y),
                "static": {
                    "converged": bool(static.diagnostics.converged),
                    "iterations": int(static.diagnostics.iterations),
                    "participation_number": float(1.0 / static.ipr),
                    "ipr": float(static.ipr),
                    "maximum_population": float(charge[source_site]),
                    "source_site": source_site,
                },
                "directions": directions,
            }
        )

    profiles = [case["directions"][direction] for case in cases for direction in ("+x", "+y")]
    metrics = [item["mass_metric"]["total_eV_fs2"] for item in profiles]
    first_gaps = [item["midpoint_spectrum"]["first_gap_eV"] for item in profiles]
    curvatures = [item["endpoint_curvature_mean_eV"] for item in profiles]
    max_endpoint = max(item["endpoint_energy_mismatch_eV"] for item in profiles)
    max_decomp = max(item["maximum_decomposition_error_eV"] for item in profiles)
    max_norm = max(item["maximum_norm_error"] for item in profiles)

    numerical_checks = {
        "all_static_relaxations_converged": bool(all(case["static"]["converged"] for case in cases)),
        "finite_positive_mass_metrics": bool(all(np.isfinite(value) and value > 0.0 for value in metrics)),
        "finite_nonnegative_midpoint_first_gaps": bool(
            all(np.isfinite(value) and value >= -1.0e-12 for value in first_gaps)
        ),
        "finite_endpoint_curvatures": bool(all(np.isfinite(value) for value in curvatures)),
        "translated_endpoint_energy_invariance": bool(max_endpoint < 1.0e-8),
        "energy_decomposition_matches_eigenvalue": bool(max_decomp < 1.0e-9),
        "electronic_norm": bool(max_norm < 1.0e-10),
    }
    numerical_pass = all(numerical_checks.values())

    payload = {
        "scope": (
            "IP0b path-specific lattice collective inertia, endpoint curvature, and midpoint low-state gap; "
            "no complete effective mass, MEP, hopping prefactor/rate, or mobility claim"
        ),
        "size": args.size,
        "image_count": args.image_count,
        "fit_points": args.fit_points,
        "solver": args.solver,
        "ratios": ratios,
        "cases": cases,
        "aggregate": {
            "minimum_mass_metric_eV_fs2": float(min(metrics)),
            "maximum_mass_metric_eV_fs2": float(max(metrics)),
            "minimum_midpoint_first_gap_eV": float(min(first_gaps)),
            "maximum_midpoint_first_gap_eV": float(max(first_gaps)),
            "minimum_endpoint_curvature_mean_eV": float(min(curvatures)),
            "maximum_endpoint_curvature_mean_eV": float(max(curvatures)),
            "maximum_endpoint_energy_mismatch_eV": float(max_endpoint),
            "maximum_decomposition_error_eV": float(max_decomp),
            "maximum_norm_error": float(max_norm),
        },
        "numerical_checks": numerical_checks,
        "numerical_pass": numerical_pass,
        "physical_interpretation_is_manual": True,
        "interpretation": (
            "Compare mass metrics, their u/vx/vy decomposition, collective harmonic scales, and midpoint gaps "
            "across anisotropy. No monotonic trend is required for numerical PASS."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# IP0b collective-inertia and midpoint-gap diagnostic",
        "",
        "The mass metric is a lattice reaction-coordinate inertia, not the complete polaron effective mass.",
        "",
        "| J0y/J0x | dir | barrier [meV] | mass metric [eV fs^2] | u frac | vx frac | vy frac | freq [THz] | midpoint gap [meV] |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        for direction in ("+x", "+y"):
            item = case["directions"][direction]
            metric = item["mass_metric"]
            frac = item["mass_metric_component_fractions"]
            freq = item["collective_harmonic_scale"]["frequency_THz"]
            freq_text = "n/a" if freq is None else f"{freq:.6f}"
            gap = item["midpoint_spectrum"]["first_gap_eV"]
            lines.append(
                f"| {case['anisotropy_ratio_j0y_over_j0x']:.3f} | {direction} | "
                f"{1000.0 * item['barrier_eV']:.6f} | {metric['total_eV_fs2']:.6e} | "
                f"{frac['intramolecular']:.4f} | {frac['vx']:.4f} | {frac['vy']:.4f} | "
                f"{freq_text} | {1000.0 * gap:.9f} |"
            )
    lines += ["", "## Numerical gates", ""]
    lines += [f"- {name}: {'PASS' if value else 'FAIL'}" for name, value in numerical_checks.items()]
    lines += [
        "",
        f"Numerical status: {'PASS' if numerical_pass else 'FAIL'}",
        "",
        "Physical trends are interpreted only after local validation.",
    ]
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    if not numerical_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
