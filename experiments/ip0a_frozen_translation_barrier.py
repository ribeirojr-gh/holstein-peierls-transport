#!/usr/bin/env python3
"""IP0a frozen one-site translation barrier scan versus transfer anisotropy.

This experiment is intentionally diagnostic.  It relaxes one static polaron for
each requested J_y/J_x ratio, translates the converged distortion by one site,
and linearly interpolates the classical lattice between the two equivalent
endpoints.  The electronic ground state is re-solved at every image.

The resulting maximum energy is an upper-bound path barrier, not a minimum
energy path, transition-state free energy, hopping activation energy, or
mobility.  No monotonic trend with anisotropy is pre-registered.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields, replace
import json
from pathlib import Path

import numpy as np

from holstein_peierls.parameters import StaticPolaronParameters
from holstein_peierls.polaron import solve_static_polaron
from holstein_peierls.translation_barrier import (
    FrozenTranslationProfile,
    TranslationEnergyDecomposition,
    frozen_translation_profile,
)


def _component_changes(profile: FrozenTranslationProfile) -> dict[str, float]:
    index = profile.barrier_image_index
    saddle = profile.images[index].energy
    first = profile.images[0].energy
    last = profile.images[-1].energy
    output: dict[str, float] = {}
    for descriptor in fields(TranslationEnergyDecomposition):
        name = descriptor.name
        reference = 0.5 * (float(getattr(first, name)) + float(getattr(last, name)))
        output[name] = float(getattr(saddle, name) - reference)
    output["lattice"] = float(saddle.lattice - 0.5 * (first.lattice + last.lattice))
    output["electronic"] = float(
        saddle.electronic - 0.5 * (first.electronic + last.electronic)
    )
    output["total"] = float(saddle.total - 0.5 * (first.total + last.total))
    return output


def _profile_dict(profile: FrozenTranslationProfile) -> dict:
    return {
        "direction": profile.direction,
        "source_site": profile.source_site,
        "target_site": profile.target_site,
        "barrier_eV": profile.barrier_eV,
        "barrier_image_index": profile.barrier_image_index,
        "barrier_fraction": profile.images[profile.barrier_image_index].fraction,
        "endpoint_energy_mismatch_eV": profile.endpoint_energy_mismatch_eV,
        "maximum_mirror_energy_mismatch_eV": profile.maximum_mirror_energy_mismatch_eV,
        "maximum_decomposition_error_eV": profile.maximum_decomposition_error_eV,
        "barrier_component_changes_eV": _component_changes(profile),
        "images": [
            {
                "fraction": image.fraction,
                "energy": asdict(image.energy),
                "total_energy_eV": image.energy.total,
                "lattice_energy_eV": image.energy.lattice,
                "electronic_energy_eV": image.energy.electronic,
                "electronic_eigenvalue_eV": image.electronic_eigenvalue_eV,
                "norm_error": image.norm_error,
                "ipr": image.ipr,
                "participation_number": image.participation_number,
                "source_population": image.source_population,
                "target_population": image.target_population,
                "charge_difference": image.charge_difference,
                "maximum_population": image.maximum_population,
                "maximum_population_site": image.maximum_population_site,
            }
            for image in profile.images
        ],
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
    parser.add_argument("--image-count", type=int, default=11)
    parser.add_argument("--solver", choices=["dense_lowest", "sparse"], default="sparse")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    ratios = [float(value) for value in args.anisotropy_ratios]
    if len(ratios) == 0 or any(not np.isfinite(value) or value <= 0.0 for value in ratios):
        raise ValueError("anisotropy ratios must be positive finite values")
    if len(set(ratios)) != len(ratios):
        raise ValueError("anisotropy ratios must be unique")
    if args.size < 4:
        raise ValueError("size must be at least 4")

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
        charge_flat = np.asarray(static.charge_density, dtype=np.float64).reshape(-1, order="C")
        source_site = int(np.argmax(charge_flat))
        profiles = {}
        for direction in ("+x", "+y"):
            profile = frozen_translation_profile(
                static.state,
                parameters,
                direction=direction,
                image_count=args.image_count,
                solver=args.solver,
                source_site=source_site,
            )
            profiles[direction] = _profile_dict(profile)

        cases.append(
            {
                "anisotropy_ratio_j0y_over_j0x": ratio,
                "j0x_eV": parameters.j0x,
                "j0y_eV": parameters.j0y,
                "alpha_interx_eV_per_A": parameters.alpha_interx,
                "alpha_intery_eV_per_A": parameters.alpha_intery,
                "static": {
                    "converged": bool(static.diagnostics.converged),
                    "iterations": int(static.diagnostics.iterations),
                    "total_energy_eV": float(static.total_energy),
                    "formation_energy_eV": float(static.formation_energy),
                    "ipr": float(static.ipr),
                    "participation_number": float(1.0 / static.ipr),
                    "legacy_ipr": float(static.legacy_ipr),
                    "maximum_charge_population": float(charge_flat[source_site]),
                    "maximum_charge_site": source_site,
                    "seeded_site": int(parameters.polaron_index),
                },
                "profiles": profiles,
            }
        )

    all_profiles = [case["profiles"][direction] for case in cases for direction in ("+x", "+y")]
    max_endpoint_mismatch = float(max(item["endpoint_energy_mismatch_eV"] for item in all_profiles))
    max_decomposition_error = float(max(item["maximum_decomposition_error_eV"] for item in all_profiles))
    max_norm_error = float(
        max(image["norm_error"] for item in all_profiles for image in item["images"])
    )
    finite_barriers = bool(
        all(np.isfinite(item["barrier_eV"]) and item["barrier_eV"] >= -1.0e-10 for item in all_profiles)
    )
    numerical_checks = {
        "all_static_relaxations_converged": all(case["static"]["converged"] for case in cases),
        "translated_endpoint_energy_invariance": max_endpoint_mismatch < 1.0e-8,
        "energy_decomposition_matches_eigenvalue": max_decomposition_error < 1.0e-9,
        "electronic_norm": max_norm_error < 1.0e-10,
        "finite_nonnegative_path_barriers": finite_barriers,
    }
    numerical_pass = all(numerical_checks.values())

    payload = {
        "scope": (
            "IP0a frozen adiabatic one-site translation path versus J0 anisotropy; "
            "upper-bound diagnostic barrier only"
        ),
        "size": args.size,
        "image_count": args.image_count,
        "solver": args.solver,
        "anisotropy_definition": "j0y/j0x with alpha_interx=alpha_intery fixed at model defaults",
        "ratios": ratios,
        "cases": cases,
        "aggregate": {
            "maximum_endpoint_energy_mismatch_eV": max_endpoint_mismatch,
            "maximum_decomposition_error_eV": max_decomposition_error,
            "maximum_norm_error": max_norm_error,
        },
        "numerical_checks": numerical_checks,
        "numerical_pass": numerical_pass,
        "physical_selection_is_manual": True,
        "interpretation": (
            "The profile is not an MEP. Barrier magnitudes and anisotropy trends are diagnostics "
            "to identify terms and initialize IP0b constrained/NEB calculations."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# IP0a frozen translation-barrier diagnostic",
        "",
        f"Control: {args.size}x{args.size}, {args.image_count} images, solver={args.solver}.",
        "",
        "The reported barriers are upper-bound frozen-path diagnostics, not minimum-energy-path barriers.",
        "",
        "| J0y/J0x | static PN | max population | barrier +x [meV] | barrier +y [meV] | saddle +x | saddle +y |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        px = case["profiles"]["+x"]
        py = case["profiles"]["+y"]
        lines.append(
            f"| {case['anisotropy_ratio_j0y_over_j0x']:.3f} | "
            f"{case['static']['participation_number']:.4f} | "
            f"{case['static']['maximum_charge_population']:.4f} | "
            f"{1000.0 * px['barrier_eV']:.6f} | {1000.0 * py['barrier_eV']:.6f} | "
            f"{px['barrier_fraction']:.3f} | {py['barrier_fraction']:.3f} |"
        )
    lines += ["", "## Numerical gates", ""]
    lines += [f"- {name}: {'PASS' if value else 'FAIL'}" for name, value in numerical_checks.items()]
    lines += [
        "",
        f"Numerical status: {'PASS' if numerical_pass else 'FAIL'}",
        "",
        "No monotonic barrier trend, hopping rate, activation energy, or mobility is asserted in IP0a.",
    ]
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    if not numerical_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
