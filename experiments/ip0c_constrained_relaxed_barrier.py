#!/usr/bin/env python3
"""IP0c constrained relaxed one-site translation barrier.

The straight endpoint translation vector defines a scalar reaction coordinate.
At each fixed fraction, all orthogonal lattice degrees of freedom are relaxed on
the instantaneous electronic ground-state surface.  The result is compared to
the IP0a frozen interpolation at the same fractions.

This is a constrained relaxed profile, not yet a formal NEB/string MEP or a
finite-temperature free-energy barrier.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from holstein_peierls.parameters import StaticPolaronParameters
from holstein_peierls.polaron import solve_static_polaron
from holstein_peierls.translation_barrier import frozen_translation_profile
from holstein_peierls.translation_relaxation import constrained_translation_profile


def _image_dict(image):
    return {
        "fraction": float(image.fraction),
        "energy_eV": float(image.energy_eV),
        "frozen_energy_eV": float(image.frozen_energy_eV),
        "energy_lowering_eV": float(image.diagnostics.energy_lowering_eV),
        "electronic_energy_eV": float(image.electronic_energy_eV),
        "ipr": float(image.ipr),
        "participation_number": float(image.participation_number),
        "source_population": float(image.source_population),
        "target_population": float(image.target_population),
        "converged": bool(image.diagnostics.converged),
        "optimizer_success": bool(image.diagnostics.optimizer_success),
        "iterations": int(image.diagnostics.iterations),
        "function_evaluations": int(image.diagnostics.function_evaluations),
        "projected_gradient_max_eV_per_A": float(
            image.diagnostics.projected_gradient_max_eV_per_A
        ),
        "reaction_coordinate_error": float(
            image.diagnostics.reaction_coordinate_error
        ),
        "message": image.diagnostics.message,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument(
        "--anisotropy-ratios", nargs="+", type=float, default=[0.15, 0.50, 1.00]
    )
    parser.add_argument("--image-count", type=int, default=7)
    parser.add_argument("--solver", choices=["dense_lowest", "sparse"], default="sparse")
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--gradient-tolerance", type=float, default=2.0e-6)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    ratios = [float(value) for value in args.anisotropy_ratios]
    if not ratios or any(not np.isfinite(value) or value <= 0.0 for value in ratios):
        raise ValueError("anisotropy ratios must be positive and finite")
    if args.size < 4:
        raise ValueError("size must be at least 4")

    center_zero = (args.size // 2) * args.size + (args.size // 2)
    base = StaticPolaronParameters(
        nx=args.size,
        ny=args.size,
        polaron_position=center_zero + 1,
    )

    cases = []
    for ratio in ratios:
        parameters = replace(base, j0y=base.j0x * ratio)
        static = solve_static_polaron(
            parameters,
            solver=args.solver,
            gradient_mode="optimized",
            legacy_convergence=False,
        )
        directions = {}
        for direction in ("+x", "+y"):
            relaxed = constrained_translation_profile(
                static.state,
                parameters,
                direction=direction,
                image_count=args.image_count,
                solver=args.solver,
                max_iterations=args.max_iterations,
                gradient_tolerance_eV_per_A=args.gradient_tolerance,
            )
            frozen = frozen_translation_profile(
                static.state,
                parameters,
                direction=direction,
                image_count=args.image_count,
                solver=args.solver,
            )
            relaxed_energies = np.asarray(
                [image.energy_eV for image in relaxed.images], dtype=np.float64
            )
            frozen_energies = np.asarray(
                [image.energy.total for image in frozen.images], dtype=np.float64
            )
            directions[direction] = {
                "relaxed_barrier_eV": float(relaxed.barrier_eV),
                "frozen_barrier_eV": float(frozen.barrier_eV),
                "barrier_reduction_eV": float(frozen.barrier_eV - relaxed.barrier_eV),
                "relaxed_barrier_fraction": float(
                    relaxed.images[relaxed.barrier_image_index].fraction
                ),
                "frozen_barrier_fraction": float(
                    frozen.images[frozen.barrier_image_index].fraction
                ),
                "endpoint_energy_mismatch_eV": float(
                    relaxed.endpoint_energy_mismatch_eV
                ),
                "maximum_reaction_coordinate_error": float(
                    relaxed.maximum_reaction_coordinate_error
                ),
                "maximum_projected_gradient_eV_per_A": float(
                    relaxed.maximum_projected_gradient_eV_per_A
                ),
                "all_images_converged": bool(relaxed.all_images_converged),
                "maximum_energy_excess_over_frozen_eV": float(
                    np.max(relaxed_energies - frozen_energies)
                ),
                "images": [_image_dict(image) for image in relaxed.images],
            }
        cases.append(
            {
                "anisotropy_ratio_j0y_over_j0x": ratio,
                "j0x_eV": float(parameters.j0x),
                "j0y_eV": float(parameters.j0y),
                "static_converged": bool(static.diagnostics.converged),
                "static_ipr": float(static.ipr),
                "static_participation_number": float(1.0 / static.ipr),
                "directions": directions,
            }
        )

    profiles = [
        case["directions"][direction]
        for case in cases
        for direction in ("+x", "+y")
    ]
    numerical_checks = {
        "all_static_relaxations_converged": all(
            case["static_converged"] for case in cases
        ),
        "all_constrained_images_converged": all(
            profile["all_images_converged"] for profile in profiles
        ),
        "translated_endpoint_energy_invariance": max(
            profile["endpoint_energy_mismatch_eV"] for profile in profiles
        ) < 1.0e-8,
        "reaction_coordinate_preserved": max(
            profile["maximum_reaction_coordinate_error"] for profile in profiles
        ) < 1.0e-9,
        "projected_gradient_converged": max(
            profile["maximum_projected_gradient_eV_per_A"] for profile in profiles
        ) < 1.0e-5,
        "relaxation_never_raises_frozen_images": max(
            profile["maximum_energy_excess_over_frozen_eV"] for profile in profiles
        ) < 1.0e-8,
        "finite_nonnegative_relaxed_barriers": all(
            np.isfinite(profile["relaxed_barrier_eV"])
            and profile["relaxed_barrier_eV"] >= -1.0e-9
            for profile in profiles
        ),
    }
    numerical_pass = all(numerical_checks.values())

    payload = {
        "scope": (
            "IP0c transversely relaxed fixed-translation hyperplane profile; "
            "not a formal MEP, free-energy barrier, hopping rate, or mobility"
        ),
        "size": int(args.size),
        "image_count": int(args.image_count),
        "solver": args.solver,
        "max_iterations": int(args.max_iterations),
        "gradient_tolerance_eV_per_A": float(args.gradient_tolerance),
        "ratios": ratios,
        "cases": cases,
        "aggregate": {
            "maximum_endpoint_energy_mismatch_eV": float(
                max(profile["endpoint_energy_mismatch_eV"] for profile in profiles)
            ),
            "maximum_reaction_coordinate_error": float(
                max(profile["maximum_reaction_coordinate_error"] for profile in profiles)
            ),
            "maximum_projected_gradient_eV_per_A": float(
                max(profile["maximum_projected_gradient_eV_per_A"] for profile in profiles)
            ),
            "maximum_energy_excess_over_frozen_eV": float(
                max(profile["maximum_energy_excess_over_frozen_eV"] for profile in profiles)
            ),
        },
        "numerical_checks": numerical_checks,
        "numerical_pass": numerical_pass,
        "physical_selection_is_manual": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# IP0c constrained relaxed translation profile",
        "",
        (
            f"Control: {args.size}x{args.size}, {args.image_count} images, "
            f"solver={args.solver}."
        ),
        "",
        "The relaxed profile is constrained to fixed projections on the endpoint translation vector; it is not yet a formal NEB/string MEP.",
        "",
        "| J0y/J0x | dir | frozen barrier [meV] | relaxed barrier [meV] | reduction [meV] | saddle s |",
        "|---:|:---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        for direction in ("+x", "+y"):
            profile = case["directions"][direction]
            lines.append(
                f"| {case['anisotropy_ratio_j0y_over_j0x']:.3f} | {direction} | "
                f"{1000.0 * profile['frozen_barrier_eV']:.6f} | "
                f"{1000.0 * profile['relaxed_barrier_eV']:.6f} | "
                f"{1000.0 * profile['barrier_reduction_eV']:.6f} | "
                f"{profile['relaxed_barrier_fraction']:.3f} |"
            )
    lines += ["", "## Numerical gates", ""]
    lines += [
        f"- {name}: {'PASS' if value else 'FAIL'}"
        for name, value in numerical_checks.items()
    ]
    lines += [
        "",
        f"Numerical status: {'PASS' if numerical_pass else 'FAIL'}",
        "",
        "Barrier trends and mechanistic interpretation remain manual.",
    ]
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    if not numerical_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
