#!/usr/bin/env python3
"""IP0d full-charge-cloud constrained translation benchmark.

The lattice remains on the same exact endpoint interpolation used in IP0a.  The
new ingredient is an explicit electronic order parameter built from the full
charge-density difference of the two translated endpoints.  A Lagrange
multiplier forces the electronic density to progress from endpoint A to B.

Reported energies exclude the auxiliary bias.  The result is still a frozen
lattice diagnostic, not a relaxed MEP or finite-temperature activation energy.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from holstein_peierls.parameters import StaticPolaronParameters
from holstein_peierls.polaron import solve_static_polaron
from holstein_peierls.translation_charge_constraint import (
    charge_constrained_frozen_profile,
)


def _profile_dict(profile) -> dict:
    endpoint_reference = 0.5 * (
        profile.images[0].unconstrained_total_energy_eV
        + profile.images[-1].unconstrained_total_energy_eV
    )
    unconstrained_energies = np.asarray(
        [image.unconstrained_total_energy_eV for image in profile.images],
        dtype=np.float64,
    )
    adiabatic_barrier = float(
        max(0.0, float(np.max(unconstrained_energies)) - endpoint_reference)
    )
    midpoint = profile.images[len(profile.images) // 2]
    return {
        "direction": profile.direction,
        "source_site": profile.source_site,
        "target_site": profile.target_site,
        "order_parameter": {
            "offset": profile.order_parameter.offset,
            "start_expectation": profile.order_parameter.start_expectation,
            "end_expectation": profile.order_parameter.end_expectation,
            "endpoint_span": (
                profile.order_parameter.start_expectation
                - profile.order_parameter.end_expectation
            ),
            "maximum_abs_weight": float(
                np.max(np.abs(profile.order_parameter.weights))
            ),
        },
        "charge_constrained_barrier_eV": profile.barrier_eV,
        "adiabatic_frozen_barrier_eV": adiabatic_barrier,
        "barrier_image_index": profile.barrier_image_index,
        "barrier_fraction": profile.images[profile.barrier_image_index].fraction,
        "endpoint_energy_mismatch_eV": profile.endpoint_energy_mismatch_eV,
        "maximum_constraint_residual": profile.maximum_constraint_residual,
        "maximum_norm_error": profile.maximum_norm_error,
        "maximum_biased_identity_error_eV": profile.maximum_biased_identity_error_eV,
        "minimum_constraint_energy_penalty_eV": profile.minimum_constraint_energy_penalty_eV,
        "midpoint": {
            "centered_coordinate": midpoint.centered_coordinate,
            "constraint_energy_penalty_eV": midpoint.constraint_energy_penalty_eV,
            "bias_lambda_eV": midpoint.bias_lambda_eV,
            "ipr": midpoint.ipr,
            "participation_number": midpoint.participation_number,
            "source_population": midpoint.source_population,
            "target_population": midpoint.target_population,
            "physical_total_energy_eV": midpoint.physical_total_energy_eV,
            "unconstrained_total_energy_eV": midpoint.unconstrained_total_energy_eV,
        },
        "images": [
            {
                "fraction": image.fraction,
                "target_expectation": image.target_expectation,
                "achieved_expectation": image.achieved_expectation,
                "centered_coordinate": image.centered_coordinate,
                "physical_total_energy_eV": image.physical_total_energy_eV,
                "unconstrained_total_energy_eV": image.unconstrained_total_energy_eV,
                "constraint_energy_penalty_eV": image.constraint_energy_penalty_eV,
                "electronic_physical_energy_eV": image.electronic_physical_energy_eV,
                "unconstrained_electronic_energy_eV": image.unconstrained_electronic_energy_eV,
                "bias_lambda_eV": image.bias_lambda_eV,
                "constraint_residual": image.constraint_residual,
                "norm_error": image.norm_error,
                "biased_energy_identity_error_eV": image.biased_energy_identity_error_eV,
                "diagonalizations": image.diagonalizations,
                "ipr": image.ipr,
                "participation_number": image.participation_number,
                "source_population": image.source_population,
                "target_population": image.target_population,
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
        default=[0.15, 0.50, 1.00],
    )
    parser.add_argument("--image-count", type=int, default=7)
    parser.add_argument("--solver", choices=["dense_lowest", "sparse"], default="sparse")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    if args.size < 4:
        raise ValueError("size must be at least 4")
    if args.image_count < 3 or args.image_count % 2 == 0:
        raise ValueError("image-count must be an odd integer >= 3")
    ratios = [float(value) for value in args.anisotropy_ratios]
    if not ratios or any(not np.isfinite(value) or value <= 0.0 for value in ratios):
        raise ValueError("anisotropy ratios must be positive finite values")

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
        directions = {}
        for direction in ("+x", "+y"):
            profile = charge_constrained_frozen_profile(
                static.state,
                parameters,
                direction=direction,
                image_count=args.image_count,
                solver=args.solver,
                expectation_tolerance=1.0e-10,
            )
            directions[direction] = _profile_dict(profile)
        cases.append(
            {
                "anisotropy_ratio_j0y_over_j0x": ratio,
                "j0x_eV": parameters.j0x,
                "j0y_eV": parameters.j0y,
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
    max_endpoint = float(max(p["endpoint_energy_mismatch_eV"] for p in profiles))
    max_constraint = float(max(p["maximum_constraint_residual"] for p in profiles))
    max_norm = float(max(p["maximum_norm_error"] for p in profiles))
    max_identity = float(max(p["maximum_biased_identity_error_eV"] for p in profiles))
    min_penalty = float(min(p["minimum_constraint_energy_penalty_eV"] for p in profiles))
    max_span_error = float(
        max(abs(p["order_parameter"]["endpoint_span"] - 2.0) for p in profiles)
    )
    max_coordinate_error = float(
        max(
            abs(image["centered_coordinate"] - (1.0 - 2.0 * image["fraction"]))
            for p in profiles
            for image in p["images"]
        )
    )
    finite_barriers = bool(
        all(
            np.isfinite(p["charge_constrained_barrier_eV"])
            and p["charge_constrained_barrier_eV"] >= -1.0e-10
            for p in profiles
        )
    )
    checks = {
        "all_static_relaxations_converged": all(case["static_converged"] for case in cases),
        "normalized_endpoint_charge_order_span": max_span_error < 1.0e-9,
        "charge_constraint_targets_achieved": max_constraint < 1.0e-8 and max_coordinate_error < 1.0e-8,
        "electronic_norm": max_norm < 1.0e-10,
        "biased_energy_identity": max_identity < 1.0e-9,
        "translated_endpoint_energy_invariance": max_endpoint < 1.0e-8,
        "constrained_state_variational_bound": min_penalty >= -1.0e-8,
        "finite_nonnegative_charge_constrained_barriers": finite_barriers,
    }
    numerical_pass = all(checks.values())

    payload = {
        "scope": (
            "IP0d full-charge-cloud constrained electronic translation on frozen lattice images; "
            "bias removed from reported physical energy; no relaxed MEP, activation energy, hopping rate, or mobility claim"
        ),
        "size": args.size,
        "image_count": args.image_count,
        "solver": args.solver,
        "ratios": ratios,
        "cases": cases,
        "aggregate": {
            "maximum_endpoint_energy_mismatch_eV": max_endpoint,
            "maximum_constraint_residual": max_constraint,
            "maximum_norm_error": max_norm,
            "maximum_biased_energy_identity_error_eV": max_identity,
            "minimum_constraint_energy_penalty_eV": min_penalty,
            "maximum_endpoint_span_error": max_span_error,
            "maximum_centered_coordinate_error": max_coordinate_error,
            "total_constraint_diagonalizations": int(
                sum(
                    image["diagonalizations"]
                    for p in profiles
                    for image in p["images"]
                )
            ),
        },
        "numerical_checks": checks,
        "numerical_pass": numerical_pass,
        "physical_selection_is_manual": True,
        "interpretation": (
            "The electronic density is explicitly forced from one translated endpoint to the other, "
            "but the lattice remains on the frozen linear interpolation. Barrier values are therefore "
            "charge-transfer diagnostics, not minimum-energy-path or free-energy barriers."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# IP0d charge-constrained frozen translation diagnostic",
        "",
        f"Control: {args.size}x{args.size}, {args.image_count} frozen lattice images, solver={args.solver}.",
        "",
        "The complete endpoint charge-density difference defines the constrained electronic coordinate. The auxiliary bias is removed from all physical energies.",
        "",
        "| J0y/J0x | dir | adiabatic frozen barrier [meV] | charge-constrained barrier [meV] | midpoint penalty [meV] | midpoint lambda [eV] | midpoint PN |",
        "|---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        for direction in ("+x", "+y"):
            p = case["directions"][direction]
            midpoint = p["midpoint"]
            lines.append(
                f"| {case['anisotropy_ratio_j0y_over_j0x']:.3f} | {direction} | "
                f"{1000.0*p['adiabatic_frozen_barrier_eV']:.6f} | "
                f"{1000.0*p['charge_constrained_barrier_eV']:.6f} | "
                f"{1000.0*midpoint['constraint_energy_penalty_eV']:.6f} | "
                f"{midpoint['bias_lambda_eV']:.6e} | "
                f"{midpoint['participation_number']:.4f} |"
            )
    lines += ["", "## Numerical gates", ""]
    lines += [f"- {name}: {'PASS' if value else 'FAIL'}" for name, value in checks.items()]
    lines += [
        "",
        f"Numerical status: {'PASS' if numerical_pass else 'FAIL'}",
        "",
        "No MEP barrier, activation energy, hopping rate, mobility, or anisotropy trend is asserted by IP0d alone.",
    ]
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    if not numerical_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
