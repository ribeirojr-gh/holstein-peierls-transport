#!/usr/bin/env python3
"""Run the S2 strict 20x20 one-polaron regression anchor."""

from __future__ import annotations

import json
from pathlib import Path
import argparse

import numpy as np

from holstein_peierls.gradients import energy_gradient
from holstein_peierls.parameters import StaticPolaronParameters
from holstein_peierls.polaron import solve_static_polaron


REFERENCE_ENERGY_EV = -0.405459057756


def center_position(nx: int, ny: int) -> int:
    return (ny // 2) * nx + (nx // 2) + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    parameters = StaticPolaronParameters(
        nx=20,
        ny=20,
        k1=16.51,
        k2=0.51,
        j0x=0.100,
        j0y=0.015,
        alpha_intra=3.0,
        alpha_interx=0.4,
        alpha_intery=0.4,
        polaron_position=center_position(20, 20),
        max_iterations=2000,
        convergence_criterion=1.0e-8,
    )
    result = solve_static_polaron(
        parameters,
        solver="sparse",
        gradient_mode="optimized",
        legacy_convergence=False,
    )
    gradient, ground = energy_gradient(
        result.state,
        parameters,
        solver="sparse",
        mode="optimized",
    )
    charge_sum = float(np.sum(result.charge_density))
    payload = {
        "scope": "S2 one-polaron static regression anchor",
        "size": [20, 20],
        "parameters": parameters.to_dict(),
        "total_energy_eV": float(result.total_energy),
        "reference_energy_eV": REFERENCE_ENERGY_EV,
        "absolute_reference_error_eV": abs(float(result.total_energy) - REFERENCE_ENERGY_EV),
        "electronic_energy_eV": float(result.electronic_energy),
        "formation_energy_eV": float(result.formation_energy),
        "ipr": float(result.ipr),
        "charge_density_sum": charge_sum,
        "final_max_gradient_eV_per_A": float(gradient.maximum_absolute_component),
        "diagnostics": {
            "iterations": int(result.diagnostics.iterations),
            "converged_u": bool(result.diagnostics.converged_u),
            "converged_vx": bool(result.diagnostics.converged_vx),
            "converged_vy": bool(result.diagnostics.converged_vy),
            "final_max_delta_u_A": float(result.diagnostics.final_max_delta_u),
            "final_max_delta_vx_A": float(result.diagnostics.final_max_delta_vx),
            "final_max_delta_vy_A": float(result.diagnostics.final_max_delta_vy),
        },
        "gates": {
            "all_coordinate_families_converged": bool(result.diagnostics.converged),
            "final_gradient_below_1e-6_eV_per_A": bool(
                gradient.maximum_absolute_component < 1.0e-6
            ),
            "charge_density_normalized_within_1e-12": bool(
                abs(charge_sum - 1.0) < 1.0e-12
            ),
            "energy_matches_reference_within_1e-8_eV": bool(
                abs(result.total_energy - REFERENCE_ENERGY_EV) <= 1.0e-8
            ),
            "ipr_positive_and_finite": bool(
                np.isfinite(result.ipr) and result.ipr > 0.0
            ),
        },
    }
    payload["sector_pass"] = bool(all(payload["gates"].values()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["sector_pass"]:
        raise SystemExit("S2 one-polaron regression gate failed")


if __name__ == "__main__":
    main()
