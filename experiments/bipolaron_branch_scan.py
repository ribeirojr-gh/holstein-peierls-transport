"""Resolve competing Holstein-Hubbard bipolaron branches on an anisotropic lattice.

Four lattice seeds are relaxed independently: onsite, intersite-x, intersite-y,
and maximally separated.  Recording every converged branch, rather than only the
lowest energy, is essential for detecting metastability and first-order branch
crossings in the adiabatic energy landscape.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from holstein_peierls.parameters import StaticPolaronParameters
from holstein_peierls.polaron import solve_static_polaron
from holstein_peierls.two_particle.bipolaron import (
    initial_distortion,
    relax_static_bipolaron,
)
from holstein_peierls.two_particle.observables import pair_observables
from holstein_peierls.two_particle.parameters import BipolaronParameters


def center_position(size: int) -> int:
    return (size // 2) * size + (size // 2) + 1


def intersite_distortion(
    parameters: BipolaronParameters,
    direction: str,
) -> np.ndarray:
    """Return two one-polaron Holstein distortions on adjacent sites."""
    u = np.zeros((parameters.ny, parameters.nx), dtype=float)
    cy, cx = divmod(parameters.pair_index, parameters.nx)
    if direction == "x":
        second = (cy, (cx + 1) % parameters.nx)
    elif direction == "y":
        second = ((cy + 1) % parameters.ny, cx)
    else:
        raise ValueError(direction)
    displacement = -parameters.alpha_intra / parameters.k1
    u[cy, cx] = displacement
    u[second] = displacement
    return u


def single_polaron_energy(size: int) -> float:
    p = StaticPolaronParameters(
        nx=size,
        ny=size,
        polaron_position=center_position(size),
        alpha_interx=0.0,
        alpha_intery=0.0,
        max_iterations=1200,
        convergence_criterion=1.0e-7,
    )
    result = solve_static_polaron(
        p,
        solver="sparse",
        gradient_mode="optimized",
        legacy_convergence=False,
    )
    return float(result.total_energy)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument(
        "--u-values",
        nargs="+",
        type=float,
        default=[0.45, 0.50, 0.525, 0.55, 0.60, 0.70, 0.80, 0.90, 1.00, 1.20, 1.50],
    )
    parser.add_argument("--output", type=Path, default=Path("bipolaron-branches"))
    args = parser.parse_args()

    e1 = single_polaron_energy(args.size)
    single = StaticPolaronParameters(
        nx=args.size,
        ny=args.size,
        polaron_position=center_position(args.size),
    )
    base = BipolaronParameters.from_polaron_parameters(single)
    base = replace(
        base,
        pair_position=center_position(args.size),
        max_iterations=900,
        convergence_criterion=1.0e-7,
        eigensolver_tolerance=2.0e-10,
    )

    rows: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []

    for hubbard_u in args.u_values:
        p = replace(base, hubbard_u=float(hubbard_u))
        seeds = {
            "onsite": initial_distortion(p, "onsite"),
            "intersite_x": intersite_distortion(p, "x"),
            "intersite_y": intersite_distortion(p, "y"),
            "separated": initial_distortion(p, "separated"),
        }
        candidates = {}
        for seed_name, seed_u in seeds.items():
            result = relax_static_bipolaron(p, initial_u=seed_u)
            obs = pair_observables(result.ground_state, p)
            binding = 2.0 * e1 - result.energy.total
            candidates[seed_name] = (result, obs, binding)
            rows.append(
                {
                    "size": args.size,
                    "U_eV": float(hubbard_u),
                    "seed": seed_name,
                    "total_energy_eV": result.energy.total,
                    "binding_vs_2polarons_eV": binding,
                    "P_onsite": obs.onsite_probability,
                    "P_nn": obs.nearest_neighbour_probability,
                    "P_nn_x": obs.nearest_neighbour_x_probability,
                    "P_nn_y": obs.nearest_neighbour_y_probability,
                    "mean_r": obs.mean_separation,
                    "rms_r": obs.rms_separation,
                    "mean_dx": obs.mean_dx,
                    "mean_dy": obs.mean_dy,
                    "one_body_ipr": obs.one_body_ipr,
                    "iterations": result.diagnostics.iterations,
                    "converged": result.diagnostics.converged,
                }
            )

        best_seed = min(candidates, key=lambda key: candidates[key][0].energy.total)
        best, obs, binding = candidates[best_seed]
        summary.append(
            {
                "size": args.size,
                "U_eV": float(hubbard_u),
                "best_seed": best_seed,
                "best_energy_eV": best.energy.total,
                "binding_vs_2polarons_eV": binding,
                "P_onsite": obs.onsite_probability,
                "P_nn_x": obs.nearest_neighbour_x_probability,
                "P_nn_y": obs.nearest_neighbour_y_probability,
                "mean_r": obs.mean_separation,
                "mean_dx": obs.mean_dx,
                "mean_dy": obs.mean_dy,
                "one_body_ipr": obs.one_body_ipr,
            }
        )
        print(
            f"U={hubbard_u:5.3f} best={best_seed:11s} "
            f"Ebind={binding:+.8f} eV P0={obs.onsite_probability:.4f} "
            f"PNNx={obs.nearest_neighbour_x_probability:.4f} "
            f"PNNy={obs.nearest_neighbour_y_probability:.4f} "
            f"<r>={obs.mean_separation:.3f}"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    branch_path = args.output / "branch_scan.csv"
    summary_path = args.output / "branch_minima.csv"
    json_path = args.output / "branch_scan.json"

    with branch_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    json_path.write_text(json.dumps({"branches": rows, "minima": summary}, indent=2) + "\n")

    print(f"Branches: {branch_path}")
    print(f"Minima: {summary_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
