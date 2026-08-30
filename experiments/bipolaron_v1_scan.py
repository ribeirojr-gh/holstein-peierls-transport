"""Scan nearest-neighbour repulsion for competing static bipolaron branches."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from holstein_peierls.lattice import LatticeState
from holstein_peierls.parameters import StaticPolaronParameters
from holstein_peierls.two_particle.interaction import interaction_expectation
from holstein_peierls.two_particle.observables import PairObservables, pair_observables
from holstein_peierls.two_particle.parameters import BipolaronParameters
from holstein_peierls.two_particle.peierls import (
    initial_lattice_state,
    relax_static_holstein_peierls_bipolaron,
)

BRANCHES = ("onsite", "intersite_x", "intersite_y", "separated")


def center_position(size: int) -> int:
    return (size // 2) * size + (size // 2) + 1


def intersite_u(parameters: BipolaronParameters, direction: str) -> np.ndarray:
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


def seed_state(parameters: BipolaronParameters, branch: str) -> LatticeState:
    if branch in ("onsite", "separated"):
        return initial_lattice_state(parameters, branch)
    if branch == "intersite_x":
        u = intersite_u(parameters, "x")
    elif branch == "intersite_y":
        u = intersite_u(parameters, "y")
    else:
        raise ValueError(branch)
    return LatticeState(u=u, vx=np.zeros_like(u), vy=np.zeros_like(u))


def classify_final_state(observables: PairObservables) -> str:
    local = observables.onsite_probability + observables.nearest_neighbour_probability
    if local < 0.10 and observables.mean_separation > 2.0:
        return "separated"
    channels = {
        "onsite": observables.onsite_probability,
        "intersite_x": observables.nearest_neighbour_x_probability,
        "intersite_y": observables.nearest_neighbour_y_probability,
    }
    label = max(channels, key=channels.get)
    return label if channels[label] >= 0.25 else "mixed"


def distortion_ratios(result, parameters: BipolaronParameters) -> tuple[float, float]:
    dx = np.roll(result.vx, -1, axis=1) - result.vx
    dy = np.roll(result.vy, -1, axis=0) - result.vy
    ratio_x = abs(parameters.alpha_interx) * float(np.max(np.abs(dx))) / abs(parameters.j0x)
    ratio_y = abs(parameters.alpha_intery) * float(np.max(np.abs(dy))) / abs(parameters.j0y)
    return ratio_x, ratio_y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--u-values", nargs="+", type=float, default=[0.525, 1.0])
    parser.add_argument(
        "--v1-values",
        nargs="+",
        type=float,
        default=[0.0, 0.005, 0.010, 0.020, 0.040, 0.060, 0.080],
    )
    parser.add_argument(
        "--branches",
        nargs="+",
        choices=BRANCHES,
        default=list(BRANCHES),
        help="Branches to relax; separated must be included for binding energies.",
    )
    parser.add_argument("--alpha-x", type=float, default=0.10)
    parser.add_argument("--alpha-y", type=float, default=0.12)
    parser.add_argument("--max-iterations", type=int, default=1200)
    parser.add_argument("--output", type=Path, default=Path("bipolaron-v1"))
    args = parser.parse_args()

    if "separated" not in args.branches:
        parser.error("--branches must include separated")

    single = StaticPolaronParameters(
        nx=args.size,
        ny=args.size,
        polaron_position=center_position(args.size),
        alpha_interx=args.alpha_x,
        alpha_intery=args.alpha_y,
    )
    base = BipolaronParameters.from_polaron_parameters(single)
    base = replace(
        base,
        pair_position=center_position(args.size),
        max_iterations=args.max_iterations,
        convergence_criterion=1.0e-7,
        gradient_convergence_criterion=1.0e-6,
        eigensolver_tolerance=2.0e-10,
    )

    rows: list[dict[str, object]] = []
    minima: list[dict[str, object]] = []
    branches = tuple(args.branches)

    for hubbard_u in args.u_values:
        for v1 in args.v1_values:
            parameters = replace(
                base,
                hubbard_u=float(hubbard_u),
                nearest_neighbor_v=float(v1),
            )
            candidates = {}
            for branch in branches:
                result = relax_static_holstein_peierls_bipolaron(
                    parameters,
                    initial_state=seed_state(parameters, branch),
                )
                obs = pair_observables(result.ground_state, parameters)
                ratio_x, ratio_y = distortion_ratios(result, parameters)
                final_state = classify_final_state(obs)
                eint = interaction_expectation(result.ground_state.wavefunction, parameters)
                candidates[branch] = (result, obs, final_state, ratio_x, ratio_y, eint)

            separated_energy = candidates["separated"][0].energy.total
            best_seed = min(candidates, key=lambda key: candidates[key][0].energy.total)
            best_energy = candidates[best_seed][0].energy.total

            for seed, (result, obs, final_state, ratio_x, ratio_y, eint) in candidates.items():
                rows.append(
                    {
                        "size": args.size,
                        "U_eV": float(hubbard_u),
                        "V1_eV": float(v1),
                        "seed": seed,
                        "final_state": final_state,
                        "total_energy_eV": result.energy.total,
                        "binding_vs_separated_eV": separated_energy - result.energy.total,
                        "energy_above_best_eV": result.energy.total - best_energy,
                        "interaction_energy_eV": eint,
                        "P_onsite": obs.onsite_probability,
                        "P_nn": obs.nearest_neighbour_probability,
                        "P_nn_x": obs.nearest_neighbour_x_probability,
                        "P_nn_y": obs.nearest_neighbour_y_probability,
                        "mean_r": obs.mean_separation,
                        "rms_r": obs.rms_separation,
                        "one_body_ipr": obs.one_body_ipr,
                        "max_delta_tx_over_Jx": ratio_x,
                        "max_delta_ty_over_Jy": ratio_y,
                        "iterations": result.diagnostics.iterations,
                        "converged": result.diagnostics.converged,
                        "final_max_gradient_eV_per_A": result.diagnostics.final_max_gradient,
                    }
                )

            result, obs, final_state, ratio_x, ratio_y, eint = candidates[best_seed]
            minima.append(
                {
                    "size": args.size,
                    "U_eV": float(hubbard_u),
                    "V1_eV": float(v1),
                    "best_seed": best_seed,
                    "best_final_state": final_state,
                    "best_energy_eV": result.energy.total,
                    "separated_energy_eV": separated_energy,
                    "binding_vs_separated_eV": separated_energy - result.energy.total,
                    "interaction_energy_eV": eint,
                    "P_onsite": obs.onsite_probability,
                    "P_nn": obs.nearest_neighbour_probability,
                    "P_nn_x": obs.nearest_neighbour_x_probability,
                    "P_nn_y": obs.nearest_neighbour_y_probability,
                    "mean_r": obs.mean_separation,
                    "one_body_ipr": obs.one_body_ipr,
                    "max_delta_tx_over_Jx": ratio_x,
                    "max_delta_ty_over_Jy": ratio_y,
                    "all_branches_converged": all(
                        candidate[0].diagnostics.converged for candidate in candidates.values()
                    ),
                }
            )
            print(
                f"U={hubbard_u:.3f} V1={v1:.4f} state={final_state:11s} "
                f"Ebind={separated_energy - result.energy.total:+.8f} eV "
                f"P0={obs.onsite_probability:.3f} PNNx={obs.nearest_neighbour_x_probability:.3f}"
            )

    args.output.mkdir(parents=True, exist_ok=True)
    branch_path = args.output / "v1_branches.csv"
    minima_path = args.output / "v1_minima.csv"
    json_path = args.output / "v1_scan.json"

    with branch_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with minima_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(minima[0].keys()))
        writer.writeheader()
        writer.writerows(minima)
    json_path.write_text(json.dumps({"branches": rows, "minima": minima}, indent=2) + "\n")

    print(f"Branches: {branch_path}")
    print(f"Minima: {minima_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
