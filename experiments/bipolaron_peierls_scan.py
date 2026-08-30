"""Scan Peierls stabilization of competing singlet bipolaron minima.

Branches are relaxed independently from onsite, intersite-x, intersite-y, and
separated Holstein seeds. The separated state in the same periodic cell is the
primary finite-size binding reference. Seed labels are never used as physical
state labels: each relaxed state is classified from its final pair observables.

Because the transfer integral is linearized in the bond displacement, the
output also records the largest hopping modulation relative to the bare
transfer integral. This diagnostic identifies parameter regions where a large
energy gain is accompanied by distortions that may lie outside the intended
linear Peierls regime.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from holstein_peierls.lattice import LatticeState
from holstein_peierls.parameters import StaticPolaronParameters
from holstein_peierls.polaron import solve_static_polaron
from holstein_peierls.two_particle.observables import PairObservables, pair_observables
from holstein_peierls.two_particle.parameters import BipolaronParameters
from holstein_peierls.two_particle.peierls import (
    initial_lattice_state,
    relax_static_holstein_peierls_bipolaron,
)


def center_position(size: int) -> int:
    return (size // 2) * size + (size // 2) + 1


def intersite_u(parameters: BipolaronParameters, direction: str) -> np.ndarray:
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


def seed_state(parameters: BipolaronParameters, branch: str) -> LatticeState:
    """Create an unstrained Peierls state with a branch-specific Holstein seed."""
    if branch == "onsite":
        return initial_lattice_state(parameters, "onsite")
    if branch == "separated":
        return initial_lattice_state(parameters, "separated")
    if branch == "intersite_x":
        u = intersite_u(parameters, "x")
    elif branch == "intersite_y":
        u = intersite_u(parameters, "y")
    else:
        raise ValueError(branch)
    return LatticeState(
        u=u,
        vx=np.zeros_like(u),
        vy=np.zeros_like(u),
    )


def classify_final_state(observables: PairObservables) -> str:
    """Return a descriptive label based only on the final pair distribution.

    This is a diagnostic classifier, not a thermodynamic phase definition.
    A state is called separated only when less than 10% of the probability is
    onsite/nearest-neighbour and the mean separation exceeds two lattice sites.
    Localized states are labelled by whichever of onsite, NN-x, or NN-y carries
    the largest probability; ambiguous residual cases are labelled ``mixed``.
    """
    local_probability = (
        observables.onsite_probability + observables.nearest_neighbour_probability
    )
    if local_probability < 0.10 and observables.mean_separation > 2.0:
        return "separated"

    channels = {
        "onsite": observables.onsite_probability,
        "intersite_x": observables.nearest_neighbour_x_probability,
        "intersite_y": observables.nearest_neighbour_y_probability,
    }
    label = max(channels, key=channels.get)
    if channels[label] < 0.25:
        return "mixed"
    return label


def single_polaron_energy(size: int, alpha_x: float, alpha_y: float) -> float:
    """Return the relaxed one-polaron energy with matching Peierls couplings."""
    p = StaticPolaronParameters(
        nx=size,
        ny=size,
        polaron_position=center_position(size),
        alpha_interx=alpha_x,
        alpha_intery=alpha_y,
        max_iterations=1500,
        convergence_criterion=1.0e-7,
    )
    result = solve_static_polaron(
        p,
        solver="sparse",
        gradient_mode="optimized",
        legacy_convergence=False,
    )
    return float(result.total_energy)


def distortion_diagnostics(
    result,
    parameters: BipolaronParameters,
) -> dict[str, float]:
    dx = np.roll(result.vx, shift=-1, axis=1) - result.vx
    dy = np.roll(result.vy, shift=-1, axis=0) - result.vy
    max_dx = float(np.max(np.abs(dx)))
    max_dy = float(np.max(np.abs(dy)))
    max_dtx = abs(parameters.alpha_interx) * max_dx
    max_dty = abs(parameters.alpha_intery) * max_dy
    ratio_x = max_dtx / abs(parameters.j0x) if parameters.j0x != 0.0 else np.inf
    ratio_y = max_dty / abs(parameters.j0y) if parameters.j0y != 0.0 else np.inf
    holstein_elastic = 0.5 * parameters.k1 * float(np.sum(np.square(result.u)))
    peierls_elastic = 0.5 * parameters.k2 * float(
        np.sum(np.square(dx)) + np.sum(np.square(dy))
    )
    return {
        "max_abs_delta_vx_A": max_dx,
        "max_abs_delta_vy_A": max_dy,
        "max_abs_delta_tx_eV": max_dtx,
        "max_abs_delta_ty_eV": max_dty,
        "max_delta_tx_over_Jx": ratio_x,
        "max_delta_ty_over_Jy": ratio_y,
        "holstein_elastic_energy_eV": holstein_elastic,
        "peierls_elastic_energy_eV": peierls_elastic,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=6)
    parser.add_argument(
        "--alpha-x-values",
        nargs="+",
        type=float,
        default=[0.0, 0.05, 0.10, 0.20, 0.40],
    )
    parser.add_argument("--alpha-y", type=float, default=0.0)
    parser.add_argument(
        "--u-values",
        nargs="+",
        type=float,
        default=[0.50, 0.525, 0.55, 0.70, 1.00],
    )
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--displacement-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--gradient-tolerance", type=float, default=2.0e-5)
    parser.add_argument("--output", type=Path, default=Path("bipolaron-peierls"))
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    minima: list[dict[str, object]] = []
    branches = ("onsite", "intersite_x", "intersite_y", "separated")
    energy_equivalence_tolerance = 1.0e-8

    for alpha_x in args.alpha_x_values:
        one_polaron = single_polaron_energy(args.size, alpha_x, args.alpha_y)
        single = StaticPolaronParameters(
            nx=args.size,
            ny=args.size,
            polaron_position=center_position(args.size),
            alpha_interx=float(alpha_x),
            alpha_intery=float(args.alpha_y),
        )
        base = BipolaronParameters.from_polaron_parameters(single)
        base = replace(
            base,
            pair_position=center_position(args.size),
            max_iterations=args.max_iterations,
            convergence_criterion=args.displacement_tolerance,
            gradient_convergence_criterion=args.gradient_tolerance,
            eigensolver_tolerance=2.0e-9,
        )

        for hubbard_u in args.u_values:
            p = replace(base, hubbard_u=float(hubbard_u))
            candidates = {}
            for branch in branches:
                result = relax_static_holstein_peierls_bipolaron(
                    p,
                    initial_state=seed_state(p, branch),
                )
                obs = pair_observables(result.ground_state, p)
                diag = distortion_diagnostics(result, p)
                final_state = classify_final_state(obs)
                candidates[branch] = (result, obs, diag, final_state)

            separated_energy = candidates["separated"][0].energy.total
            best_seed = min(
                candidates,
                key=lambda key: candidates[key][0].energy.total,
            )
            best_energy = candidates[best_seed][0].energy.total
            equivalent_best_seeds = [
                seed
                for seed, (result, _, _, _) in candidates.items()
                if abs(result.energy.total - best_energy) <= energy_equivalence_tolerance
            ]

            for branch, (result, obs, diag, final_state) in candidates.items():
                rows.append(
                    {
                        "size": args.size,
                        "alpha_x_eV_per_A": float(alpha_x),
                        "alpha_y_eV_per_A": float(args.alpha_y),
                        "U_eV": float(hubbard_u),
                        "seed": branch,
                        "final_state": final_state,
                        "total_energy_eV": result.energy.total,
                        "separated_branch_energy_eV": separated_energy,
                        "binding_vs_separated_branch_eV": separated_energy
                        - result.energy.total,
                        "binding_vs_2polarons_eV": 2.0 * one_polaron
                        - result.energy.total,
                        "energy_above_best_eV": result.energy.total - best_energy,
                        "P_onsite": obs.onsite_probability,
                        "P_nn": obs.nearest_neighbour_probability,
                        "P_nn_x": obs.nearest_neighbour_x_probability,
                        "P_nn_y": obs.nearest_neighbour_y_probability,
                        "mean_r": obs.mean_separation,
                        "rms_r": obs.rms_separation,
                        "one_body_ipr": obs.one_body_ipr,
                        **diag,
                        "iterations": result.diagnostics.iterations,
                        "converged": result.diagnostics.converged,
                        "final_max_update_A": result.diagnostics.final_max_update,
                        "final_max_gradient_eV_per_A": result.diagnostics.final_max_gradient,
                    }
                )

            best, obs, diag, best_final_state = candidates[best_seed]
            minima.append(
                {
                    "size": args.size,
                    "alpha_x_eV_per_A": float(alpha_x),
                    "alpha_y_eV_per_A": float(args.alpha_y),
                    "U_eV": float(hubbard_u),
                    "best_seed": best_seed,
                    "best_final_state": best_final_state,
                    "equivalent_best_seeds": ";".join(equivalent_best_seeds),
                    "best_energy_eV": best.energy.total,
                    "separated_branch_energy_eV": separated_energy,
                    "best_binding_vs_separated_branch_eV": separated_energy
                    - best.energy.total,
                    "binding_vs_2polarons_eV": 2.0 * one_polaron
                    - best.energy.total,
                    "P_onsite": obs.onsite_probability,
                    "P_nn": obs.nearest_neighbour_probability,
                    "P_nn_x": obs.nearest_neighbour_x_probability,
                    "P_nn_y": obs.nearest_neighbour_y_probability,
                    "mean_r": obs.mean_separation,
                    "rms_r": obs.rms_separation,
                    "one_body_ipr": obs.one_body_ipr,
                    **diag,
                    "best_final_max_gradient_eV_per_A": best.diagnostics.final_max_gradient,
                    "max_branch_gradient_eV_per_A": max(
                        result.diagnostics.final_max_gradient
                        for result, _, _, _ in candidates.values()
                    ),
                    "all_branches_converged": all(
                        result.diagnostics.converged
                        for result, _, _, _ in candidates.values()
                    ),
                }
            )
            print(
                f"alpha_x={alpha_x:.3f} U={hubbard_u:.4f} "
                f"state={best_final_state:11s} seed={best_seed:11s} "
                f"Ebind_sep={separated_energy - best.energy.total:+.7f} eV "
                f"P0={obs.onsite_probability:.3f} "
                f"PNNx={obs.nearest_neighbour_x_probability:.3f} "
                f"dtx/Jx={diag['max_delta_tx_over_Jx']:.3f} "
                f"gmax={best.diagnostics.final_max_gradient:.2e} eV/A"
            )

    args.output.mkdir(parents=True, exist_ok=True)
    branch_path = args.output / "peierls_branch_scan.csv"
    minima_path = args.output / "peierls_branch_minima.csv"
    json_path = args.output / "peierls_branch_scan.json"

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
