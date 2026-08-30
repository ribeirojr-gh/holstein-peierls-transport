"""Scan screened long-range repulsion for competing static bipolaron branches."""

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
from holstein_peierls.two_particle.interaction import (
    interaction_expectation,
    minimum_image_offsets,
    pair_interaction_matrix,
)
from holstein_peierls.two_particle.observables import PairObservables, pair_observables
from holstein_peierls.two_particle.parameters import BipolaronParameters
from holstein_peierls.two_particle.peierls import (
    initial_lattice_state,
    relax_static_holstein_peierls_bipolaron,
)

BRANCHES = ("onsite", "intersite_x", "intersite_y", "diagonal", "separated")


def center_position(size: int) -> int:
    return (size // 2) * size + (size // 2) + 1


def two_site_seed(
    parameters: BipolaronParameters,
    offset_y: int,
    offset_x: int,
) -> LatticeState:
    u = np.zeros((parameters.ny, parameters.nx), dtype=float)
    cy, cx = divmod(parameters.pair_index, parameters.nx)
    second = ((cy + offset_y) % parameters.ny, (cx + offset_x) % parameters.nx)
    displacement = -parameters.alpha_intra / parameters.k1
    u[cy, cx] = displacement
    u[second] = displacement
    return LatticeState(u=u, vx=np.zeros_like(u), vy=np.zeros_like(u))


def seed_state(parameters: BipolaronParameters, branch: str) -> LatticeState:
    if branch in ("onsite", "separated"):
        return initial_lattice_state(parameters, branch)
    if branch == "intersite_x":
        return two_site_seed(parameters, 0, 1)
    if branch == "intersite_y":
        return two_site_seed(parameters, 1, 0)
    if branch == "diagonal":
        return two_site_seed(parameters, 1, 1)
    raise ValueError(branch)


def diagonal_probability(
    wavefunction: np.ndarray,
    parameters: BipolaronParameters,
) -> float:
    dx, dy = minimum_image_offsets(parameters)
    diagonal = (dx == 1) & (dy == 1)
    return float(np.sum(np.square(wavefunction)[diagonal]))


def classify_final_state(observables: PairObservables, p_diagonal: float) -> str:
    local = (
        observables.onsite_probability
        + observables.nearest_neighbour_probability
        + p_diagonal
    )
    if local < 0.10 and observables.mean_separation > 2.0:
        return "separated"
    channels = {
        "onsite": observables.onsite_probability,
        "intersite_x": observables.nearest_neighbour_x_probability,
        "intersite_y": observables.nearest_neighbour_y_probability,
        "diagonal": p_diagonal,
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
        "--epsilon-values",
        nargs="+",
        type=float,
        default=[3.0, 10.0, 30.0, 100.0, 300.0],
    )
    parser.add_argument("--ax", type=float, default=7.0)
    parser.add_argument("--ay", type=float, default=7.0)
    parser.add_argument("--v1", type=float, default=0.0)
    parser.add_argument(
        "--branches", nargs="+", choices=BRANCHES, default=list(BRANCHES)
    )
    parser.add_argument("--j0x", type=float, default=0.0575)
    parser.add_argument("--j0y", type=float, default=0.0575)
    parser.add_argument("--alpha-x", type=float, default=0.10)
    parser.add_argument("--alpha-y", type=float, default=0.10)
    parser.add_argument("--max-iterations", type=int, default=1200)
    parser.add_argument("--output", type=Path, default=Path("bipolaron-long-range"))
    args = parser.parse_args()

    single = StaticPolaronParameters(
        nx=args.size,
        ny=args.size,
        polaron_position=center_position(args.size),
        j0x=args.j0x,
        j0y=args.j0y,
        alpha_interx=args.alpha_x,
        alpha_intery=args.alpha_y,
    )
    polaron = solve_static_polaron(
        single,
        solver="sparse",
        gradient_mode="optimized",
        legacy_convergence=False,
    )
    two_polaron_reference = 2.0 * polaron.total_energy

    base = BipolaronParameters.from_polaron_parameters(
        single,
        long_range_coulomb=True,
        lattice_spacing_x_angstrom=args.ax,
        lattice_spacing_y_angstrom=args.ay,
        relative_permittivity=args.epsilon_values[0],
    )
    base = replace(
        base,
        pair_position=center_position(args.size),
        nearest_neighbor_v=args.v1,
        max_iterations=args.max_iterations,
        convergence_criterion=1.0e-7,
        gradient_convergence_criterion=1.0e-6,
        eigensolver_tolerance=2.0e-10,
    )

    rows: list[dict[str, object]] = []
    minima: list[dict[str, object]] = []

    for hubbard_u in args.u_values:
        for epsilon_r in args.epsilon_values:
            parameters = replace(
                base,
                hubbard_u=float(hubbard_u),
                relative_permittivity=float(epsilon_r),
            )
            interaction = pair_interaction_matrix(parameters)
            center = parameters.pair_index
            cy, cx = divmod(center, parameters.nx)
            right = cy * parameters.nx + (cx + 1) % parameters.nx
            down = ((cy + 1) % parameters.ny) * parameters.nx + cx
            diagonal = ((cy + 1) % parameters.ny) * parameters.nx + (cx + 1) % parameters.nx
            scales = {
                "Vx_eV": float(interaction[center, right]),
                "Vy_eV": float(interaction[center, down]),
                "Vdiag_eV": float(interaction[center, diagonal]),
            }

            candidates = {}
            for branch in args.branches:
                result = relax_static_holstein_peierls_bipolaron(
                    parameters,
                    initial_state=seed_state(parameters, branch),
                )
                obs = pair_observables(result.ground_state, parameters)
                pdiag = diagonal_probability(result.ground_state.wavefunction, parameters)
                ratio_x, ratio_y = distortion_ratios(result, parameters)
                final_state = classify_final_state(obs, pdiag)
                eint = interaction_expectation(result.ground_state.wavefunction, parameters)
                candidates[branch] = (
                    result,
                    obs,
                    pdiag,
                    final_state,
                    ratio_x,
                    ratio_y,
                    eint,
                )

            best_seed = min(candidates, key=lambda key: candidates[key][0].energy.total)
            best_energy = candidates[best_seed][0].energy.total
            separated_energy = (
                candidates["separated"][0].energy.total
                if "separated" in candidates
                else np.nan
            )

            for seed, (result, obs, pdiag, final_state, ratio_x, ratio_y, eint) in candidates.items():
                rows.append(
                    {
                        "size": args.size,
                        "U_eV": float(hubbard_u),
                        "epsilon_r": float(epsilon_r),
                        "ax_A": args.ax,
                        "ay_A": args.ay,
                        "V1_override_eV": args.v1,
                        **scales,
                        "seed": seed,
                        "final_state": final_state,
                        "total_energy_eV": result.energy.total,
                        "binding_vs_2polaron_eV": two_polaron_reference - result.energy.total,
                        "binding_vs_finite_separated_eV": separated_energy - result.energy.total,
                        "energy_above_best_eV": result.energy.total - best_energy,
                        "interaction_energy_eV": eint,
                        "P_onsite": obs.onsite_probability,
                        "P_nn_x": obs.nearest_neighbour_x_probability,
                        "P_nn_y": obs.nearest_neighbour_y_probability,
                        "P_diagonal": pdiag,
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

            result, obs, pdiag, final_state, ratio_x, ratio_y, eint = candidates[best_seed]
            minima.append(
                {
                    "size": args.size,
                    "U_eV": float(hubbard_u),
                    "epsilon_r": float(epsilon_r),
                    "ax_A": args.ax,
                    "ay_A": args.ay,
                    "V1_override_eV": args.v1,
                    **scales,
                    "one_polaron_energy_eV": polaron.total_energy,
                    "two_polaron_reference_eV": two_polaron_reference,
                    "best_seed": best_seed,
                    "best_final_state": final_state,
                    "best_energy_eV": result.energy.total,
                    "binding_vs_2polaron_eV": two_polaron_reference - result.energy.total,
                    "finite_separated_energy_eV": separated_energy,
                    "finite_separated_excess_eV": separated_energy - two_polaron_reference,
                    "interaction_energy_eV": eint,
                    "P_onsite": obs.onsite_probability,
                    "P_nn_x": obs.nearest_neighbour_x_probability,
                    "P_nn_y": obs.nearest_neighbour_y_probability,
                    "P_diagonal": pdiag,
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
                f"U={hubbard_u:.3f} eps={epsilon_r:7.2f} "
                f"Vnn={scales['Vx_eV']:.6f} state={final_state:11s} "
                f"Ebind(2P)={two_polaron_reference - result.energy.total:+.8f} eV "
                f"Esep-2P={separated_energy - two_polaron_reference:+.8f} eV"
            )

    args.output.mkdir(parents=True, exist_ok=True)
    branch_path = args.output / "long_range_branches.csv"
    minima_path = args.output / "long_range_minima.csv"
    json_path = args.output / "long_range_scan.json"

    with branch_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with minima_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(minima[0].keys()))
        writer.writeheader()
        writer.writerows(minima)
    json_path.write_text(json.dumps({"branches": rows, "minima": minima}, indent=2) + "\n")

    print(f"One-polaron energy: {polaron.total_energy:.12f} eV")
    print(f"Two-polaron reference: {two_polaron_reference:.12f} eV")
    print(f"Branches: {branch_path}")
    print(f"Minima: {minima_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
