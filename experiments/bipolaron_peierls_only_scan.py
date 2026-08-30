"""Probe pure Peierls self-trapping and two-carrier pairing.

This experiment sets the intramolecular Holstein coupling to zero and seeds
localized intermolecular distortions explicitly. The zero-distortion state is
also included, so the scan can distinguish genuine Peierls self-trapping from
an artefact of the symmetry-breaking seed.

For every coupling point the one-polaron reference is relaxed from several
Peierls seeds and the lowest result is used in the two-polaron dissociation
energy 2 E_1 - E_2. The two-particle solver is then relaxed from bond-centred
and separated seeds. Final-state labels are assigned from pair observables, not
from the seed name.
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
from holstein_peierls.polaron import PolaronResult, solve_static_polaron
from holstein_peierls.two_particle.observables import PairObservables, pair_observables
from holstein_peierls.two_particle.parameters import BipolaronParameters
from holstein_peierls.two_particle.peierls import (
    HolsteinPeierlsBipolaronResult,
    relax_static_holstein_peierls_bipolaron,
)


def center_position(size: int) -> int:
    return (size // 2) * size + (size // 2) + 1


def center_index(size: int) -> int:
    return center_position(size) - 1


def opposite_index(size: int) -> int:
    cy, cx = divmod(center_index(size), size)
    return ((cy + size // 2) % size) * size + ((cx + size // 2) % size)


def add_bond_seed(
    state: LatticeState,
    *,
    center: int,
    direction: str,
    amplitude: float,
) -> None:
    """Add a local coordinate contrast across one selected bond.

    ``amplitude`` is the direct coordinate difference introduced across the
    central bond. Neighbouring bond differences follow from the fact that the
    Peierls variables are molecular coordinates rather than independent bond
    variables.
    """
    ny, nx = state.shape
    cy, cx = divmod(center, nx)
    half = 0.5 * amplitude
    if direction == "x":
        state.vx[cy, cx] -= half
        state.vx[cy, (cx + 1) % nx] += half
    elif direction == "y":
        state.vy[cy, cx] -= half
        state.vy[(cy + 1) % ny, cx] += half
    else:
        raise ValueError(direction)


def peierls_seed(
    size: int,
    mode: str,
    amplitude: float,
) -> LatticeState:
    state = LatticeState.zeros(size, size)
    if mode == "zero":
        return state

    centers = [center_index(size)]
    if mode.startswith("separated_"):
        centers.append(opposite_index(size))
        mode = mode.removeprefix("separated_")

    if mode == "x":
        directions = ("x",)
    elif mode == "y":
        directions = ("y",)
    elif mode == "xy":
        directions = ("x", "y")
    else:
        raise ValueError(mode)

    for site in centers:
        for direction in directions:
            add_bond_seed(
                state,
                center=site,
                direction=direction,
                amplitude=amplitude,
            )
    return state


def classify_final_state(observables: PairObservables) -> str:
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


def distortion_diagnostics(
    state: LatticeState,
    *,
    alpha_x: float,
    alpha_y: float,
    j0x: float,
    j0y: float,
) -> dict[str, float]:
    dx = np.roll(state.vx, shift=-1, axis=1) - state.vx
    dy = np.roll(state.vy, shift=-1, axis=0) - state.vy
    max_dx = float(np.max(np.abs(dx)))
    max_dy = float(np.max(np.abs(dy)))
    max_dtx = abs(alpha_x) * max_dx
    max_dty = abs(alpha_y) * max_dy
    return {
        "max_abs_delta_vx_A": max_dx,
        "max_abs_delta_vy_A": max_dy,
        "max_abs_delta_tx_eV": max_dtx,
        "max_abs_delta_ty_eV": max_dty,
        "max_delta_tx_over_Jx": max_dtx / abs(j0x) if j0x else np.inf,
        "max_delta_ty_over_Jy": max_dty / abs(j0y) if j0y else np.inf,
    }


def one_polaron_reference(
    parameters: StaticPolaronParameters,
    seed_amplitudes: list[float],
) -> tuple[str, float, PolaronResult, dict[str, float]]:
    candidates: list[tuple[str, float, PolaronResult, dict[str, float]]] = []
    modes = ("zero", "x", "y", "xy")
    for mode in modes:
        amplitudes = [0.0] if mode == "zero" else seed_amplitudes
        for amplitude in amplitudes:
            seed = peierls_seed(parameters.nx, mode, amplitude)
            result = solve_static_polaron(
                parameters,
                initial_state=seed,
                solver="sparse",
                gradient_mode="optimized",
                legacy_convergence=False,
                apply_legacy_seed=False,
            )
            diag = distortion_diagnostics(
                result.state,
                alpha_x=parameters.alpha_interx,
                alpha_y=parameters.alpha_intery,
                j0x=parameters.j0x,
                j0y=parameters.j0y,
            )
            candidates.append((mode, amplitude, result, diag))
    return min(candidates, key=lambda item: item[2].total_energy)


def pair_candidates(
    parameters: BipolaronParameters,
    seed_amplitudes: list[float],
) -> list[
    tuple[
        str,
        float,
        HolsteinPeierlsBipolaronResult,
        PairObservables,
        dict[str, float],
        str,
    ]
]:
    candidates = []
    modes = ("zero", "x", "y", "xy", "separated_x", "separated_y", "separated_xy")
    for mode in modes:
        amplitudes = [0.0] if mode == "zero" else seed_amplitudes
        for amplitude in amplitudes:
            seed = peierls_seed(parameters.nx, mode, amplitude)
            result = relax_static_holstein_peierls_bipolaron(
                parameters,
                initial_state=seed,
            )
            obs = pair_observables(result.ground_state, parameters)
            diag = distortion_diagnostics(
                result.lattice,
                alpha_x=parameters.alpha_interx,
                alpha_y=parameters.alpha_intery,
                j0x=parameters.j0x,
                j0y=parameters.j0y,
            )
            candidates.append(
                (mode, amplitude, result, obs, diag, classify_final_state(obs))
            )
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=6)
    parser.add_argument(
        "--alpha-x-values",
        nargs="+",
        type=float,
        default=[0.10, 0.20, 0.30, 0.40],
    )
    parser.add_argument("--alpha-y", type=float, default=0.0)
    parser.add_argument(
        "--u-values",
        nargs="+",
        type=float,
        default=[0.0, 0.5, 1.0],
    )
    parser.add_argument(
        "--seed-amplitudes",
        nargs="+",
        type=float,
        default=[0.02, 0.05],
    )
    parser.add_argument("--max-iterations", type=int, default=1200)
    parser.add_argument("--displacement-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--gradient-tolerance", type=float, default=2.0e-5)
    parser.add_argument("--output", type=Path, default=Path("bipolaron-peierls-only"))
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    minima: list[dict[str, object]] = []
    one_rows: list[dict[str, object]] = []

    for alpha_x in args.alpha_x_values:
        single_parameters = StaticPolaronParameters(
            nx=args.size,
            ny=args.size,
            polaron_position=center_position(args.size),
            alpha_intra=0.0,
            alpha_interx=float(alpha_x),
            alpha_intery=float(args.alpha_y),
            max_iterations=args.max_iterations,
            convergence_criterion=args.displacement_tolerance,
        )
        single_mode, single_amp, single_result, single_diag = one_polaron_reference(
            single_parameters,
            [float(value) for value in args.seed_amplitudes],
        )
        one_rows.append(
            {
                "size": args.size,
                "alpha_x_eV_per_A": float(alpha_x),
                "alpha_y_eV_per_A": float(args.alpha_y),
                "best_seed": single_mode,
                "seed_amplitude_A": single_amp,
                "energy_eV": single_result.total_energy,
                "ipr": single_result.ipr,
                "iterations": single_result.diagnostics.iterations,
                "converged_u": single_result.diagnostics.converged_u,
                "converged_vx": single_result.diagnostics.converged_vx,
                "converged_vy": single_result.diagnostics.converged_vy,
                **single_diag,
            }
        )

        base = BipolaronParameters.from_polaron_parameters(single_parameters)
        base = replace(
            base,
            pair_position=center_position(args.size),
            gradient_convergence_criterion=args.gradient_tolerance,
            eigensolver_tolerance=2.0e-9,
        )

        print(
            f"alpha_x={alpha_x:.3f} single_seed={single_mode:4s} "
            f"E1={single_result.total_energy:+.8f} eV "
            f"IPR={single_result.ipr:.5f} "
            f"dtx/Jx={single_diag['max_delta_tx_over_Jx']:.3f}"
        )

        for hubbard_u in args.u_values:
            parameters = replace(base, hubbard_u=float(hubbard_u))
            candidates = pair_candidates(
                parameters,
                [float(value) for value in args.seed_amplitudes],
            )
            best = min(candidates, key=lambda item: item[2].energy.total)
            best_energy = best[2].energy.total
            binding_two_polarons = 2.0 * single_result.total_energy - best_energy

            for mode, amplitude, result, obs, diag, final_state in candidates:
                rows.append(
                    {
                        "size": args.size,
                        "alpha_x_eV_per_A": float(alpha_x),
                        "alpha_y_eV_per_A": float(args.alpha_y),
                        "U_eV": float(hubbard_u),
                        "seed": mode,
                        "seed_amplitude_A": amplitude,
                        "final_state": final_state,
                        "total_energy_eV": result.energy.total,
                        "energy_above_best_eV": result.energy.total - best_energy,
                        "binding_vs_2polarons_eV": 2.0 * single_result.total_energy
                        - result.energy.total,
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

            mode, amplitude, result, obs, diag, final_state = best
            minima.append(
                {
                    "size": args.size,
                    "alpha_x_eV_per_A": float(alpha_x),
                    "alpha_y_eV_per_A": float(args.alpha_y),
                    "U_eV": float(hubbard_u),
                    "best_seed": mode,
                    "best_seed_amplitude_A": amplitude,
                    "best_final_state": final_state,
                    "best_energy_eV": best_energy,
                    "single_polaron_energy_eV": single_result.total_energy,
                    "binding_vs_2polarons_eV": binding_two_polarons,
                    "P_onsite": obs.onsite_probability,
                    "P_nn": obs.nearest_neighbour_probability,
                    "P_nn_x": obs.nearest_neighbour_x_probability,
                    "P_nn_y": obs.nearest_neighbour_y_probability,
                    "mean_r": obs.mean_separation,
                    "rms_r": obs.rms_separation,
                    "one_body_ipr": obs.one_body_ipr,
                    **diag,
                    "converged": result.diagnostics.converged,
                    "final_max_gradient_eV_per_A": result.diagnostics.final_max_gradient,
                }
            )
            print(
                f"  U={hubbard_u:.3f} state={final_state:11s} "
                f"seed={mode:12s} Ebind_2p={binding_two_polarons:+.7f} eV "
                f"P0={obs.onsite_probability:.3f} "
                f"PNNx={obs.nearest_neighbour_x_probability:.3f} "
                f"dtx/Jx={diag['max_delta_tx_over_Jx']:.3f}"
            )

    args.output.mkdir(parents=True, exist_ok=True)
    one_path = args.output / "peierls_only_single_polaron.csv"
    branch_path = args.output / "peierls_only_pair_branches.csv"
    minima_path = args.output / "peierls_only_pair_minima.csv"
    json_path = args.output / "peierls_only_scan.json"

    with one_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(one_rows[0].keys()))
        writer.writeheader()
        writer.writerows(one_rows)
    with branch_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with minima_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(minima[0].keys()))
        writer.writeheader()
        writer.writerows(minima)
    json_path.write_text(
        json.dumps({"single_polaron": one_rows, "branches": rows, "minima": minima}, indent=2)
        + "\n"
    )

    print(f"Single polarons: {one_path}")
    print(f"Pair branches: {branch_path}")
    print(f"Pair minima: {minima_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
