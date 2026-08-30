"""Relax one selected Holstein-Peierls bipolaron branch for size scaling."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from holstein_peierls.lattice import LatticeState
from holstein_peierls.parameters import StaticPolaronParameters
from holstein_peierls.two_particle.observables import pair_observables
from holstein_peierls.two_particle.parameters import BipolaronParameters
from holstein_peierls.two_particle.peierls import relax_static_holstein_peierls_bipolaron


def center_position(size: int) -> int:
    return (size // 2) * size + (size // 2) + 1


def branch_seed(parameters: BipolaronParameters, branch: str) -> LatticeState:
    u = np.zeros((parameters.ny, parameters.nx), dtype=float)
    cy, cx = divmod(parameters.pair_index, parameters.nx)
    displacement = -parameters.alpha_intra / parameters.k1
    if branch == "intersite_x":
        u[cy, cx] = displacement
        u[cy, (cx + 1) % parameters.nx] = displacement
    elif branch == "separated":
        u[cy, cx] = displacement
        u[(cy + parameters.ny // 2) % parameters.ny,
          (cx + parameters.nx // 2) % parameters.nx] = displacement
    else:
        raise ValueError(f"unsupported branch: {branch}")
    return LatticeState(u=u, vx=np.zeros_like(u), vy=np.zeros_like(u))


def distortion_ratios(result, parameters: BipolaronParameters) -> tuple[float, float]:
    dx = np.roll(result.vx, shift=-1, axis=1) - result.vx
    dy = np.roll(result.vy, shift=-1, axis=0) - result.vy
    ratio_x = abs(parameters.alpha_interx) * float(np.max(np.abs(dx))) / abs(parameters.j0x)
    ratio_y = abs(parameters.alpha_intery) * float(np.max(np.abs(dy))) / abs(parameters.j0y)
    return ratio_x, ratio_y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--u", type=float, required=True)
    parser.add_argument("--branch", choices=("intersite_x", "separated"), required=True)
    parser.add_argument("--alpha-x", type=float, default=0.10)
    parser.add_argument("--alpha-y", type=float, default=0.12)
    parser.add_argument("--max-iterations", type=int, default=1200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    single = StaticPolaronParameters(
        nx=args.size,
        ny=args.size,
        polaron_position=center_position(args.size),
        alpha_interx=args.alpha_x,
        alpha_intery=args.alpha_y,
    )
    parameters = BipolaronParameters.from_polaron_parameters(single, hubbard_u=args.u)
    parameters = replace(
        parameters,
        pair_position=center_position(args.size),
        max_iterations=args.max_iterations,
        convergence_criterion=1.0e-6,
        gradient_convergence_criterion=2.0e-5,
        eigensolver_tolerance=2.0e-9,
    )

    result = relax_static_holstein_peierls_bipolaron(
        parameters,
        initial_state=branch_seed(parameters, args.branch),
    )
    obs = pair_observables(result.ground_state, parameters)
    ratio_x, ratio_y = distortion_ratios(result, parameters)
    record = {
        "size": args.size,
        "alpha_x_eV_per_A": args.alpha_x,
        "alpha_y_eV_per_A": args.alpha_y,
        "U_eV": args.u,
        "branch": args.branch,
        "total_energy_eV": result.energy.total,
        "P_onsite": obs.onsite_probability,
        "P_nn_x": obs.nearest_neighbour_x_probability,
        "P_nn_y": obs.nearest_neighbour_y_probability,
        "mean_r": obs.mean_separation,
        "rms_r": obs.rms_separation,
        "one_body_ipr": obs.one_body_ipr,
        "max_delta_tx_over_Jx": ratio_x,
        "max_delta_ty_over_Jy": ratio_y,
        "iterations": result.diagnostics.iterations,
        "converged": result.diagnostics.converged,
        "final_max_update_A": result.diagnostics.final_max_update,
        "final_max_gradient_eV_per_A": result.diagnostics.final_max_gradient,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / "branch_result.json"
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, sort_keys=True))
    print(f"JSON: {output}")


if __name__ == "__main__":
    main()
