"""Large-cell branch benchmark with a branch-consistent initial Krylov vector.

The physical Hamiltonian, RPROP update, convergence tolerances, and eigensolver
are identical to the experimental two-particle solver. The only numerical
specialization is the first ``eigsh`` starting vector, chosen to match the
onsite, intersite-x, intersite-y, diagonal, or separated branch under
validation. Subsequent solves reuse the preceding converged eigenvector.

The optional long-range mode uses the same frozen-equilibrium minimum-image
Coulomb interaction as the research solver and records binding relative to two
independently relaxed polarons.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from holstein_peierls.lattice import LatticeState
from holstein_peierls.parameters import StaticPolaronParameters
from holstein_peierls.polaron import solve_static_polaron
from holstein_peierls.two_particle.bipolaron import BipolaronRelaxationDiagnostics
from holstein_peierls.two_particle.interaction import (
    interaction_expectation,
    minimum_image_offsets,
    pair_interaction_matrix,
)
from holstein_peierls.two_particle.observables import pair_observables
from holstein_peierls.two_particle.parameters import BipolaronParameters
from holstein_peierls.two_particle.peierls import (
    HolsteinPeierlsBipolaronResult,
    _rprop_update,
    energy_gradient,
    solve_holstein_peierls_ground_state,
    total_energy,
)

BRANCHES = ("onsite", "intersite_x", "intersite_y", "diagonal", "separated")


def center_position(size: int) -> int:
    return (size // 2) * size + (size // 2) + 1


def branch_indices(parameters: BipolaronParameters, branch: str) -> tuple[int, int]:
    first = parameters.pair_index
    cy, cx = divmod(first, parameters.nx)
    if branch == "onsite":
        second = first
    elif branch == "intersite_x":
        second = cy * parameters.nx + (cx + 1) % parameters.nx
    elif branch == "intersite_y":
        second = ((cy + 1) % parameters.ny) * parameters.nx + cx
    elif branch == "diagonal":
        second = (
            ((cy + 1) % parameters.ny) * parameters.nx
            + (cx + 1) % parameters.nx
        )
    elif branch == "separated":
        second = (
            ((cy + parameters.ny // 2) % parameters.ny) * parameters.nx
            + (cx + parameters.nx // 2) % parameters.nx
        )
    else:
        raise ValueError(f"unsupported branch: {branch}")
    return first, second


def branch_seed(parameters: BipolaronParameters, branch: str) -> LatticeState:
    u = np.zeros((parameters.ny, parameters.nx), dtype=float)
    first, second = branch_indices(parameters, branch)
    if first == second:
        y, x = divmod(first, parameters.nx)
        u[y, x] = -2.0 * parameters.alpha_intra / parameters.k1
    else:
        displacement = -parameters.alpha_intra / parameters.k1
        for site in (first, second):
            y, x = divmod(site, parameters.nx)
            u[y, x] = displacement
    return LatticeState(u=u, vx=np.zeros_like(u), vy=np.zeros_like(u))


def branch_wavefunction(parameters: BipolaronParameters, branch: str) -> np.ndarray:
    first, second = branch_indices(parameters, branch)
    psi = np.zeros((parameters.n_sites, parameters.n_sites), dtype=float)
    if first == second:
        psi[first, first] = 1.0
    else:
        amplitude = 1.0 / np.sqrt(2.0)
        psi[first, second] = amplitude
        psi[second, first] = amplitude
    return psi


def diagonal_probability(
    wavefunction: np.ndarray,
    parameters: BipolaronParameters,
) -> float:
    dx, dy = minimum_image_offsets(parameters)
    mask = (dx == 1) & (dy == 1)
    return float(np.sum(np.square(wavefunction)[mask]))


def relax_seeded(
    parameters: BipolaronParameters,
    *,
    branch: str,
) -> HolsteinPeierlsBipolaronResult:
    """Replicate core relaxation with only the first Krylov vector specialized."""
    lattice = branch_seed(parameters, branch)
    previous_u = np.zeros_like(lattice.u)
    previous_vx = np.zeros_like(lattice.vx)
    previous_vy = np.zeros_like(lattice.vy)
    step_u = np.full_like(lattice.u, parameters.update_start)
    step_vx = np.full_like(lattice.vx, parameters.update_start)
    step_vy = np.full_like(lattice.vy, parameters.update_start)

    current_state = solve_holstein_peierls_ground_state(
        lattice,
        parameters,
        initial_wavefunction=branch_wavefunction(parameters, branch),
    )
    converged = False
    final_max_update = np.inf
    final_max_gradient = np.inf
    final_energy = None

    for iteration in range(1, parameters.max_iterations + 1):
        gradient, _ = energy_gradient(
            lattice,
            parameters,
            ground_state=current_state,
        )
        delta_u, _ = _rprop_update(
            lattice.u, gradient.u, previous_u, step_u, parameters
        )
        delta_vx, _ = _rprop_update(
            lattice.vx, gradient.vx, previous_vx, step_vx, parameters
        )
        delta_vy, _ = _rprop_update(
            lattice.vy, gradient.vy, previous_vy, step_vy, parameters
        )

        next_state = solve_holstein_peierls_ground_state(
            lattice,
            parameters,
            initial_wavefunction=current_state.wavefunction,
        )
        final_energy, _ = total_energy(
            lattice,
            parameters,
            ground_state=next_state,
        )
        final_gradient, _ = energy_gradient(
            lattice,
            parameters,
            ground_state=next_state,
        )
        current_state = next_state

        final_max_update = float(
            max(
                np.max(np.abs(delta_u)),
                np.max(np.abs(delta_vx)),
                np.max(np.abs(delta_vy)),
            )
        )
        final_max_gradient = final_gradient.maximum_absolute_component
        converged = (
            final_max_update < parameters.convergence_criterion
            and final_max_gradient < parameters.gradient_convergence_criterion
        )
        if converged:
            break

    if final_energy is None:
        final_energy, _ = total_energy(
            lattice,
            parameters,
            ground_state=current_state,
        )

    return HolsteinPeierlsBipolaronResult(
        lattice=lattice,
        ground_state=current_state,
        energy=final_energy,
        diagnostics=BipolaronRelaxationDiagnostics(
            iterations=iteration,
            converged=converged,
            final_max_update=final_max_update,
            final_max_gradient=final_max_gradient,
        ),
    )


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
    parser.add_argument("--v1", type=float, default=0.0)
    parser.add_argument("--branch", choices=BRANCHES, required=True)
    parser.add_argument("--j0x", type=float, default=0.100)
    parser.add_argument("--j0y", type=float, default=0.015)
    parser.add_argument("--alpha-x", type=float, default=0.10)
    parser.add_argument("--alpha-y", type=float, default=0.12)
    parser.add_argument("--long-range", action="store_true")
    parser.add_argument("--ax", type=float)
    parser.add_argument("--ay", type=float)
    parser.add_argument("--epsilon-r", type=float)
    parser.add_argument("--max-iterations", type=int, default=1200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.long_range:
        missing = [
            name
            for name, value in (("--ax", args.ax), ("--ay", args.ay), ("--epsilon-r", args.epsilon_r))
            if value is None
        ]
        if missing:
            parser.error("--long-range requires " + ", ".join(missing))

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

    parameters = BipolaronParameters.from_polaron_parameters(
        single,
        hubbard_u=args.u,
        nearest_neighbor_v=args.v1,
        long_range_coulomb=args.long_range,
        lattice_spacing_x_angstrom=args.ax if args.long_range else None,
        lattice_spacing_y_angstrom=args.ay if args.long_range else None,
        relative_permittivity=args.epsilon_r if args.long_range else None,
    )
    parameters = replace(
        parameters,
        pair_position=center_position(args.size),
        max_iterations=args.max_iterations,
        convergence_criterion=1.0e-8,
        gradient_convergence_criterion=1.0e-6,
        eigensolver_tolerance=1.0e-11,
    )

    result = relax_seeded(parameters, branch=args.branch)
    obs = pair_observables(result.ground_state, parameters)
    pdiag = diagonal_probability(result.ground_state.wavefunction, parameters)
    ratio_x, ratio_y = distortion_ratios(result, parameters)
    eint = interaction_expectation(result.ground_state.wavefunction, parameters)
    interaction = pair_interaction_matrix(parameters)
    first = parameters.pair_index
    cy, cx = divmod(first, parameters.nx)
    right = cy * parameters.nx + (cx + 1) % parameters.nx
    down = ((cy + 1) % parameters.ny) * parameters.nx + cx
    diag = ((cy + 1) % parameters.ny) * parameters.nx + (cx + 1) % parameters.nx

    record = {
        "size": args.size,
        "Jx_eV": args.j0x,
        "Jy_eV": args.j0y,
        "alpha_x_eV_per_A": args.alpha_x,
        "alpha_y_eV_per_A": args.alpha_y,
        "U_eV": args.u,
        "V1_eV": args.v1,
        "long_range_coulomb": args.long_range,
        "ax_A": args.ax if args.long_range else None,
        "ay_A": args.ay if args.long_range else None,
        "epsilon_r": args.epsilon_r if args.long_range else None,
        "Vx_eV": float(interaction[first, right]),
        "Vy_eV": float(interaction[first, down]),
        "Vdiag_eV": float(interaction[first, diag]),
        "branch": args.branch,
        "one_polaron_energy_eV": polaron.total_energy,
        "two_polaron_reference_eV": two_polaron_reference,
        "total_energy_eV": result.energy.total,
        "binding_vs_2polaron_eV": two_polaron_reference - result.energy.total,
        "interaction_energy_eV": eint,
        "P_onsite": obs.onsite_probability,
        "P_nn": obs.nearest_neighbour_probability,
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
        "final_max_update_A": result.diagnostics.final_max_update,
        "final_max_gradient_eV_per_A": result.diagnostics.final_max_gradient,
        "convergence_criterion_A": parameters.convergence_criterion,
        "gradient_convergence_criterion_eV_per_A": parameters.gradient_convergence_criterion,
        "eigensolver_tolerance": parameters.eigensolver_tolerance,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / "branch_result.json"
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, sort_keys=True))
    print(f"JSON: {output}")

    if not result.diagnostics.converged:
        raise SystemExit(
            "strict large-cell branch validation did not converge within "
            f"{parameters.max_iterations} iterations"
        )


if __name__ == "__main__":
    main()
