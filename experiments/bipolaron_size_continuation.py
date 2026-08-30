"""Finite-size validation by continuation of a localized bipolaron.

A converged 20x20 intersite-x state is embedded into a 40x40 periodic cell and
then fully re-relaxed. No large-cell lattice or electronic degree of freedom is
frozen. The dissociation reference is twice the independently relaxed
one-polaron energy in the same cell, which is the thermodynamic two-polaron
reference and avoids an unnecessary correlated calculation for two carriers
already placed infinitely far apart in the finite-size limit.
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
from holstein_peierls.two_particle.observables import pair_observables
from holstein_peierls.two_particle.parameters import BipolaronParameters
from holstein_peierls.two_particle.peierls import (
    HolsteinPeierlsBipolaronResult,
    _rprop_update,
    energy_gradient,
    solve_holstein_peierls_ground_state,
    total_energy,
)


def center_position(size: int) -> int:
    return (size // 2) * size + (size // 2) + 1


def intersite_seed(parameters: BipolaronParameters) -> tuple[LatticeState, np.ndarray]:
    first = parameters.pair_index
    cy, cx = divmod(first, parameters.nx)
    second = cy * parameters.nx + (cx + 1) % parameters.nx

    u = np.zeros((parameters.ny, parameters.nx), dtype=float)
    displacement = -parameters.alpha_intra / parameters.k1
    u[cy, cx] = displacement
    sy, sx = divmod(second, parameters.nx)
    u[sy, sx] = displacement
    lattice = LatticeState(u=u, vx=np.zeros_like(u), vy=np.zeros_like(u))

    psi = np.zeros((parameters.n_sites, parameters.n_sites), dtype=float)
    amplitude = 1.0 / np.sqrt(2.0)
    psi[first, second] = amplitude
    psi[second, first] = amplitude
    return lattice, psi


def relax_from_seed(
    parameters: BipolaronParameters,
    lattice_seed: LatticeState,
    wavefunction_seed: np.ndarray,
) -> HolsteinPeierlsBipolaronResult:
    lattice = lattice_seed.copy()
    previous_u = np.zeros_like(lattice.u)
    previous_vx = np.zeros_like(lattice.vx)
    previous_vy = np.zeros_like(lattice.vy)
    step_u = np.full_like(lattice.u, parameters.update_start)
    step_vx = np.full_like(lattice.vx, parameters.update_start)
    step_vy = np.full_like(lattice.vy, parameters.update_start)

    current_state = solve_holstein_peierls_ground_state(
        lattice,
        parameters,
        initial_wavefunction=wavefunction_seed,
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


def parameters_for(size: int, hubbard_u: float, alpha_x: float, alpha_y: float) -> BipolaronParameters:
    single = StaticPolaronParameters(
        nx=size,
        ny=size,
        polaron_position=center_position(size),
        alpha_interx=alpha_x,
        alpha_intery=alpha_y,
    )
    parameters = BipolaronParameters.from_polaron_parameters(single, hubbard_u=hubbard_u)
    return replace(
        parameters,
        pair_position=center_position(size),
        max_iterations=1200,
        convergence_criterion=1.0e-6,
        gradient_convergence_criterion=2.0e-5,
        eigensolver_tolerance=2.0e-9,
    )


def embed_state(
    result: HolsteinPeierlsBipolaronResult,
    large_parameters: BipolaronParameters,
) -> tuple[LatticeState, np.ndarray]:
    small_ny, small_nx = result.lattice.shape
    if large_parameters.nx < small_nx or large_parameters.ny < small_ny:
        raise ValueError("large cell must not be smaller than continuation cell")
    if (large_parameters.nx - small_nx) % 2 or (large_parameters.ny - small_ny) % 2:
        raise ValueError("cell-size differences must be even for centered embedding")

    ox = (large_parameters.nx - small_nx) // 2
    oy = (large_parameters.ny - small_ny) // 2
    u = np.zeros((large_parameters.ny, large_parameters.nx), dtype=float)
    vx = np.zeros_like(u)
    vy = np.zeros_like(u)

    # Peierls coordinates have a uniform-shift gauge freedom. Remove the tiny
    # cell-average gauge before embedding so that the exterior zero reference
    # does not introduce a spurious boundary step.
    u[oy : oy + small_ny, ox : ox + small_nx] = result.u
    vx_small = result.vx - float(np.mean(result.vx))
    vy_small = result.vy - float(np.mean(result.vy))
    vx[oy : oy + small_ny, ox : ox + small_nx] = vx_small
    vy[oy : oy + small_ny, ox : ox + small_nx] = vy_small

    small_sites = np.arange(small_nx * small_ny).reshape(small_ny, small_nx)
    large_sites = np.arange(large_parameters.n_sites).reshape(
        large_parameters.ny, large_parameters.nx
    )
    mapping = np.empty(small_nx * small_ny, dtype=int)
    mapping[small_sites.ravel()] = large_sites[
        oy : oy + small_ny, ox : ox + small_nx
    ].ravel()

    psi_large = np.zeros(
        (large_parameters.n_sites, large_parameters.n_sites),
        dtype=float,
    )
    psi_large[np.ix_(mapping, mapping)] = result.ground_state.wavefunction
    return LatticeState(u=u, vx=vx, vy=vy), psi_large


def one_polaron_energy(size: int, alpha_x: float, alpha_y: float) -> float:
    parameters = StaticPolaronParameters(
        nx=size,
        ny=size,
        polaron_position=center_position(size),
        alpha_interx=alpha_x,
        alpha_intery=alpha_y,
        max_iterations=1800,
        convergence_criterion=1.0e-7,
    )
    result = solve_static_polaron(
        parameters,
        solver="sparse",
        gradient_mode="optimized",
        legacy_convergence=False,
    )
    return float(result.total_energy)


def diagnostics(result: HolsteinPeierlsBipolaronResult, parameters: BipolaronParameters) -> dict[str, float | int | bool]:
    obs = pair_observables(result.ground_state, parameters)
    dx = np.roll(result.vx, shift=-1, axis=1) - result.vx
    dy = np.roll(result.vy, shift=-1, axis=0) - result.vy
    ratio_x = abs(parameters.alpha_interx) * float(np.max(np.abs(dx))) / abs(parameters.j0x)
    ratio_y = abs(parameters.alpha_intery) * float(np.max(np.abs(dy))) / abs(parameters.j0y)
    return {
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--u", type=float, required=True)
    parser.add_argument("--small-size", type=int, default=20)
    parser.add_argument("--large-size", type=int, default=40)
    parser.add_argument("--alpha-x", type=float, default=0.10)
    parser.add_argument("--alpha-y", type=float, default=0.12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    small_p = parameters_for(args.small_size, args.u, args.alpha_x, args.alpha_y)
    small_lattice, small_psi = intersite_seed(small_p)
    small_result = relax_from_seed(small_p, small_lattice, small_psi)
    if not small_result.diagnostics.converged:
        raise RuntimeError("small-cell continuation state did not converge")

    large_p = parameters_for(args.large_size, args.u, args.alpha_x, args.alpha_y)
    large_lattice, large_psi = embed_state(small_result, large_p)
    large_result = relax_from_seed(large_p, large_lattice, large_psi)
    if not large_result.diagnostics.converged:
        raise RuntimeError("large-cell continuation state did not converge")

    e1_small = one_polaron_energy(args.small_size, args.alpha_x, args.alpha_y)
    e1_large = one_polaron_energy(args.large_size, args.alpha_x, args.alpha_y)
    record = {
        "U_eV": args.u,
        "alpha_x_eV_per_A": args.alpha_x,
        "alpha_y_eV_per_A": args.alpha_y,
        "small_size": args.small_size,
        "large_size": args.large_size,
        "small_single_polaron_energy_eV": e1_small,
        "large_single_polaron_energy_eV": e1_large,
        "small_binding_vs_2polarons_eV": 2.0 * e1_small - small_result.energy.total,
        "large_binding_vs_2polarons_eV": 2.0 * e1_large - large_result.energy.total,
        "small": diagnostics(small_result, small_p),
        "large": diagnostics(large_result, large_p),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "continuation_result.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, sort_keys=True))
    print(f"JSON: {path}")


if __name__ == "__main__":
    main()
