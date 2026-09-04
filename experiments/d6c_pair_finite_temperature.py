#!/usr/bin/env python3
"""D6c finite-temperature pair BAOAB benchmark.

This gate verifies transfer of the already validated D4 classical bath to the
D6 pair sectors.  It is not an electronic-equilibrium or transport benchmark.
The control uses a small 4x4 lattice so several independent stochastic seeds
remain inexpensive while the pair Hilbert space is nontrivial (dimension 256).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from holstein_peierls.dynamics.ehrenfest import LatticeVelocity
from holstein_peierls.dynamics.langevin import LangevinBath
from holstein_peierls.dynamics.pair_coupled import (
    MovingPairHamiltonianFactory,
    PairCoupledState,
    PairLatticeMasses,
    pair_dynamic_total_energy,
)
from holstein_peierls.dynamics.pair_frozen import (
    bipolaron_exchange_symmetry_error,
    bipolaron_one_body_density_matrix,
    exciton_one_body_density_matrices,
)
from holstein_peierls.dynamics.pair_thermal import (
    pair_coupled_baoab_step,
    pair_kinetic_temperature_K,
    pair_thermostatted_degrees_of_freedom,
    pair_zero_mode_means,
    project_pair_zero_modes,
)
from holstein_peierls.exciton.parameters import ExcitonParameters
from holstein_peierls.exciton.solver import solve_exciton_ground_state
from holstein_peierls.lattice import LatticeState
from holstein_peierls.two_particle.parameters import BipolaronParameters
from holstein_peierls.two_particle.peierls import solve_holstein_peierls_ground_state


def center_position(size: int) -> int:
    return (size // 2) * size + size // 2 + 1


def lattice_control(size: int) -> LatticeState:
    y, x = np.indices((size, size), dtype=np.float64)
    return LatticeState(
        u=8.0e-3 * np.cos(0.71 * x + 0.43 * y),
        vx=3.0e-3 * np.sin(0.91 * x - 0.23 * y),
        vy=2.5e-3 * np.cos(0.31 * x + 0.83 * y),
    )


def parameters_for(sector: str, size: int):
    if sector == "bipolaron":
        return BipolaronParameters(
            nx=size,
            ny=size,
            pair_position=center_position(size),
            hubbard_u=0.22,
            nearest_neighbor_v=0.04,
        )
    return ExcitonParameters(
        nx=size,
        ny=size,
        exciton_position=center_position(size),
        electron_j0x=0.100,
        electron_j0y=0.015,
        hole_j0x=0.082,
        hole_j0y=0.021,
        electron_alpha_intra=3.0,
        hole_alpha_intra=2.4,
        electron_alpha_interx=0.4,
        electron_alpha_intery=0.4,
        hole_alpha_interx=0.31,
        hole_alpha_intery=0.28,
        onsite_attraction=0.30,
        nearest_neighbor_attraction=0.05,
    )


def initial_state(sector: str, size: int):
    lattice = lattice_control(size)
    parameters = parameters_for(sector, size)
    if sector == "bipolaron":
        ground = solve_holstein_peierls_ground_state(lattice, parameters)
    else:
        ground = solve_exciton_ground_state(lattice, parameters)
    raw = PairCoupledState(
        lattice=lattice,
        velocity=LatticeVelocity.zeros(size, size),
        electronic_state=np.asarray(ground.wavefunction, dtype=np.complex128).ravel(order="C"),
    )
    return project_pair_zero_modes(raw), parameters


def constraint_error(sector: str, state: PairCoupledState, n_sites: int) -> float:
    if sector == "bipolaron":
        gamma = bipolaron_one_body_density_matrix(state.electronic_state, n_sites)
        return max(
            bipolaron_exchange_symmetry_error(state.electronic_state, n_sites),
            abs(complex(np.trace(gamma)) - 2.0),
        )
    gamma_e, gamma_h = exciton_one_body_density_matrices(
        state.electronic_state, n_sites
    )
    return max(
        abs(complex(np.trace(gamma_e)) - 1.0),
        abs(complex(np.trace(gamma_h)) - 1.0),
    )


def run_trajectory(
    sector: str,
    size: int,
    masses: PairLatticeMasses,
    bath: LangevinBath,
    seed: int,
    dt_fs: float,
    final_time_fs: float,
    burn_in_fs: float,
    sample_interval_fs: float,
    krylov_dimension: int,
) -> dict:
    initial, parameters = initial_state(sector, size)
    current = PairCoupledState(
        initial.lattice.copy(), initial.velocity.copy(), initial.electronic_state.copy()
    )
    factory = MovingPairHamiltonianFactory(sector, parameters)
    rng = np.random.default_rng(seed)
    dof = pair_thermostatted_degrees_of_freedom(parameters, "project")
    initial_energy = pair_dynamic_total_energy(
        initial, sector, parameters, masses, factory=factory
    ).total
    steps = int(round(final_time_fs / dt_fs))
    burn_step = int(round(burn_in_fs / dt_fs))
    sample_stride = max(1, int(round(sample_interval_fs / dt_fs)))
    bath_heat = 0.0
    max_residual = 0.0
    max_norm_error = 0.0
    max_constraint = 0.0
    max_zero_mode = 0.0
    max_excursion = 0.0
    temperatures: list[float] = []
    initial_arrays = (
        initial.lattice.u.copy(), initial.lattice.vx.copy(), initial.lattice.vy.copy()
    )
    h_eval = 0
    h_apply = 0
    start = perf_counter()
    for step in range(steps):
        result = pair_coupled_baoab_step(
            current,
            sector,
            parameters,
            masses,
            bath,
            rng,
            step * dt_fs,
            dt_fs,
            zero_mode_policy="project",
            krylov_dimension=krylov_dimension,
            factory=factory,
        )
        current = result.state
        bath_heat += result.bath_heat_eV
        h_eval += result.hamiltonian_evaluations
        h_apply += result.hamiltonian_applications
        if (step + 1) % sample_stride == 0 or step + 1 == steps:
            energy = pair_dynamic_total_energy(
                current, sector, parameters, masses, factory=factory
            ).total
            residual = (energy - initial_energy) - bath_heat
            max_residual = max(max_residual, abs(residual))
            max_norm_error = max(
                max_norm_error,
                abs(float(np.linalg.norm(current.electronic_state)) - 1.0),
            )
            max_constraint = max(
                max_constraint,
                constraint_error(sector, current, parameters.n_sites),
            )
            max_zero_mode = max(
                max_zero_mode,
                max(abs(value) for value in pair_zero_mode_means(current).values()),
            )
            max_excursion = max(
                max_excursion,
                float(np.max(np.abs(current.lattice.u - initial_arrays[0]))),
                float(np.max(np.abs(current.lattice.vx - initial_arrays[1]))),
                float(np.max(np.abs(current.lattice.vy - initial_arrays[2]))),
            )
            if step + 1 > burn_step:
                temperatures.append(
                    pair_kinetic_temperature_K(
                        current.velocity,
                        masses,
                        degrees_of_freedom=dof,
                    )
                )
    final_energy = pair_dynamic_total_energy(
        current, sector, parameters, masses, factory=factory
    ).total
    final_residual = (final_energy - initial_energy) - bath_heat
    if not temperatures:
        raise RuntimeError("no post-burn-in temperature samples were collected")
    return {
        "sector": sector,
        "seed": int(seed),
        "steps": steps,
        "elapsed_seconds": float(perf_counter() - start),
        "kinetic_degrees_of_freedom": dof,
        "mean_temperature_K": float(np.mean(temperatures)),
        "std_temperature_K": float(np.std(temperatures)),
        "bath_heat_eV": float(bath_heat),
        "matter_energy_change_eV": float(final_energy - initial_energy),
        "final_energy_residual_eV": float(final_residual),
        "maximum_absolute_energy_residual_eV": float(max_residual),
        "maximum_norm_error": float(max_norm_error),
        "maximum_sector_constraint_error": float(max_constraint),
        "maximum_zero_mode_mean": float(max_zero_mode),
        "maximum_lattice_excursion_A": float(max_excursion),
        "hamiltonian_evaluations": int(h_eval),
        "hamiltonian_applications": int(h_apply),
    }


def aggregate(sector: str, runs: list[dict]) -> dict:
    return {
        "sector": sector,
        "trajectory_count": len(runs),
        "mean_temperature_K": float(np.mean([r["mean_temperature_K"] for r in runs])),
        "std_across_trajectory_means_K": float(
            np.std([r["mean_temperature_K"] for r in runs])
        ),
        "maximum_absolute_energy_residual_eV": float(
            max(r["maximum_absolute_energy_residual_eV"] for r in runs)
        ),
        "maximum_norm_error": float(max(r["maximum_norm_error"] for r in runs)),
        "maximum_sector_constraint_error": float(
            max(r["maximum_sector_constraint_error"] for r in runs)
        ),
        "maximum_zero_mode_mean": float(max(r["maximum_zero_mode_mean"] for r in runs)),
        "maximum_lattice_excursion_A": float(
            max(r["maximum_lattice_excursion_A"] for r in runs)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--temperature-K", type=float, default=300.0)
    parser.add_argument("--gamma-u-per-fs", type=float, default=0.01)
    parser.add_argument("--gamma-v-per-fs", type=float, default=0.01)
    parser.add_argument("--dt-fs", type=float, default=0.2)
    parser.add_argument("--final-time-fs", type=float, default=2000.0)
    parser.add_argument("--burn-in-fs", type=float, default=500.0)
    parser.add_argument("--sample-interval-fs", type=float, default=5.0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260905, 20260906, 20260907, 20260908])
    parser.add_argument("--krylov-dimension", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    masses = PairLatticeMasses(75000.0, 150000.0)
    bath = LangevinBath(args.temperature_K, args.gamma_u_per_fs, args.gamma_v_per_fs)
    runs = []
    for sector in ("bipolaron", "exciton"):
        for seed in args.seeds:
            runs.append(
                run_trajectory(
                    sector,
                    args.size,
                    masses,
                    bath,
                    seed,
                    args.dt_fs,
                    args.final_time_fs,
                    args.burn_in_fs,
                    args.sample_interval_fs,
                    args.krylov_dimension,
                )
            )
    aggregates = [
        aggregate(sector, [r for r in runs if r["sector"] == sector])
        for sector in ("bipolaron", "exciton")
    ]

    checks: dict[str, bool] = {}
    for item in aggregates:
        prefix = item["sector"]
        checks[f"{prefix}_temperature"] = 240.0 <= item["mean_temperature_K"] <= 360.0
        checks[f"{prefix}_energy_balance"] = item["maximum_absolute_energy_residual_eV"] < 1.0e-4
        checks[f"{prefix}_norm"] = item["maximum_norm_error"] < 1.0e-10
        checks[f"{prefix}_sector_constraints"] = item["maximum_sector_constraint_error"] < 1.0e-10
        checks[f"{prefix}_zero_modes"] = item["maximum_zero_mode_mean"] < 1.0e-12
        checks[f"{prefix}_bounded_lattice"] = item["maximum_lattice_excursion_A"] < 2.0
    closure_pass = all(checks.values())
    payload = {
        "size": args.size,
        "temperature_K": args.temperature_K,
        "gamma_u_per_fs": args.gamma_u_per_fs,
        "gamma_v_per_fs": args.gamma_v_per_fs,
        "dt_fs": args.dt_fs,
        "final_time_fs": args.final_time_fs,
        "burn_in_fs": args.burn_in_fs,
        "sample_interval_fs": args.sample_interval_fs,
        "seeds": args.seeds,
        "krylov_dimension": args.krylov_dimension,
        "masses_eV_fs2_per_A2": {
            "intramolecular": masses.intramolecular,
            "intermolecular": masses.intermolecular,
        },
        "runs": runs,
        "aggregates": aggregates,
        "closure_checks": checks,
        "closure_pass": closure_pass,
        "interpretation": (
            "D6c validates lattice-bath transfer and pair-sector numerical constraints only; "
            "it is not an electronic thermalization or transport claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# D6c finite-temperature pair benchmark",
        "",
        f"Control: {args.size}x{args.size}, T={args.temperature_K:g} K, dt={args.dt_fs:g} fs, "
        f"time={args.final_time_fs/1000:g} ps, burn-in={args.burn_in_fs/1000:g} ps.",
        "",
        "| sector | mean T [K] | max |dE-Q| [eV] | max norm err | max constraint err | max zero mode | max excursion [A] |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in aggregates:
        lines.append(
            f"| {item['sector']} | {item['mean_temperature_K']:.3f} | "
            f"{item['maximum_absolute_energy_residual_eV']:.3e} | "
            f"{item['maximum_norm_error']:.3e} | "
            f"{item['maximum_sector_constraint_error']:.3e} | "
            f"{item['maximum_zero_mode_mean']:.3e} | "
            f"{item['maximum_lattice_excursion_A']:.3e} |"
        )
    lines += ["", "## Pre-registered closure", ""]
    lines += [f"- {name}: {'PASS' if value else 'FAIL'}" for name, value in checks.items()]
    lines += ["", f"Overall: {'PASS' if closure_pass else 'FAIL'}", ""]
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    if not closure_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
