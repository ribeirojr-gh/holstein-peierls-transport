#!/usr/bin/env python3
"""D6d diagnostic benchmark for pair electronic thermalization under D6c dynamics.

The trajectory is exactly the validated D6c coherent pair Ehrenfest + BAOAB
scheme.  D6d only samples instantaneous many-body adiabatic populations and
compares them with canonical and uniform references.  No collapse, reweighting,
decoherence, or additional force is applied.

For the bipolaron, diagonalization is restricted to the symmetric spatial
singlet sector.  For the distinguishable e-h exciton, the full ordered N**2
space is used.
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
    project_pair_zero_modes,
)
from holstein_peierls.dynamics.pair_thermalization import diagnose_pair_thermalization
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
    gamma_e, gamma_h = exciton_one_body_density_matrices(state.electronic_state, n_sites)
    return max(
        abs(complex(np.trace(gamma_e)) - 1.0),
        abs(complex(np.trace(gamma_h)) - 1.0),
    )


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def finite_mean(values: list[float]) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    return float(np.mean(finite)) if finite.size else float("nan")


def segment_mean(values: list[float], which: str) -> float:
    count = max(1, len(values) // 3)
    segment = values[:count] if which == "early" else values[-count:]
    return mean(segment)


def trend_slope_per_ps(times_fs: list[float], values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    times = np.asarray(times_fs, dtype=np.float64) * 1.0e-3
    data = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(times) & np.isfinite(data)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    slope, _ = np.polyfit(times[mask], data[mask], 1)
    return float(slope)


def run_trajectory(
    sector: str,
    size: int,
    masses: PairLatticeMasses,
    bath: LangevinBath,
    seed: int,
    dt_fs: float,
    final_time_fs: float,
    burn_in_fs: float,
    diagnostic_interval_fs: float,
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
    initial_norm = float(np.linalg.norm(initial.electronic_state))

    steps = int(round(final_time_fs / dt_fs))
    burn_steps = int(round(burn_in_fs / dt_fs))
    stride = int(round(diagnostic_interval_fs / dt_fs))
    if steps <= 0 or not np.isclose(steps * dt_fs, final_time_fs):
        raise ValueError("final_time_fs must be an integer multiple of dt_fs")
    if burn_steps < 0 or burn_steps >= steps:
        raise ValueError("burn_in_fs must lie in [0, final_time_fs)")
    if stride <= 0 or not np.isclose(stride * dt_fs, diagnostic_interval_fs):
        raise ValueError("diagnostic_interval_fs must be an integer multiple of dt_fs")

    times: list[float] = []
    heating: list[float] = []
    beta_ratios: list[float] = []
    tv_canonical: list[float] = []
    js_canonical: list[float] = []
    tv_uniform: list[float] = []
    entropies: list[float] = []
    participations: list[float] = []
    ground_populations: list[float] = []
    canonical_ground: list[float] = []
    temperatures: list[float] = []
    sector_dimensions: list[int] = []

    bath_heat = 0.0
    max_norm_error = 0.0
    max_constraint = 0.0
    max_energy_residual = 0.0
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
        max_norm_error = max(
            max_norm_error,
            abs(float(np.linalg.norm(current.electronic_state)) - initial_norm),
        )
        max_constraint = max(
            max_constraint,
            constraint_error(sector, current, parameters.n_sites),
        )

        if (step + 1) % stride != 0:
            continue
        matter_energy = pair_dynamic_total_energy(
            current, sector, parameters, masses, factory=factory
        ).total
        max_energy_residual = max(
            max_energy_residual,
            abs((matter_energy - initial_energy) - bath_heat),
        )
        if step + 1 <= burn_steps:
            continue

        snapshot = diagnose_pair_thermalization(
            current.lattice,
            sector,
            parameters,
            current.electronic_state,
            bath.temperature_K,
            factory=factory,
        )
        times.append(float((step + 1) * dt_fs))
        heating.append(snapshot.heating_coordinate)
        beta_ratios.append(snapshot.beta_eff_over_beta_bath)
        tv_canonical.append(snapshot.tv_to_canonical)
        js_canonical.append(snapshot.js_to_canonical)
        tv_uniform.append(snapshot.tv_to_uniform)
        entropies.append(snapshot.normalized_occupation_entropy)
        participations.append(snapshot.occupation_participation_number)
        ground_populations.append(snapshot.ground_manifold_population)
        canonical_ground.append(snapshot.canonical_ground_manifold_population)
        sector_dimensions.append(snapshot.physical_sector_dimension)
        temperatures.append(
            pair_kinetic_temperature_K(
                current.velocity,
                masses,
                degrees_of_freedom=dof,
            )
        )

    if not times:
        raise RuntimeError("D6d produced no post-burn-in diagnostic samples")
    closer_uniform = [float(u < c) for u, c in zip(tv_uniform, tv_canonical)]
    nonpositive_beta = [float(value <= 0.0) for value in beta_ratios]
    return {
        "sector": sector,
        "seed": int(seed),
        "steps": int(steps),
        "elapsed_seconds": float(perf_counter() - start),
        "diagnostic_samples": len(times),
        "physical_sector_dimension": int(sector_dimensions[0]),
        "maximum_electronic_norm_error": float(max_norm_error),
        "maximum_sector_constraint_error": float(max_constraint),
        "maximum_sampled_abs_energy_bath_residual_eV": float(max_energy_residual),
        "lattice_temperature_mean_K": mean(temperatures),
        "lattice_temperature_std_K": std(temperatures),
        "heating_coordinate_mean": mean(heating),
        "heating_coordinate_early_mean": segment_mean(heating, "early"),
        "heating_coordinate_late_mean": segment_mean(heating, "late"),
        "heating_coordinate_slope_per_ps": trend_slope_per_ps(times, heating),
        "beta_eff_over_beta_bath_mean_finite": finite_mean(beta_ratios),
        "nonpositive_beta_fraction": mean(nonpositive_beta),
        "tv_to_canonical_mean": mean(tv_canonical),
        "js_to_canonical_mean": mean(js_canonical),
        "tv_to_uniform_mean": mean(tv_uniform),
        "closer_to_uniform_fraction": mean(closer_uniform),
        "normalized_occupation_entropy_mean": mean(entropies),
        "occupation_participation_number_mean": mean(participations),
        "ground_manifold_population_mean": mean(ground_populations),
        "canonical_ground_manifold_population_mean": mean(canonical_ground),
        "sample_times_fs": times,
        "heating_coordinate_samples": heating,
    }


def aggregate(sector: str, records: list[dict]) -> dict:
    scalar_keys = (
        "lattice_temperature_mean_K",
        "heating_coordinate_mean",
        "heating_coordinate_early_mean",
        "heating_coordinate_late_mean",
        "heating_coordinate_slope_per_ps",
        "beta_eff_over_beta_bath_mean_finite",
        "nonpositive_beta_fraction",
        "tv_to_canonical_mean",
        "js_to_canonical_mean",
        "tv_to_uniform_mean",
        "closer_to_uniform_fraction",
        "normalized_occupation_entropy_mean",
        "occupation_participation_number_mean",
        "ground_manifold_population_mean",
        "canonical_ground_manifold_population_mean",
    )
    result = {
        "sector": sector,
        "trajectory_count": len(records),
        "total_diagnostic_samples": int(sum(r["diagnostic_samples"] for r in records)),
        "physical_sector_dimension": int(records[0]["physical_sector_dimension"]),
        "maximum_electronic_norm_error": float(max(r["maximum_electronic_norm_error"] for r in records)),
        "maximum_sector_constraint_error": float(max(r["maximum_sector_constraint_error"] for r in records)),
        "maximum_sampled_abs_energy_bath_residual_eV": float(max(r["maximum_sampled_abs_energy_bath_residual_eV"] for r in records)),
    }
    for key in scalar_keys:
        values = np.asarray([r[key] for r in records], dtype=np.float64)
        finite = values[np.isfinite(values)]
        result[f"{key}_ensemble_mean"] = float(np.mean(finite)) if finite.size else float("nan")
        result[f"{key}_ensemble_std"] = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--temperature-K", type=float, default=300.0)
    parser.add_argument("--gamma-u-per-fs", type=float, default=0.01)
    parser.add_argument("--gamma-v-per-fs", type=float, default=0.01)
    parser.add_argument("--dt-fs", type=float, default=0.2)
    parser.add_argument("--final-time-fs", type=float, default=4000.0)
    parser.add_argument("--burn-in-fs", type=float, default=1000.0)
    parser.add_argument("--diagnostic-interval-fs", type=float, default=50.0)
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
                    args.diagnostic_interval_fs,
                    args.krylov_dimension,
                )
            )
    aggregates = [aggregate(sector, [r for r in runs if r["sector"] == sector]) for sector in ("bipolaron", "exciton")]

    numerical_checks = {}
    for item in aggregates:
        prefix = item["sector"]
        numerical_checks[f"{prefix}_temperature"] = 240.0 <= item["lattice_temperature_mean_K_ensemble_mean"] <= 360.0
        numerical_checks[f"{prefix}_energy_balance"] = item["maximum_sampled_abs_energy_bath_residual_eV"] < 1.0e-4
        numerical_checks[f"{prefix}_norm"] = item["maximum_electronic_norm_error"] < 1.0e-10
        numerical_checks[f"{prefix}_sector_constraints"] = item["maximum_sector_constraint_error"] < 1.0e-10
    numerical_pass = all(numerical_checks.values())
    payload = {
        "size": args.size,
        "temperature_K": args.temperature_K,
        "gamma_u_per_fs": args.gamma_u_per_fs,
        "gamma_v_per_fs": args.gamma_v_per_fs,
        "dt_fs": args.dt_fs,
        "final_time_fs": args.final_time_fs,
        "burn_in_fs": args.burn_in_fs,
        "diagnostic_interval_fs": args.diagnostic_interval_fs,
        "seeds": args.seeds,
        "krylov_dimension": args.krylov_dimension,
        "runs": runs,
        "aggregates": aggregates,
        "numerical_checks": numerical_checks,
        "numerical_pass": numerical_pass,
        "physical_selection_is_manual": True,
        "interpretation": (
            "D6d is diagnostic only. PASS means execution and preservation of validated D6c numerical controls; "
            "the heating/canonical metrics must be interpreted manually before any pair decoherence model is introduced."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# D6d pair electronic thermalization diagnostic",
        "",
        f"Control: {args.size}x{args.size}, T={args.temperature_K:g} K, dt={args.dt_fs:g} fs, time={args.final_time_fs/1000:g} ps, burn-in={args.burn_in_fs/1000:g} ps.",
        "",
        "| sector | physical dim | <Tlat> [K] | <heating> | early | late | slope/ps | TV canonical | TV uniform | ground pop | canonical ground |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in aggregates:
        lines.append(
            f"| {item['sector']} | {item['physical_sector_dimension']} | "
            f"{item['lattice_temperature_mean_K_ensemble_mean']:.3f} | "
            f"{item['heating_coordinate_mean_ensemble_mean']:.6g} | "
            f"{item['heating_coordinate_early_mean_ensemble_mean']:.6g} | "
            f"{item['heating_coordinate_late_mean_ensemble_mean']:.6g} | "
            f"{item['heating_coordinate_slope_per_ps_ensemble_mean']:.3e} | "
            f"{item['tv_to_canonical_mean_ensemble_mean']:.6g} | "
            f"{item['tv_to_uniform_mean_ensemble_mean']:.6g} | "
            f"{item['ground_manifold_population_mean_ensemble_mean']:.6g} | "
            f"{item['canonical_ground_manifold_population_mean_ensemble_mean']:.6g} |"
        )
    lines += ["", "Numerical gates: " + ("PASS" if numerical_pass else "FAIL"), "", "Physical selection remains manual.", ""]
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    if not numerical_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
