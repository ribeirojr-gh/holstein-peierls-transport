"""D5a diagnostic benchmark for electronic thermalization under D4 dynamics.

The trajectory itself is exactly the validated D4b coherent Ehrenfest + BAOAB
scheme.  D5a adds only coarse-grained instantaneous eigendecompositions and
occupation diagnostics.  No electronic collapse, reweighting, decoherence, or
additional force is applied.

The default control uses a 20x20 relaxed one-polaron state, a 300 K lattice
bath, projected intermolecular zero modes, four independent RNG seeds, 10 ps of
propagation, 2 ps burn-in, and diagnostics every 100 fs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from holstein_peierls.dynamics.coupled import CoupledEhrenfestState
from holstein_peierls.dynamics.ehrenfest import (
    LatticeVelocity,
    dynamic_total_energy,
)
from holstein_peierls.dynamics.electronic_thermalization import (
    diagnose_electronic_thermalization,
)
from holstein_peierls.dynamics.langevin import LangevinBath, kinetic_temperature_K
from holstein_peierls.dynamics.thermal import (
    coupled_baoab_step,
    project_inter_molecular_zero_modes,
    thermostatted_kinetic_degrees_of_freedom,
)
from holstein_peierls.electronic import solve_ground_state
from holstein_peierls.parameters import StaticPolaronParameters
from holstein_peierls.polaron import solve_static_polaron


def _prepare_initial_state(
    size: int,
    zero_mode_policy: str,
) -> tuple[CoupledEhrenfestState, StaticPolaronParameters, dict[str, float | int | bool]]:
    center = (size // 2) * size + (size // 2) + 1
    parameters = StaticPolaronParameters(nx=size, ny=size, polaron_position=center)
    static = solve_static_polaron(
        parameters,
        solver="sparse",
        gradient_mode="optimized",
        legacy_convergence=False,
    )
    ground = solve_ground_state(static.state, parameters, solver="sparse")
    state = CoupledEhrenfestState(
        static.state.copy(),
        LatticeVelocity.zeros(size, size),
        np.asarray(ground.wavefunction, dtype=np.complex128),
    )
    if zero_mode_policy == "project":
        state = project_inter_molecular_zero_modes(state)
    metadata = {
        "static_iterations": static.diagnostics.iterations,
        "static_converged": static.diagnostics.converged,
        "static_total_energy_eV": static.total_energy,
        "static_formation_energy_eV": static.formation_energy,
    }
    return state, parameters, metadata


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _finite_mean(values: list[float]) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    return float(np.mean(finite)) if finite.size else float("nan")


def _finite_median(values: list[float]) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    return float(np.median(finite)) if finite.size else float("nan")


def _trend_slope_per_ps(times_fs: list[float], values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    times_ps = np.asarray(times_fs, dtype=np.float64) * 1.0e-3
    data = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(times_ps) & np.isfinite(data)
    if np.count_nonzero(finite) < 2:
        return float("nan")
    slope, _ = np.polyfit(times_ps[finite], data[finite], 1)
    return float(slope)


def _segment_mean(values: list[float], which: str) -> float:
    if not values:
        return float("nan")
    count = max(1, len(values) // 3)
    segment = values[:count] if which == "early" else values[-count:]
    return _mean(segment)


def _run_seed(
    initial: CoupledEhrenfestState,
    parameters: StaticPolaronParameters,
    bath: LangevinBath,
    *,
    dt_fs: float,
    final_time_fs: float,
    burn_in_fs: float,
    diagnostic_interval_fs: float,
    seed: int,
    zero_mode_policy: str,
    krylov_dimension: int,
) -> dict:
    steps = int(round(final_time_fs / dt_fs))
    burn_steps = int(round(burn_in_fs / dt_fs))
    diagnostic_stride = int(round(diagnostic_interval_fs / dt_fs))
    if steps <= 0 or not np.isclose(steps * dt_fs, final_time_fs):
        raise ValueError("final_time_fs must be a positive integer multiple of dt_fs")
    if burn_steps < 0 or burn_steps >= steps:
        raise ValueError("burn_in_fs must lie in [0, final_time_fs)")
    if diagnostic_stride <= 0 or not np.isclose(diagnostic_stride * dt_fs, diagnostic_interval_fs):
        raise ValueError("diagnostic_interval_fs must be a positive integer multiple of dt_fs")

    current = CoupledEhrenfestState(
        initial.lattice.copy(),
        initial.velocity.copy(),
        initial.electronic_state.copy(),
    )
    rng = np.random.default_rng(seed)
    initial_norm = float(np.linalg.norm(current.electronic_state))
    initial_energy = dynamic_total_energy(
        current.lattice,
        current.velocity,
        parameters,
        current.electronic_state,
    ).total
    dof = thermostatted_kinetic_degrees_of_freedom(parameters, zero_mode_policy)

    times: list[float] = []
    heating: list[float] = []
    beta_ratios: list[float] = []
    effective_temperatures: list[float] = []
    tv_canonical: list[float] = []
    js_canonical: list[float] = []
    tv_uniform: list[float] = []
    entropies: list[float] = []
    participations: list[float] = []
    ground_populations: list[float] = []
    canonical_ground_populations: list[float] = []
    propagated_energies: list[float] = []
    canonical_energies: list[float] = []
    uniform_energies: list[float] = []
    lattice_temperatures: list[float] = []

    bath_heat = 0.0
    max_norm_error = 0.0
    max_energy_bath_residual = 0.0
    h_evaluations = 0
    h_applications = 0
    start = perf_counter()

    for step in range(steps):
        result = coupled_baoab_step(
            current,
            parameters,
            bath,
            rng,
            step * dt_fs,
            dt_fs,
            zero_mode_policy=zero_mode_policy,
            electronic_method="cfm4_lanczos",
            krylov_dimension=krylov_dimension,
        )
        current = result.state
        bath_heat += result.bath_heat_eV
        h_evaluations += result.hamiltonian_evaluations
        h_applications += result.hamiltonian_applications
        max_norm_error = max(
            max_norm_error,
            abs(float(np.linalg.norm(current.electronic_state)) - initial_norm),
        )

        if (step + 1) % diagnostic_stride != 0:
            continue

        matter_energy = dynamic_total_energy(
            current.lattice,
            current.velocity,
            parameters,
            current.electronic_state,
        ).total
        max_energy_bath_residual = max(
            max_energy_bath_residual,
            abs(matter_energy - initial_energy - bath_heat),
        )

        if step + 1 <= burn_steps:
            continue

        time_fs = (step + 1) * dt_fs
        snapshot = diagnose_electronic_thermalization(
            current.lattice,
            parameters,
            current.electronic_state,
            bath.temperature_K,
        )
        times.append(float(time_fs))
        heating.append(snapshot.heating_coordinate)
        beta_ratios.append(snapshot.beta_eff_over_beta_bath)
        effective_temperatures.append(snapshot.effective_temperature_K)
        tv_canonical.append(snapshot.tv_to_canonical)
        js_canonical.append(snapshot.js_to_canonical)
        tv_uniform.append(snapshot.tv_to_uniform)
        entropies.append(snapshot.normalized_occupation_entropy)
        participations.append(snapshot.occupation_participation_number)
        ground_populations.append(snapshot.ground_manifold_population)
        canonical_ground_populations.append(snapshot.canonical_ground_manifold_population)
        propagated_energies.append(snapshot.propagated_energy_eV)
        canonical_energies.append(snapshot.canonical_energy_eV)
        uniform_energies.append(snapshot.uniform_energy_eV)
        lattice_temperatures.append(
            kinetic_temperature_K(
                current.velocity,
                parameters,
                degrees_of_freedom=dof,
            )
        )

    elapsed = perf_counter() - start
    if not times:
        raise RuntimeError("D5a produced no post-burn-in diagnostic samples")

    closer_to_uniform = [
        float(uniform_distance < canonical_distance)
        for uniform_distance, canonical_distance in zip(tv_uniform, tv_canonical)
    ]
    nonpositive_beta = [float(value <= 0.0) for value in beta_ratios]

    return {
        "seed": int(seed),
        "elapsed_seconds": float(elapsed),
        "steps": int(steps),
        "diagnostic_samples": int(len(times)),
        "diagnostic_interval_fs": float(diagnostic_interval_fs),
        "hamiltonian_evaluations": int(h_evaluations),
        "hamiltonian_applications": int(h_applications),
        "maximum_electronic_norm_error": float(max_norm_error),
        "maximum_sampled_abs_energy_bath_residual_eV": float(max_energy_bath_residual),
        "lattice_temperature_mean_K": _mean(lattice_temperatures),
        "lattice_temperature_std_K": _std(lattice_temperatures),
        "heating_coordinate_mean": _mean(heating),
        "heating_coordinate_std": _std(heating),
        "heating_coordinate_early_mean": _segment_mean(heating, "early"),
        "heating_coordinate_late_mean": _segment_mean(heating, "late"),
        "heating_coordinate_slope_per_ps": _trend_slope_per_ps(times, heating),
        "beta_eff_over_beta_bath_mean_finite": _finite_mean(beta_ratios),
        "beta_eff_over_beta_bath_median_finite": _finite_median(beta_ratios),
        "nonpositive_beta_fraction": _mean(nonpositive_beta),
        "effective_temperature_mean_finite_K": _finite_mean(effective_temperatures),
        "tv_to_canonical_mean": _mean(tv_canonical),
        "js_to_canonical_mean": _mean(js_canonical),
        "tv_to_uniform_mean": _mean(tv_uniform),
        "closer_to_uniform_fraction": _mean(closer_to_uniform),
        "normalized_occupation_entropy_mean": _mean(entropies),
        "occupation_participation_number_mean": _mean(participations),
        "ground_manifold_population_mean": _mean(ground_populations),
        "canonical_ground_manifold_population_mean": _mean(canonical_ground_populations),
        "propagated_electronic_energy_mean_eV": _mean(propagated_energies),
        "canonical_electronic_energy_mean_eV": _mean(canonical_energies),
        "uniform_electronic_energy_mean_eV": _mean(uniform_energies),
        "sample_times_fs": times,
        "heating_coordinate_samples": heating,
        "beta_eff_over_beta_bath_samples": beta_ratios,
        "tv_to_canonical_samples": tv_canonical,
        "tv_to_uniform_samples": tv_uniform,
    }


def _aggregate(records: list[dict]) -> dict[str, float | int]:
    keys = (
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
    aggregate: dict[str, float | int] = {
        "trajectory_count": len(records),
        "total_diagnostic_samples": int(sum(record["diagnostic_samples"] for record in records)),
        "maximum_electronic_norm_error": float(
            max(record["maximum_electronic_norm_error"] for record in records)
        ),
        "maximum_sampled_abs_energy_bath_residual_eV": float(
            max(record["maximum_sampled_abs_energy_bath_residual_eV"] for record in records)
        ),
    }
    for key in keys:
        values = np.asarray([record[key] for record in records], dtype=np.float64)
        finite = values[np.isfinite(values)]
        aggregate[f"{key}_ensemble_mean"] = float(np.mean(finite)) if finite.size else float("nan")
        aggregate[f"{key}_ensemble_std"] = (
            float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
        )
    return aggregate


def _markdown(payload: dict) -> str:
    lines = [
        "# D5a electronic thermalization diagnostic",
        "",
        f"Lattice: {payload['size']}x{payload['size']}; bath: {payload['temperature_K']} K",
        f"gamma_u=gamma_v={payload['gamma_u_per_fs']} fs^-1; dt={payload['dt_fs']} fs",
        f"Time: {payload['final_time_fs']} fs; burn-in: {payload['burn_in_fs']} fs; diagnostic interval: {payload['diagnostic_interval_fs']} fs",
        f"Zero-mode policy: {payload['zero_mode_policy']}; electronic propagation: CF4-Lanczos m={payload['krylov_dimension']}",
        "",
        "D5a is diagnostic only: the electronic dynamics is unchanged from D4b.",
        "",
        "| seed | <Tlat> [K] | <heating> | early | late | slope/ps | <beta_eff/beta_bath> | TV canonical | TV uniform | closer uniform fraction | entropy/logN | max norm err |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in payload["trajectories"]:
        lines.append(
            "| {seed} | {lattice_temperature_mean_K:.4f} | {heating_coordinate_mean:.6g} | {heating_coordinate_early_mean:.6g} | {heating_coordinate_late_mean:.6g} | {heating_coordinate_slope_per_ps:.3e} | {beta_eff_over_beta_bath_mean_finite:.6g} | {tv_to_canonical_mean:.6g} | {tv_to_uniform_mean:.6g} | {closer_to_uniform_fraction:.4f} | {normalized_occupation_entropy_mean:.6g} | {maximum_electronic_norm_error:.3e} |".format(**record)
        )
    aggregate = payload["ensemble"]
    lines += [
        "",
        "## Ensemble summary",
        "",
        f"Trajectories: {aggregate['trajectory_count']}; post-burn-in diagnostic samples: {aggregate['total_diagnostic_samples']}.",
        f"Mean heating coordinate: {aggregate['heating_coordinate_mean_ensemble_mean']:.6g} +/- {aggregate['heating_coordinate_mean_ensemble_std']:.3g} across trajectories.",
        f"Mean finite beta_eff/beta_bath: {aggregate['beta_eff_over_beta_bath_mean_finite_ensemble_mean']:.6g} +/- {aggregate['beta_eff_over_beta_bath_mean_finite_ensemble_std']:.3g}.",
        f"Mean TV distance to canonical: {aggregate['tv_to_canonical_mean_ensemble_mean']:.6g}; to uniform: {aggregate['tv_to_uniform_mean_ensemble_mean']:.6g}.",
        f"Mean fraction of snapshots closer to uniform than canonical: {aggregate['closer_to_uniform_fraction_ensemble_mean']:.6g}.",
        f"Maximum electronic norm error: {aggregate['maximum_electronic_norm_error']:.3e}.",
        "",
        "No automatic physical pass/fail threshold is applied. These quantities are the D5a evidence used to decide whether D5b is required.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--temperature-K", type=float, default=300.0)
    parser.add_argument("--gamma-u-per-fs", type=float, default=0.01)
    parser.add_argument("--gamma-v-per-fs", type=float, default=0.01)
    parser.add_argument("--dt-fs", type=float, default=0.2)
    parser.add_argument("--final-time-fs", type=float, default=10000.0)
    parser.add_argument("--burn-in-fs", type=float, default=2000.0)
    parser.add_argument("--diagnostic-interval-fs", type=float, default=100.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260903, 20260904, 20260905, 20260906])
    parser.add_argument("--zero-mode-policy", choices=("retain", "project"), default="project")
    parser.add_argument("--krylov-dimension", type=int, default=6)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    bath = LangevinBath(
        temperature_K=args.temperature_K,
        gamma_u_per_fs=args.gamma_u_per_fs,
        gamma_v_per_fs=args.gamma_v_per_fs,
    )
    initial, parameters, static_metadata = _prepare_initial_state(
        args.size,
        args.zero_mode_policy,
    )
    trajectories = [
        _run_seed(
            initial,
            parameters,
            bath,
            dt_fs=args.dt_fs,
            final_time_fs=args.final_time_fs,
            burn_in_fs=args.burn_in_fs,
            diagnostic_interval_fs=args.diagnostic_interval_fs,
            seed=seed,
            zero_mode_policy=args.zero_mode_policy,
            krylov_dimension=args.krylov_dimension,
        )
        for seed in args.seeds
    ]
    payload = {
        "size": args.size,
        "temperature_K": args.temperature_K,
        "gamma_u_per_fs": args.gamma_u_per_fs,
        "gamma_v_per_fs": args.gamma_v_per_fs,
        "dt_fs": args.dt_fs,
        "final_time_fs": args.final_time_fs,
        "burn_in_fs": args.burn_in_fs,
        "diagnostic_interval_fs": args.diagnostic_interval_fs,
        "zero_mode_policy": args.zero_mode_policy,
        "krylov_dimension": args.krylov_dimension,
        "seeds": args.seeds,
        "static": static_metadata,
        "trajectories": trajectories,
        "ensemble": _aggregate(trajectories),
    }
    text = _markdown(payload)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
