"""D5b benchmark of fixed-interval instantaneous decoherence controls.

The benchmark compares IDC-DP, IDC-BM and IDC-MA on the same 20x20, 300 K D4
control used for D5a. The default decoherence interval of 100 fs is a numerical
control, not a material-specific prediction. Sensitivity to the decoherence time
is deliberately left to the next gate after the algorithms themselves are
validated.

For a collapsed stochastic trajectory, a post-collapse pure adiabatic state is
not expected to resemble a canonical mixed distribution in a single
realization. Therefore equilibrium diagnostics are sampled immediately BEFORE
each IDC event. Separately, the expected and realized post-collapse electronic
energies are recorded. Ensemble/time aggregation then tests whether repeated IDC
events counteract the systematic Ehrenfest overheating observed in D5a.
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
from holstein_peierls.dynamics.electronic_decoherence import (
    apply_instantaneous_decoherence,
)
from holstein_peierls.dynamics.electronic_thermalization import (
    canonical_occupations,
    diagnose_electronic_thermalization,
)
from holstein_peierls.dynamics.langevin import LangevinBath, kinetic_temperature_K
from holstein_peierls.dynamics.thermal import (
    coupled_baoab_step,
    project_inter_molecular_zero_modes,
    thermostatted_kinetic_degrees_of_freedom,
)
from holstein_peierls.dynamics.thermal_decoherence import decoherence_interval_steps
from holstein_peierls.electronic import solve_ground_state
from holstein_peierls.parameters import StaticPolaronParameters
from holstein_peierls.polaron import solve_static_polaron


def _prepare_initial_state(
    size: int,
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
    state = project_inter_molecular_zero_modes(state)
    metadata = {
        "static_iterations": static.diagnostics.iterations,
        "static_converged": static.diagnostics.converged,
        "static_total_energy_eV": static.total_energy,
        "static_formation_energy_eV": static.formation_energy,
    }
    return state, parameters, metadata


def _finite_mean(values: list[float]) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    return float(np.mean(finite)) if finite.size else float("nan")


def _window_mean(values: list[float], fraction: float, *, tail: bool) -> float:
    if not values:
        return float("nan")
    count = max(1, int(round(fraction * len(values))))
    selected = values[-count:] if tail else values[:count]
    return float(np.mean(selected))


def _slope_per_ps(times_fs: list[float], values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    x = np.asarray(times_fs, dtype=np.float64) / 1000.0
    y = np.asarray(values, dtype=np.float64)
    x = x - float(np.mean(x))
    denominator = float(np.dot(x, x))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(x, y - float(np.mean(y))) / denominator)


def _energy_coordinate(
    energy_eV: float,
    canonical_energy_eV: float,
    infinite_temperature_energy_eV: float,
) -> float:
    denominator = infinite_temperature_energy_eV - canonical_energy_eV
    if abs(denominator) <= 1.0e-14:
        return float("nan")
    return float((energy_eV - canonical_energy_eV) / denominator)


def _run_scheme_trajectory(
    initial: CoupledEhrenfestState,
    parameters: StaticPolaronParameters,
    bath: LangevinBath,
    *,
    scheme: str,
    dt_fs: float,
    final_time_fs: float,
    burn_in_fs: float,
    decoherence_interval_fs: float,
    lattice_seed: int,
    decoherence_seed: int,
    krylov_dimension: int,
) -> dict[str, float | int | str]:
    steps = int(round(final_time_fs / dt_fs))
    if steps <= 0 or not np.isclose(steps * dt_fs, final_time_fs):
        raise ValueError("final_time_fs must be an integer multiple of dt_fs")
    burn_steps = int(round(burn_in_fs / dt_fs))
    if burn_steps < 0 or burn_steps >= steps:
        raise ValueError("burn_in_fs must lie in [0, final_time_fs)")
    event_stride = decoherence_interval_steps(decoherence_interval_fs, dt_fs)

    current = CoupledEhrenfestState(
        initial.lattice.copy(),
        initial.velocity.copy(),
        initial.electronic_state.copy(),
    )
    lattice_rng = np.random.default_rng(lattice_seed)
    decoherence_rng = np.random.default_rng(decoherence_seed)
    initial_energy = dynamic_total_energy(
        current.lattice, current.velocity, parameters, current.electronic_state
    ).total
    initial_norm = float(np.linalg.norm(current.electronic_state))
    dof = thermostatted_kinetic_degrees_of_freedom(parameters, "project")

    lattice_heat = 0.0
    electronic_exchange = 0.0
    max_norm_error = 0.0
    max_generalized_residual = 0.0
    events_total = 0

    sample_times: list[float] = []
    temperatures: list[float] = []
    pre_heating: list[float] = []
    pre_beta_ratio: list[float] = []
    pre_tv_canonical: list[float] = []
    pre_tv_uniform: list[float] = []
    pre_ground_population: list[float] = []
    canonical_ground_population: list[float] = []
    expected_post_heating: list[float] = []
    realized_post_heating: list[float] = []
    collapse_energy_jumps: list[float] = []

    start = perf_counter()
    for step in range(steps):
        time_fs = step * dt_fs
        thermal = coupled_baoab_step(
            current,
            parameters,
            bath,
            lattice_rng,
            time_fs,
            dt_fs,
            zero_mode_policy="project",
            electronic_method="cfm4_lanczos",
            krylov_dimension=krylov_dimension,
        )
        current = thermal.state
        lattice_heat += thermal.bath_heat_eV
        max_norm_error = max(
            max_norm_error,
            abs(float(np.linalg.norm(current.electronic_state)) - initial_norm),
        )

        if (step + 1) % event_stride == 0:
            event_time = (step + 1) * dt_fs
            pre = diagnose_electronic_thermalization(
                current.lattice,
                parameters,
                current.electronic_state,
                bath.temperature_K,
            )
            event = apply_instantaneous_decoherence(
                current.lattice,
                parameters,
                current.electronic_state,
                bath.temperature_K,
                scheme,
                decoherence_rng,
            )
            events_total += 1

            canonical = canonical_occupations(event.adiabatic_energies_eV, bath.temperature_K)
            canonical_energy = float(np.dot(canonical, event.adiabatic_energies_eV))
            infinite_energy = float(np.mean(event.adiabatic_energies_eV))
            expected_after = float(
                np.dot(event.collapse_probabilities, event.adiabatic_energies_eV)
            )
            expected_coordinate = _energy_coordinate(
                expected_after, canonical_energy, infinite_energy
            )
            realized_coordinate = _energy_coordinate(
                event.electronic_energy_after_eV, canonical_energy, infinite_energy
            )

            electronic_exchange += event.electronic_environment_exchange_eV
            current = CoupledEhrenfestState(
                current.lattice.copy(),
                current.velocity.copy(),
                event.electronic_state.copy(),
            )

            total_energy = dynamic_total_energy(
                current.lattice,
                current.velocity,
                parameters,
                current.electronic_state,
            ).total
            residual = total_energy - initial_energy - lattice_heat - electronic_exchange
            max_generalized_residual = max(max_generalized_residual, abs(float(residual)))

            if step + 1 > burn_steps:
                sample_times.append(float(event_time))
                temperatures.append(
                    kinetic_temperature_K(
                        current.velocity,
                        parameters,
                        degrees_of_freedom=dof,
                    )
                )
                pre_heating.append(float(pre.heating_coordinate))
                pre_beta_ratio.append(float(pre.beta_eff_over_beta_bath))
                pre_tv_canonical.append(float(pre.tv_to_canonical))
                pre_tv_uniform.append(float(pre.tv_to_uniform))
                pre_ground_population.append(float(pre.ground_manifold_population))
                canonical_ground_population.append(
                    float(pre.canonical_ground_manifold_population)
                )
                expected_post_heating.append(float(expected_coordinate))
                realized_post_heating.append(float(realized_coordinate))
                collapse_energy_jumps.append(
                    float(event.electronic_environment_exchange_eV)
                )

    elapsed = perf_counter() - start
    final_energy = dynamic_total_energy(
        current.lattice, current.velocity, parameters, current.electronic_state
    ).total
    final_residual = final_energy - initial_energy - lattice_heat - electronic_exchange

    closer_uniform = [u < c for c, u in zip(pre_tv_canonical, pre_tv_uniform)]
    nonpositive_beta = [value <= 0.0 for value in pre_beta_ratio if np.isfinite(value)]
    return {
        "scheme": scheme,
        "lattice_seed": int(lattice_seed),
        "decoherence_seed": int(decoherence_seed),
        "dt_fs": float(dt_fs),
        "final_time_fs": float(final_time_fs),
        "burn_in_fs": float(burn_in_fs),
        "decoherence_interval_fs": float(decoherence_interval_fs),
        "steps": int(steps),
        "events_total": int(events_total),
        "post_burn_samples": int(len(pre_heating)),
        "elapsed_seconds": float(elapsed),
        "temperature_mean_K": float(np.mean(temperatures)),
        "temperature_std_K": float(np.std(temperatures, ddof=1)) if len(temperatures) > 1 else 0.0,
        "pre_heating_coordinate_mean": float(np.mean(pre_heating)),
        "pre_heating_coordinate_early_mean": _window_mean(pre_heating, 0.25, tail=False),
        "pre_heating_coordinate_late_mean": _window_mean(pre_heating, 0.25, tail=True),
        "pre_heating_coordinate_slope_per_ps": _slope_per_ps(sample_times, pre_heating),
        "pre_beta_eff_over_beta_bath_mean_finite": _finite_mean(pre_beta_ratio),
        "pre_nonpositive_beta_fraction": float(np.mean(nonpositive_beta)) if nonpositive_beta else 0.0,
        "pre_tv_to_canonical_mean": float(np.mean(pre_tv_canonical)),
        "pre_tv_to_uniform_mean": float(np.mean(pre_tv_uniform)),
        "pre_closer_to_uniform_fraction": float(np.mean(closer_uniform)),
        "pre_ground_manifold_population_mean": float(np.mean(pre_ground_population)),
        "canonical_ground_manifold_population_mean": float(np.mean(canonical_ground_population)),
        "expected_post_heating_coordinate_mean": float(np.mean(expected_post_heating)),
        "realized_post_heating_coordinate_mean": float(np.mean(realized_post_heating)),
        "collapse_energy_jump_mean_eV": float(np.mean(collapse_energy_jumps)),
        "collapse_energy_jump_std_eV": float(np.std(collapse_energy_jumps, ddof=1)) if len(collapse_energy_jumps) > 1 else 0.0,
        "accumulated_lattice_bath_heat_eV": float(lattice_heat),
        "accumulated_electronic_environment_exchange_eV": float(electronic_exchange),
        "final_generalized_energy_residual_eV": float(final_residual),
        "maximum_event_abs_generalized_energy_residual_eV": float(max_generalized_residual),
        "maximum_electronic_norm_error": float(max_norm_error),
    }


def _aggregate(records: list[dict]) -> dict[str, dict[str, float | int]]:
    metrics = [
        "temperature_mean_K",
        "pre_heating_coordinate_mean",
        "pre_heating_coordinate_early_mean",
        "pre_heating_coordinate_late_mean",
        "pre_heating_coordinate_slope_per_ps",
        "pre_beta_eff_over_beta_bath_mean_finite",
        "pre_nonpositive_beta_fraction",
        "pre_tv_to_canonical_mean",
        "pre_tv_to_uniform_mean",
        "pre_closer_to_uniform_fraction",
        "pre_ground_manifold_population_mean",
        "canonical_ground_manifold_population_mean",
        "expected_post_heating_coordinate_mean",
        "realized_post_heating_coordinate_mean",
        "collapse_energy_jump_mean_eV",
        "accumulated_electronic_environment_exchange_eV",
        "maximum_event_abs_generalized_energy_residual_eV",
        "maximum_electronic_norm_error",
    ]
    result: dict[str, dict[str, float | int]] = {}
    for scheme in ("dp", "bm", "ma"):
        subset = [record for record in records if record["scheme"] == scheme]
        aggregate: dict[str, float | int] = {
            "trajectory_count": len(subset),
            "total_post_burn_samples": int(sum(int(record["post_burn_samples"]) for record in subset)),
        }
        for metric in metrics:
            values = np.asarray([float(record[metric]) for record in subset], dtype=np.float64)
            aggregate[f"{metric}_ensemble_mean"] = float(np.mean(values))
            aggregate[f"{metric}_ensemble_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        result[scheme] = aggregate
    return result


def _markdown(payload: dict) -> str:
    lines = [
        "# D5b instantaneous-decoherence benchmark",
        "",
        f"Lattice: {payload['size']}x{payload['size']}; T={payload['temperature_K']} K; dt={payload['dt_fs']} fs",
        f"IDC interval: {payload['decoherence_interval_fs']} fs (numerical control); zero-mode policy: project",
        "",
        "| scheme | trajectories | samples | <T> [K] | pre heat | pre heat late | slope [ps^-1] | beta_eff/beta | TV canonical | TV uniform | closer uniform | ground pop | canonical ground | expected post heat | realized post heat | max balance residual [eV] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scheme in ("dp", "bm", "ma"):
        a = payload["ensemble"][scheme]
        lines.append(
            "| {scheme} | {n} | {samples} | {temp:.3f} | {heat:.4f} | {late:.4f} | {slope:.4f} | {beta:.4f} | {tvc:.4f} | {tvu:.4f} | {closer:.3f} | {ground:.4f} | {cground:.4f} | {epost:.4f} | {rpost:.4f} | {res:.3e} |".format(
                scheme=scheme.upper(),
                n=a["trajectory_count"],
                samples=a["total_post_burn_samples"],
                temp=a["temperature_mean_K_ensemble_mean"],
                heat=a["pre_heating_coordinate_mean_ensemble_mean"],
                late=a["pre_heating_coordinate_late_mean_ensemble_mean"],
                slope=a["pre_heating_coordinate_slope_per_ps_ensemble_mean"],
                beta=a["pre_beta_eff_over_beta_bath_mean_finite_ensemble_mean"],
                tvc=a["pre_tv_to_canonical_mean_ensemble_mean"],
                tvu=a["pre_tv_to_uniform_mean_ensemble_mean"],
                closer=a["pre_closer_to_uniform_fraction_ensemble_mean"],
                ground=a["pre_ground_manifold_population_mean_ensemble_mean"],
                cground=a["canonical_ground_manifold_population_mean_ensemble_mean"],
                epost=a["expected_post_heating_coordinate_mean_ensemble_mean"],
                rpost=a["realized_post_heating_coordinate_mean_ensemble_mean"],
                res=a["maximum_event_abs_generalized_energy_residual_eV_ensemble_mean"],
            )
        )
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
    parser.add_argument("--decoherence-interval-fs", type=float, default=100.0)
    parser.add_argument("--lattice-seeds", type=int, nargs="+", default=[20260903, 20260904, 20260905, 20260906])
    parser.add_argument("--decoherence-seed-offset", type=int, default=1000000)
    parser.add_argument("--krylov-dimension", type=int, default=6)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    bath = LangevinBath(
        temperature_K=args.temperature_K,
        gamma_u_per_fs=args.gamma_u_per_fs,
        gamma_v_per_fs=args.gamma_v_per_fs,
    )
    initial, parameters, static_metadata = _prepare_initial_state(args.size)
    trajectories = []
    for scheme_index, scheme in enumerate(("dp", "bm", "ma")):
        for seed in args.lattice_seeds:
            trajectories.append(
                _run_scheme_trajectory(
                    initial,
                    parameters,
                    bath,
                    scheme=scheme,
                    dt_fs=args.dt_fs,
                    final_time_fs=args.final_time_fs,
                    burn_in_fs=args.burn_in_fs,
                    decoherence_interval_fs=args.decoherence_interval_fs,
                    lattice_seed=seed,
                    decoherence_seed=args.decoherence_seed_offset + 10000 * scheme_index + seed,
                    krylov_dimension=args.krylov_dimension,
                )
            )

    payload = {
        "size": args.size,
        "temperature_K": args.temperature_K,
        "gamma_u_per_fs": args.gamma_u_per_fs,
        "gamma_v_per_fs": args.gamma_v_per_fs,
        "dt_fs": args.dt_fs,
        "final_time_fs": args.final_time_fs,
        "burn_in_fs": args.burn_in_fs,
        "decoherence_interval_fs": args.decoherence_interval_fs,
        "lattice_seeds": args.lattice_seeds,
        "krylov_dimension": args.krylov_dimension,
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
