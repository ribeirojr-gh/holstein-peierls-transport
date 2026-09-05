#!/usr/bin/env python3
"""D6e pair IDC screening benchmark.

DP, BM and MA are compared at one common *numerical-control* decoherence
interval.  The benchmark does not select a material-specific decoherence time
and does not assume that the one-polaron D5 result transfers to pair dynamics.

Diagnostics are evaluated from the pre-collapse populations already returned by
each IDC event, avoiding a redundant diagonalization.  The bipolaron event basis
is restricted to the symmetric spatial singlet sector; the exciton event basis
is the full distinguishable e-h sector.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from holstein_peierls.dynamics.electronic_thermalization import (
    canonical_occupations,
    distribution_energy_eV,
    effective_inverse_temperature_eV_inv,
    total_variation_distance,
)
from holstein_peierls.dynamics.langevin import BOLTZMANN_EV_PER_K, LangevinBath
from holstein_peierls.dynamics.pair_coupled import (
    MovingPairHamiltonianFactory,
    PairCoupledState,
    PairLatticeMasses,
    pair_dynamic_total_energy,
)
from holstein_peierls.dynamics.pair_decoherence import apply_pair_instantaneous_decoherence
from holstein_peierls.dynamics.pair_thermal import (
    pair_coupled_baoab_step,
    pair_kinetic_temperature_K,
    pair_thermostatted_degrees_of_freedom,
)


def _load_d6d_helpers():
    path = Path(__file__).with_name("d6d_pair_electronic_thermalization.py")
    spec = importlib.util.spec_from_file_location("d6d_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load D6d helper module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _segment_mean(values: list[float], which: str) -> float:
    count = max(1, len(values) // 3)
    data = values[:count] if which == "early" else values[-count:]
    return _mean(data)


def _slope_per_ps(times_fs: list[float], values: list[float]) -> float:
    times = np.asarray(times_fs, dtype=np.float64) * 1.0e-3
    data = np.asarray(values, dtype=np.float64)
    if data.size < 2:
        return float("nan")
    slope, _ = np.polyfit(times, data, 1)
    return float(slope)


def _precollapse_metrics(event, temperature_K: float) -> dict[str, float]:
    energies = event.adiabatic_energies_eV
    population = event.adiabatic_populations_before
    canonical = canonical_occupations(energies, temperature_K)
    uniform = np.full(energies.size, 1.0 / energies.size, dtype=np.float64)
    propagated = distribution_energy_eV(energies, population)
    canonical_energy = distribution_energy_eV(energies, canonical)
    uniform_energy = distribution_energy_eV(energies, uniform)
    denominator = uniform_energy - canonical_energy
    heating = float((propagated - canonical_energy) / denominator)
    expected_after = float(np.dot(event.collapse_probabilities, energies))
    expected_post_heating = float((expected_after - canonical_energy) / denominator)
    realized_post_heating = float((event.electronic_energy_after_eV - canonical_energy) / denominator)
    ground = float(energies[0])
    ground_mask = np.abs(energies - ground) <= 1.0e-10
    ground_population = float(np.sum(population[ground_mask]))
    canonical_ground = float(np.sum(canonical[ground_mask]))
    beta_eff = effective_inverse_temperature_eV_inv(energies, propagated)
    beta_bath = 1.0 / (BOLTZMANN_EV_PER_K * temperature_K)
    return {
        "heating": heating,
        "tv_canonical": total_variation_distance(population, canonical),
        "tv_uniform": total_variation_distance(population, uniform),
        "ground_population": ground_population,
        "canonical_ground_population": canonical_ground,
        "ground_mismatch": abs(ground_population - canonical_ground),
        "beta_ratio": float(beta_eff / beta_bath) if np.isfinite(beta_eff) else float(beta_eff),
        "expected_post_heating": expected_post_heating,
        "realized_post_heating": realized_post_heating,
    }


def run_trajectory(
    sector: str,
    scheme: str,
    size: int,
    masses: PairLatticeMasses,
    bath: LangevinBath,
    seed: int,
    dt_fs: float,
    final_time_fs: float,
    burn_in_fs: float,
    decoherence_interval_fs: float,
    krylov_dimension: int,
) -> dict:
    helpers = _load_d6d_helpers()
    initial, parameters = helpers.initial_state(sector, size)
    current = PairCoupledState(
        initial.lattice.copy(), initial.velocity.copy(), initial.electronic_state.copy()
    )
    factory = MovingPairHamiltonianFactory(sector, parameters)
    lattice_rng = np.random.default_rng(seed)
    scheme_code = {"dp": 1, "bm": 2, "ma": 3}[scheme]
    sector_code = 0 if sector == "bipolaron" else 1000
    idc_rng = np.random.default_rng(seed + 100_000 + 10_000 * scheme_code + sector_code)
    steps = int(round(final_time_fs / dt_fs))
    burn_steps = int(round(burn_in_fs / dt_fs))
    stride = int(round(decoherence_interval_fs / dt_fs))
    if not np.isclose(stride * dt_fs, decoherence_interval_fs, rtol=0.0, atol=1.0e-12):
        raise ValueError("decoherence interval must be an integer multiple of dt")
    dof = pair_thermostatted_degrees_of_freedom(parameters, "project")
    initial_energy = pair_dynamic_total_energy(initial, sector, parameters, masses, factory=factory).total

    lattice_heat = 0.0
    electronic_exchange = 0.0
    times: list[float] = []
    heating: list[float] = []
    tv_can: list[float] = []
    tv_uniform: list[float] = []
    ground_mismatch: list[float] = []
    beta_ratio: list[float] = []
    expected_post: list[float] = []
    realized_post: list[float] = []
    temperatures: list[float] = []
    max_residual = 0.0
    max_norm_error = 0.0
    max_constraint = 0.0
    event_count = 0
    start = perf_counter()

    for step in range(steps):
        thermal = pair_coupled_baoab_step(
            current, sector, parameters, masses, bath, lattice_rng,
            step * dt_fs, dt_fs, zero_mode_policy="project",
            krylov_dimension=krylov_dimension, factory=factory,
        )
        current = thermal.state
        lattice_heat += thermal.bath_heat_eV
        if (step + 1) % stride != 0:
            continue

        event = apply_pair_instantaneous_decoherence(
            current.lattice, sector, parameters, current.electronic_state,
            bath.temperature_K, scheme, idc_rng, factory=factory,
        )
        if step + 1 > burn_steps:
            metrics = _precollapse_metrics(event, bath.temperature_K)
            times.append((step + 1) * dt_fs)
            heating.append(metrics["heating"])
            tv_can.append(metrics["tv_canonical"])
            tv_uniform.append(metrics["tv_uniform"])
            ground_mismatch.append(metrics["ground_mismatch"])
            beta_ratio.append(metrics["beta_ratio"])
            expected_post.append(metrics["expected_post_heating"])
            realized_post.append(metrics["realized_post_heating"])
            temperatures.append(
                pair_kinetic_temperature_K(current.velocity, masses, degrees_of_freedom=dof)
            )

        current = PairCoupledState(
            current.lattice, current.velocity, event.electronic_state.copy()
        )
        electronic_exchange += event.electronic_environment_exchange_eV
        event_count += 1
        energy = pair_dynamic_total_energy(current, sector, parameters, masses, factory=factory).total
        max_residual = max(max_residual, abs((energy - initial_energy) - lattice_heat - electronic_exchange))
        max_norm_error = max(max_norm_error, abs(float(np.linalg.norm(current.electronic_state)) - 1.0))
        max_constraint = max(max_constraint, helpers.constraint_error(sector, current, parameters.n_sites))

    if not heating:
        raise RuntimeError("no post-burn-in IDC samples")
    return {
        "sector": sector,
        "scheme": scheme,
        "seed": seed,
        "elapsed_seconds": float(perf_counter() - start),
        "idc_events": event_count,
        "diagnostic_samples": len(heating),
        "mean_lattice_temperature_K": _mean(temperatures),
        "mean_pre_heating": _mean(heating),
        "early_pre_heating": _segment_mean(heating, "early"),
        "late_pre_heating": _segment_mean(heating, "late"),
        "heating_slope_per_ps": _slope_per_ps(times, heating),
        "mean_tv_canonical": _mean(tv_can),
        "mean_tv_uniform": _mean(tv_uniform),
        "mean_ground_mismatch": _mean(ground_mismatch),
        "mean_beta_ratio_finite": _mean([v for v in beta_ratio if np.isfinite(v)]),
        "mean_expected_post_heating": _mean(expected_post),
        "mean_realized_post_heating": _mean(realized_post),
        "electronic_environment_exchange_eV": float(electronic_exchange),
        "electronic_exchange_rate_eV_per_ps": float(electronic_exchange / (final_time_fs * 1.0e-3)),
        "maximum_generalized_balance_residual_eV": float(max_residual),
        "maximum_norm_error": float(max_norm_error),
        "maximum_sector_constraint_error": float(max_constraint),
    }


def aggregate(runs: list[dict]) -> list[dict]:
    result = []
    keys = [
        "mean_lattice_temperature_K", "mean_pre_heating", "early_pre_heating",
        "late_pre_heating", "heating_slope_per_ps", "mean_tv_canonical",
        "mean_tv_uniform", "mean_ground_mismatch", "mean_beta_ratio_finite",
        "mean_expected_post_heating", "mean_realized_post_heating",
        "electronic_exchange_rate_eV_per_ps",
    ]
    for sector in ("bipolaron", "exciton"):
        for scheme in ("dp", "bm", "ma"):
            group = [r for r in runs if r["sector"] == sector and r["scheme"] == scheme]
            item = {"sector": sector, "scheme": scheme, "trajectory_count": len(group)}
            for key in keys:
                values = [r[key] for r in group]
                item[key] = _mean(values)
                item[f"{key}_std"] = _std(values)
            item["maximum_generalized_balance_residual_eV"] = max(r["maximum_generalized_balance_residual_eV"] for r in group)
            item["maximum_norm_error"] = max(r["maximum_norm_error"] for r in group)
            item["maximum_sector_constraint_error"] = max(r["maximum_sector_constraint_error"] for r in group)
            result.append(item)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=4)
    p.add_argument("--temperature-K", type=float, default=300.0)
    p.add_argument("--gamma-u-per-fs", type=float, default=0.01)
    p.add_argument("--gamma-v-per-fs", type=float, default=0.01)
    p.add_argument("--dt-fs", type=float, default=0.2)
    p.add_argument("--final-time-fs", type=float, default=4000.0)
    p.add_argument("--burn-in-fs", type=float, default=1000.0)
    p.add_argument("--decoherence-interval-fs", type=float, default=100.0)
    p.add_argument("--seeds", nargs="+", type=int, default=[20260905, 20260906, 20260907, 20260908])
    p.add_argument("--krylov-dimension", type=int, default=8)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--markdown", type=Path, required=True)
    args = p.parse_args()

    masses = PairLatticeMasses(75000.0, 150000.0)
    bath = LangevinBath(args.temperature_K, args.gamma_u_per_fs, args.gamma_v_per_fs)
    runs = []
    for sector in ("bipolaron", "exciton"):
        for scheme in ("dp", "bm", "ma"):
            for seed in args.seeds:
                runs.append(run_trajectory(
                    sector, scheme, args.size, masses, bath, seed, args.dt_fs,
                    args.final_time_fs, args.burn_in_fs, args.decoherence_interval_fs,
                    args.krylov_dimension,
                ))
    aggregates = aggregate(runs)
    checks = {}
    for item in aggregates:
        prefix = f"{item['sector']}_{item['scheme']}"
        checks[prefix + "_temperature"] = 240.0 <= item["mean_lattice_temperature_K"] <= 360.0
        checks[prefix + "_balance"] = item["maximum_generalized_balance_residual_eV"] < 1.0e-4
        checks[prefix + "_norm"] = item["maximum_norm_error"] < 1.0e-10
        checks[prefix + "_constraints"] = item["maximum_sector_constraint_error"] < 1.0e-10
    numerical_pass = all(checks.values())
    payload = vars(args).copy()
    payload["output"] = str(args.output)
    payload["markdown"] = str(args.markdown)
    payload.update(
        runs=runs,
        aggregates=aggregates,
        numerical_checks=checks,
        numerical_pass=numerical_pass,
        physical_selection_is_manual=True,
        no_weighted_selection_score=True,
        interpretation=(
            "D6e is a one-interval screening benchmark. Numerical PASS does not select "
            "DP/BM/MA or validate the 100 fs interval as a material parameter."
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# D6e pair IDC screening", "",
        f"Control: {args.size}x{args.size}, {args.temperature_K:g} K, dt={args.dt_fs:g} fs, t_d={args.decoherence_interval_fs:g} fs, time={args.final_time_fs/1000:g} ps.", "",
        "| sector | scheme | <T> [K] | <pre heat> | late | slope/ps | TV can | ground mismatch | expected post heat | Qe rate [eV/ps] | max balance [eV] |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in aggregates:
        lines.append(
            f"| {item['sector']} | {item['scheme']} | {item['mean_lattice_temperature_K']:.3f} | "
            f"{item['mean_pre_heating']:.5f} | {item['late_pre_heating']:.5f} | "
            f"{item['heating_slope_per_ps']:.3e} | {item['mean_tv_canonical']:.5f} | "
            f"{item['mean_ground_mismatch']:.5f} | {item['mean_expected_post_heating']:.5f} | "
            f"{item['electronic_exchange_rate_eV_per_ps']:.3e} | "
            f"{item['maximum_generalized_balance_residual_eV']:.3e} |"
        )
    lines += ["", "Numerical gates: " + ("PASS" if numerical_pass else "FAIL"), "", "Physical selection remains manual; no weighted score is used.", ""]
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    if not numerical_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
