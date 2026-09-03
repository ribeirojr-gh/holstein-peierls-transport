"""Reproducible D4b finite-temperature coupled-bath benchmark.

The default control starts from the relaxed 20x20 one-polaron state, sets all
lattice velocities to zero, couples only the classical lattice to a 300 K
Markovian Langevin bath, and propagates the electron coherently with the
D2/D3-selected CF4-Lanczos method.

No transport coefficient is inferred by this benchmark. Its purpose is to
validate thermal statistics, energy exchange with the bath, electronic norm,
zero-mode handling, timestep sensitivity, and multi-picosecond stability.
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
    lattice_kinetic_energy,
)
from holstein_peierls.dynamics.langevin import LangevinBath, kinetic_temperature_K
from holstein_peierls.dynamics.thermal import (
    coupled_baoab_step,
    project_inter_molecular_zero_modes,
    thermostatted_kinetic_degrees_of_freedom,
    zero_mode_means,
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


def _ipr(psi: np.ndarray) -> float:
    probability = np.abs(psi) ** 2
    probability = probability / float(np.sum(probability))
    return float(np.sum(probability * probability))


def _population(psi: np.ndarray) -> np.ndarray:
    probability = np.abs(psi) ** 2
    return np.asarray(probability / float(np.sum(probability)), dtype=np.float64)


def _run_trajectory(
    initial: CoupledEhrenfestState,
    parameters: StaticPolaronParameters,
    bath: LangevinBath,
    *,
    dt_fs: float,
    final_time_fs: float,
    burn_in_fs: float,
    seed: int,
    zero_mode_policy: str,
    sample_stride: int,
    krylov_dimension: int,
) -> dict[str, float | int | str]:
    steps = int(round(final_time_fs / dt_fs))
    if steps <= 0 or not np.isclose(steps * dt_fs, final_time_fs):
        raise ValueError("final_time_fs must be a positive integer multiple of dt_fs")
    burn_steps = int(round(burn_in_fs / dt_fs))
    if burn_steps < 0 or burn_steps >= steps:
        raise ValueError("burn_in_fs must lie in [0, final_time_fs)")
    if sample_stride <= 0:
        raise ValueError("sample_stride must be positive")

    current = CoupledEhrenfestState(
        initial.lattice.copy(),
        initial.velocity.copy(),
        initial.electronic_state.copy(),
    )
    rng = np.random.default_rng(seed)
    initial_energy = dynamic_total_energy(
        current.lattice,
        current.velocity,
        parameters,
        current.electronic_state,
    ).total
    initial_norm = float(np.linalg.norm(current.electronic_state))
    initial_population = _population(current.electronic_state)
    initial_ipr = _ipr(current.electronic_state)
    initial_lattice = current.lattice.copy()

    dof = thermostatted_kinetic_degrees_of_freedom(parameters, zero_mode_policy)
    bath_heat = 0.0
    temperatures: list[float] = []
    energies: list[float] = []
    kinetic_energies: list[float] = []
    residuals: list[float] = []
    max_norm_error = 0.0
    max_zero_mode_mean = 0.0
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
        means = zero_mode_means(current)
        max_zero_mode_mean = max(
            max_zero_mode_mean,
            *(abs(value) for value in means.values()),
        )

        if (step + 1) % sample_stride == 0 or step + 1 == steps:
            energy = dynamic_total_energy(
                current.lattice,
                current.velocity,
                parameters,
                current.electronic_state,
            ).total
            residual = energy - initial_energy - bath_heat
            residuals.append(float(residual))
            if step + 1 > burn_steps:
                temperatures.append(
                    kinetic_temperature_K(
                        current.velocity,
                        parameters,
                        degrees_of_freedom=dof,
                    )
                )
                energies.append(float(energy))
                kinetic_energies.append(
                    lattice_kinetic_energy(current.velocity, parameters)
                )
    elapsed = perf_counter() - start

    final_energy = dynamic_total_energy(
        current.lattice,
        current.velocity,
        parameters,
        current.electronic_state,
    ).total
    final_population = _population(current.electronic_state)
    final_ipr = _ipr(current.electronic_state)
    final_means = zero_mode_means(current)

    max_excursion = max(
        float(np.max(np.abs(current.lattice.u - initial_lattice.u))),
        float(np.max(np.abs(current.lattice.vx - initial_lattice.vx))),
        float(np.max(np.abs(current.lattice.vy - initial_lattice.vy))),
    )

    return {
        "dt_fs": float(dt_fs),
        "final_time_fs": float(final_time_fs),
        "burn_in_fs": float(burn_in_fs),
        "steps": steps,
        "sample_stride": int(sample_stride),
        "seed": int(seed),
        "zero_mode_policy": zero_mode_policy,
        "kinetic_degrees_of_freedom": int(dof),
        "elapsed_seconds": float(elapsed),
        "hamiltonian_evaluations": int(h_evaluations),
        "hamiltonian_applications": int(h_applications),
        "initial_energy_eV": float(initial_energy),
        "final_energy_eV": float(final_energy),
        "matter_energy_change_eV": float(final_energy - initial_energy),
        "accumulated_bath_heat_eV": float(bath_heat),
        "final_energy_bath_residual_eV": float(final_energy - initial_energy - bath_heat),
        "maximum_sampled_abs_energy_bath_residual_eV": float(
            max((abs(value) for value in residuals), default=0.0)
        ),
        "temperature_mean_K": float(np.mean(temperatures)),
        "temperature_std_K": float(np.std(temperatures, ddof=1)) if len(temperatures) > 1 else 0.0,
        "temperature_samples": int(len(temperatures)),
        "mean_lattice_kinetic_energy_eV": float(np.mean(kinetic_energies)),
        "mean_matter_energy_eV": float(np.mean(energies)),
        "matter_energy_std_eV": float(np.std(energies, ddof=1)) if len(energies) > 1 else 0.0,
        "maximum_electronic_norm_error": float(max_norm_error),
        "initial_ipr": float(initial_ipr),
        "final_ipr": float(final_ipr),
        "population_l2_change": float(np.linalg.norm(final_population - initial_population)),
        "maximum_lattice_coordinate_excursion_A": float(max_excursion),
        "maximum_zero_mode_mean": float(max_zero_mode_mean),
        "final_zero_mode_vx_A": float(final_means["vx_A"]),
        "final_zero_mode_vy_A": float(final_means["vy_A"]),
        "final_zero_mode_vx_velocity_A_per_fs": float(final_means["vx_velocity_A_per_fs"]),
        "final_zero_mode_vy_velocity_A_per_fs": float(final_means["vy_velocity_A_per_fs"]),
    }


def _markdown(payload: dict) -> str:
    lines = [
        "# D4b coupled finite-temperature bath benchmark",
        "",
        f"Lattice: {payload['size']}x{payload['size']}",
        f"Bath: {payload['temperature_K']} K; gamma_u={payload['gamma_u_per_fs']} fs^-1; gamma_v={payload['gamma_v_per_fs']} fs^-1",
        f"Zero-mode policy: {payload['zero_mode_policy']}; RNG seed: {payload['seed']}",
        "Field: 0; electronic method: CF4-Lanczos",
        "",
        "| dt [fs] | time [fs] | elapsed [s] | <Tkin> [K] | std(T) [K] | max norm err | max |dE-Qbath| [eV] | final dE-Qbath [eV] | max zero-mode mean | IPR final | pop L2 change |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in payload["runs"]:
        lines.append(
            "| {dt_fs:.4g} | {final_time_fs:.6g} | {elapsed_seconds:.6g} | {temperature_mean_K:.6g} | {temperature_std_K:.6g} | {maximum_electronic_norm_error:.3e} | {maximum_sampled_abs_energy_bath_residual_eV:.3e} | {final_energy_bath_residual_eV:.3e} | {maximum_zero_mode_mean:.3e} | {final_ipr:.8f} | {population_l2_change:.3e} |".format(**record)
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--temperature-K", type=float, default=300.0)
    parser.add_argument("--gamma-u-per-fs", type=float, default=0.01)
    parser.add_argument("--gamma-v-per-fs", type=float, default=0.01)
    parser.add_argument("--dt-values", type=float, nargs="+", default=[0.2])
    parser.add_argument("--final-time-fs", type=float, default=10000.0)
    parser.add_argument("--burn-in-fs", type=float, default=2000.0)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--zero-mode-policy", choices=("retain", "project"), default="project")
    parser.add_argument("--sample-stride", type=int, default=10)
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
    runs = [
        _run_trajectory(
            initial,
            parameters,
            bath,
            dt_fs=dt,
            final_time_fs=args.final_time_fs,
            burn_in_fs=args.burn_in_fs,
            seed=args.seed,
            zero_mode_policy=args.zero_mode_policy,
            sample_stride=args.sample_stride,
            krylov_dimension=args.krylov_dimension,
        )
        for dt in args.dt_values
    ]
    payload = {
        "size": args.size,
        "temperature_K": args.temperature_K,
        "gamma_u_per_fs": args.gamma_u_per_fs,
        "gamma_v_per_fs": args.gamma_v_per_fs,
        "zero_mode_policy": args.zero_mode_policy,
        "seed": args.seed,
        "krylov_dimension": args.krylov_dimension,
        "static": static_metadata,
        "runs": runs,
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
