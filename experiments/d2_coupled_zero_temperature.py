"""Reproducible D2 zero-field, zero-temperature coupled dynamics benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from holstein_peierls.dynamics.coupled import (
    CoupledEhrenfestState,
    coupled_verlet_step,
    integrate_coupled_dop853,
)
from holstein_peierls.dynamics.ehrenfest import LatticeVelocity, dynamic_total_energy
from holstein_peierls.electronic import solve_ground_state
from holstein_peierls.parameters import StaticPolaronParameters
from holstein_peierls.polaron import solve_static_polaron


def _prepare_initial_state(
    size: int,
    kick_velocity_a_per_fs: float,
) -> tuple[CoupledEhrenfestState, StaticPolaronParameters, dict[str, float | int | bool]]:
    center = (size // 2) * size + (size // 2) + 1
    parameters = StaticPolaronParameters(
        nx=size,
        ny=size,
        polaron_position=center,
    )
    static = solve_static_polaron(
        parameters,
        solver="sparse",
        gradient_mode="optimized",
        legacy_convergence=False,
    )
    ground = solve_ground_state(
        static.state,
        parameters,
        solver="sparse",
    )
    velocity = LatticeVelocity.zeros(size, size)
    flat = velocity.u.ravel(order="C")
    flat[parameters.polaron_index] = float(kick_velocity_a_per_fs)
    state = CoupledEhrenfestState(
        static.state.copy(),
        velocity,
        np.asarray(ground.wavefunction, dtype=np.complex128),
    )
    metadata = {
        "static_iterations": static.diagnostics.iterations,
        "static_converged": static.diagnostics.converged,
        "static_total_energy_eV": static.total_energy,
        "static_formation_energy_eV": static.formation_energy,
    }
    return state, parameters, metadata


def _phase_aligned_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    overlap = np.vdot(reference, candidate)
    aligned = candidate
    if abs(overlap) > 0.0:
        aligned = candidate * np.conj(overlap) / abs(overlap)
    return float(np.linalg.norm(aligned - reference) / np.linalg.norm(reference))


def _lattice_error(reference, candidate) -> float:
    numerator = (
        np.linalg.norm(candidate.u - reference.u) ** 2
        + np.linalg.norm(candidate.vx - reference.vx) ** 2
        + np.linalg.norm(candidate.vy - reference.vy) ** 2
    )
    scale = (
        np.linalg.norm(reference.u) ** 2
        + np.linalg.norm(reference.vx) ** 2
        + np.linalg.norm(reference.vy) ** 2
    )
    return float(np.sqrt(numerator / max(scale, 1.0e-30)))


def _velocity_error(reference, candidate) -> float:
    numerator = (
        np.linalg.norm(candidate.u - reference.u) ** 2
        + np.linalg.norm(candidate.vx - reference.vx) ** 2
        + np.linalg.norm(candidate.vy - reference.vy) ** 2
    )
    scale = (
        np.linalg.norm(reference.u) ** 2
        + np.linalg.norm(reference.vx) ** 2
        + np.linalg.norm(reference.vy) ** 2
    )
    return float(np.sqrt(numerator / max(scale, 1.0e-30)))


def _run_split(
    initial: CoupledEhrenfestState,
    parameters: StaticPolaronParameters,
    final_time_fs: float,
    dt_fs: float,
    method: str,
    krylov_dimension: int,
) -> tuple[dict[str, float | int | str], CoupledEhrenfestState]:
    steps = int(round(final_time_fs / dt_fs))
    if not np.isclose(steps * dt_fs, final_time_fs):
        raise ValueError("final time must be an integer multiple of dt")
    current = CoupledEhrenfestState(
        initial.lattice.copy(),
        initial.velocity.copy(),
        initial.electronic_state.copy(),
    )
    initial_energy = dynamic_total_energy(
        current.lattice,
        current.velocity,
        parameters,
        current.electronic_state,
    ).total
    initial_norm = float(np.linalg.norm(current.electronic_state))
    max_drift = 0.0
    h_evaluations = 0
    h_applications = 0
    start = perf_counter()
    for _ in range(steps):
        current, evaluations, applications = coupled_verlet_step(
            current,
            parameters,
            dt_fs,
            electronic_method=method,
            krylov_dimension=krylov_dimension,
        )
        h_evaluations += evaluations
        h_applications += applications
        energy = dynamic_total_energy(
            current.lattice,
            current.velocity,
            parameters,
            current.electronic_state,
        ).total
        max_drift = max(max_drift, abs(energy - initial_energy))
    elapsed = perf_counter() - start
    final_energy = dynamic_total_energy(
        current.lattice,
        current.velocity,
        parameters,
        current.electronic_state,
    ).total
    record: dict[str, float | int | str] = {
        "method": method,
        "dt_fs": dt_fs,
        "steps": steps,
        "elapsed_seconds": float(elapsed),
        "hamiltonian_evaluations": h_evaluations,
        "hamiltonian_applications": h_applications,
        "initial_energy_eV": initial_energy,
        "final_energy_eV": final_energy,
        "final_energy_drift_eV": final_energy - initial_energy,
        "maximum_absolute_energy_drift_eV": max_drift,
        "relative_max_energy_drift": max_drift / max(abs(initial_energy), 1.0e-30),
        "electronic_norm_error": abs(
            float(np.linalg.norm(current.electronic_state)) - initial_norm
        ),
    }
    return record, current


def _markdown(payload: dict) -> str:
    lines = [
        "# D2 coupled zero-temperature benchmark",
        "",
        f"Lattice: {payload['size']}x{payload['size']}",
        f"Final time: {payload['final_time_fs']} fs",
        f"Initial local u-velocity kick: {payload['kick_velocity_a_per_fs']} A/fs",
        "Field: 0; thermostat: none",
        "",
        "| method | dt [fs] | elapsed [s] | H evals | H apps | max |dE| [eV] | relative max drift | norm error | electronic error vs DOP853 | lattice error vs DOP853 | velocity error vs DOP853 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in payload["split_results"]:
        lines.append(
            "| {method} | {dt_fs:.4g} | {elapsed_seconds:.6g} | {hamiltonian_evaluations} | {hamiltonian_applications} | {maximum_absolute_energy_drift_eV:.3e} | {relative_max_energy_drift:.3e} | {electronic_norm_error:.3e} | {electronic_error_vs_reference:.3e} | {lattice_error_vs_reference:.3e} | {velocity_error_vs_reference:.3e} |".format(**record)
        )
    ref = payload["reference"]
    lines += [
        "",
        "## Tightened coupled DOP853 reference",
        "",
        f"Elapsed: {ref['elapsed_seconds']:.6g} s; RHS evaluations: {ref['rhs_evaluations']}; final energy drift: {ref['final_energy_drift_eV']:.3e} eV; norm error: {ref['electronic_norm_error']:.3e}.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--final-time-fs", type=float, default=10.0)
    parser.add_argument("--dt-values", type=float, nargs="+", default=[0.2, 0.1, 0.05])
    parser.add_argument("--kick-velocity-a-per-fs", type=float, default=2.0e-4)
    parser.add_argument("--krylov-dimension", type=int, default=6)
    parser.add_argument("--reference-rtol", type=float, default=1.0e-9)
    parser.add_argument("--reference-atol", type=float, default=1.0e-11)
    parser.add_argument("--reference-max-step-fs", type=float, default=0.02)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    initial, parameters, static_metadata = _prepare_initial_state(
        args.size,
        args.kick_velocity_a_per_fs,
    )
    initial_energy = dynamic_total_energy(
        initial.lattice,
        initial.velocity,
        parameters,
        initial.electronic_state,
    ).total
    initial_norm = float(np.linalg.norm(initial.electronic_state))

    reference = integrate_coupled_dop853(
        initial,
        parameters,
        final_time_fs=args.final_time_fs,
        rtol=args.reference_rtol,
        atol=args.reference_atol,
        max_step_fs=args.reference_max_step_fs,
    )
    if not reference.success:
        raise RuntimeError(reference.message)
    reference_energy = dynamic_total_energy(
        reference.state.lattice,
        reference.state.velocity,
        parameters,
        reference.state.electronic_state,
    ).total
    reference_record = {
        "elapsed_seconds": reference.elapsed_seconds,
        "rhs_evaluations": reference.rhs_evaluations,
        "hamiltonian_evaluations": reference.hamiltonian_evaluations,
        "initial_energy_eV": initial_energy,
        "final_energy_eV": reference_energy,
        "final_energy_drift_eV": reference_energy - initial_energy,
        "electronic_norm_error": abs(
            float(np.linalg.norm(reference.state.electronic_state)) - initial_norm
        ),
    }

    split_results = []
    for dt in args.dt_values:
        for method in ("cfm4_lanczos", "rk4"):
            record, final_state = _run_split(
                initial,
                parameters,
                args.final_time_fs,
                dt,
                method,
                args.krylov_dimension,
            )
            record["electronic_error_vs_reference"] = _phase_aligned_error(
                reference.state.electronic_state,
                final_state.electronic_state,
            )
            record["lattice_error_vs_reference"] = _lattice_error(
                reference.state.lattice,
                final_state.lattice,
            )
            record["velocity_error_vs_reference"] = _velocity_error(
                reference.state.velocity,
                final_state.velocity,
            )
            split_results.append(record)

    payload = {
        "size": args.size,
        "final_time_fs": args.final_time_fs,
        "kick_velocity_a_per_fs": args.kick_velocity_a_per_fs,
        "krylov_dimension": args.krylov_dimension,
        "static": static_metadata,
        "reference": reference_record,
        "split_results": split_results,
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
