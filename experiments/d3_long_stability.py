"""Long field-driven D3 stability check on the relaxed 20x20 polaron."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from holstein_peierls.dynamics.coupled import CoupledEhrenfestState
from holstein_peierls.dynamics.driven import (
    coupled_field_verlet_step,
    field_dynamic_total_energy,
)
from holstein_peierls.dynamics.ehrenfest import LatticeVelocity
from holstein_peierls.dynamics.field import UniformElectricField2D
from holstein_peierls.electronic import solve_ground_state
from holstein_peierls.parameters import StaticPolaronParameters
from holstein_peierls.polaron import solve_static_polaron


def _ipr(psi: np.ndarray) -> float:
    density = np.abs(psi) ** 2
    density /= np.sum(density)
    return float(np.sum(density * density))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--field-mv-per-a", type=float, default=2.0)
    parser.add_argument("--field-angle-rad", type=float, default=0.0)
    parser.add_argument("--final-time-fs", type=float, default=10_000.0)
    parser.add_argument("--dt-fs", type=float, default=0.2)
    parser.add_argument("--krylov-dimension", type=int, default=6)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    size = args.size
    center = (size // 2) * size + (size // 2) + 1
    parameters = StaticPolaronParameters(nx=size, ny=size, polaron_position=center)
    static = solve_static_polaron(
        parameters,
        solver="sparse",
        gradient_mode="optimized",
        legacy_convergence=False,
    )
    if not static.diagnostics.converged:
        raise RuntimeError("static-polaron preparation did not converge")
    ground = solve_ground_state(static.state, parameters, solver="sparse")
    current = CoupledEhrenfestState(
        static.state.copy(),
        LatticeVelocity.zeros(size, size),
        np.asarray(ground.wavefunction, dtype=np.complex128),
    )
    field = UniformElectricField2D.from_millivolt_per_angstrom(
        args.field_mv_per_a,
        args.field_angle_rad,
        ax_angstrom=3.0,
        ay_angstrom=3.0,
    )

    steps = int(round(args.final_time_fs / args.dt_fs))
    if not np.isclose(steps * args.dt_fs, args.final_time_fs):
        raise ValueError("final time must be an integer multiple of dt")

    initial_lattice = current.lattice.copy()
    initial_density = np.abs(current.electronic_state) ** 2
    initial_norm = float(np.linalg.norm(current.electronic_state))
    initial_ipr = _ipr(current.electronic_state)
    initial_energy = field_dynamic_total_energy(
        current.lattice,
        current.velocity,
        parameters,
        current.electronic_state,
        field,
        0.0,
    ).total

    accumulated_work = 0.0
    maximum_balance_residual = 0.0
    maximum_norm_error = 0.0
    maximum_coordinate_excursion = 0.0
    h_evaluations = 0
    h_applications = 0
    time_fs = 0.0

    start = perf_counter()
    for _ in range(steps):
        current, work, evaluations, applications = coupled_field_verlet_step(
            current,
            parameters,
            field,
            time_fs,
            args.dt_fs,
            electronic_method="cfm4_lanczos",
            krylov_dimension=args.krylov_dimension,
        )
        accumulated_work += work
        h_evaluations += evaluations
        h_applications += applications
        time_fs += args.dt_fs

        matter_energy = field_dynamic_total_energy(
            current.lattice,
            current.velocity,
            parameters,
            current.electronic_state,
            field,
            time_fs,
        ).total
        residual = (matter_energy - initial_energy) - accumulated_work
        maximum_balance_residual = max(maximum_balance_residual, abs(residual))
        maximum_norm_error = max(
            maximum_norm_error,
            abs(float(np.linalg.norm(current.electronic_state)) - initial_norm),
        )
        maximum_coordinate_excursion = max(
            maximum_coordinate_excursion,
            float(np.max(np.abs(current.lattice.u - initial_lattice.u))),
            float(np.max(np.abs(current.lattice.vx - initial_lattice.vx))),
            float(np.max(np.abs(current.lattice.vy - initial_lattice.vy))),
        )
    elapsed = perf_counter() - start

    final_energy = field_dynamic_total_energy(
        current.lattice,
        current.velocity,
        parameters,
        current.electronic_state,
        field,
        args.final_time_fs,
    ).total
    final_density = np.abs(current.electronic_state) ** 2
    matter_energy_change = final_energy - initial_energy
    final_balance_residual = matter_energy_change - accumulated_work
    phase_x, phase_y = field.phases(args.final_time_fs)

    payload = {
        "size": size,
        "field_mV_per_A": args.field_mv_per_a,
        "field_angle_radians": args.field_angle_rad,
        "final_phase_x_radians": phase_x,
        "final_phase_y_radians": phase_y,
        "final_time_fs": args.final_time_fs,
        "dt_fs": args.dt_fs,
        "steps": steps,
        "krylov_dimension": args.krylov_dimension,
        "elapsed_seconds": float(elapsed),
        "static_iterations": static.diagnostics.iterations,
        "static_converged": static.diagnostics.converged,
        "initial_matter_energy_eV": initial_energy,
        "final_matter_energy_eV": final_energy,
        "matter_energy_change_eV": matter_energy_change,
        "field_work_eV": accumulated_work,
        "final_energy_work_residual_eV": final_balance_residual,
        "maximum_absolute_energy_work_residual_eV": maximum_balance_residual,
        "maximum_electronic_norm_error": maximum_norm_error,
        "initial_ipr": initial_ipr,
        "final_ipr": _ipr(current.electronic_state),
        "population_l2_change": float(np.linalg.norm(final_density - initial_density)),
        "maximum_coordinate_excursion_A": maximum_coordinate_excursion,
        "hamiltonian_evaluations": h_evaluations,
        "hamiltonian_applications": h_applications,
    }
    markdown = "\n".join(
        [
            "# D3 long field-driven zero-temperature stability",
            "",
            f"- lattice: {size}x{size}",
            f"- field: {args.field_mv_per_a} mV/A at angle {args.field_angle_rad} rad",
            f"- trajectory: {args.final_time_fs / 1000.0:.3f} ps at dt={args.dt_fs} fs ({steps} steps)",
            "- initial lattice velocity: zero; thermostat: none",
            f"- elapsed: {elapsed:.6g} s, one CPU thread in CI",
            f"- matter-energy change: {matter_energy_change:.6e} eV",
            f"- accumulated field work: {accumulated_work:.6e} eV",
            f"- max |dE-W|: {maximum_balance_residual:.6e} eV",
            f"- final dE-W: {final_balance_residual:.6e} eV",
            f"- max electronic norm error: {maximum_norm_error:.6e}",
            f"- IPR: {initial_ipr:.8f} -> {payload['final_ipr']:.8f}",
            f"- population L2 change: {payload['population_l2_change']:.6e}",
            f"- maximum lattice-coordinate excursion: {maximum_coordinate_excursion:.6e} A",
            f"- H evaluations/applications: {h_evaluations}/{h_applications}",
            "",
        ]
    )
    print(markdown)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
