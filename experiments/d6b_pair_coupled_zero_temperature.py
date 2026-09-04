#!/usr/bin/env python3
"""D6b zero-temperature moving-lattice pair benchmark.

The benchmark intentionally uses a small 4x4 control so a tightened full-system
DOP853 reference remains affordable.  It measures time-step convergence,
Krylov-dimension sensitivity, total-energy stability, norm preservation and the
sector-specific pair constraints before any finite-temperature or field-driven
extension is attempted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from holstein_peierls.dynamics.ehrenfest import LatticeVelocity
from holstein_peierls.dynamics.pair_coupled import (
    MovingPairHamiltonianFactory,
    PairCoupledState,
    PairLatticeMasses,
    integrate_pair_coupled_dop853,
    integrate_pair_coupled_verlet,
    pair_coupled_verlet_step,
    pair_dynamic_total_energy,
)
from holstein_peierls.dynamics.pair_frozen import (
    bipolaron_exchange_symmetry_error,
    bipolaron_one_body_density_matrix,
    exciton_one_body_density_matrices,
)
from holstein_peierls.dynamics.time_dependent import compare_time_dependent_states
from holstein_peierls.exciton.parameters import ExcitonParameters
from holstein_peierls.exciton.solver import solve_exciton_ground_state
from holstein_peierls.lattice import LatticeState
from holstein_peierls.two_particle.parameters import BipolaronParameters
from holstein_peierls.two_particle.peierls import solve_holstein_peierls_ground_state


def lattice_control(size: int) -> LatticeState:
    y, x = np.indices((size, size), dtype=np.float64)
    return LatticeState(
        u=8.0e-3 * np.cos(0.71 * x + 0.43 * y),
        vx=3.0e-3 * np.sin(0.91 * x - 0.23 * y),
        vy=2.5e-3 * np.cos(0.31 * x + 0.83 * y),
    )


def center_position(size: int) -> int:
    y = size // 2
    x = size // 2
    return y * size + x + 1


def bipolaron_parameters(size: int) -> BipolaronParameters:
    return BipolaronParameters(
        nx=size,
        ny=size,
        pair_position=center_position(size),
        hubbard_u=0.22,
        nearest_neighbor_v=0.04,
    )


def exciton_parameters(size: int) -> ExcitonParameters:
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
    velocity = LatticeVelocity.zeros(size, size)
    velocity.u[size // 2, size // 2] = 1.5e-4
    if sector == "bipolaron":
        parameters = bipolaron_parameters(size)
        ground = solve_holstein_peierls_ground_state(lattice, parameters)
    else:
        parameters = exciton_parameters(size)
        ground = solve_exciton_ground_state(lattice, parameters)
    state = PairCoupledState(
        lattice=lattice,
        velocity=velocity,
        electronic_state=np.asarray(
            ground.wavefunction, dtype=np.complex128
        ).ravel(order="C"),
    )
    return state, parameters


def state_errors(reference: PairCoupledState, candidate: PairCoupledState) -> dict:
    electronic = compare_time_dependent_states(
        reference.electronic_state, candidate.electronic_state
    )
    lattice_error = max(
        float(np.max(np.abs(candidate.lattice.u - reference.lattice.u))),
        float(np.max(np.abs(candidate.lattice.vx - reference.lattice.vx))),
        float(np.max(np.abs(candidate.lattice.vy - reference.lattice.vy))),
    )
    velocity_error = max(
        float(np.max(np.abs(candidate.velocity.u - reference.velocity.u))),
        float(np.max(np.abs(candidate.velocity.vx - reference.velocity.vx))),
        float(np.max(np.abs(candidate.velocity.vy - reference.velocity.vy))),
    )
    return {
        "electronic_phase_aligned_error": electronic.phase_aligned_state_error,
        "electronic_fidelity": electronic.fidelity,
        "lattice_max_error_A": lattice_error,
        "velocity_max_error_A_per_fs": velocity_error,
    }


def constraint_metrics(sector: str, state: PairCoupledState, n_sites: int) -> dict:
    norm_error = abs(float(np.linalg.norm(state.electronic_state)) - 1.0)
    result = {"norm_error": norm_error}
    if sector == "bipolaron":
        gamma = bipolaron_one_body_density_matrix(state.electronic_state, n_sites)
        result.update(
            exchange_symmetry_error=bipolaron_exchange_symmetry_error(
                state.electronic_state, n_sites
            ),
            rdm_trace_error=abs(complex(np.trace(gamma)) - 2.0),
        )
    else:
        gamma_e, gamma_h = exciton_one_body_density_matrices(
            state.electronic_state, n_sites
        )
        result.update(
            electron_rdm_trace_error=abs(complex(np.trace(gamma_e)) - 1.0),
            hole_rdm_trace_error=abs(complex(np.trace(gamma_h)) - 1.0),
        )
    return result


def run_sector(
    sector: str,
    size: int,
    masses: PairLatticeMasses,
    reference_time_fs: float,
    dt_values: list[float],
    krylov_dimensions: list[int],
    stability_time_fs: float,
    stability_dt_fs: float,
    sample_stride: int,
) -> dict:
    initial, parameters = initial_state(sector, size)
    factory = MovingPairHamiltonianFactory(sector, parameters)
    e0 = pair_dynamic_total_energy(
        initial, sector, parameters, masses, factory=factory
    ).total

    reference = integrate_pair_coupled_dop853(
        initial,
        sector,
        parameters,
        masses,
        final_time_fs=reference_time_fs,
        rtol=2.0e-11,
        atol=2.0e-13,
        max_step_fs=min(0.02, min(dt_values) / 4.0),
    )
    if not reference.success:
        raise RuntimeError(f"{sector} DOP853 reference failed: {reference.message}")

    dt_runs = []
    for dt in dt_values:
        steps = int(round(reference_time_fs / dt))
        if abs(steps * dt - reference_time_fs) > 1.0e-12:
            raise ValueError("reference_time_fs must be commensurate with every dt")
        result = integrate_pair_coupled_verlet(
            initial,
            sector,
            parameters,
            masses,
            dt_fs=dt,
            steps=steps,
            krylov_dimension=8,
        )
        energy = pair_dynamic_total_energy(
            result.state, sector, parameters, masses, factory=factory
        ).total
        dt_runs.append(
            {
                "dt_fs": dt,
                "steps": steps,
                "elapsed_seconds": result.elapsed_seconds,
                "energy_change_eV": energy - e0,
                "hamiltonian_evaluations": result.hamiltonian_evaluations,
                "hamiltonian_applications": result.hamiltonian_applications,
                **state_errors(reference.state, result.state),
                **constraint_metrics(sector, result.state, parameters.n_sites),
            }
        )

    krylov_runs = []
    krylov_dt = max(dt_values)
    krylov_steps = int(round(reference_time_fs / krylov_dt))
    for dimension in krylov_dimensions:
        result = integrate_pair_coupled_verlet(
            initial,
            sector,
            parameters,
            masses,
            dt_fs=krylov_dt,
            steps=krylov_steps,
            krylov_dimension=dimension,
        )
        krylov_runs.append(
            {
                "krylov_dimension": dimension,
                "dt_fs": krylov_dt,
                "elapsed_seconds": result.elapsed_seconds,
                **state_errors(reference.state, result.state),
                **constraint_metrics(sector, result.state, parameters.n_sites),
            }
        )

    current = PairCoupledState(
        initial.lattice.copy(), initial.velocity.copy(), initial.electronic_state.copy()
    )
    initial_arrays = (
        initial.lattice.u.copy(), initial.lattice.vx.copy(), initial.lattice.vy.copy()
    )
    stability_steps = int(round(stability_time_fs / stability_dt_fs))
    if abs(stability_steps * stability_dt_fs - stability_time_fs) > 1.0e-12:
        raise ValueError("stability_time_fs must be commensurate with stability_dt_fs")
    max_energy_drift = 0.0
    max_norm_error = 0.0
    max_constraint_error = 0.0
    start = perf_counter()
    for step in range(stability_steps):
        current, _, _ = pair_coupled_verlet_step(
            current,
            sector,
            parameters,
            masses,
            stability_dt_fs,
            krylov_dimension=8,
            factory=factory,
        )
        if (step + 1) % sample_stride == 0 or step + 1 == stability_steps:
            energy = pair_dynamic_total_energy(
                current, sector, parameters, masses, factory=factory
            ).total
            max_energy_drift = max(max_energy_drift, abs(energy - e0))
            metrics = constraint_metrics(sector, current, parameters.n_sites)
            max_norm_error = max(max_norm_error, float(metrics["norm_error"]))
            for key, value in metrics.items():
                if key != "norm_error":
                    max_constraint_error = max(max_constraint_error, abs(complex(value)))
    stability_elapsed = perf_counter() - start
    final_energy = pair_dynamic_total_energy(
        current, sector, parameters, masses, factory=factory
    ).total
    max_lattice_excursion = max(
        float(np.max(np.abs(current.lattice.u - initial_arrays[0]))),
        float(np.max(np.abs(current.lattice.vx - initial_arrays[1]))),
        float(np.max(np.abs(current.lattice.vy - initial_arrays[2]))),
    )

    return {
        "sector": sector,
        "n_sites": parameters.n_sites,
        "hilbert_dimension": parameters.n_sites**2,
        "initial_total_energy_eV": e0,
        "reference": {
            "elapsed_seconds": reference.elapsed_seconds,
            "rhs_evaluations": reference.rhs_evaluations,
            "steps": reference.steps,
            **constraint_metrics(sector, reference.state, parameters.n_sites),
        },
        "dt_runs": dt_runs,
        "krylov_runs": krylov_runs,
        "stability": {
            "dt_fs": stability_dt_fs,
            "time_fs": stability_time_fs,
            "steps": stability_steps,
            "elapsed_seconds": stability_elapsed,
            "maximum_absolute_energy_drift_eV": max_energy_drift,
            "final_energy_change_eV": final_energy - e0,
            "maximum_norm_error": max_norm_error,
            "maximum_sector_constraint_error": max_constraint_error,
            "maximum_lattice_excursion_A": max_lattice_excursion,
            **constraint_metrics(sector, current, parameters.n_sites),
        },
    }


def closure_checks(sectors: list[dict]) -> dict:
    checks = {}
    for sector in sectors:
        name = sector["sector"]
        runs = sorted(sector["dt_runs"], key=lambda row: row["dt_fs"], reverse=True)
        by_dt = {row["dt_fs"]: row for row in runs}
        if 0.2 in by_dt and 0.1 in by_dt:
            coarse = max(
                by_dt[0.2]["electronic_phase_aligned_error"],
                by_dt[0.2]["lattice_max_error_A"],
            )
            fine = max(
                by_dt[0.1]["electronic_phase_aligned_error"],
                by_dt[0.1]["lattice_max_error_A"],
            )
            checks[f"{name}_timestep_improves"] = fine <= 0.6 * coarse
        krylov = {row["krylov_dimension"]: row for row in sector["krylov_runs"]}
        if 8 in krylov:
            checks[f"{name}_m8_accuracy"] = (
                krylov[8]["electronic_phase_aligned_error"] < 1.0e-5
            )
        stable = sector["stability"]
        checks[f"{name}_energy_stability"] = (
            stable["maximum_absolute_energy_drift_eV"] < 1.0e-4
        )
        checks[f"{name}_norm"] = stable["maximum_norm_error"] < 1.0e-10
        checks[f"{name}_sector_constraints"] = (
            stable["maximum_sector_constraint_error"] < 1.0e-10
        )
        checks[f"{name}_bounded_lattice"] = (
            np.isfinite(stable["maximum_lattice_excursion_A"])
            and stable["maximum_lattice_excursion_A"] < 0.2
        )
    return checks


def markdown_report(payload: dict) -> str:
    lines = [
        "# D6b coupled zero-temperature pair benchmark",
        "",
        f"Lattice: {payload['size']}x{payload['size']}; reference time: {payload['reference_time_fs']} fs",
        "",
    ]
    for sector in payload["sectors"]:
        lines += [f"## {sector['sector']}", ""]
        lines.append(
            "| dt [fs] | electronic error | lattice error [A] | energy change [eV] | norm error |"
        )
        lines.append("|---:|---:|---:|---:|---:|")
        for row in sector["dt_runs"]:
            lines.append(
                f"| {row['dt_fs']:.3f} | {row['electronic_phase_aligned_error']:.3e} | "
                f"{row['lattice_max_error_A']:.3e} | {row['energy_change_eV']:.3e} | "
                f"{row['norm_error']:.3e} |"
            )
        stable = sector["stability"]
        lines += [
            "",
            f"Stability ({stable['time_fs']:.1f} fs, dt={stable['dt_fs']:.3f} fs): "
            f"max |dE|={stable['maximum_absolute_energy_drift_eV']:.3e} eV, "
            f"max norm error={stable['maximum_norm_error']:.3e}, "
            f"max sector-constraint error={stable['maximum_sector_constraint_error']:.3e}.",
            "",
        ]
    lines += [
        "## Pre-registered closure",
        "",
        *(f"- {name}: {'PASS' if value else 'FAIL'}" for name, value in payload["closure_checks"].items()),
        "",
        f"Overall: {'PASS' if payload['closure_pass'] else 'FAIL'}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--reference-time-fs", type=float, default=4.0)
    parser.add_argument("--dt-values", type=float, nargs="+", default=[0.2, 0.1, 0.05])
    parser.add_argument("--krylov-dimensions", type=int, nargs="+", default=[6, 8, 12])
    parser.add_argument("--stability-time-fs", type=float, default=100.0)
    parser.add_argument("--stability-dt-fs", type=float, default=0.2)
    parser.add_argument("--sample-stride", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    masses = PairLatticeMasses.from_legacy_as2(7.5e10, 1.5e11)
    sectors = [
        run_sector(
            sector,
            args.size,
            masses,
            args.reference_time_fs,
            list(args.dt_values),
            list(args.krylov_dimensions),
            args.stability_time_fs,
            args.stability_dt_fs,
            args.sample_stride,
        )
        for sector in ("bipolaron", "exciton")
    ]
    checks = closure_checks(sectors)
    payload = {
        "size": args.size,
        "reference_time_fs": args.reference_time_fs,
        "dt_values": list(args.dt_values),
        "krylov_dimensions": list(args.krylov_dimensions),
        "stability_time_fs": args.stability_time_fs,
        "stability_dt_fs": args.stability_dt_fs,
        "masses_eV_fs2_per_A2": {
            "intramolecular": masses.intramolecular,
            "intermolecular": masses.intermolecular,
        },
        "sectors": sectors,
        "closure_checks": checks,
        "closure_pass": bool(all(checks.values())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")
    args.markdown.write_text(markdown_report(payload), encoding="utf-8")
    print(markdown_report(payload))
    if not payload["closure_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
