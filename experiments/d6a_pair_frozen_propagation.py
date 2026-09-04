"""D6a frozen-H propagation benchmark for bipolaron and exciton sectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from holstein_peierls.dynamics.frozen import exact_spectral_step, lanczos_exponential_step
from holstein_peierls.dynamics.pair_frozen import (
    bipolaron_exchange_symmetry_error,
    bipolaron_frozen_action,
    bipolaron_one_body_density_matrix,
    dense_hamiltonian_from_action,
    electronic_energy_expectation,
    exciton_frozen_action,
    exciton_one_body_density_matrices,
    normalized_pair_state,
    symmetrized_bipolaron_state,
)
from holstein_peierls.exciton.parameters import ExcitonParameters
from holstein_peierls.lattice import LatticeState
from holstein_peierls.two_particle.parameters import BipolaronParameters


def _lattice(size: int) -> LatticeState:
    y, x = np.indices((size, size), dtype=np.float64)
    scale = max(1.0, float(size - 1))
    return LatticeState(
        u=1.0e-2 * np.cos(0.9 * x / scale + 0.6 * y / scale),
        vx=4.0e-3 * np.sin(1.1 * x / scale - 0.3 * y / scale),
        vy=3.0e-3 * np.cos(0.4 * x / scale + 1.0 * y / scale),
    )


def _bipolaron_parameters(size: int) -> BipolaronParameters:
    center = (size // 2) * size + (size // 2) + 1
    return BipolaronParameters(
        nx=size,
        ny=size,
        pair_position=center,
        hubbard_u=0.22,
        nearest_neighbor_v=0.04,
    )


def _exciton_parameters(size: int) -> ExcitonParameters:
    center = (size // 2) * size + (size // 2) + 1
    return ExcitonParameters(
        nx=size,
        ny=size,
        exciton_position=center,
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


def _phase_aligned_error(reference: np.ndarray, state: np.ndarray) -> float:
    overlap = np.vdot(reference, state)
    if abs(overlap) == 0.0:
        return float(np.linalg.norm(state - reference))
    aligned = state * np.exp(-1.0j * np.angle(overlap))
    return float(np.linalg.norm(aligned - reference))


def _propagate(action, initial: np.ndarray, dt_fs: float, steps: int, krylov_dimension: int):
    current = initial.copy()
    action.reset()
    start = perf_counter()
    maximum_norm_error = 0.0
    for _ in range(steps):
        result = lanczos_exponential_step(
            action,
            current,
            dt_fs,
            krylov_dimension=krylov_dimension,
        )
        current = result.state
        maximum_norm_error = max(maximum_norm_error, abs(float(np.linalg.norm(current)) - 1.0))
    elapsed = perf_counter() - start
    return current, elapsed, action.applications, maximum_norm_error


def _sector_record(
    sector: str,
    action,
    initial: np.ndarray,
    dt_fs: float,
    steps: int,
    dimensions: list[int],
) -> dict:
    dense = dense_hamiltonian_from_action(action)
    hermiticity_error = float(np.max(np.abs(dense - dense.conj().T)))
    reference = exact_spectral_step(dense, initial, dt_fs * steps).state
    reference_energy = electronic_energy_expectation(action, reference)
    runs = []
    for dimension in dimensions:
        state, elapsed, applications, max_norm_error = _propagate(
            action, initial, dt_fs, steps, dimension
        )
        fidelity = float(abs(np.vdot(reference, state)) ** 2)
        energy = electronic_energy_expectation(action, state)
        record = {
            "krylov_dimension": int(dimension),
            "elapsed_seconds": float(elapsed),
            "hamiltonian_applications": int(applications),
            "maximum_norm_error": float(max_norm_error),
            "final_norm_error": float(abs(np.linalg.norm(state) - 1.0)),
            "phase_aligned_state_error": _phase_aligned_error(reference, state),
            "fidelity": fidelity,
            "energy_error_eV": float(abs(energy - reference_energy)),
        }
        if sector == "bipolaron":
            gamma = bipolaron_one_body_density_matrix(state, action.n_sites)
            record["exchange_symmetry_error"] = bipolaron_exchange_symmetry_error(
                state, action.n_sites
            )
            record["one_body_rdm_trace_error"] = float(abs(np.trace(gamma) - 2.0))
        else:
            gamma_e, gamma_h = exciton_one_body_density_matrices(state, action.n_sites)
            record["electron_rdm_trace_error"] = float(abs(np.trace(gamma_e) - 1.0))
            record["hole_rdm_trace_error"] = float(abs(np.trace(gamma_h) - 1.0))
        runs.append(record)
    return {
        "sector": sector,
        "n_sites": int(action.n_sites),
        "hilbert_dimension": int(action.dimension),
        "hermiticity_error": hermiticity_error,
        "reference_energy_eV": float(reference_energy),
        "runs": runs,
    }


def _markdown(payload: dict) -> str:
    lines = [
        "# D6a frozen pair propagation benchmark",
        "",
        f"Lattice: {payload['size']}x{payload['size']}; dt={payload['dt_fs']} fs; steps={payload['steps']}",
        "",
        "| sector | dim | Krylov m | phase-aligned error | fidelity | max norm error | energy error [eV] | H applications |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sector in payload["sectors"]:
        for run in sector["runs"]:
            lines.append(
                "| {sector} | {dim} | {m} | {err:.3e} | {fid:.12f} | {norm:.3e} | {energy:.3e} | {apps} |".format(
                    sector=sector["sector"],
                    dim=sector["hilbert_dimension"],
                    m=run["krylov_dimension"],
                    err=run["phase_aligned_state_error"],
                    fid=run["fidelity"],
                    norm=run["maximum_norm_error"],
                    energy=run["energy_error_eV"],
                    apps=run["hamiltonian_applications"],
                )
            )
    lines.extend(
        [
            "",
            f"Pre-registered D6a numerical closure: {'PASS' if payload['closure_pass'] else 'FAIL'}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=3)
    parser.add_argument("--dt-fs", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--krylov-dimensions", type=int, nargs="+", default=[6, 8, 12, 16])
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    if args.size < 2:
        raise ValueError("D6a dense reference requires size >= 2")
    lattice = _lattice(args.size)
    rng = np.random.default_rng(args.seed)

    bip_action = bipolaron_frozen_action(lattice, _bipolaron_parameters(args.size))
    bip_initial = symmetrized_bipolaron_state(
        rng.normal(size=bip_action.dimension) + 1.0j * rng.normal(size=bip_action.dimension),
        bip_action.n_sites,
    )

    exc_action = exciton_frozen_action(lattice, _exciton_parameters(args.size))
    exc_initial_matrix = np.zeros((exc_action.n_sites, exc_action.n_sites), dtype=np.complex128)
    exc_initial_matrix[0, min(4, exc_action.n_sites - 1)] = 1.0
    exc_initial = normalized_pair_state(exc_initial_matrix, exc_action.n_sites)

    sectors = [
        _sector_record(
            "bipolaron",
            bip_action,
            bip_initial,
            args.dt_fs,
            args.steps,
            args.krylov_dimensions,
        ),
        _sector_record(
            "exciton",
            exc_action,
            exc_initial,
            args.dt_fs,
            args.steps,
            args.krylov_dimensions,
        ),
    ]

    final_dimension = max(args.krylov_dimensions)
    final_runs = [
        next(run for run in sector["runs"] if run["krylov_dimension"] == final_dimension)
        for sector in sectors
    ]
    checks = {
        "hermiticity": all(sector["hermiticity_error"] < 1.0e-12 for sector in sectors),
        "state_accuracy": all(run["phase_aligned_state_error"] < 1.0e-8 for run in final_runs),
        "norm": all(run["maximum_norm_error"] < 1.0e-10 for run in final_runs),
        "energy": all(run["energy_error_eV"] < 1.0e-9 for run in final_runs),
        "bipolaron_symmetry": final_runs[0]["exchange_symmetry_error"] < 1.0e-10,
        "bipolaron_rdm_trace": final_runs[0]["one_body_rdm_trace_error"] < 1.0e-12,
        "exciton_rdm_traces": (
            final_runs[1]["electron_rdm_trace_error"] < 1.0e-12
            and final_runs[1]["hole_rdm_trace_error"] < 1.0e-12
        ),
    }
    payload = {
        "size": args.size,
        "dt_fs": args.dt_fs,
        "steps": args.steps,
        "total_time_fs": args.dt_fs * args.steps,
        "seed": args.seed,
        "krylov_dimensions": args.krylov_dimensions,
        "sectors": sectors,
        "closure_checks": checks,
        "closure_pass": bool(all(checks.values())),
    }
    text = _markdown(payload)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(text, encoding="utf-8")
    if not payload["closure_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
