"""Accuracy/cost sweep for the deterministic D0a frozen-H propagators.

The sweep fixes the total physical propagation time while varying the time step,
lattice size, and Krylov dimension.  Each case constructs one exact spectral
reference over the total time, then times the candidate propagators without
renormalizing their states.

This is a benchmark/validation driver, not a production dynamics entry point.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np

from holstein_peierls.dynamics.frozen import (
    CountingMatrixHamiltonian,
    PropagationStep,
    cfm4_frozen_limit_step,
    compare_to_reference,
    crank_nicolson_step,
    exact_spectral_step,
    lanczos_exponential_step,
    rk4_step,
    rkf78_step,
)
from holstein_peierls.hamiltonian import build_dense_hamiltonian, build_sparse_hamiltonian
from holstein_peierls.lattice import LatticeState
from holstein_peierls.parameters import StaticPolaronParameters


def _deterministic_problem(size: int) -> tuple[np.ndarray, Any, np.ndarray]:
    parameters = StaticPolaronParameters(nx=size, ny=size, polaron_position=1)
    y, x = np.indices((size, size), dtype=np.float64)
    state = LatticeState.zeros(size, size)
    state.u[:] = 0.015 * np.cos(2.0 * np.pi * x / size) + 0.010 * np.sin(
        2.0 * np.pi * y / size
    )
    state.vx[:] = 0.008 * np.sin(2.0 * np.pi * (x + y) / size)
    state.vy[:] = 0.006 * np.cos(2.0 * np.pi * (x - y) / size)
    dense = build_dense_hamiltonian(state, parameters).astype(np.complex128)
    sparse = build_sparse_hamiltonian(state, parameters).astype(np.complex128)

    index = np.arange(size * size, dtype=np.float64)
    psi = (
        np.exp(-0.5 * ((index - 0.37 * size * size) / max(1.0, 0.11 * size * size)) ** 2)
        * np.exp(1.0j * (0.173 * index + 0.011 * index * index / (size * size)))
    )
    psi += 0.23 * np.exp(1.0j * 0.413 * index)
    psi = np.asarray(psi / np.linalg.norm(psi), dtype=np.complex128)
    return dense, sparse, psi


def _propagate(
    initial: np.ndarray,
    steps: int,
    stepper: Callable[[np.ndarray], PropagationStep],
) -> tuple[np.ndarray, dict[str, int | float | None]]:
    psi = initial.copy()
    hpsi = eig = solves = workspace = 0
    max_embedded: float | None = None
    max_krylov: int | None = None
    start = perf_counter()
    for _ in range(steps):
        result = stepper(psi)
        psi = result.state
        hpsi += result.hamiltonian_applications
        eig += result.eigendecompositions
        solves += result.linear_solves
        workspace = max(workspace, result.approximate_workspace_bytes)
        if result.embedded_error_estimate is not None:
            max_embedded = (
                result.embedded_error_estimate
                if max_embedded is None
                else max(max_embedded, result.embedded_error_estimate)
            )
        if result.krylov_dimension_used is not None:
            max_krylov = (
                result.krylov_dimension_used
                if max_krylov is None
                else max(max_krylov, result.krylov_dimension_used)
            )
    elapsed = perf_counter() - start
    return psi, {
        "elapsed_seconds": float(elapsed),
        "seconds_per_step": float(elapsed / steps),
        "hamiltonian_applications": int(hpsi),
        "eigendecompositions": int(eig),
        "linear_solves": int(solves),
        "approximate_workspace_bytes": int(workspace),
        "maximum_embedded_error_estimate": max_embedded,
        "maximum_krylov_dimension_used": max_krylov,
    }


def _record(
    *,
    method: str,
    size: int,
    dt_fs: float,
    steps: int,
    dense: np.ndarray,
    initial: np.ndarray,
    reference: np.ndarray,
    stepper: Callable[[np.ndarray], PropagationStep],
) -> dict[str, Any]:
    candidate, work = _propagate(initial, steps, stepper)
    metrics = compare_to_reference(reference, candidate, dense)
    return {
        "size": size,
        "dimension": size * size,
        "method": method,
        "dt_fs": dt_fs,
        "steps": steps,
        "total_time_fs": dt_fs * steps,
        **work,
        "metrics": asdict(metrics),
    }


def run_sweep(
    *,
    total_time_fs: float,
    krylov_dimensions: tuple[int, ...],
) -> dict[str, Any]:
    cases = [
        (4, 0.05), (4, 0.10), (4, 0.20),
        (8, 0.05), (8, 0.10), (8, 0.20),
        (12, 0.05), (12, 0.10), (12, 0.20),
        (20, 0.10),
    ]
    records: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []

    problems: dict[int, tuple[np.ndarray, Any, np.ndarray]] = {}
    for size, dt_fs in cases:
        dense, sparse, initial = problems.setdefault(size, _deterministic_problem(size))
        steps = int(round(total_time_fs / dt_fs))
        if not np.isclose(steps * dt_fs, total_time_fs, rtol=0.0, atol=1.0e-12):
            raise ValueError("total_time_fs must be an integer multiple of every dt")

        start = perf_counter()
        reference = exact_spectral_step(dense, initial, total_time_fs).state
        reference_elapsed = perf_counter() - start
        references.append(
            {
                "size": size,
                "dimension": size * size,
                "dt_fs": dt_fs,
                "steps": steps,
                "reference_elapsed_seconds": float(reference_elapsed),
            }
        )

        records.append(
            _record(
                method="legacy_spectral",
                size=size,
                dt_fs=dt_fs,
                steps=steps,
                dense=dense,
                initial=initial,
                reference=reference,
                stepper=lambda psi, h=dense, dt=dt_fs: exact_spectral_step(h, psi, dt),
            )
        )
        rk4_action = CountingMatrixHamiltonian(sparse)
        records.append(
            _record(
                method="rk4",
                size=size,
                dt_fs=dt_fs,
                steps=steps,
                dense=dense,
                initial=initial,
                reference=reference,
                stepper=lambda psi, a=rk4_action, dt=dt_fs: rk4_step(a, psi, dt),
            )
        )
        rkf_action = CountingMatrixHamiltonian(sparse)
        records.append(
            _record(
                method="rkf78_order8",
                size=size,
                dt_fs=dt_fs,
                steps=steps,
                dense=dense,
                initial=initial,
                reference=reference,
                stepper=lambda psi, a=rkf_action, dt=dt_fs: rkf78_step(a, psi, dt),
            )
        )
        records.append(
            _record(
                method="crank_nicolson",
                size=size,
                dt_fs=dt_fs,
                steps=steps,
                dense=dense,
                initial=initial,
                reference=reference,
                stepper=lambda psi, h=sparse, dt=dt_fs: crank_nicolson_step(h, psi, dt),
            )
        )

        for krylov_dimension in krylov_dimensions:
            lanczos_action = CountingMatrixHamiltonian(sparse)
            records.append(
                _record(
                    method=f"lanczos_m{krylov_dimension}",
                    size=size,
                    dt_fs=dt_fs,
                    steps=steps,
                    dense=dense,
                    initial=initial,
                    reference=reference,
                    stepper=lambda psi, a=lanczos_action, dt=dt_fs, m=krylov_dimension: lanczos_exponential_step(
                        a, psi, dt, krylov_dimension=m
                    ),
                )
            )
            cfm_action = CountingMatrixHamiltonian(sparse)
            records.append(
                _record(
                    method=f"cfm4_frozen_m{krylov_dimension}",
                    size=size,
                    dt_fs=dt_fs,
                    steps=steps,
                    dense=dense,
                    initial=initial,
                    reference=reference,
                    stepper=lambda psi, a=cfm_action, dt=dt_fs, m=krylov_dimension: cfm4_frozen_limit_step(
                        a, psi, dt, krylov_dimension=m
                    ),
                )
            )

    return {
        "total_time_fs": total_time_fs,
        "krylov_dimensions": list(krylov_dimensions),
        "references": references,
        "records": records,
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# D0a frozen-H propagator sweep",
        "",
        f"Fixed total physical time: `{payload['total_time_fs']} fs`.",
        "",
        "| size | dt [fs] | method | time/step [ms] | Hpsi/step | norm err | 1-F | phase err | E err [eV] |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for record in payload["records"]:
        metrics = record["metrics"]
        steps = record["steps"]
        lines.append(
            "| {size} | {dt:.2f} | {method} | {time:.5g} | {hpsi:.3g} | {norm:.3e} | {infidelity:.3e} | {phase:.3e} | {energy:.3e} |".format(
                size=record["size"],
                dt=record["dt_fs"],
                method=record["method"],
                time=1.0e3 * record["seconds_per_step"],
                hpsi=record["hamiltonian_applications"] / steps,
                norm=metrics["norm_error"],
                infidelity=max(0.0, 1.0 - metrics["fidelity"]),
                phase=metrics["phase_aligned_state_error"],
                energy=metrics["energy_error"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-time-fs", type=float, default=2.0)
    parser.add_argument("--krylov-dimensions", nargs="+", type=int, default=[4, 6, 8, 12])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    if args.total_time_fs <= 0.0:
        raise ValueError("total time must be positive")
    krylov_dimensions = tuple(sorted(set(args.krylov_dimensions)))
    if not krylov_dimensions or min(krylov_dimensions) <= 0:
        raise ValueError("Krylov dimensions must be positive")

    payload = run_sweep(
        total_time_fs=args.total_time_fs,
        krylov_dimensions=krylov_dimensions,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(_markdown(payload), encoding="utf-8")
    print(_markdown(payload))


if __name__ == "__main__":
    main()
