"""Reproducible CPU microbenchmarks for the static Holstein-Peierls solver.

The benchmark separates matrix construction, electronic ground-state solution,
and analytical-gradient evaluation. It deliberately does not time a complete
RPROP relaxation because convergence iteration counts can differ slightly across
linear-algebra libraries and are a scientific diagnostic rather than a pure
kernel-performance measure.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Callable

import numpy as np

from holstein_peierls.electronic import solve_ground_state
from holstein_peierls.gradients import energy_gradient
from holstein_peierls.hamiltonian import build_dense_hamiltonian, build_sparse_hamiltonian
from holstein_peierls.lattice import LatticeState
from holstein_peierls.parameters import StaticPolaronParameters


def _time(callable_: Callable[[], object], repeats: int) -> float:
    samples: list[float] = []
    for _ in range(repeats):
        start = perf_counter()
        callable_()
        samples.append(perf_counter() - start)
    return median(samples)


def _state(size: int, seed: int) -> LatticeState:
    rng = np.random.default_rng(seed + size)
    shape = (size, size)
    return LatticeState(
        u=rng.normal(scale=0.02, size=shape),
        vx=rng.normal(scale=0.02, size=shape),
        vy=rng.normal(scale=0.02, size=shape),
    )


def run_benchmarks(
    sizes: list[int],
    solvers: list[str],
    repeats: int,
    dense_max_sites: int,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for size in sizes:
        n_sites = size * size
        parameters = replace(
            StaticPolaronParameters(),
            nx=size,
            ny=size,
            polaron_position=(n_sites // 2) + 1,
        )
        state = _state(size, seed)

        sparse_build = _time(
            lambda: build_sparse_hamiltonian(state, parameters), repeats
        )
        rows.append(
            {
                "size": size,
                "n_sites": n_sites,
                "kernel": "hamiltonian_build",
                "method": "sparse_csr",
                "seconds": sparse_build,
            }
        )

        if n_sites <= dense_max_sites:
            dense_build = _time(
                lambda: build_dense_hamiltonian(state, parameters), repeats
            )
            rows.append(
                {
                    "size": size,
                    "n_sites": n_sites,
                    "kernel": "hamiltonian_build",
                    "method": "dense",
                    "seconds": dense_build,
                }
            )

        ground_for_gradient = solve_ground_state(
            state, parameters, solver="sparse"
        )
        for mode in ("reference", "optimized"):
            gradient_time = _time(
                lambda mode=mode: energy_gradient(
                    state,
                    parameters,
                    solver="sparse",
                    ground_state=ground_for_gradient,
                    mode=mode,
                ),
                repeats,
            )
            rows.append(
                {
                    "size": size,
                    "n_sites": n_sites,
                    "kernel": "gradient",
                    "method": mode,
                    "seconds": gradient_time,
                }
            )

        for solver in solvers:
            if solver.startswith("dense") and n_sites > dense_max_sites:
                continue
            solve_time = _time(
                lambda solver=solver: solve_ground_state(
                    state, parameters, solver=solver
                ),
                repeats,
            )
            rows.append(
                {
                    "size": size,
                    "n_sites": n_sites,
                    "kernel": "ground_state",
                    "method": solver,
                    "seconds": solve_time,
                }
            )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[20, 40, 80])
    parser.add_argument(
        "--solvers",
        nargs="+",
        choices=("dense_full", "dense_lowest", "sparse"),
        default=["dense_lowest", "sparse"],
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dense-max-sites", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results"))
    args = parser.parse_args()

    rows = run_benchmarks(
        sizes=args.sizes,
        solvers=args.solvers,
        repeats=args.repeats,
        dense_max_sites=args.dense_max_sites,
        seed=args.seed,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output / "static_cpu.json"
    csv_path = args.output / "static_cpu.csv"
    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("size", "n_sites", "kernel", "method", "seconds"),
        )
        writer.writeheader()
        writer.writerows(rows)

    print("size  sites  kernel              method          seconds")
    print("----  -----  ------------------  --------------  --------")
    for row in rows:
        print(
            f"{row['size']:>4}  {row['n_sites']:>5}  "
            f"{row['kernel']:<18}  {row['method']:<14}  "
            f"{row['seconds']:.6f}"
        )
    print(f"\nJSON: {json_path}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()
