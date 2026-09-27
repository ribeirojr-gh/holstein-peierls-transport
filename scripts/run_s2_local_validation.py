#!/usr/bin/env python3
"""Self-contained local fallback for S2 unified static-sector regression."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy


def _run(name: str, command: list[str], root: Path) -> dict:
    print(f"\n=== {name} ===", flush=True)
    started = perf_counter()
    proc = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    elapsed = float(perf_counter() - started)
    (root / f"{name}.log").write_text(proc.stdout, encoding="utf-8")
    print(proc.stdout, end="", flush=True)
    return {
        "name": name,
        "command": command,
        "return_code": int(proc.returncode),
        "status": "pass" if proc.returncode == 0 else "fail",
        "elapsed_s": elapsed,
    }


def _git(args: list[str]) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path("s2-local-validation") / stamp
    data = root / "sector-data"
    aggregate = root / "aggregate"
    data.mkdir(parents=True, exist_ok=True)
    aggregate.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    gates = []

    gates.append(
        _run(
            "s2-pycompile",
            [
                py, "-m", "py_compile",
                "experiments/s2_polaron_regression.py",
                "experiments/s2_static_aggregate.py",
                "src/holstein_peierls/s2_static_regression.py",
            ],
            root,
        )
    )
    gates.append(
        _run(
            "s2-focused-pytest",
            [
                py, "-m", "pytest",
                "tests/test_s2_static_regression.py",
                "tests/test_s1r_root_manifold.py",
                "tests/test_long_range_coulomb.py",
                "tests/test_exciton_solver.py",
                "tests/test_exciton_relaxation.py",
                "tests/test_exciton_branches.py",
                "-q",
            ],
            root,
        )
    )
    gates.append(_run("full-pytest", [py, "-m", "pytest"], root))

    commands = [
        (
            "s2-polaron",
            [
                py, "experiments/s2_polaron_regression.py",
                "--output", str(data / "s2-polaron.json"),
            ],
        ),
        (
            "s2-bipolaron",
            [
                py, "experiments/bipolaron_branch_benchmark_seeded.py",
                "--size", "40",
                "--u", "1.0",
                "--v1", "0.0",
                "--branch", "intersite_x",
                "--j0x", "0.100",
                "--j0y", "0.015",
                "--alpha-x", "0.10",
                "--alpha-y", "0.12",
                "--max-iterations", "1200",
                "--output", str(data / "s2-bipolaron"),
            ],
        ),
        (
            "s2-exciton",
            [
                py, "experiments/exciton_reference_branch_benchmark.py",
                "--size", "20",
                "--onsite-attraction", "0.525",
                "--max-iterations", "1600",
                "--mode", "frenkel",
                "--mode", "diagonal",
                "--output", str(data / "s2-exciton.json"),
            ],
        ),
    ]

    spin_cases = [
        ("preconditioned", "singlet", "onsite", "9"),
        ("rprop", "singlet", "onsite", "8"),
        ("preconditioned", "triplet", "bond_x", "0"),
        ("rprop", "triplet", "onsite", "0"),
    ]
    for optimizer, multiplicity, seed, root_id in spin_cases:
        stem = f"s2-spin-{optimizer}-{multiplicity}"
        commands.append(
            (
                stem,
                [
                    py, "experiments/s1r_root_branch.py",
                    "--structural-optimizer", optimizer,
                    "--multiplicity", multiplicity,
                    "--structural-seed", seed,
                    "--root-seed-id", root_id,
                    "--output", str(data / f"{stem}.json"),
                    "--arrays", str(data / f"{stem}.npz"),
                ],
            )
        )

    if all(g["status"] == "pass" for g in gates):
        for name, command in commands:
            gate = _run(name, command, root)
            gates.append(gate)
            if gate["status"] != "pass":
                break

    aggregate_json = aggregate / "s2-aggregate.json"
    if all(g["status"] == "pass" for g in gates):
        gates.append(
            _run(
                "s2-aggregate",
                [
                    py, "experiments/s2_static_aggregate.py",
                    "--input-dir", str(data),
                    "--output", str(aggregate_json),
                    "--markdown", str(aggregate / "s2-aggregate.md"),
                    "--full-pytest-passed",
                ],
                root,
            )
        )

    payload = (
        json.loads(aggregate_json.read_text(encoding="utf-8"))
        if aggregate_json.exists()
        else None
    )
    summary = {
        "metadata": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git(["rev-parse", "HEAD"]),
            "git_branch": _git(["branch", "--show-current"]),
            "git_status": _git(["status", "--porcelain"]),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "thread_environment": {
                key: os.environ.get(key)
                for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
            },
        },
        "execution_integrity": bool(
            payload is not None and all(g["status"] == "pass" for g in gates)
        ),
        "s2_pass": None if payload is None else bool(payload["summary"]["s2_pass"]),
        "gates": gates,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nS2 local artifact: {root}")
    print(f"Execution integrity: {summary['execution_integrity']}")
    print(f"S2 pass: {summary['s2_pass']}")
    if not summary["execution_integrity"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
