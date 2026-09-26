#!/usr/bin/env python3
"""Self-contained local fallback for the complete S1R root-manifold audit."""

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
    log = root / f"{name}.log"
    log.write_text(proc.stdout, encoding="utf-8")
    print(proc.stdout, end="", flush=True)
    return {
        "name": name,
        "command": command,
        "return_code": int(proc.returncode),
        "status": "pass" if proc.returncode == 0 else "fail",
        "elapsed_s": elapsed,
        "log": str(log),
    }


def _git(args: list[str]) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path("s1r-local-validation") / stamp
    branches = root / "branches"
    aggregate = root / "aggregate"
    root.mkdir(parents=True, exist_ok=True)
    branches.mkdir(parents=True, exist_ok=True)
    aggregate.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    gates: list[dict] = []

    gates.append(
        _run(
            "s1r-pycompile",
            [
                py,
                "-m",
                "py_compile",
                "src/holstein_peierls/spin_adapted/excitation_reference.py",
                "src/holstein_peierls/spin_adapted/relaxation_control.py",
                "src/holstein_peierls/spin_adapted/s1r_root_manifold.py",
                "experiments/s1r_seed_preflight.py",
                "experiments/s1r_root_branch.py",
                "experiments/s1r_root_aggregate.py",
            ],
            root,
        )
    )
    gates.append(
        _run(
            "s1r-focused-pytest",
            [
                py,
                "-m",
                "pytest",
                "tests/test_s1r_root_manifold.py",
                "tests/test_spin_relaxation_control.py",
                "tests/test_excitation_reference.py",
                "tests/test_s1_rprop_bridge.py",
                "tests/test_s1p_provenance.py",
                "-q",
            ],
            root,
        )
    )
    full = _run("full-pytest", [py, "-m", "pytest"], root)
    gates.append(full)
    preflight = _run(
        "s1r-seed-preflight",
        [
            py,
            "experiments/s1r_seed_preflight.py",
            "--output",
            str(root / "s1r-seed-preflight.json"),
            "--markdown",
            str(root / "s1r-seed-preflight.md"),
        ],
        root,
    )
    gates.append(preflight)

    validation_pass = all(gate["status"] == "pass" for gate in gates)

    if validation_pass:
        for optimizer in ("preconditioned", "rprop"):
            for structural_seed in ("onsite", "bond_x", "bond_y"):
                for root_seed_id in range(16):
                    name = (
                        f"s1r-singlet-{optimizer}-{structural_seed}-"
                        f"root{root_seed_id:02d}"
                    )
                    gates.append(
                        _run(
                            name,
                            [
                                py,
                                "experiments/s1r_root_branch.py",
                                "--structural-optimizer",
                                optimizer,
                                "--multiplicity",
                                "singlet",
                                "--structural-seed",
                                structural_seed,
                                "--root-seed-id",
                                str(root_seed_id),
                                "--output",
                                str(branches / f"{name}.json"),
                                "--arrays",
                                str(branches / f"{name}.npz"),
                            ],
                            root,
                        )
                    )

        for optimizer in ("preconditioned", "rprop"):
            for structural_seed in ("onsite", "bond_x", "bond_y"):
                name = f"s1r-triplet-{optimizer}-{structural_seed}-root00"
                gates.append(
                    _run(
                        name,
                        [
                            py,
                            "experiments/s1r_root_branch.py",
                            "--structural-optimizer",
                            optimizer,
                            "--multiplicity",
                            "triplet",
                            "--structural-seed",
                            structural_seed,
                            "--root-seed-id",
                            "0",
                            "--output",
                            str(branches / f"{name}.json"),
                            "--arrays",
                            str(branches / f"{name}.npz"),
                        ],
                        root,
                    )
                )

    branch_json_count = len(list(branches.glob("s1r-*.json")))
    aggregate_gate = None
    if validation_pass and branch_json_count == 102:
        command = [
            py,
            "experiments/s1r_root_aggregate.py",
            "--input-dir",
            str(branches),
            "--output",
            str(aggregate / "s1r-aggregate.json"),
            "--markdown",
            str(aggregate / "s1r-aggregate.md"),
            "--arrays",
            str(aggregate / "s1r-aggregate.npz"),
        ]
        if full["status"] == "pass":
            command.append("--full-pytest-passed")
        aggregate_gate = _run("s1r-aggregate", command, root)
        gates.append(aggregate_gate)

    aggregate_json = aggregate / "s1r-aggregate.json"
    payload = (
        json.loads(aggregate_json.read_text(encoding="utf-8"))
        if aggregate_json.exists()
        else None
    )
    failed = [gate["name"] for gate in gates if gate["status"] != "pass"]
    execution_integrity = bool(
        validation_pass
        and branch_json_count == 102
        and aggregate_gate is not None
        and aggregate_gate["status"] == "pass"
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
                for key in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            },
        },
        "execution_integrity": "pass" if execution_integrity else "fail",
        "branch_json_count": branch_json_count,
        "failed_execution_gates": failed,
        "s1r_primary_pass": (
            None if payload is None else bool(payload["summary"]["s1r_pass"])
        ),
        "gates": gates,
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nS1R local artifact: {root}")
    print(f"Execution integrity: {summary['execution_integrity'].upper()}")
    print(f"S1R primary pass: {summary['s1r_primary_pass']}")
    if not execution_integrity:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
