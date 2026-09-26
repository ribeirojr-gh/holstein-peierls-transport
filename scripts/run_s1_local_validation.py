#!/usr/bin/env python3
"""Self-contained local fallback for S1 spin-adapted RPROP bridge validation."""

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


def _run(name: str, command: list[str], directory: Path) -> dict:
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
    log = directory / f"{name}.log"
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
    root = Path("s1-local-validation") / stamp
    branches = root / "branches"
    aggregate = root / "aggregate"
    branches.mkdir(parents=True, exist_ok=True)
    aggregate.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    gates = []

    gates.append(
        _run(
            "s1-pycompile",
            [
                py,
                "-m",
                "py_compile",
                "src/holstein_peierls/spin_adapted/relaxation_control.py",
                "src/holstein_peierls/spin_adapted/rprop_bridge.py",
                "experiments/s1_spin_adapted_rprop_branch.py",
                "experiments/s1_spin_adapted_rprop_aggregate.py",
            ],
            root,
        )
    )
    gates.append(
        _run(
            "s1-focused-pytest",
            [
                py,
                "-m",
                "pytest",
                "tests/test_s1_rprop_bridge.py",
                "tests/test_spin_relaxation_control.py",
                "tests/test_spin_adapted_static.py",
                "-q",
            ],
            root,
        )
    )
    gates.append(_run("full-pytest", [py, "-m", "pytest"], root))

    if all(gate["status"] == "pass" for gate in gates):
        for optimizer in ("preconditioned", "rprop"):
            for multiplicity in ("singlet", "triplet"):
                for seed in ("onsite", "bond_x", "bond_y"):
                    name = f"s1-{optimizer}-{multiplicity}-{seed}"
                    output = branches / f"{name}.json"
                    gate = _run(
                        name,
                        [
                            py,
                            "experiments/s1_spin_adapted_rprop_branch.py",
                            "--structural-optimizer",
                            optimizer,
                            "--multiplicity",
                            multiplicity,
                            "--seed",
                            seed,
                            "--output",
                            str(output),
                        ],
                        root,
                    )
                    gates.append(gate)
                    if gate["status"] != "pass":
                        break
                if gates[-1]["status"] != "pass":
                    break
            if gates[-1]["status"] != "pass":
                break

    aggregate_json = aggregate / "s1-aggregate.json"
    if all(gate["status"] == "pass" for gate in gates):
        gates.append(
            _run(
                "s1-aggregate",
                [
                    py,
                    "experiments/s1_spin_adapted_rprop_aggregate.py",
                    "--input-dir",
                    str(branches),
                    "--output",
                    str(aggregate_json),
                    "--markdown",
                    str(aggregate / "s1-aggregate.md"),
                ],
                root,
            )
        )

    failed = [gate["name"] for gate in gates if gate["status"] != "pass"]
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
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "thread_environment": {
                key: os.environ.get(key, "unset")
                for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
            },
        },
        "execution_integrity": "pass" if not failed and payload is not None else "fail",
        "failed_gates": failed,
        "s1_primary_pass": (
            None if payload is None else bool(payload["summary"]["s1_pass"])
        ),
        "gates": gates,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nS1 local artifact: {root}")
    print(f"Execution integrity: {summary['execution_integrity'].upper()}")
    print(f"S1 primary pass: {summary['s1_primary_pass']}")
    if summary["execution_integrity"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
