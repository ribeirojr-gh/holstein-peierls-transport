#!/usr/bin/env python3
"""Run IP0b collective-inertia and midpoint-gap diagnostics locally."""

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


def _run(name: str, command: list[str], output_dir: Path) -> dict:
    print(f"\n=== {name} ===", flush=True)
    start = perf_counter()
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    elapsed = perf_counter() - start
    log_path = output_dir / f"{name}.log"
    log_path.write_text(completed.stdout, encoding="utf-8")
    print(completed.stdout, end="")
    status = "pass" if completed.returncode == 0 else "fail"
    print(f"--- {name}: {status.upper()} ({elapsed:.3f} s) ---", flush=True)
    return {
        "name": name,
        "command": command,
        "return_code": int(completed.returncode),
        "status": status,
        "elapsed_s": float(elapsed),
        "log": str(log_path),
    }


def _git_value(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = root / "ip0b-local-validation" / stamp
    output_dir.mkdir(parents=True, exist_ok=False)

    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_value(["rev-parse", "HEAD"]),
        "git_branch": _git_value(["branch", "--show-current"]),
        "git_status": _git_value(["status", "--short"]),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.machine(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "scope": (
            "IP0b path-specific lattice collective inertia, endpoint harmonic scale, and midpoint low-state gap; "
            "no complete effective mass, MEP, hopping rate, or mobility claim"
        ),
    }

    python = sys.executable
    gates = [
        _run(
            "ip0b-pycompile",
            [
                python,
                "-m",
                "py_compile",
                "src/holstein_peierls/translation_inertia.py",
                "experiments/ip0b_collective_inertia_and_gap.py",
            ],
            output_dir,
        ),
        _run(
            "ip0b-focused-pytest",
            [
                python,
                "-m",
                "pytest",
                "tests/test_ip0b_translation_inertia.py",
                "tests/test_ip0_translation_barrier.py",
                "tests/test_static_solver.py",
                "-q",
            ],
            output_dir,
        ),
        _run("full-pytest", [python, "-m", "pytest"], output_dir),
    ]

    artifact = output_dir / "ip0b-collective-inertia-gap.json"
    markdown = output_dir / "ip0b-collective-inertia-gap.md"
    gates.append(
        _run(
            "ip0b-collective-inertia-gap-benchmark",
            [
                python,
                "experiments/ip0b_collective_inertia_and_gap.py",
                "--size",
                "20",
                "--anisotropy-ratios",
                "0.15",
                "0.30",
                "0.50",
                "0.70",
                "1.00",
                "--image-count",
                "21",
                "--fit-points",
                "5",
                "--solver",
                "sparse",
                "--output",
                str(artifact),
                "--markdown",
                str(markdown),
            ],
            output_dir,
        )
    )

    failed = [gate["name"] for gate in gates if gate["status"] != "pass"]
    numerical_pass = False
    numerical_checks = None
    aggregate = None
    if artifact.exists():
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        numerical_pass = bool(payload.get("numerical_pass", False))
        numerical_checks = payload.get("numerical_checks")
        aggregate = payload.get("aggregate")
        if not numerical_pass and "ip0b-collective-inertia-gap-benchmark" not in failed:
            failed.append("ip0b-numerical-gates")

    summary = {
        "metadata": metadata,
        "overall_status": "pass" if not failed else "fail",
        "failed_gates": failed,
        "gates": gates,
        "ip0b_artifact": str(artifact.relative_to(root)) if artifact.exists() else None,
        "ip0b_numerical_pass": numerical_pass,
        "ip0b_numerical_checks": numerical_checks,
        "ip0b_aggregate": aggregate,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== IP0B LOCAL VALIDATION SUMMARY ===")
    print(f"Output directory: {output_dir}")
    print(f"Overall status: {summary['overall_status'].upper()}")
    if failed:
        print("Failed gates: " + ", ".join(failed))
        raise SystemExit(1)
    print("All IP0b execution and numerical gates passed; collective-inertia/gap interpretation remains manual.")


if __name__ == "__main__":
    main()
