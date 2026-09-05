"""Run TP1 periodic transport-observable validation locally.

The runner keeps BLAS/OpenMP libraries single-threaded, executes the focused TP0
and TP1 regression tests, the complete pytest suite, and the stochastic TP1
bond-current/displacement benchmark.  Hosted GitHub Actions are intentionally
not required.
"""

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
    output_dir = root / "tp1-local-validation" / stamp
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
            "TP1 PBC-safe one-polaron probability current, particle velocity, and "
            "unwrapped displacement; no mobility or steady-state claim"
        ),
    }

    python = sys.executable
    focused = [
        "tests/test_electric_field.py",
        "tests/test_d3_field_driven.py",
        "tests/test_d4_coupled_bath.py",
        "tests/test_d5b_instantaneous_decoherence.py",
        "tests/test_tp0_driven_thermal_decoherence.py",
        "tests/test_tp1_transport_observables.py",
    ]
    gates = [
        _run("tp1-focused-pytest", [python, "-m", "pytest", *focused, "-q"], output_dir),
        _run("full-pytest", [python, "-m", "pytest"], output_dir),
    ]

    artifact = output_dir / "tp1-transport-observables.json"
    markdown = output_dir / "tp1-transport-observables.md"
    gates.append(
        _run(
            "tp1-transport-observables-benchmark",
            [
                python,
                "experiments/tp1_transport_observables.py",
                "--size",
                "4",
                "--temperature-K",
                "300",
                "--gamma-u-per-fs",
                "0.01",
                "--gamma-v-per-fs",
                "0.01",
                "--field-mv-per-A",
                "2.0",
                "--dt-fs",
                "0.2",
                "--final-time-fs",
                "4000",
                "--burn-in-fs",
                "1000",
                "--sample-interval-fs",
                "10",
                "--decoherence-interval-fs",
                "180",
                "--seeds",
                "20260905",
                "20260906",
                "20260907",
                "20260908",
                "--krylov-dimension",
                "6",
                "--output",
                str(artifact),
                "--markdown",
                str(markdown),
            ],
            output_dir,
        )
    )

    failed = [gate["name"] for gate in gates if gate["status"] != "pass"]
    closure_pass = False
    checks = None
    aggregate = None
    if artifact.exists():
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        closure_pass = bool(payload.get("closure_pass", False))
        checks = payload.get("closure_checks")
        aggregate = payload.get("aggregate")
        if not closure_pass and "tp1-transport-observables-benchmark" not in failed:
            failed.append("tp1-closure-gates")

    summary = {
        "metadata": metadata,
        "overall_status": "pass" if not failed else "fail",
        "failed_gates": failed,
        "gates": gates,
        "tp1_artifact": str(artifact.relative_to(root)) if artifact.exists() else None,
        "tp1_closure_pass": closure_pass,
        "tp1_closure_checks": checks,
        "tp1_aggregate": aggregate,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n=== TP1 LOCAL VALIDATION SUMMARY ===")
    print(f"Output directory: {output_dir}")
    print(f"Overall status: {summary['overall_status'].upper()}")
    if failed:
        print("Failed gates: " + ", ".join(failed))
        raise SystemExit(1)
    print("All TP1 periodic-current/displacement and regression gates passed.")


if __name__ == "__main__":
    main()
