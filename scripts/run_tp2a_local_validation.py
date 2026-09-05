#!/usr/bin/env python3
"""Run TP2a paired-field linear-response validation locally.

The runner keeps BLAS/OpenMP single threaded, records provenance, executes the
focused transport tests and full regression suite, and then runs the 20x20
paired-field screening benchmark. Hosted GitHub Actions are intentionally not
required.
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
    output_dir = root / "tp2a-local-validation" / stamp
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
            "TP2a 20x20 paired +/- field one-polaron statistical screening; "
            "no converged mobility, steady-state, material, pair-transport, threading, or GPU claim"
        ),
    }

    python = sys.executable
    focused = [
        "tests/test_tp0_driven_thermal_decoherence.py",
        "tests/test_tp1_transport_observables.py",
        "tests/test_tp2_linear_response.py",
        "tests/test_electric_field.py",
        "tests/test_d4_coupled_bath.py",
        "tests/test_d5b_instantaneous_decoherence.py",
    ]
    gates = [
        _run("tp2a-focused-pytest", [python, "-m", "pytest", *focused, "-q"], output_dir),
        _run("full-pytest", [python, "-m", "pytest"], output_dir),
    ]

    artifact = output_dir / "tp2a-paired-field-linear-response.json"
    markdown = output_dir / "tp2a-paired-field-linear-response.md"
    gates.append(
        _run(
            "tp2a-paired-field-linear-response-benchmark",
            [
                python,
                "experiments/tp2a_paired_field_linear_response.py",
                "--size",
                "20",
                "--temperature-K",
                "300",
                "--gamma-u-per-fs",
                "0.01",
                "--gamma-v-per-fs",
                "0.01",
                "--field-magnitudes-mv-per-A",
                "0.5",
                "1.0",
                "2.0",
                "--dt-fs",
                "0.2",
                "--final-time-fs",
                "6000",
                "--burn-in-fs",
                "2000",
                "--temperature-sample-interval-fs",
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
    response = None
    if artifact.exists():
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        closure_pass = bool(payload.get("closure_pass", False))
        checks = payload.get("closure_checks")
        aggregate = payload.get("aggregate_numerics")
        response = payload.get("response")
        if not closure_pass and "tp2a-paired-field-linear-response-benchmark" not in failed:
            failed.append("tp2a-closure-gates")

    summary = {
        "metadata": metadata,
        "overall_status": "pass" if not failed else "fail",
        "failed_gates": failed,
        "gates": gates,
        "tp2a_artifact": str(artifact.relative_to(root)) if artifact.exists() else None,
        "tp2a_closure_pass": closure_pass,
        "tp2a_closure_checks": checks,
        "tp2a_aggregate_numerics": aggregate,
        "tp2a_response": response,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n=== TP2A LOCAL VALIDATION SUMMARY ===")
    print(f"Output directory: {output_dir}")
    print(f"Overall status: {summary['overall_status'].upper()}")
    if failed:
        print("Failed gates: " + ", ".join(failed))
        raise SystemExit(1)
    print("All TP2a paired-field numerical and regression gates passed.")
    print("Inspect the response statistics before deciding TP2b; PASS is not a mobility claim.")


if __name__ == "__main__":
    main()
