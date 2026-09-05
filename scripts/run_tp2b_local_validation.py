#!/usr/bin/env python3
"""Run TP2b paired-field mobility-convergence validation locally.

The runner separates numerical correctness from statistical mobility readiness.
A numerically correct benchmark exits successfully even when the pre-registered
mobility-readiness diagnostics report NOT CONVERGED; that outcome is scientific
information rather than a software failure.
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
    output_dir = root / "tp2b-local-validation" / stamp
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
            "TP2b 20x20 long paired +/- field one-polaron convergence study; "
            "mobility readiness is reported separately from numerical pass"
        ),
    }

    python = sys.executable
    gates = [
        _run(
            "tp2b-compile",
            [python, "-m", "py_compile", "experiments/tp2b_paired_field_convergence.py"],
            output_dir,
        ),
        _run(
            "tp2b-focused-pytest",
            [
                python,
                "-m",
                "pytest",
                "tests/test_tp0_driven_thermal_decoherence.py",
                "tests/test_tp1_transport_observables.py",
                "tests/test_tp2_linear_response.py",
                "tests/test_electric_field.py",
                "tests/test_d4_coupled_bath.py",
                "tests/test_d5b_instantaneous_decoherence.py",
                "-q",
            ],
            output_dir,
        ),
        _run("full-pytest", [python, "-m", "pytest"], output_dir),
    ]

    artifact = output_dir / "tp2b-paired-field-convergence.json"
    markdown = output_dir / "tp2b-paired-field-convergence.md"
    gates.append(
        _run(
            "tp2b-paired-field-convergence-benchmark",
            [
                python,
                "experiments/tp2b_paired_field_convergence.py",
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
                "12000",
                "--burn-in-fs",
                "2000",
                "--checkpoint-times-fs",
                "6000",
                "9000",
                "12000",
                "--temperature-sample-interval-fs",
                "20",
                "--decoherence-interval-fs",
                "180",
                "--seeds",
                "20260905",
                "20260906",
                "20260907",
                "20260908",
                "20260909",
                "20260910",
                "20260911",
                "20260912",
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
    numerical_pass = False
    mobility_ready = False
    numerical_checks = None
    readiness_checks = None
    aggregate = None
    final_response = None
    convergence = None
    if artifact.exists():
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        numerical_pass = bool(payload.get("numerical_pass", False))
        mobility_ready = bool(payload.get("mobility_ready", False))
        numerical_checks = payload.get("numerical_checks")
        readiness_checks = payload.get("mobility_readiness_checks")
        aggregate = payload.get("aggregate_numerics")
        final_response = payload.get("final_response")
        convergence = payload.get("convergence_diagnostics")
        if not numerical_pass and "tp2b-paired-field-convergence-benchmark" not in failed:
            failed.append("tp2b-numerical-closure")

    summary = {
        "metadata": metadata,
        "overall_status": "pass" if not failed else "fail",
        "failed_gates": failed,
        "gates": gates,
        "tp2b_artifact": str(artifact.relative_to(root)) if artifact.exists() else None,
        "tp2b_numerical_pass": numerical_pass,
        "tp2b_mobility_ready": mobility_ready,
        "tp2b_numerical_checks": numerical_checks,
        "tp2b_mobility_readiness_checks": readiness_checks,
        "tp2b_aggregate_numerics": aggregate,
        "tp2b_final_response": final_response,
        "tp2b_convergence_diagnostics": convergence,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== TP2B LOCAL VALIDATION SUMMARY ===")
    print(f"Output directory: {output_dir}")
    print(f"Numerical status: {'PASS' if not failed else 'FAIL'}")
    print(f"Mobility readiness: {'PASS' if mobility_ready else 'NOT CONVERGED'}")
    if failed:
        print("Failed numerical gates: " + ", ".join(failed))
        raise SystemExit(1)
    if mobility_ready:
        print("TP2b numerical and pre-registered mobility-readiness diagnostics passed.")
    else:
        print("TP2b is numerically valid, but mobility readiness remains unresolved.")


if __name__ == "__main__":
    main()
