#!/usr/bin/env python3
"""Run IP1a zero-field thermal-hopping validation locally.

The runner avoids hosted GitHub Actions.  It performs syntax checks, focused
regressions, the complete pytest suite, and then the expensive 20x20 temperature
screen.  A numerical PASS does not require any hop to occur.
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
    output_dir = root / "ip1a-local-validation" / stamp
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
            "IP1a zero-field 20x20 finite-temperature persistent hopping and "
            "transient local transfer-anisotropy screen; no nonzero-hop, diffusion, "
            "activation-energy, or mobility requirement"
        ),
    }

    python = sys.executable
    gates = [
        _run(
            "ip1a-pycompile",
            [
                python,
                "-m",
                "py_compile",
                "src/holstein_peierls/dynamics/hopping_observables.py",
                "experiments/ip1a_zero_field_thermal_hopping.py",
            ],
            output_dir,
        ),
        _run(
            "ip1a-focused-pytest",
            [
                python,
                "-m",
                "pytest",
                "tests/test_ip1a_hopping_observables.py",
                "tests/test_d4_coupled_bath.py",
                "tests/test_d5b_instantaneous_decoherence.py",
                "tests/test_tp1_transport_observables.py",
                "tests/test_static_solver.py",
                "-q",
            ],
            output_dir,
        ),
        _run("full-pytest", [python, "-m", "pytest"], output_dir),
    ]

    artifact = output_dir / "ip1a-zero-field-thermal-hopping.json"
    markdown = output_dir / "ip1a-zero-field-thermal-hopping.md"
    gates.append(
        _run(
            "ip1a-zero-field-thermal-hopping-benchmark",
            [
                python,
                "experiments/ip1a_zero_field_thermal_hopping.py",
                "--size",
                "20",
                "--temperatures-K",
                "100",
                "200",
                "300",
                "400",
                "500",
                "--isotropic-ratio",
                "1.0",
                "--reference-ratio",
                "0.15",
                "--reference-temperature-K",
                "300",
                "--dt-fs",
                "0.2",
                "--final-time-fs",
                "20000",
                "--burn-in-fs",
                "2000",
                "--sample-interval-fs",
                "2.0",
                "--persistence-fs",
                "20.0",
                "--prehop-window-fs",
                "100.0",
                "--decoherence-interval-fs",
                "180.0",
                "--lattice-seeds",
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
    numerical_pass = False
    numerical_checks = None
    condition_summary = None
    if artifact.exists():
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        numerical_pass = bool(payload.get("numerical_pass", False))
        numerical_checks = payload.get("numerical_checks")
        condition_summary = [
            {
                "anisotropy_ratio": condition["anisotropy_ratio"],
                "temperature_K": condition["temperature_K"],
                "aggregate": condition["aggregate"],
            }
            for condition in payload.get("conditions", [])
        ]
        if not numerical_pass and "ip1a-zero-field-thermal-hopping-benchmark" not in failed:
            failed.append("ip1a-numerical-gates")

    summary = {
        "metadata": metadata,
        "overall_status": "pass" if not failed else "fail",
        "failed_gates": failed,
        "gates": gates,
        "ip1a_artifact": str(artifact.relative_to(root)) if artifact.exists() else None,
        "ip1a_numerical_pass": numerical_pass,
        "ip1a_numerical_checks": numerical_checks,
        "condition_summary": condition_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== IP1A LOCAL VALIDATION SUMMARY ===")
    print(f"Output directory: {output_dir}")
    print(f"Overall status: {summary['overall_status'].upper()}")
    if failed:
        print("Failed gates: " + ", ".join(failed))
        raise SystemExit(1)
    print("All IP1a execution and numerical gates passed.")
    print("Persistent-hop counts and transient-anisotropy trends require manual physical interpretation.")


if __name__ == "__main__":
    main()
