#!/usr/bin/env python3
"""Run local validation for IP1g controlled single-relocation wake dynamics."""

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
    log = output_dir / f"{name}.log"
    log.write_text(completed.stdout, encoding="utf-8")
    print(completed.stdout, end="")
    status = "pass" if completed.returncode == 0 else "fail"
    print(f"[{name}] {status.upper()} ({elapsed:.2f} s)", flush=True)
    return {
        "name": name,
        "command": command,
        "return_code": int(completed.returncode),
        "status": status,
        "elapsed_s": float(elapsed),
        "log": str(log),
    }


def _git(command: list[str]) -> str:
    result = subprocess.run(
        ["git", *command],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path("ip1g-local-validation") / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    gates: list[dict] = []

    gates.append(
        _run(
            "ip1g-pycompile",
            [
                python,
                "-m",
                "py_compile",
                "src/holstein_peierls/dynamics/single_relocation.py",
                "experiments/ip1g_single_relocation_wake.py",
            ],
            output_dir,
        )
    )
    gates.append(
        _run(
            "ip1g-focused-pytest",
            [
                python,
                "-m",
                "pytest",
                "tests/test_ip1g_single_relocation.py",
                "tests/test_ip1f_phonon_wake.py",
                "tests/test_ip1p_phonon_recurrence.py",
                "tests/test_numerical_validation.py",
                "tests/test_d2_coupled_dynamics.py",
                "tests/test_static_solver.py",
                "-q",
            ],
            output_dir,
        )
    )
    gates.append(_run("full-pytest", [python, "-m", "pytest"], output_dir))

    recurrence_json = output_dir / "ip1g-phonon-recurrence.json"
    recurrence_md = output_dir / "ip1g-phonon-recurrence.md"
    gates.append(
        _run(
            "ip1g-phonon-recurrence-preflight",
            [
                python,
                "experiments/ip1p_phonon_recurrence_audit.py",
                "--nx",
                "40",
                "--ny",
                "40",
                "--lattice-spacing-A",
                "3.0",
                "--gamma-v-per-fs",
                "0.0",
                "--event-half-window-fs",
                "5000.0",
                "--planned-final-time-fs",
                "5000.0",
                "--output",
                str(recurrence_json),
                "--markdown",
                str(recurrence_md),
            ],
            output_dir,
        )
    )

    recurrence = (
        json.loads(recurrence_json.read_text(encoding="utf-8"))
        if recurrence_json.exists()
        else None
    )
    recurrence_pass = bool(
        recurrence is not None
        and recurrence.get("planned_run_finishes_before_stationary_wrap", False)
    )

    result_json = output_dir / "ip1g-single-relocation-wake.json"
    result_md = output_dir / "ip1g-single-relocation-wake.md"
    profiles = output_dir / "ip1g-single-relocation-wake-profiles.npz"
    if recurrence_pass and all(gate["status"] == "pass" for gate in gates):
        gates.append(
            _run(
                "ip1g-single-relocation-wake-benchmark",
                [
                    python,
                    "experiments/ip1g_single_relocation_wake.py",
                    "--size",
                    "40",
                    "--ratios",
                    "1.0",
                    "0.15",
                    "--dt-fs",
                    "0.2",
                    "--final-time-fs",
                    "5000",
                    "--sample-interval-fs",
                    "2.0",
                    "--krylov-dimension",
                    "6",
                    "--output",
                    str(result_json),
                    "--markdown",
                    str(result_md),
                    "--profiles",
                    str(profiles),
                ],
                output_dir,
            )
        )

    payload = (
        json.loads(result_json.read_text(encoding="utf-8"))
        if result_json.exists()
        else None
    )
    failed = [gate["name"] for gate in gates if gate["status"] != "pass"]
    numerical_pass = bool(payload is not None and payload.get("numerical_pass", False))
    overall = bool(
        not failed
        and recurrence_pass
        and numerical_pass
        and profiles.exists()
    )
    summary = {
        "metadata": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git(["rev-parse", "HEAD"]),
            "git_branch": _git(["branch", "--show-current"]),
            "git_status": _git(["status", "--porcelain"]),
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "processor": platform.machine(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "thread_environment": {
                key: os.environ.get(key, "unset")
                for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
            },
            "scope": "IP1g deterministic controlled single-relocation lattice-radiation validation",
        },
        "overall_status": "pass" if overall else "fail",
        "failed_gates": failed,
        "gates": gates,
        "recurrence_preflight_pass": recurrence_pass,
        "ip1g_artifact": str(result_json) if result_json.exists() else None,
        "ip1g_profiles": str(profiles) if profiles.exists() else None,
        "ip1g_numerical_pass": numerical_pass,
        "ip1g_numerical_checks": payload.get("numerical_checks") if payload else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nValidation artifact: {output_dir}")
    print(f"Overall IP1g status: {summary['overall_status'].upper()}")
    print(f"Recurrence preflight: {'PASS' if recurrence_pass else 'FAIL'}")
    if payload is not None:
        print(f"IP1g numerical status: {'PASS' if numerical_pass else 'FAIL'}")
    if not overall:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
