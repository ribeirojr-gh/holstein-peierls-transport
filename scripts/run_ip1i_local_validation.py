#!/usr/bin/env python3
"""Run local validation for IP1i deterministic field-driven transport screen."""

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


def _git(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path("ip1i-local-validation") / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    gates: list[dict] = []

    gates.append(
        _run(
            "ip1i-pycompile",
            [python, "-m", "py_compile", "experiments/ip1i_field_driven_transport_screen.py"],
            output_dir,
        )
    )
    gates.append(
        _run(
            "ip1i-focused-pytest",
            [
                python,
                "-m",
                "pytest",
                "tests/test_d3_field_driven.py",
                "tests/test_tp1_transport_observables.py",
                "tests/test_ip1a_hopping_observables.py",
                "tests/test_numerical_validation.py",
                "tests/test_static_solver.py",
                "-q",
            ],
            output_dir,
        )
    )
    gates.append(_run("full-pytest", [python, "-m", "pytest"], output_dir))

    recurrence_json = output_dir / "ip1i-phonon-recurrence.json"
    recurrence_md = output_dir / "ip1i-phonon-recurrence.md"
    gates.append(
        _run(
            "ip1i-phonon-recurrence-preflight",
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
                "0.0",
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

    artifact = output_dir / "ip1i-field-driven-transport-screen.json"
    markdown = output_dir / "ip1i-field-driven-transport-screen.md"
    if all(gate["status"] == "pass" for gate in gates):
        gates.append(
            _run(
                "ip1i-field-driven-transport-screen-benchmark",
                [
                    python,
                    "experiments/ip1i_field_driven_transport_screen.py",
                    "--size",
                    "40",
                    "--ratios",
                    "1.0",
                    "0.15",
                    "--fields-mv-per-A",
                    "2.0",
                    "5.0",
                    "10.0",
                    "--dt-fs",
                    "0.2",
                    "--final-time-fs",
                    "5000",
                    "--tracker-sample-interval-fs",
                    "2.0",
                    "--energy-sample-interval-fs",
                    "10.0",
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

    payload = json.loads(artifact.read_text(encoding="utf-8")) if artifact.exists() else None
    recurrence = json.loads(recurrence_json.read_text(encoding="utf-8")) if recurrence_json.exists() else None
    failed = [gate["name"] for gate in gates if gate["status"] != "pass"]
    recurrence_pass = bool(
        recurrence is not None and recurrence.get("planned_run_finishes_before_stationary_wrap", False)
    )
    numerical_pass = bool(payload is not None and payload.get("numerical_pass", False))
    overall = bool(not failed and recurrence_pass and numerical_pass)
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
            "scope": "IP1i 40x40 deterministic field-driven moving-polaron protocol screen",
        },
        "overall_status": "pass" if overall else "fail",
        "failed_gates": failed,
        "recurrence_preflight_pass": recurrence_pass,
        "gates": gates,
        "ip1i_artifact": str(artifact) if artifact.exists() else None,
        "ip1i_numerical_pass": numerical_pass,
        "qualified_condition_count": int(payload.get("qualified_condition_count", 0)) if payload else 0,
        "condition_summary": (
            [
                {
                    "anisotropy_ratio": item["anisotropy_ratio"],
                    "field_mV_per_A": item["field_mV_per_A"],
                    "tp1_displacement_x_A": item["tp1_displacement_x_A"],
                    "nearest_neighbor_event_count": item["nearest_neighbor_event_count"],
                    "x_nearest_neighbor_fraction": item["x_nearest_neighbor_fraction"],
                    "persistent_net_dx_sites": item["persistent_net_dx_sites"],
                    "median_nn_event_gap_fs": item["median_nn_event_gap_fs"],
                    "longest_postevent_quiet_interval_fs": item["longest_postevent_quiet_interval_fs"],
                    "transport_qualified": item["transport_qualified"],
                    "numerical_pass": item["numerical_pass"],
                }
                for item in payload["conditions"]
            ]
            if payload is not None
            else None
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nValidation artifact: {output_dir}")
    print(f"Overall IP1i status: {summary['overall_status'].upper()}")
    print(f"Recurrence preflight: {'PASS' if recurrence_pass else 'FAIL'}")
    print(f"Transport-qualified controls: {summary['qualified_condition_count']}")
    if not overall:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
