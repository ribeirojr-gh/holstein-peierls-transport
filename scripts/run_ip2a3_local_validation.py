#!/usr/bin/env python3
"""Self-contained local numerical audit of IP2a-3 timestep refinement."""

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
    proc = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    elapsed = float(perf_counter() - start)
    log = output_dir / f"{name}.log"
    log.write_text(proc.stdout, encoding="utf-8")
    print(proc.stdout, end="", flush=True)
    status = "pass" if proc.returncode == 0 else "fail"
    print(f"[{name}] {status.upper()} ({elapsed:.2f} s)", flush=True)
    return {
        "name": name,
        "command": command,
        "return_code": int(proc.returncode),
        "status": status,
        "elapsed_s": elapsed,
        "log": str(log),
    }


def _git(args: list[str]) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = Path("ip2a3-local-validation") / stamp
    directory.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    gates = []
    gates.append(
        _run(
            "ip2a3-pycompile",
            [
                py, "-m", "py_compile",
                "src/holstein_peierls/dynamics/energy_work_refinement.py",
                "experiments/ip2a3_timestep_work_balance.py",
            ],
            directory,
        )
    )
    gates.append(
        _run(
            "ip2a3-focused-pytest",
            [
                py, "-m", "pytest",
                "tests/test_ip2a3_energy_work_refinement.py",
                "tests/test_ip2a2_ensemble_calibration.py",
                "tests/test_ip2a_ensemble_preparation.py",
                "tests/test_d3_field_driven.py",
                "tests/test_numerical_validation.py",
                "-q",
            ],
            directory,
        )
    )
    gates.append(_run("full-pytest", [py, "-m", "pytest"], directory))

    recurrence_json = directory / "ip2a3-phonon-recurrence.json"
    recurrence_md = directory / "ip2a3-phonon-recurrence.md"
    gates.append(
        _run(
            "ip2a3-phonon-recurrence-preflight",
            [
                py,
                "experiments/ip1p_phonon_recurrence_audit.py",
                "--nx", "40", "--ny", "40",
                "--lattice-spacing-A", "3.0",
                "--gamma-v-per-fs", "0.0",
                "--event-half-window-fs", "500.0",
                "--planned-final-time-fs", "4000.0",
                "--output", str(recurrence_json),
                "--markdown", str(recurrence_md),
            ],
            directory,
        )
    )

    artifact = directory / "ip2a3-timestep-work-balance.json"
    markdown = directory / "ip2a3-timestep-work-balance.md"
    arrays = directory / "ip2a3-timestep-work-balance.npz"
    if all(item["status"] == "pass" for item in gates):
        gates.append(
            _run(
                "ip2a3-timestep-work-balance",
                [
                    py, "experiments/ip2a3_timestep_work_balance.py",
                    "--size", "40",
                    "--field-mv-per-A", "10.0",
                    "--dt-grid-fs", "0.2", "0.1", "0.05",
                    "--output", str(artifact),
                    "--markdown", str(markdown),
                    "--arrays", str(arrays),
                ],
                directory,
            )
        )

    payload = json.loads(artifact.read_text(encoding="utf-8")) if artifact.exists() else None
    recurrence = json.loads(recurrence_json.read_text(encoding="utf-8")) if recurrence_json.exists() else None
    recurrence_pass = bool(
        recurrence is not None
        and recurrence.get("planned_run_finishes_before_stationary_wrap", False)
    )
    failed = [item["name"] for item in gates if item["status"] != "pass"]
    numerical_pass = bool(payload is not None and payload.get("numerical_pass", False))
    overall = bool(not failed and recurrence_pass and numerical_pass and arrays.exists())
    diagnostic = payload.get("diagnostic_decision") if payload else None
    summary = {
        "metadata": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git(["rev-parse", "HEAD"]),
            "git_branch": _git(["branch", "--show-current"]),
            "git_status": _git(["status", "--porcelain"]),
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "thread_environment": {
                key: os.environ.get(key, "unset")
                for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
            },
            "scope": "IP2a-3 pre-event energy/work numerical dt diagnostic",
        },
        "overall_status": "pass" if overall else "fail",
        "failed_gates": failed,
        "recurrence_preflight_pass": recurrence_pass,
        "gates": gates,
        "ip2a3_artifact": str(artifact) if artifact.exists() else None,
        "ip2a3_markdown": str(markdown) if markdown.exists() else None,
        "ip2a3_arrays": str(arrays) if arrays.exists() else None,
        "ip2a3_numerical_pass": numerical_pass,
        "ip2a3_numerical_checks": payload.get("numerical_checks") if payload else None,
        "diagnostic_decision": diagnostic,
        "original_ip2a2_selection_remains_failed": True,
    }
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nValidation artifact: {directory}")
    print(f"Execution integrity: {summary['overall_status'].upper()}")
    if diagnostic:
        print(
            "Numerical refinement: "
            f"{'PASS' if diagnostic['numerical_refinement_succeeded'] else 'FAIL'}"
        )
        print(f"Candidate refined dt: {diagnostic['candidate_refined_dt_fs']}")
    print("IP2a-2 production-energy selection remains FAIL.")
    if not overall:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
