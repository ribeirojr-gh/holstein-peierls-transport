#!/usr/bin/env python3
"""Run the IP2a-4 refined-dt complete pre-intervention calibration locally."""

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
    try:
        completed = subprocess.run(
            ["git", *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )
        return completed.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path("ip2a4-local-validation") / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    gates: list[dict] = []

    gates.append(
        _run(
            "ip2a4-pycompile",
            [
                python,
                "-m",
                "py_compile",
                "src/holstein_peierls/dynamics/ensemble_calibration.py",
                "experiments/ip2a4_refined_complete_calibration.py",
            ],
            output_dir,
        )
    )
    gates.append(
        _run(
            "ip2a4-focused-pytest",
            [
                python,
                "-m",
                "pytest",
                "tests/test_ip2a4_protocol.py",
                "tests/test_ip2a2_ensemble_calibration.py",
                "tests/test_ip2a_ensemble_preparation.py",
                "tests/test_d3_field_driven.py",
                "tests/test_numerical_validation.py",
                "tests/test_static_solver.py",
                "-q",
            ],
            output_dir,
        )
    )
    gates.append(_run("full-pytest", [python, "-m", "pytest"], output_dir))

    recurrence_json = output_dir / "ip2a4-phonon-recurrence.json"
    recurrence_md = output_dir / "ip2a4-phonon-recurrence.md"
    gates.append(
        _run(
            "ip2a4-phonon-recurrence-preflight",
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
                "500.0",
                "--planned-final-time-fs",
                "4000.0",
                "--output",
                str(recurrence_json),
                "--markdown",
                str(recurrence_md),
            ],
            output_dir,
        )
    )

    artifact = output_dir / "ip2a4-refined-complete-calibration.json"
    markdown = output_dir / "ip2a4-refined-complete-calibration.md"
    arrays = output_dir / "ip2a4-refined-complete-calibration.npz"
    if all(gate["status"] == "pass" for gate in gates):
        gates.append(
            _run(
                "ip2a4-refined-complete-calibration",
                [
                    python,
                    "experiments/ip2a4_refined_complete_calibration.py",
                    "--size",
                    "40",
                    "--field-mv-per-A",
                    "10.0",
                    "--dt-fs",
                    "0.1",
                    "--search-time-fs",
                    "4000.0",
                    "--sample-interval-fs",
                    "2.0",
                    "--energy-sample-interval-fs",
                    "10.0",
                    "--krylov-dimension",
                    "6",
                    "--candidate-energies-eV",
                    "1e-5",
                    "3e-5",
                    "1e-4",
                    "--output",
                    str(artifact),
                    "--markdown",
                    str(markdown),
                    "--arrays",
                    str(arrays),
                ],
                output_dir,
            )
        )

    payload = json.loads(artifact.read_text(encoding="utf-8")) if artifact.exists() else None
    recurrence = json.loads(recurrence_json.read_text(encoding="utf-8")) if recurrence_json.exists() else None
    failed = [gate["name"] for gate in gates if gate["status"] != "pass"]
    recurrence_pass = bool(
        recurrence is not None
        and recurrence.get("planned_run_finishes_before_stationary_wrap", False)
    )
    numerical_pass = bool(payload is not None and payload.get("numerical_pass", False))
    overall = bool(not failed and recurrence_pass and numerical_pass and arrays.exists())
    selection_succeeded = bool(
        payload is not None
        and payload.get("selection", {}).get("selection_succeeded", False)
    )

    compact = None
    if payload is not None:
        compact = {
            "selection": payload["selection"],
            "elapsed_seconds": payload["elapsed_seconds"],
            "candidate_energies_eV": payload["candidate_energies_eV"],
            "pilot_member_ids": payload["pilot_member_ids"],
        }

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
            "scope": "IP2a-4 refined-dt complete pre-intervention calibration",
        },
        "overall_status": "pass" if overall else "fail",
        "failed_gates": failed,
        "recurrence_preflight_pass": recurrence_pass,
        "gates": gates,
        "ip2a4_artifact": str(artifact) if artifact.exists() else None,
        "ip2a4_markdown": str(markdown) if markdown.exists() else None,
        "ip2a4_arrays": str(arrays) if arrays.exists() else None,
        "ip2a4_numerical_pass": numerical_pass,
        "ip2a4_production_energy_selection_succeeded": selection_succeeded,
        "prior_ip2a2_selection_remains_failed": True,
        "ip2a4_numerical_checks": payload.get("numerical_checks") if payload else None,
        "compact_summary": compact,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nValidation artifact: {output_dir}")
    print(f"Overall IP2a-4 numerical status: {summary['overall_status'].upper()}")
    print(f"Recurrence preflight: {'PASS' if recurrence_pass else 'FAIL'}")
    if compact is not None:
        selection = compact["selection"]
        print(f"Selection succeeded: {selection['selection_succeeded']}")
        print(f"Selected energy: {selection['selected_energy_eV']}")

    if not overall:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
