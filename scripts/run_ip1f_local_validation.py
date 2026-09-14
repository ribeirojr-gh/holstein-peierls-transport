#!/usr/bin/env python3
"""Run IP1f carrier-centered phonon-wake validation locally."""

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
    print(f"[{name}] {status.upper()} ({elapsed:.2f} s)", flush=True)
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
    output_dir = Path("ip1f-local-validation") / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    python = sys.executable

    gates: list[dict] = []
    gates.append(
        _run(
            "ip1f-pycompile",
            [
                python,
                "-m",
                "py_compile",
                "src/holstein_peierls/dynamics/phonon_wake.py",
                "experiments/ip1f_carrier_centered_phonon_wake.py",
            ],
            output_dir,
        )
    )
    gates.append(
        _run(
            "ip1f-focused-pytest",
            [
                python,
                "-m",
                "pytest",
                "tests/test_ip1f_phonon_wake.py",
                "tests/test_ip1p_phonon_recurrence.py",
                "tests/test_ip1d_matched_precursors.py",
                "tests/test_ip1c_event_conditioned_hopping.py",
                "tests/test_ip1a_hopping_observables.py",
                "tests/test_tp1_transport_observables.py",
                "tests/test_d4_coupled_bath.py",
                "tests/test_d5b_instantaneous_decoherence.py",
                "tests/test_numerical_validation.py",
                "tests/test_static_solver.py",
                "-q",
            ],
            output_dir,
        )
    )
    gates.append(_run("full-pytest", [python, "-m", "pytest"], output_dir))

    recurrence_json = output_dir / "ip1f-phonon-recurrence.json"
    recurrence_md = output_dir / "ip1f-phonon-recurrence.md"
    gates.append(
        _run(
            "ip1f-phonon-recurrence-preflight",
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
                "0.002",
                "--event-half-window-fs",
                "2000.0",
                "--planned-final-time-fs",
                "15000.0",
                "--output",
                str(recurrence_json),
                "--markdown",
                str(recurrence_md),
            ],
            output_dir,
        )
    )

    recurrence_pass = False
    if recurrence_json.exists():
        recurrence = json.loads(recurrence_json.read_text(encoding="utf-8"))
        recurrence_pass = bool(recurrence["planned_run_finishes_before_stationary_wrap"])

    artifact = output_dir / "ip1f-carrier-centered-phonon-wake.json"
    markdown = output_dir / "ip1f-carrier-centered-phonon-wake.md"
    profiles = output_dir / "ip1f-carrier-centered-phonon-wake-profiles.npz"
    if recurrence_pass and all(gate["status"] == "pass" for gate in gates):
        gates.append(
            _run(
                "ip1f-carrier-centered-phonon-wake-benchmark",
                [
                    python,
                    "experiments/ip1f_carrier_centered_phonon_wake.py",
                    "--size",
                    "40",
                    "--temperature-K",
                    "300",
                    "--ratios",
                    "1.0",
                    "0.15",
                    "--gamma-v-values",
                    "0.01",
                    "0.002",
                    "--gamma-u-per-fs",
                    "0.01",
                    "--dt-fs",
                    "0.2",
                    "--final-time-fs",
                    "15000",
                    "--burn-in-fs",
                    "5000",
                    "--sample-interval-fs",
                    "2.0",
                    "--decoherence-interval-fs",
                    "180.0",
                    "--lattice-seeds",
                    "20260905",
                    "20260906",
                    "--krylov-dimension",
                    "6",
                    "--output",
                    str(artifact),
                    "--markdown",
                    str(markdown),
                    "--profiles",
                    str(profiles),
                ],
                output_dir,
            )
        )

    payload = json.loads(artifact.read_text(encoding="utf-8")) if artifact.exists() else None
    failed = [gate["name"] for gate in gates if gate["status"] != "pass"]
    numerical_pass = bool(payload is not None and payload.get("numerical_pass", False))
    overall = bool(not failed and recurrence_pass and numerical_pass and profiles.exists())

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_value(["rev-parse", "HEAD"]),
        "git_branch": _git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_status": _git_value(["status", "--porcelain"]),
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
        "scope": "IP1f 40x40 carrier-centered phonon-wake characterization; passive lattice-energy/current diagnostics only; no mobility/diffusion/activation/rate claim",
    }
    summary = {
        "metadata": metadata,
        "overall_status": "pass" if overall else "fail",
        "failed_gates": failed,
        "recurrence_preflight_pass": recurrence_pass,
        "gates": gates,
        "ip1f_artifact": str(artifact) if artifact.exists() else None,
        "ip1f_markdown": str(markdown) if markdown.exists() else None,
        "ip1f_profiles": str(profiles) if profiles.exists() else None,
        "ip1f_numerical_pass": numerical_pass,
        "ip1f_numerical_checks": payload.get("numerical_checks") if payload is not None else None,
        "condition_summary": (
            [
                {
                    "gamma_v_per_fs": condition["gamma_v_per_fs"],
                    "anisotropy_ratio": condition["anisotropy_ratio"],
                    "aggregate": condition["aggregate"],
                }
                for condition in payload["conditions"]
            ]
            if payload is not None
            else None
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nValidation artifact: {output_dir}")
    print(f"Overall runner status: {summary['overall_status'].upper()}")
    print(f"Recurrence preflight: {'PASS' if recurrence_pass else 'FAIL'}")
    if payload is not None:
        print(f"IP1f numerical status: {'PASS' if numerical_pass else 'FAIL'}")

    if not overall:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
