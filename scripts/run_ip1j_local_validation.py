#!/usr/bin/env python3
"""Run IP1j first-natural-hop field-driven wake validation locally."""

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
        result = subprocess.run(
            ["git", *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path("ip1j-local-validation") / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    gates: list[dict] = []

    gates.append(
        _run(
            "ip1j-pycompile",
            [
                python,
                "-m",
                "py_compile",
                "src/holstein_peierls/dynamics/event_wake.py",
                "experiments/ip1j_field_driven_first_hop_wake.py",
            ],
            output_dir,
        )
    )
    gates.append(
        _run(
            "ip1j-focused-pytest",
            [
                python,
                "-m",
                "pytest",
                "tests/test_ip1j_event_wake.py",
                "tests/test_ip1h_wavepacket_flux.py",
                "tests/test_ip1f_phonon_wake.py",
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

    recurrence_json = output_dir / "ip1j-phonon-recurrence.json"
    recurrence_md = output_dir / "ip1j-phonon-recurrence.md"
    gates.append(
        _run(
            "ip1j-phonon-recurrence-preflight",
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
                "1500.0",
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

    artifact = output_dir / "ip1j-field-driven-first-hop-wake.json"
    markdown = output_dir / "ip1j-field-driven-first-hop-wake.md"
    profiles = output_dir / "ip1j-field-driven-first-hop-wake-profiles.npz"
    if all(gate["status"] == "pass" for gate in gates):
        gates.append(
            _run(
                "ip1j-field-driven-first-hop-wake-benchmark",
                [
                    python,
                    "experiments/ip1j_field_driven_first_hop_wake.py",
                    "--size",
                    "40",
                    "--ratios",
                    "1.0",
                    "0.15",
                    "--field-mv-per-A",
                    "10.0",
                    "--dt-fs",
                    "0.2",
                    "--final-time-fs",
                    "5000",
                    "--sample-interval-fs",
                    "2.0",
                    "--energy-sample-interval-fs",
                    "10.0",
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
    recurrence = json.loads(recurrence_json.read_text(encoding="utf-8")) if recurrence_json.exists() else None
    recurrence_pass = bool(
        recurrence is not None and recurrence.get("planned_run_finishes_before_stationary_wrap", False)
    )
    numerical_pass = bool(payload is not None and payload.get("numerical_pass", False))
    overall = bool(not failed and recurrence_pass and numerical_pass and profiles.exists())

    condition_summary = None
    if payload is not None:
        condition_summary = []
        for condition in payload["conditions"]:
            wake = condition.get("wake_analysis")
            condition_summary.append(
                {
                    "anisotropy_ratio": condition["anisotropy_ratio"],
                    "tp1_displacement_x_A": condition["tp1_displacement_x_A"],
                    "persistent_x_event_count": condition["persistent_x_event_count"],
                    "first_event_direction": (
                        wake["first_persistent_x_event"]["direction"] if wake else None
                    ),
                    "post_window_duration_fs": wake["post_window_duration_fs"] if wake else None,
                    "d2_positive_energy_directionality": (
                        wake["d2_positive_energy_directionality"] if wake else None
                    ),
                    "backward_packet_validated": wake["backward_packet_validated"] if wake else False,
                    "forward_packet_validated": wake["forward_packet_validated"] if wake else False,
                    "retrograde_branch_vphonon_dot_vcarrier_negative": (
                        wake["retrograde_branch_vphonon_dot_vcarrier_negative"] if wake else False
                    ),
                }
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
            "scope": "IP1j 40x40 T=0 10mV/A first-natural-hop event-conditioned lattice-radiation validation",
        },
        "overall_status": "pass" if overall else "fail",
        "failed_gates": failed,
        "recurrence_preflight_pass": recurrence_pass,
        "gates": gates,
        "ip1j_artifact": str(artifact) if artifact.exists() else None,
        "ip1j_markdown": str(markdown) if markdown.exists() else None,
        "ip1j_profiles": str(profiles) if profiles.exists() else None,
        "ip1j_numerical_pass": numerical_pass,
        "ip1j_numerical_checks": payload.get("numerical_checks") if payload else None,
        "condition_summary": condition_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nValidation artifact: {output_dir}")
    print(f"Overall IP1j status: {summary['overall_status'].upper()}")
    print(f"Recurrence preflight: {'PASS' if recurrence_pass else 'FAIL'}")
    if payload is not None:
        print(f"IP1j numerical status: {'PASS' if numerical_pass else 'FAIL'}")

    if not overall:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
