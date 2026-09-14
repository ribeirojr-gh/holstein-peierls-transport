#!/usr/bin/env python3
"""Run the IP1m isotropic single-hop wake-memory validation locally."""

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
    output_dir = Path("ip1m-local-validation") / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    gates: list[dict] = []

    gates.append(
        _run(
            "ip1m-pycompile",
            [
                python,
                "-m",
                "py_compile",
                "src/holstein_peierls/dynamics/single_hop_memory.py",
                "experiments/ip1m_isotropic_single_hop_memory.py",
            ],
            output_dir,
        )
    )
    gates.append(
        _run(
            "ip1m-focused-pytest",
            [
                python,
                "-m",
                "pytest",
                "tests/test_ip1m_single_hop_memory.py",
                "tests/test_ip1l_interhop_memory.py",
                "tests/test_ip1k_event_background.py",
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

    recurrence_json = output_dir / "ip1m-phonon-recurrence.json"
    recurrence_md = output_dir / "ip1m-phonon-recurrence.md"
    gates.append(
        _run(
            "ip1m-phonon-recurrence-preflight",
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
                "1100.0",
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

    artifact = output_dir / "ip1m-isotropic-single-hop-memory.json"
    markdown = output_dir / "ip1m-isotropic-single-hop-memory.md"
    trajectory = output_dir / "ip1m-isotropic-single-hop-memory-trajectory.npz"
    if all(gate["status"] == "pass" for gate in gates):
        gates.append(
            _run(
                "ip1m-isotropic-single-hop-memory-benchmark",
                [
                    python,
                    "experiments/ip1m_isotropic_single_hop_memory.py",
                    "--size",
                    "40",
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
                    "--trajectory",
                    str(trajectory),
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
    overall = bool(not failed and recurrence_pass and numerical_pass and trajectory.exists())

    physical = payload.get("physical_decision") if payload else None
    condition = payload.get("condition") if payload else None
    compact = None
    if condition is not None and condition.get("analysis") is not None:
        analysis = condition["analysis"]
        event = analysis["event"]
        memory = analysis["d2_memory_gate"]
        packet = analysis["first_hop_backward_packet"]
        envelope = analysis["spatial_envelope_fit"]
        compact = {
            "event_time_fs": event["transition_start_time_fs"],
            "event_direction": event["direction"],
            "backward_lag_fs": packet["lag_fs"],
            "backward_speed_sites_per_ps": packet["speed_sites_per_ps"],
            "backward_correlation": packet["correlation"],
            "backward_packet_qualified": packet["packet_qualified"],
            "late_memory_present": memory["late_memory_present"],
            "sustained_late_memory": memory["sustained_late_memory"],
            "late_passing_bin_count": memory["late_passing_bin_count"],
            "late_passing_fraction": memory["late_passing_fraction"],
            "latest_passing_center_fs": memory["latest_passing_center_fs"],
            "envelope_fit_available": envelope["available"],
            "envelope_speed_sites_per_ps": envelope["speed_sites_per_ps"],
            "envelope_r_squared": envelope["r_squared"],
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
            "scope": "IP1m 40x40 isotropic T=0 10mV/A natural single-hop wake persistence and spatial envelope",
        },
        "overall_status": "pass" if overall else "fail",
        "failed_gates": failed,
        "recurrence_preflight_pass": recurrence_pass,
        "gates": gates,
        "ip1m_artifact": str(artifact) if artifact.exists() else None,
        "ip1m_markdown": str(markdown) if markdown.exists() else None,
        "ip1m_trajectory": str(trajectory) if trajectory.exists() else None,
        "ip1m_numerical_pass": numerical_pass,
        "ip1m_numerical_checks": condition.get("numerical_checks") if condition else None,
        "physical_decision": physical,
        "compact_physical_summary": compact,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nValidation artifact: {output_dir}")
    print(f"Overall IP1m numerical status: {summary['overall_status'].upper()}")
    print(f"Recurrence preflight: {'PASS' if recurrence_pass else 'FAIL'}")
    if physical is not None:
        print(
            "Single-hop long-lived retrograde memory: "
            + ("YES" if physical["single_hop_long_lived_retrograde_memory"] else "NO")
        )

    if not overall:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
