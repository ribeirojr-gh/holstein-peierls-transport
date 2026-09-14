#!/usr/bin/env python3
"""Run the IP1n gauge-continuous field-release validation locally."""

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
    output_dir = Path("ip1n-local-validation") / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    gates: list[dict] = []

    gates.append(
        _run(
            "ip1n-pycompile",
            [
                python,
                "-m",
                "py_compile",
                "src/holstein_peierls/dynamics/field_release.py",
                "experiments/ip1n_gauge_continuous_field_release.py",
            ],
            output_dir,
        )
    )
    gates.append(
        _run(
            "ip1n-focused-pytest",
            [
                python,
                "-m",
                "pytest",
                "tests/test_ip1n_field_release.py",
                "tests/test_ip1m_single_hop_memory.py",
                "tests/test_ip1l_interhop_memory.py",
                "tests/test_ip1k_event_background.py",
                "tests/test_ip1j_event_wake.py",
                "tests/test_ip1h_wavepacket_flux.py",
                "tests/test_d3_field_driven.py",
                "tests/test_tp1_transport_observables.py",
                "tests/test_numerical_validation.py",
                "tests/test_static_solver.py",
                "-q",
            ],
            output_dir,
        )
    )
    gates.append(_run("full-pytest", [python, "-m", "pytest"], output_dir))

    recurrence_json = output_dir / "ip1n-phonon-recurrence.json"
    recurrence_md = output_dir / "ip1n-phonon-recurrence.md"
    gates.append(
        _run(
            "ip1n-phonon-recurrence-preflight",
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
                "1000.0",
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

    artifact = output_dir / "ip1n-gauge-continuous-field-release.json"
    markdown = output_dir / "ip1n-gauge-continuous-field-release.md"
    trajectory = output_dir / "ip1n-gauge-continuous-field-release-trajectories.npz"
    if all(gate["status"] == "pass" for gate in gates):
        gates.append(
            _run(
                "ip1n-gauge-continuous-field-release-benchmark",
                [
                    python,
                    "experiments/ip1n_gauge_continuous_field_release.py",
                    "--size",
                    "40",
                    "--field-mv-per-A",
                    "10.0",
                    "--dt-fs",
                    "0.2",
                    "--search-time-fs",
                    "4000.0",
                    "--continuation-fs",
                    "2000.0",
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
    compact = None
    if payload is not None and payload.get("analysis") is not None:
        analysis = payload["analysis"]
        on = analysis["field_on"]
        off = analysis["released"]
        compact = {
            "event_start_time_fs": analysis["event"]["transition_start_time_fs"],
            "switch_time_fs": analysis["switch_time_fs"],
            "phase_continuity_error": analysis["phase_continuity_error"],
            "hamiltonian_continuity_max_abs_eV": analysis[
                "hamiltonian_continuity_max_abs_eV"
            ],
            "released_external_work_eV": off["accumulated_external_work_eV"],
            "released_energy_drift_eV": off["maximum_energy_balance_residual_eV"],
            "field_on_energy_work_residual_eV": on["maximum_energy_balance_residual_eV"],
            "field_on_late_passing_bins": on["sustained_pattern_gate"]["late_passing_bin_count"],
            "released_late_passing_bins": off["sustained_pattern_gate"]["late_passing_bin_count"],
            "field_on_sustained_memory": on["sustained_pattern_gate"]["sustained_pattern_memory"],
            "released_sustained_memory": off["sustained_pattern_gate"]["sustained_pattern_memory"],
            "late_released_to_on_amplitude_ratio": analysis[
                "late_released_to_on_amplitude_ratio"
            ],
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
            "scope": "IP1n 40x40 isotropic gauge-continuous field release after the first natural hop",
        },
        "overall_status": "pass" if overall else "fail",
        "failed_gates": failed,
        "recurrence_preflight_pass": recurrence_pass,
        "gates": gates,
        "ip1n_artifact": str(artifact) if artifact.exists() else None,
        "ip1n_markdown": str(markdown) if markdown.exists() else None,
        "ip1n_trajectory": str(trajectory) if trajectory.exists() else None,
        "ip1n_numerical_pass": numerical_pass,
        "ip1n_numerical_checks": payload.get("numerical_checks") if payload else None,
        "physical_decision": physical,
        "compact_physical_summary": compact,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nValidation artifact: {output_dir}")
    print(f"Overall IP1n numerical status: {summary['overall_status'].upper()}")
    print(f"Recurrence preflight: {'PASS' if recurrence_pass else 'FAIL'}")
    if physical is not None:
        print(
            "Autonomous stored lattice memory: "
            + ("YES" if physical["autonomous_stored_lattice_memory"] else "NO")
        )

    if not overall:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
