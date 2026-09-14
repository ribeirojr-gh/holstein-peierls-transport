#!/usr/bin/env python3
"""Run the IP1t post-recrossing direction-control validation locally."""

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
    output_dir = Path("ip1t-local-validation") / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    gates: list[dict] = []

    gates.append(
        _run(
            "ip1t-pycompile",
            [
                python,
                "-m",
                "py_compile",
                "src/holstein_peierls/dynamics/recrossing_control.py",
                "experiments/ip1t_post_recrossing_direction_control.py",
            ],
            output_dir,
        )
    )
    gates.append(
        _run(
            "ip1t-focused-pytest",
            [
                python,
                "-m",
                "pytest",
                "tests/test_ip1t_recrossing_control.py",
                "tests/test_ip1s_direction_reversal.py",
                "tests/test_ip1r_traveling_current_attribution.py",
                "tests/test_ip1q_mode_memory.py",
                "tests/test_ip1o_frozen_electronic_surface.py",
                "tests/test_ip1n_field_release.py",
                "tests/test_d3_field_driven.py",
                "tests/test_numerical_validation.py",
                "tests/test_static_solver.py",
                "-q",
            ],
            output_dir,
        )
    )
    gates.append(_run("full-pytest", [python, "-m", "pytest"], output_dir))

    recurrence_json = output_dir / "ip1t-phonon-recurrence.json"
    recurrence_md = output_dir / "ip1t-phonon-recurrence.md"
    gates.append(
        _run(
            "ip1t-phonon-recurrence-preflight",
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
                "2500.0",
                "--planned-final-time-fs",
                "6000.0",
                "--output",
                str(recurrence_json),
                "--markdown",
                str(recurrence_md),
            ],
            output_dir,
        )
    )

    artifact = output_dir / "ip1t-post-recrossing-direction-control.json"
    markdown = output_dir / "ip1t-post-recrossing-direction-control.md"
    trajectory = output_dir / "ip1t-post-recrossing-direction-control-trajectories.npz"
    if all(gate["status"] == "pass" for gate in gates):
        gates.append(
            _run(
                "ip1t-post-recrossing-direction-control-benchmark",
                [
                    python,
                    "experiments/ip1t_post_recrossing_direction_control.py",
                    "--size",
                    "40",
                    "--field-mv-per-A",
                    "10.0",
                    "--dt-fs",
                    "0.2",
                    "--search-time-fs",
                    "4000.0",
                    "--common-search-time-fs",
                    "1500.0",
                    "--continuation-fs",
                    "2000.0",
                    "--sample-interval-fs",
                    "2.0",
                    "--current-sample-interval-fs",
                    "10.0",
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

    compact = None
    if payload is not None and payload.get("analysis") is not None:
        analysis = payload["analysis"]
        decision = payload["physical_decision"]
        divergence = analysis["population_divergence"]
        compact = {
            "first_natural_event": analysis["first_natural_event"],
            "common_recrossing_event": analysis["common_recrossing_event"],
            "native_first_x_event": analysis["native"]["first_x_event_within_commitment_window"],
            "reversed_first_x_event": analysis["reversed"]["first_x_event_within_commitment_window"],
            "return_commitment_changed": decision["return_commitment_changed"],
            "electronic_state_diverged": decision["electronic_state_diverged"],
            "directional_memory_controls_recrossing_commitment": decision[
                "directional_memory_controls_recrossing_commitment"
            ],
            "maximum_l1_first_2ps": divergence["maximum_l1_first_2ps"],
            "first_l1_ge_0p25_offset_fs": divergence["first_l1_ge_0p25_offset_fs"],
            "native_max_energy_drift_eV": analysis["native"]["maximum_energy_drift_eV"],
            "reversed_max_energy_drift_eV": analysis["reversed"]["maximum_energy_drift_eV"],
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
            "scope": "IP1t 40x40 post-recrossing energy-preserving Peierls direction control",
        },
        "overall_status": "pass" if overall else "fail",
        "failed_gates": failed,
        "recurrence_preflight_pass": recurrence_pass,
        "gates": gates,
        "ip1t_artifact": str(artifact) if artifact.exists() else None,
        "ip1t_markdown": str(markdown) if markdown.exists() else None,
        "ip1t_trajectory": str(trajectory) if trajectory.exists() else None,
        "ip1t_numerical_pass": numerical_pass,
        "ip1t_numerical_checks": payload.get("numerical_checks") if payload else None,
        "compact_physical_summary": compact,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nValidation artifact: {output_dir}")
    print(f"Overall IP1t numerical status: {summary['overall_status'].upper()}")
    print(f"Recurrence preflight: {'PASS' if recurrence_pass else 'FAIL'}")
    if compact is not None:
        print(
            "Primary causal classification: "
            f"{'PASS' if compact['directional_memory_controls_recrossing_commitment'] else 'FAIL'}"
        )
        print(f"Max L1 first 2 ps: {compact['maximum_l1_first_2ps']:.6f}")

    if not overall:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
