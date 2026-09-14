#!/usr/bin/env python3
"""Run the IP1k natural-wake replication/background validation locally."""

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
    output_dir = Path("ip1k-local-validation") / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    gates: list[dict] = []

    gates.append(
        _run(
            "ip1k-pycompile",
            [
                python,
                "-m",
                "py_compile",
                "src/holstein_peierls/dynamics/event_background.py",
                "experiments/ip1k_natural_wake_background.py",
                "experiments/run_ip1k_natural_wake_background.py",
            ],
            output_dir,
        )
    )
    gates.append(
        _run(
            "ip1k-focused-pytest",
            [
                python,
                "-m",
                "pytest",
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

    recurrence_json = output_dir / "ip1k-phonon-recurrence.json"
    recurrence_md = output_dir / "ip1k-phonon-recurrence.md"
    gates.append(
        _run(
            "ip1k-phonon-recurrence-preflight",
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
                "800.0",
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

    artifact = output_dir / "ip1k-natural-wake-background.json"
    markdown = output_dir / "ip1k-natural-wake-background.md"
    profiles = output_dir / "ip1k-natural-wake-background-profiles.npz"
    if all(gate["status"] == "pass" for gate in gates):
        gates.append(
            _run(
                "ip1k-natural-wake-background-benchmark",
                [
                    python,
                    "experiments/run_ip1k_natural_wake_background.py",
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
    recurrence = json.loads(recurrence_json.read_text(encoding="utf-8")) if recurrence_json.exists() else None
    failed = [gate["name"] for gate in gates if gate["status"] != "pass"]
    recurrence_pass = bool(
        recurrence is not None and recurrence.get("planned_run_finishes_before_stationary_wrap", False)
    )
    numerical_pass = bool(payload is not None and payload.get("numerical_pass", False))
    overall = bool(not failed and recurrence_pass and numerical_pass and profiles.exists())

    condition_summary = None
    replication = None
    if payload is not None:
        replication = payload.get("replication_decision")
        condition_summary = []
        for condition in payload["conditions"]:
            event_summaries = []
            for analysis in condition["complete_event_analyses"]:
                back = analysis["real_metrics"]["backward_d1_to_d2_delay"]
                event_summaries.append(
                    {
                        "event_time_fs": analysis["event"]["transition_start_time_fs"],
                        "direction": analysis["event"]["direction"],
                        "backward_lag_fs": back["lag_fs"],
                        "backward_speed_sites_per_ps": back["speed_sites_per_ps"],
                        "backward_correlation": back["correlation"],
                        "backward_packet_qualified": analysis["backward_packet_qualified"],
                        "backward_positive_energy_percentile": analysis[
                            "backward_positive_energy_percentile"
                        ],
                        "backward_peak_flux_percentile": analysis[
                            "backward_peak_flux_percentile"
                        ],
                        "background_separated": analysis["background_separated"],
                    }
                )
            condition_summary.append(
                {
                    "anisotropy_ratio": condition["anisotropy_ratio"],
                    "tp1_displacement_x_A": condition["tp1_displacement_x_A"],
                    "persistent_x_event_count": condition["persistent_x_event_count"],
                    "complete_real_event_count": condition["complete_real_event_count"],
                    "pseudo_event_count": condition["pseudo_event_count"],
                    "background_separated_event_count": condition[
                        "background_separated_event_count"
                    ],
                    "events": event_summaries,
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
            "scope": "IP1k 40x40 T=0 10mV/A natural-hop wake replication against matched same-trajectory event-free background",
        },
        "overall_status": "pass" if overall else "fail",
        "failed_gates": failed,
        "recurrence_preflight_pass": recurrence_pass,
        "gates": gates,
        "ip1k_artifact": str(artifact) if artifact.exists() else None,
        "ip1k_markdown": str(markdown) if markdown.exists() else None,
        "ip1k_profiles": str(profiles) if profiles.exists() else None,
        "ip1k_numerical_pass": numerical_pass,
        "ip1k_numerical_checks": payload.get("numerical_checks") if payload else None,
        "replication_decision": replication,
        "condition_summary": condition_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nValidation artifact: {output_dir}")
    print(f"Overall IP1k numerical status: {summary['overall_status'].upper()}")
    print(f"Recurrence preflight: {'PASS' if recurrence_pass else 'FAIL'}")
    if replication is not None:
        print(
            "Replicated event-associated retrograde evidence: "
            + ("YES" if replication["replicated_event_associated_retrograde_evidence"] else "NO")
        )

    if not overall:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
