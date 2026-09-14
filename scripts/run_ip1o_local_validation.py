#!/usr/bin/env python3
"""Run the IP1o frozen-electronic-surface validation locally."""

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
    output_dir = Path("ip1o-local-validation") / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    gates: list[dict] = []

    gates.append(
        _run(
            "ip1o-pycompile",
            [
                python,
                "-m",
                "py_compile",
                "src/holstein_peierls/dynamics/frozen_electronic_surface.py",
                "experiments/ip1o_frozen_electronic_surface.py",
            ],
            output_dir,
        )
    )
    gates.append(
        _run(
            "ip1o-focused-pytest",
            [
                python,
                "-m",
                "pytest",
                "tests/test_ip1o_frozen_electronic_surface.py",
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

    recurrence_json = output_dir / "ip1o-phonon-recurrence.json"
    recurrence_md = output_dir / "ip1o-phonon-recurrence.md"
    gates.append(
        _run(
            "ip1o-phonon-recurrence-preflight",
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

    artifact = output_dir / "ip1o-frozen-electronic-surface.json"
    markdown = output_dir / "ip1o-frozen-electronic-surface.md"
    trajectory = output_dir / "ip1o-frozen-electronic-surface-trajectories.npz"
    if all(gate["status"] == "pass" for gate in gates):
        gates.append(
            _run(
                "ip1o-frozen-electronic-surface-benchmark",
                [
                    python,
                    "experiments/ip1o_frozen_electronic_surface.py",
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
    analysis = payload.get("analysis") if payload else None
    compact = None
    if analysis is not None:
        full = analysis["fully_coupled_released"]
        frozen = analysis["frozen_electronic_surface"]
        compact = {
            "event_start_time_fs": analysis["event"]["transition_start_time_fs"],
            "switch_time_fs": analysis["switch_time_fs"],
            "phase_continuity_error": analysis["phase_continuity_error"],
            "force_continuity_error": analysis["initial_force_continuity_max_abs"],
            "fully_coupled_post_switch_event_count": len(full["persistent_events_after_switch"]),
            "fully_coupled_late_passing_bins": full["sustained_pattern_gate"][
                "late_passing_bin_count"
            ],
            "frozen_late_passing_bins": frozen["sustained_pattern_gate"][
                "late_passing_bin_count"
            ],
            "fully_coupled_sustained_memory": full["sustained_pattern_gate"][
                "sustained_pattern_memory"
            ],
            "frozen_sustained_memory": frozen["sustained_pattern_gate"][
                "sustained_pattern_memory"
            ],
            "frozen_energy_drift_eV": frozen["maximum_energy_balance_residual_eV"],
            "frozen_electronic_state_bitwise_unchanged": frozen[
                "electronic_state_bitwise_unchanged"
            ],
            "late_frozen_to_full_amplitude_ratio": analysis[
                "late_frozen_to_full_amplitude_ratio"
            ],
            "late_trailing_vector_cosine_similarity_mean": analysis[
                "late_trailing_vector_cosine_similarity_mean"
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
            "scope": "IP1o 40x40 isotropic frozen-electronic-surface zero-power causal memory control",
        },
        "overall_status": "pass" if overall else "fail",
        "failed_gates": failed,
        "recurrence_preflight_pass": recurrence_pass,
        "gates": gates,
        "ip1o_artifact": str(artifact) if artifact.exists() else None,
        "ip1o_markdown": str(markdown) if markdown.exists() else None,
        "ip1o_trajectory": str(trajectory) if trajectory.exists() else None,
        "ip1o_numerical_pass": numerical_pass,
        "ip1o_numerical_checks": payload.get("numerical_checks") if payload else None,
        "physical_decision": physical,
        "compact_physical_summary": compact,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nValidation artifact: {output_dir}")
    print(f"Overall IP1o numerical status: {summary['overall_status'].upper()}")
    print(f"Recurrence preflight: {'PASS' if recurrence_pass else 'FAIL'}")
    if physical is not None:
        print(
            "Lattice-encoded memory without electronic redistribution: "
            + (
                "YES"
                if physical["lattice_encoded_memory_without_electronic_redistribution"]
                else "NO"
            )
        )

    if not overall:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
