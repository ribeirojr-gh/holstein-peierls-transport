#!/usr/bin/env python3
"""Run IP1e finite-size and intermolecular bath-memory screening locally."""

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


def _recurrence_command(
    python: str,
    *,
    size: int,
    gamma_v: float,
    final_time_fs: float,
    output: Path,
    markdown: Path,
) -> list[str]:
    return [
        python,
        "experiments/ip1p_phonon_recurrence_audit.py",
        "--nx",
        str(size),
        "--ny",
        str(size),
        "--lattice-spacing-A",
        "3.0",
        "--gamma-v-per-fs",
        str(gamma_v),
        "--event-half-window-fs",
        "500.0",
        "--planned-final-time-fs",
        str(final_time_fs),
        "--output",
        str(output),
        "--markdown",
        str(markdown),
    ]


def _dynamics_command(
    python: str,
    *,
    size: int,
    gamma_v: float,
    final_time_fs: float,
    burn_in_fs: float,
    output: Path,
    markdown: Path,
) -> list[str]:
    return [
        python,
        "experiments/ip1d_matched_counterfactual_precursors.py",
        "--size",
        str(size),
        "--temperatures-K",
        "300",
        "--isotropic-ratio",
        "1.0",
        "--reference-ratio",
        "0.15",
        "--reference-temperature-K",
        "300",
        "--gamma-u-per-fs",
        "0.01",
        "--gamma-v-per-fs",
        str(gamma_v),
        "--dt-fs",
        "0.2",
        "--final-time-fs",
        str(final_time_fs),
        "--burn-in-fs",
        str(burn_in_fs),
        "--sample-interval-fs",
        "2.0",
        "--decoherence-interval-fs",
        "180.0",
        "--lattice-spacing-A",
        "3.0",
        "--lattice-seeds",
        "20260905",
        "20260906",
        "--krylov-dimension",
        "6",
        "--output",
        str(output),
        "--markdown",
        str(markdown),
    ]


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path("ip1e-local-validation") / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    python = sys.executable

    gates: list[dict] = []
    gates.append(
        _run(
            "ip1e-pycompile",
            [
                python,
                "-m",
                "py_compile",
                "experiments/ip1e_compare_finite_size_bath_memory.py",
                "experiments/ip1d_matched_counterfactual_precursors.py",
                "experiments/ip1p_phonon_recurrence_audit.py",
            ],
            output_dir,
        )
    )
    gates.append(
        _run(
            "ip1e-focused-pytest",
            [
                python,
                "-m",
                "pytest",
                "tests/test_ip1p_phonon_recurrence.py",
                "tests/test_ip1d_matched_precursors.py",
                "tests/test_ip1c_event_conditioned_hopping.py",
                "tests/test_ip1b_dressed_hopping.py",
                "tests/test_ip1a_hopping_observables.py",
                "tests/test_tp1_transport_observables.py",
                "tests/test_d4_coupled_bath.py",
                "tests/test_d5b_instantaneous_decoherence.py",
                "tests/test_static_solver.py",
                "-q",
            ],
            output_dir,
        )
    )
    gates.append(_run("full-pytest", [python, "-m", "pytest"], output_dir))

    protocols = {
        "baseline20": {"size": 20, "gamma_v": 0.01, "final": 10000.0, "burn": 2000.0},
        "large40": {"size": 40, "gamma_v": 0.01, "final": 15000.0, "burn": 5000.0},
        "weak40": {"size": 40, "gamma_v": 0.002, "final": 15000.0, "burn": 5000.0},
    }

    recurrence_payloads: dict[str, dict] = {}
    dynamics_payloads: dict[str, dict] = {}
    artifact_paths: dict[str, Path] = {}

    for name, cfg in protocols.items():
        recurrence_json = output_dir / f"{name}-phonon-recurrence.json"
        recurrence_md = output_dir / f"{name}-phonon-recurrence.md"
        gates.append(
            _run(
                f"{name}-phonon-recurrence",
                _recurrence_command(
                    python,
                    size=int(cfg["size"]),
                    gamma_v=float(cfg["gamma_v"]),
                    final_time_fs=float(cfg["final"]),
                    output=recurrence_json,
                    markdown=recurrence_md,
                ),
                output_dir,
            )
        )
        if recurrence_json.exists():
            recurrence_payloads[name] = json.loads(recurrence_json.read_text(encoding="utf-8"))

    recurrence_guard_pass = bool(
        len(recurrence_payloads) == len(protocols)
        and all(
            item["planned_run_finishes_before_stationary_wrap"]
            and item["planned_complete_event_windows_finish_before_stationary_wrap"]
            for item in recurrence_payloads.values()
        )
    )

    if recurrence_guard_pass:
        for name, cfg in protocols.items():
            artifact = output_dir / f"{name}-ip1d.json"
            markdown = output_dir / f"{name}-ip1d.md"
            artifact_paths[name] = artifact
            gates.append(
                _run(
                    f"{name}-dynamics",
                    _dynamics_command(
                        python,
                        size=int(cfg["size"]),
                        gamma_v=float(cfg["gamma_v"]),
                        final_time_fs=float(cfg["final"]),
                        burn_in_fs=float(cfg["burn"]),
                        output=artifact,
                        markdown=markdown,
                    ),
                    output_dir,
                )
            )
            if artifact.exists():
                dynamics_payloads[name] = json.loads(artifact.read_text(encoding="utf-8"))

    comparison_json = output_dir / "ip1e-finite-size-bath-memory-comparison.json"
    comparison_md = output_dir / "ip1e-finite-size-bath-memory-comparison.md"
    if len(dynamics_payloads) == len(protocols) and all(
        bool(payload.get("numerical_pass", False)) for payload in dynamics_payloads.values()
    ):
        gates.append(
            _run(
                "ip1e-comparison",
                [
                    python,
                    "experiments/ip1e_compare_finite_size_bath_memory.py",
                    "--baseline20",
                    str(artifact_paths["baseline20"]),
                    "--large40",
                    str(artifact_paths["large40"]),
                    "--weak40",
                    str(artifact_paths["weak40"]),
                    "--temperature-K",
                    "300",
                    "--ratios",
                    "1.0",
                    "0.15",
                    "--output",
                    str(comparison_json),
                    "--markdown",
                    str(comparison_md),
                ],
                output_dir,
            )
        )

    failed = [gate["name"] for gate in gates if gate["status"] != "pass"]
    dynamics_numerical_pass = bool(
        len(dynamics_payloads) == len(protocols)
        and all(bool(payload.get("numerical_pass", False)) for payload in dynamics_payloads.values())
    )
    comparison_exists = comparison_json.exists()
    overall = bool(not failed and recurrence_guard_pass and dynamics_numerical_pass and comparison_exists)

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
        "scope": "IP1e finite-size and intermolecular bath-memory sensitivity screen; no mobility/diffusion/activation/rate claim",
    }
    summary = {
        "metadata": metadata,
        "overall_status": "pass" if overall else "fail",
        "failed_gates": failed,
        "recurrence_guard_pass": recurrence_guard_pass,
        "dynamics_numerical_pass": dynamics_numerical_pass,
        "protocols": protocols,
        "recurrence": recurrence_payloads,
        "gates": gates,
        "comparison_artifact": str(comparison_json) if comparison_exists else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nValidation artifact: {output_dir}")
    print(f"Overall runner status: {summary['overall_status'].upper()}")
    print(f"Recurrence guard: {'PASS' if recurrence_guard_pass else 'FAIL'}")
    print(f"Underlying dynamics: {'PASS' if dynamics_numerical_pass else 'FAIL'}")

    if not overall:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
