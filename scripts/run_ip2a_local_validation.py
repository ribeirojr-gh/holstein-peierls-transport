#!/usr/bin/env python3
"""Run the IP2a ensemble-construction validation locally."""

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
    output_dir = Path("ip2a-local-validation") / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    gates: list[dict] = []

    gates.append(
        _run(
            "ip2a-pycompile",
            [
                python,
                "-m",
                "py_compile",
                "src/holstein_peierls/dynamics/ensemble_preparation.py",
                "experiments/ip2a_ensemble_construction.py",
            ],
            output_dir,
        )
    )
    gates.append(
        _run(
            "ip2a-focused-pytest",
            [
                python,
                "-m",
                "pytest",
                "tests/test_ip2a_ensemble_preparation.py",
                "tests/test_ip1q_mode_memory.py",
                "tests/test_numerical_validation.py",
                "tests/test_static_solver.py",
                "-q",
            ],
            output_dir,
        )
    )
    gates.append(_run("full-pytest", [python, "-m", "pytest"], output_dir))

    artifact = output_dir / "ip2a-ensemble-construction.json"
    markdown = output_dir / "ip2a-ensemble-construction.md"
    arrays = output_dir / "ip2a-ensemble-construction.npz"
    if all(gate["status"] == "pass" for gate in gates):
        gates.append(
            _run(
                "ip2a-ensemble-construction",
                [
                    python,
                    "experiments/ip2a_ensemble_construction.py",
                    "--size",
                    "40",
                    "--ensemble-size",
                    "32",
                    "--candidate-energies-eV",
                    "1e-5",
                    "3e-5",
                    "1e-4",
                    "--vx-energy-fraction",
                    "0.5",
                    "--max-mode-index",
                    "4",
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
    failed = [gate["name"] for gate in gates if gate["status"] != "pass"]
    numerical_pass = bool(payload is not None and payload.get("numerical_pass", False))
    overall = bool(not failed and numerical_pass and arrays.exists())

    compact = None
    if payload is not None:
        compact = {
            "candidate_energies_eV": payload["candidate_energies_eV"],
            "ensemble_size": payload["ensemble_size"],
            "maximum_energy_error_eV": payload["maximum_energy_error_eV"],
            "maximum_matter_increment_error_eV": payload[
                "maximum_matter_increment_error_eV"
            ],
            "maximum_zero_mode_A_per_fs": payload["maximum_zero_mode_A_per_fs"],
            "maximum_opposite_pair_error_A_per_fs": payload[
                "maximum_opposite_pair_error_A_per_fs"
            ],
            "minimum_distinct_base_pair_l2_distance_A_per_fs": payload[
                "minimum_distinct_base_pair_l2_distance_A_per_fs"
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
            "scope": "IP2a deterministic fixed-energy low-q Peierls ensemble construction",
        },
        "overall_status": "pass" if overall else "fail",
        "failed_gates": failed,
        "gates": gates,
        "ip2a_artifact": str(artifact) if artifact.exists() else None,
        "ip2a_markdown": str(markdown) if markdown.exists() else None,
        "ip2a_arrays": str(arrays) if arrays.exists() else None,
        "ip2a_numerical_pass": numerical_pass,
        "ip2a_numerical_checks": payload.get("numerical_checks") if payload else None,
        "compact_summary": compact,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nValidation artifact: {output_dir}")
    print(f"Overall IP2a numerical status: {summary['overall_status'].upper()}")
    if compact is not None:
        print(f"Candidate energies: {compact['candidate_energies_eV']}")
        print(f"Max matter increment error: {compact['maximum_matter_increment_error_eV']:.3e} eV")

    if not overall:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
