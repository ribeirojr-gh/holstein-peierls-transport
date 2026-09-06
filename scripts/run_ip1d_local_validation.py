#!/usr/bin/env python3
"""Run IP1d matched-counterfactual precursor validation locally.

The runner avoids hosted GitHub Actions and records a self-contained artifact.
Physical precursor strengths are never promoted to numerical PASS gates.
"""

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
    output_dir = Path("ip1d-local-validation") / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    python = sys.executable

    gates: list[dict] = []
    gates.append(
        _run(
            "ip1d-pycompile",
            [
                python,
                "-m",
                "py_compile",
                "src/holstein_peierls/dynamics/matched_precursors.py",
                "experiments/ip1d_matched_counterfactual_precursors.py",
            ],
            output_dir,
        )
    )
    gates.append(
        _run(
            "ip1d-focused-pytest",
            [
                python,
                "-m",
                "pytest",
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

    artifact = output_dir / "ip1d-matched-counterfactual-precursors.json"
    markdown = output_dir / "ip1d-matched-counterfactual-precursors.md"
    gates.append(
        _run(
            "ip1d-matched-counterfactual-precursor-benchmark",
            [
                python,
                "experiments/ip1d_matched_counterfactual_precursors.py",
                "--size",
                "20",
                "--temperatures-K",
                "100",
                "300",
                "500",
                "--isotropic-ratio",
                "1.0",
                "--reference-ratio",
                "0.15",
                "--reference-temperature-K",
                "300",
                "--dt-fs",
                "0.2",
                "--final-time-fs",
                "20000",
                "--burn-in-fs",
                "2000",
                "--sample-interval-fs",
                "2.0",
                "--decoherence-interval-fs",
                "180.0",
                "--lattice-spacing-A",
                "3.0",
                "--lattice-seeds",
                "20260905",
                "20260906",
                "20260907",
                "20260908",
                "--krylov-dimension",
                "6",
                "--output",
                str(artifact),
                "--markdown",
                str(markdown),
            ],
            output_dir,
        )
    )

    payload = json.loads(artifact.read_text(encoding="utf-8")) if artifact.exists() else None
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
        "scope": "IP1d zero-field 20x20 matched counterfactual precursor validation; same dynamics as IP1c; no mobility/diffusion/activation claim",
    }
    failed = [gate["name"] for gate in gates if gate["status"] != "pass"]
    summary = {
        "metadata": metadata,
        "overall_status": "pass" if not failed else "fail",
        "failed_gates": failed,
        "gates": gates,
        "ip1d_artifact": str(artifact) if artifact.exists() else None,
        "ip1d_numerical_pass": bool(payload["numerical_pass"]) if payload is not None else False,
        "ip1d_numerical_checks": payload["numerical_checks"] if payload is not None else None,
        "condition_summary": (
            [
                {
                    "anisotropy_ratio": condition["anisotropy_ratio"],
                    "temperature_K": condition["temperature_K"],
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
    if payload is not None:
        print(f"IP1d numerical status: {'PASS' if payload['numerical_pass'] else 'FAIL'}")

    if failed or payload is None or not payload["numerical_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
