#!/usr/bin/env python3
"""Run IP1b dressed-polaron hopping validation locally.

This runner intentionally avoids hosted GitHub Actions.  It performs syntax
checks, focused regressions, the complete pytest suite, and then the 20x20
zero-field dressed-polaron benchmark.  A numerical PASS never requires a
nonzero dressed-hop count.
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
        env=os.environ.copy(),
    )
    elapsed = perf_counter() - start
    log = output_dir / f"{name}.log"
    log.write_text(completed.stdout, encoding="utf-8")
    print(completed.stdout, end="")
    print(f"[{name}] return={completed.returncode} elapsed={elapsed:.3f}s", flush=True)
    return {
        "name": name,
        "command": command,
        "return_code": int(completed.returncode),
        "status": "pass" if completed.returncode == 0 else "fail",
        "elapsed_s": float(elapsed),
        "log": str(log),
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
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path("ip1b-local-validation") / stamp
    output_dir.mkdir(parents=True, exist_ok=False)
    artifact = output_dir / "ip1b-dressed-polaron-hopping.json"
    markdown = output_dir / "ip1b-dressed-polaron-hopping.md"

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_value(["rev-parse", "HEAD"]),
        "git_branch": _git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_status": _git_value(["status", "--short"]),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.machine(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "thread_environment": {
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "scope": (
            "IP1b zero-field 20x20 dressed-polaron hopping validation with "
            "independent electronic/lattice centers, persistence sensitivity, "
            "lag matching, and future-bond rank diagnostics; no mobility, "
            "diffusion, activation-energy, or nonzero-hop requirement"
        ),
    }

    gates: list[dict] = []
    python = sys.executable
    gates.append(
        _run(
            "ip1b-pycompile",
            [
                python,
                "-m",
                "py_compile",
                "src/holstein_peierls/dynamics/dressed_hopping.py",
                "experiments/ip1b_dressed_polaron_hopping.py",
            ],
            output_dir,
        )
    )
    if gates[-1]["return_code"] == 0:
        gates.append(
            _run(
                "ip1b-focused-pytest",
                [
                    python,
                    "-m",
                    "pytest",
                    "tests/test_ip1b_dressed_hopping.py",
                    "tests/test_ip1a_hopping_observables.py",
                    "tests/test_d4_coupled_bath.py",
                    "tests/test_d5b_instantaneous_decoherence.py",
                    "tests/test_static_solver.py",
                    "-q",
                ],
                output_dir,
            )
        )
    if all(gate["return_code"] == 0 for gate in gates):
        gates.append(
            _run(
                "full-pytest",
                [python, "-m", "pytest"],
                output_dir,
            )
        )
    if all(gate["return_code"] == 0 for gate in gates):
        gates.append(
            _run(
                "ip1b-dressed-polaron-hopping-benchmark",
                [
                    python,
                    "experiments/ip1b_dressed_polaron_hopping.py",
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
                    "--prehop-window-fs",
                    "100.0",
                    "--decoherence-interval-fs",
                    "180.0",
                    "--minimum-template-amplitude",
                    "0.05",
                    "--minimum-template-relative-gap",
                    "0.01",
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

    payload = None
    if artifact.exists():
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    failed = [gate["name"] for gate in gates if gate["return_code"] != 0]
    numerical_pass = bool(payload is not None and payload.get("numerical_pass", False))
    if payload is not None and not numerical_pass:
        failed.append("ip1b-numerical-checks")
    overall = "pass" if not failed and numerical_pass else "fail"
    summary = {
        "metadata": metadata,
        "overall_status": overall,
        "failed_gates": failed,
        "gates": gates,
        "ip1b_artifact": str(artifact) if artifact.exists() else None,
        "ip1b_numerical_pass": numerical_pass,
        "ip1b_numerical_checks": payload.get("numerical_checks") if payload else None,
        "condition_summary": (
            [
                {
                    "anisotropy_ratio": condition["anisotropy_ratio"],
                    "temperature_K": condition["temperature_K"],
                    "aggregate": condition["aggregate"],
                }
                for condition in payload["conditions"]
            ]
            if payload
            else None
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== IP1B LOCAL VALIDATION SUMMARY ===")
    print(f"Artifact directory: {output_dir}")
    print(f"Overall status: {overall.upper()}")
    if failed:
        print("Failed gates: " + ", ".join(failed))
    else:
        print("All IP1b numerical and regression gates passed.")
        print(
            "Inspect persistence sensitivity, lattice/electronic event matching, "
            "lag distributions, and future-bond predictability before making any "
            "physical hopping claim."
        )
    raise SystemExit(0 if overall == "pass" else 2)


if __name__ == "__main__":
    main()
