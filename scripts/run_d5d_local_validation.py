"""Run the final D5 closure locally without consuming GitHub Actions minutes.

This runner restores the full project regression gate after the D5b/D5c focused
sensitivity work: D5 tests, full pytest, D0a, six strict S0 relaxations, and the
10 ps four-seed IDC-BM closure benchmark at the explicit numerical-control
interval ``t_d = 180 fs``.
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
    print(f"--- {name}: {status.upper()} ({elapsed:.3f} s) ---", flush=True)
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
        result = subprocess.run(
            ["git", *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = root / "d5d-local-validation" / stamp
    output_dir.mkdir(parents=True, exist_ok=False)

    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_value(["rev-parse", "HEAD"]),
        "git_branch": _git_value(["branch", "--show-current"]),
        "git_status": _git_value(["status", "--short"]),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.machine(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "d5_reference_scheme": "IDC-BM",
        "decoherence_interval_fs": 180.0,
        "decoherence_interval_is_material_calibrated": False,
    }

    python = sys.executable
    gates: list[dict] = []

    gates.append(
        _run(
            "d5-focused-pytest",
            [
                python,
                "-m",
                "pytest",
                "tests/test_d5_electronic_thermalization.py",
                "tests/test_d5b_instantaneous_decoherence.py",
                "tests/test_d5c_decoherence_interval_sweep.py",
                "-q",
            ],
            output_dir,
        )
    )
    gates.append(_run("full-pytest", [python, "-m", "pytest"], output_dir))

    d0a_artifact = output_dir / "d0a-frozen.json"
    gates.append(
        _run(
            "d0a-frozen",
            [
                python,
                "experiments/d0a_frozen_propagators.py",
                "--nx",
                "4",
                "--ny",
                "4",
                "--dt-fs",
                "0.1",
                "--steps",
                "20",
                "--krylov-dimension",
                "8",
                "--output",
                str(d0a_artifact),
            ],
            output_dir,
        )
    )

    for multiplicity in ("singlet", "triplet"):
        for seed in ("onsite", "bond_x", "bond_y"):
            artifact = output_dir / f"{multiplicity}-{seed}.json"
            gates.append(
                _run(
                    f"s0-{multiplicity}-{seed}",
                    [
                        python,
                        "experiments/spin_adapted_isotropic_relaxation.py",
                        "--size",
                        "4",
                        "--staggered-gap",
                        "2.0",
                        "--multiplicity",
                        multiplicity,
                        "--seed",
                        seed,
                        "--require-converged",
                        "--output",
                        str(artifact),
                    ],
                    output_dir,
                )
            )

    closure_artifact = output_dir / "d5d-bm-closure.json"
    closure_markdown = output_dir / "d5d-bm-closure.md"
    gates.append(
        _run(
            "d5d-bm-20x20-10ps",
            [
                python,
                "experiments/d5d_bm_closure.py",
                "--size",
                "20",
                "--temperature-K",
                "300.0",
                "--gamma-u-per-fs",
                "0.01",
                "--gamma-v-per-fs",
                "0.01",
                "--dt-fs",
                "0.2",
                "--final-time-fs",
                "10000.0",
                "--burn-in-fs",
                "2000.0",
                "--decoherence-interval-fs",
                "180.0",
                "--lattice-seeds",
                "20260903",
                "20260904",
                "20260905",
                "20260906",
                "--krylov-dimension",
                "6",
                "--output",
                str(closure_artifact),
                "--markdown",
                str(closure_markdown),
            ],
            output_dir,
        )
    )

    failed = [gate["name"] for gate in gates if gate["status"] != "pass"]
    closure_payload = None
    if closure_artifact.exists():
        closure_payload = json.loads(closure_artifact.read_text(encoding="utf-8"))
        if not bool(closure_payload.get("closure_pass", False)):
            failed.append("d5d-physical-closure")

    summary = {
        "metadata": metadata,
        "overall_status": "pass" if not failed else "fail",
        "failed_gates": failed,
        "gates": gates,
        "d0a_artifact": str(d0a_artifact.relative_to(root)) if d0a_artifact.exists() else None,
        "d5d_artifact": str(closure_artifact.relative_to(root)) if closure_artifact.exists() else None,
        "d5d_closure_pass": (
            bool(closure_payload.get("closure_pass", False))
            if closure_payload is not None
            else False
        ),
        "d5d_closure_checks": (
            closure_payload.get("closure_checks") if closure_payload is not None else None
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n=== D5D FINAL LOCAL VALIDATION SUMMARY ===")
    print(f"Output directory: {output_dir}")
    print(f"Overall status: {summary['overall_status'].upper()}")
    if failed:
        print("Failed gates: " + ", ".join(failed))
        raise SystemExit(1)
    print("All numerical, regression, and pre-registered D5 physical closure gates passed.")


if __name__ == "__main__":
    main()
