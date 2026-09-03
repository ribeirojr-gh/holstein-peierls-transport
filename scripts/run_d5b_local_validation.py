"""Run local D5b IDC validation without consuming GitHub Actions minutes.

Runner PASS means only that the implemented numerical gates executed correctly.
The scientific selection among IDC-DP, IDC-BM and IDC-MA is made afterward from
the generated ensemble JSON; it is not hard-coded into this runner.
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
    output_dir = root / "d5b-local-validation" / stamp
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
            key: os.environ.get(key) for key in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
        "physical_interpretation_gate": (
            "none: runner PASS confirms execution/regressions only; scheme selection "
            "requires post-run analysis of d5b-instantaneous-decoherence.json"
        ),
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
                "-q",
            ],
            output_dir,
        )
    )
    gates.append(_run("full-pytest", [python, "-m", "pytest"], output_dir))

    artifact = output_dir / "d5b-instantaneous-decoherence.json"
    markdown = output_dir / "d5b-instantaneous-decoherence.md"
    gates.append(
        _run(
            "d5b-20x20-ensemble",
            [
                python,
                "experiments/d5b_instantaneous_decoherence.py",
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
                "100.0",
                "--lattice-seeds",
                "20260903",
                "20260904",
                "20260905",
                "20260906",
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

    failed = [gate["name"] for gate in gates if gate["status"] != "pass"]
    summary = {
        "metadata": metadata,
        "overall_status": "pass" if not failed else "fail",
        "failed_gates": failed,
        "gates": gates,
        "d5b_artifact": str(artifact.relative_to(root)) if artifact.exists() else None,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n=== D5B LOCAL VALIDATION SUMMARY ===")
    print(f"Output directory: {output_dir}")
    print(f"Overall execution/regression status: {summary['overall_status'].upper()}")
    if failed:
        print("Failed gates: " + ", ".join(failed))
        raise SystemExit(1)
    print("All requested numerical gates passed.")
    print("Physical IDC scheme selection requires analysis of the generated JSON.")


if __name__ == "__main__":
    main()
