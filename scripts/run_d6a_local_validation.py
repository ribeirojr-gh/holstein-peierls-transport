"""Run local D6a frozen pair-dynamics validation without GitHub Actions."""

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
    output_dir = root / "d6a-local-validation" / stamp
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
        "scope": "D6a frozen-H pair propagation only; no moving lattice, bath, field, or decoherence",
    }

    python = sys.executable
    gates: list[dict] = []
    gates.append(
        _run(
            "d6a-focused-pytest",
            [python, "-m", "pytest", "tests/test_d6_pair_frozen.py", "-q"],
            output_dir,
        )
    )
    gates.append(_run("full-pytest", [python, "-m", "pytest"], output_dir))

    artifact = output_dir / "d6a-pair-frozen.json"
    markdown = output_dir / "d6a-pair-frozen.md"
    gates.append(
        _run(
            "d6a-pair-frozen-benchmark",
            [
                python,
                "experiments/d6a_pair_frozen_propagation.py",
                "--size",
                "3",
                "--dt-fs",
                "0.2",
                "--steps",
                "40",
                "--krylov-dimensions",
                "6",
                "8",
                "12",
                "16",
                "--seed",
                "20260905",
                "--output",
                str(artifact),
                "--markdown",
                str(markdown),
            ],
            output_dir,
        )
    )

    failed = [gate["name"] for gate in gates if gate["status"] != "pass"]
    benchmark_closure = False
    benchmark_checks = None
    if artifact.exists():
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        benchmark_closure = bool(payload.get("closure_pass", False))
        benchmark_checks = payload.get("closure_checks")
        if not benchmark_closure and "d6a-pair-frozen-benchmark" not in failed:
            failed.append("d6a-pair-frozen-physical-closure")

    summary = {
        "metadata": metadata,
        "overall_status": "pass" if not failed else "fail",
        "failed_gates": failed,
        "gates": gates,
        "d6a_artifact": str(artifact.relative_to(root)) if artifact.exists() else None,
        "d6a_closure_pass": benchmark_closure,
        "d6a_closure_checks": benchmark_checks,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n=== D6A LOCAL VALIDATION SUMMARY ===")
    print(f"Output directory: {output_dir}")
    print(f"Overall status: {summary['overall_status'].upper()}")
    if failed:
        print("Failed gates: " + ", ".join(failed))
        raise SystemExit(1)
    print("All D6a numerical, regression, and frozen-pair closure gates passed.")


if __name__ == "__main__":
    main()
