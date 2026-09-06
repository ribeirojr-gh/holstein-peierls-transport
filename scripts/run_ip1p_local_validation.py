#!/usr/bin/env python3
"""Run the lightweight IP1p phonon-recurrence audit locally."""

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
    return {
        "name": name,
        "command": command,
        "return_code": int(completed.returncode),
        "status": "pass" if completed.returncode == 0 else "fail",
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
    output_dir = Path("ip1p-local-validation") / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    python = sys.executable

    gates = [
        _run(
            "ip1p-pycompile",
            [
                python,
                "-m",
                "py_compile",
                "src/holstein_peierls/dynamics/phonon_recurrence.py",
                "experiments/ip1p_phonon_recurrence_audit.py",
            ],
            output_dir,
        ),
        _run(
            "ip1p-focused-pytest",
            [python, "-m", "pytest", "tests/test_ip1p_phonon_recurrence.py", "-q"],
            output_dir,
        ),
        _run("full-pytest", [python, "-m", "pytest"], output_dir),
    ]

    artifact = output_dir / "ip1p-phonon-recurrence-audit.json"
    markdown = output_dir / "ip1p-phonon-recurrence-audit.md"
    gates.append(
        _run(
            "ip1p-phonon-recurrence-audit",
            [
                python,
                "experiments/ip1p_phonon_recurrence_audit.py",
                "--nx",
                "20",
                "--ny",
                "20",
                "--lattice-spacing-A",
                "3.0",
                "--gamma-v-per-fs",
                "0.01",
                "--event-half-window-fs",
                "500",
                "--planned-final-time-fs",
                "10000",
                "--output",
                str(artifact),
                "--markdown",
                str(markdown),
            ],
            output_dir,
        )
    )

    payload = json.loads(artifact.read_text(encoding="utf-8")) if artifact.exists() else None
    failed = [gate["name"] for gate in gates if gate["status"] != "pass"]
    summary = {
        "metadata": {
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
            "scope": "IP1p harmonic PBC phonon-recurrence and Langevin-damping audit before IP1d",
        },
        "overall_status": "pass" if not failed else "fail",
        "failed_gates": failed,
        "gates": gates,
        "audit": payload,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nValidation artifact: {output_dir}")
    print(f"Overall runner status: {summary['overall_status'].upper()}")
    if failed or payload is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
