#!/usr/bin/env python3
"""Validate the IP1e size-aware posthoc recheck without rerunning dynamics."""

from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    args = parser.parse_args()

    artifact_dir = args.artifact_dir.resolve()
    required = [artifact_dir / f"{name}-ip1d.json" for name in ("baseline20", "large40", "weak40")]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing IP1e artifacts: " + ", ".join(missing))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path("ip1e-recheck-local-validation") / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    python = sys.executable

    gates: list[dict] = []
    gates.append(
        _run(
            "ip1e-recheck-pycompile",
            [
                python,
                "-m",
                "py_compile",
                "src/holstein_peierls/dynamics/numerical_validation.py",
                "experiments/ip1e_recheck_finite_size_energy_balance.py",
            ],
            output_dir,
        )
    )
    gates.append(
        _run(
            "ip1e-recheck-focused-pytest",
            [python, "-m", "pytest", "tests/test_numerical_validation.py", "-q"],
            output_dir,
        )
    )
    gates.append(_run("full-pytest", [python, "-m", "pytest"], output_dir))

    audited_json = output_dir / "ip1e-size-aware-recheck.json"
    audited_md = output_dir / "ip1e-size-aware-recheck.md"
    gates.append(
        _run(
            "ip1e-size-aware-recheck",
            [
                python,
                "experiments/ip1e_recheck_finite_size_energy_balance.py",
                str(artifact_dir),
                "--output",
                str(audited_json),
                "--markdown",
                str(audited_md),
            ],
            output_dir,
        )
    )

    payload = json.loads(audited_json.read_text(encoding="utf-8")) if audited_json.exists() else None
    failed = [gate["name"] for gate in gates if gate["status"] != "pass"]
    overall = bool(not failed and payload is not None and payload["rechecked_numerical_pass"])
    summary = {
        "metadata": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
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
            "source_artifact_dir": str(artifact_dir),
            "scope": "IP1e no-dynamics size-aware recheck of completed finite-size artifacts",
        },
        "overall_status": "pass" if overall else "fail",
        "failed_gates": failed,
        "gates": gates,
        "audited_artifact": str(audited_json) if audited_json.exists() else None,
        "rechecked_numerical_pass": bool(payload["rechecked_numerical_pass"]) if payload else False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nValidation artifact: {output_dir}")
    print(f"Overall recheck status: {summary['overall_status'].upper()}")
    if not overall:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
