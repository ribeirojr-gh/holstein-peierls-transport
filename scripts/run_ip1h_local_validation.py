#!/usr/bin/env python3
"""Run the no-dynamics IP1h fixed-boundary flux reanalysis locally."""

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


def _git(command: list[str]) -> str:
    result = subprocess.run(
        ["git", *command], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _discover_ip1g(root: Path) -> Path:
    required = {
        "ip1g-single-relocation-wake.json",
        "ip1g-single-relocation-wake-profiles.npz",
    }
    candidates = sorted(path for path in root.glob("*") if path.is_dir())
    for path in reversed(candidates):
        if all((path / name).exists() for name in required):
            return path
    raise FileNotFoundError("no complete IP1g validation directory found")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path, nargs="?", default=None)
    args = parser.parse_args()

    source = args.artifact_dir.resolve() if args.artifact_dir is not None else _discover_ip1g(Path("ip1g-local-validation"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path("ip1h-local-validation") / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    gates: list[dict] = []

    gates.append(
        _run(
            "ip1h-pycompile",
            [
                python,
                "-m",
                "py_compile",
                "src/holstein_peierls/dynamics/wavepacket_flux.py",
                "experiments/ip1h_reanalyze_boundary_flux.py",
            ],
            output_dir,
        )
    )
    gates.append(
        _run(
            "ip1h-focused-pytest",
            [python, "-m", "pytest", "tests/test_ip1h_wavepacket_flux.py", "-q"],
            output_dir,
        )
    )
    gates.append(_run("full-pytest", [python, "-m", "pytest"], output_dir))

    result_json = output_dir / "ip1h-boundary-flux.json"
    result_md = output_dir / "ip1h-boundary-flux.md"
    if all(gate["status"] == "pass" for gate in gates):
        gates.append(
            _run(
                "ip1h-boundary-flux-reanalysis",
                [
                    python,
                    "experiments/ip1h_reanalyze_boundary_flux.py",
                    str(source),
                    "--output",
                    str(result_json),
                    "--markdown",
                    str(result_md),
                ],
                output_dir,
            )
        )

    payload = json.loads(result_json.read_text(encoding="utf-8")) if result_json.exists() else None
    failed = [gate["name"] for gate in gates if gate["status"] != "pass"]
    overall = bool(not failed and payload is not None and payload.get("numerical_pass", False))
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
            "scope": "IP1h no-dynamics fixed-boundary lattice-radiation flux reanalysis",
            "source_ip1g_artifact_dir": str(source),
        },
        "overall_status": "pass" if overall else "fail",
        "failed_gates": failed,
        "gates": gates,
        "ip1h_artifact": str(result_json) if result_json.exists() else None,
        "ip1h_numerical_pass": bool(payload.get("numerical_pass", False)) if payload else False,
        "ip1h_numerical_checks": payload.get("numerical_checks") if payload else None,
        "records": payload.get("records") if payload else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nValidation artifact: {output_dir}")
    print(f"Overall IP1h status: {summary['overall_status'].upper()}")
    if not overall:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
