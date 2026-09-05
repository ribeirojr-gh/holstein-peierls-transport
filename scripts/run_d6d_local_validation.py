"""Run local D6d pair electronic-thermalization diagnostics without Actions."""

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
    output_dir = root / "d6d-local-validation" / stamp
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
        "scope": (
            "D6d diagnostic-only pair electronic thermalization under validated D6c BAOAB; "
            "no pair decoherence, field, mobility, or transport fit"
        ),
    }

    python = sys.executable
    gates: list[dict] = []
    gates.append(
        _run(
            "d6-focused-pytest",
            [
                python,
                "-m",
                "pytest",
                "tests/test_d6_pair_frozen.py",
                "tests/test_d6_pair_coupled.py",
                "tests/test_d6_pair_thermal.py",
                "tests/test_d6_pair_thermalization.py",
                "-q",
            ],
            output_dir,
        )
    )
    gates.append(_run("full-pytest", [python, "-m", "pytest"], output_dir))

    artifact = output_dir / "d6d-pair-electronic-thermalization.json"
    markdown = output_dir / "d6d-pair-electronic-thermalization.md"
    gates.append(
        _run(
            "d6d-pair-electronic-thermalization-benchmark",
            [
                python,
                "experiments/d6d_pair_electronic_thermalization.py",
                "--size",
                "4",
                "--temperature-K",
                "300",
                "--gamma-u-per-fs",
                "0.01",
                "--gamma-v-per-fs",
                "0.01",
                "--dt-fs",
                "0.2",
                "--final-time-fs",
                "4000",
                "--burn-in-fs",
                "1000",
                "--diagnostic-interval-fs",
                "50",
                "--seeds",
                "20260905",
                "20260906",
                "20260907",
                "20260908",
                "--krylov-dimension",
                "8",
                "--output",
                str(artifact),
                "--markdown",
                str(markdown),
            ],
            output_dir,
        )
    )

    failed = [gate["name"] for gate in gates if gate["status"] != "pass"]
    numerical_pass = False
    numerical_checks = None
    if artifact.exists():
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        numerical_pass = bool(payload.get("numerical_pass", False))
        numerical_checks = payload.get("numerical_checks")
        if not numerical_pass and "d6d-pair-electronic-thermalization-benchmark" not in failed:
            failed.append("d6d-numerical-controls")

    summary = {
        "metadata": metadata,
        "overall_status": "pass" if not failed else "fail",
        "failed_gates": failed,
        "gates": gates,
        "d6d_artifact": str(artifact.relative_to(root)) if artifact.exists() else None,
        "d6d_numerical_pass": numerical_pass,
        "d6d_numerical_checks": numerical_checks,
        "physical_selection_is_manual": True,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n=== D6D LOCAL VALIDATION SUMMARY ===")
    print(f"Output directory: {output_dir}")
    print(f"Overall status: {summary['overall_status'].upper()}")
    if failed:
        print("Failed gates: " + ", ".join(failed))
        raise SystemExit(1)
    print("All D6d execution/regression/numerical-control gates passed.")
    print("Electronic-equilibrium interpretation remains a manual scientific decision.")


if __name__ == "__main__":
    main()
