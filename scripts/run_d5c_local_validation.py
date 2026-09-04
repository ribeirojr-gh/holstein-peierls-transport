"""Run local D5c decoherence-interval sensitivity without GitHub Actions.

Runner PASS confirms only that the numerical/regression gates and sensitivity
sweep executed successfully.  The physical selection of IDC-BM/IDC-MA and of a
decoherence interval is made afterward from the generated JSON.
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
    output_dir = root / "d5c-local-validation" / stamp
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
        "physical_interpretation_gate": (
            "none: runner PASS confirms execution/regressions only; BM/MA and t_d "
            "selection require post-run analysis of d5c-decoherence-interval-sweep.json"
        ),
    }

    python = sys.executable
    gates: list[dict] = []
    gates.append(
        _run(
            "d5c-focused-pytest",
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

    artifact = output_dir / "d5c-decoherence-interval-sweep.json"
    markdown = output_dir / "d5c-decoherence-interval-sweep.md"
    gates.append(
        _run(
            "d5c-20x20-interval-sweep",
            [
                python,
                "experiments/d5c_decoherence_interval_sweep.py",
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
                "6000.0",
                "--burn-in-fs",
                "2000.0",
                "--decoherence-intervals-fs",
                "50.0",
                "100.0",
                "180.0",
                "250.0",
                "500.0",
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
        "d5c_artifact": str(artifact.relative_to(root)) if artifact.exists() else None,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n=== D5C LOCAL VALIDATION SUMMARY ===")
    print(f"Output directory: {output_dir}")
    print(f"Overall execution/regression status: {summary['overall_status'].upper()}")
    if failed:
        print("Failed gates: " + ", ".join(failed))
        raise SystemExit(1)
    print("All requested numerical gates passed.")
    print("Physical BM/MA and decoherence-interval selection requires analysis of the generated JSON.")


if __name__ == "__main__":
    main()
