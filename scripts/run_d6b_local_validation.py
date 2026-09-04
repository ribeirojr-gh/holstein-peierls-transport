"""Run local D6b coupled pair-dynamics validation without GitHub Actions."""

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
    output_dir = root / "d6b-local-validation" / stamp
    output_dir.mkdir(parents=True, exist_ok=False)

    # Override rather than setdefault: the validation metadata must describe the
    # actual intended single-thread environment even if the parent shell had a
    # different BLAS setting.
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
            "D6b zero-temperature moving-lattice bipolaron/exciton dynamics; "
            "no thermostat, decoherence, electric field, or transport fit"
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
                "-q",
            ],
            output_dir,
        )
    )
    gates.append(_run("full-pytest", [python, "-m", "pytest"], output_dir))

    artifact = output_dir / "d6b-pair-coupled.json"
    markdown = output_dir / "d6b-pair-coupled.md"
    gates.append(
        _run(
            "d6b-pair-coupled-benchmark",
            [
                python,
                "experiments/d6b_pair_coupled_zero_temperature.py",
                "--size",
                "4",
                "--reference-time-fs",
                "4.0",
                "--dt-values",
                "0.2",
                "0.1",
                "0.05",
                "--krylov-dimensions",
                "6",
                "8",
                "12",
                "--stability-time-fs",
                "100.0",
                "--stability-dt-fs",
                "0.2",
                "--sample-stride",
                "10",
                "--output",
                str(artifact),
                "--markdown",
                str(markdown),
            ],
            output_dir,
        )
    )

    failed = [gate["name"] for gate in gates if gate["status"] != "pass"]
    closure = False
    checks = None
    if artifact.exists():
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        closure = bool(payload.get("closure_pass", False))
        checks = payload.get("closure_checks")
        if not closure and "d6b-pair-coupled-benchmark" not in failed:
            failed.append("d6b-pair-coupled-physical-closure")

    summary = {
        "metadata": metadata,
        "overall_status": "pass" if not failed else "fail",
        "failed_gates": failed,
        "gates": gates,
        "d6b_artifact": str(artifact.relative_to(root)) if artifact.exists() else None,
        "d6b_closure_pass": closure,
        "d6b_closure_checks": checks,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n=== D6B LOCAL VALIDATION SUMMARY ===")
    print(f"Output directory: {output_dir}")
    print(f"Overall status: {summary['overall_status'].upper()}")
    if failed:
        print("Failed gates: " + ", ".join(failed))
        raise SystemExit(1)
    print("All D6b numerical, regression, and coupled-pair closure gates passed.")


if __name__ == "__main__":
    main()
