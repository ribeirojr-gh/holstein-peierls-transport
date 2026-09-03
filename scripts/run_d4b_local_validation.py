"""Run D4b coupled-bath validation locally without GitHub Actions.

This runner is intentionally separate from the lightweight repository CI
replacement. It executes the focused D4a+D4b tests, the full pytest suite, a
short 20x20 timestep/statistics comparison, and the 10 ps projected-zero-mode
300 K stability gate.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Sequence


def _run(command: Sequence[str], cwd: Path) -> tuple[int, str, float]:
    start = time.perf_counter()
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
        check=False,
    )
    return completed.returncode, completed.stdout, time.perf_counter() - start


def _git_value(repo: Path, *args: str) -> str:
    code, output, _ = _run(("git", *args), repo)
    return output.strip() if code == 0 else "unknown"


def _package_version(name: str) -> str:
    try:
        module = __import__(name)
        return str(module.__version__)
    except Exception:
        return "unavailable"


def _gate(
    name: str,
    command: Sequence[str],
    repo: Path,
    output_dir: Path,
) -> dict[str, object]:
    print(f"\n=== {name} ===", flush=True)
    print(" ".join(command), flush=True)
    code, output, elapsed = _run(command, repo)
    log = output_dir / f"{name}.log"
    log.write_text(output, encoding="utf-8")
    print(output, end="" if output.endswith("\n") else "\n", flush=True)
    status = "PASS" if code == 0 else "FAIL"
    print(f"--- {name}: {status} ({elapsed:.3f} s) ---", flush=True)
    return {
        "name": name,
        "command": list(command),
        "return_code": code,
        "status": status.lower(),
        "elapsed_s": elapsed,
        "log": str(log.relative_to(repo)),
    }


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = repo / "d4b-local-validation" / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)

    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_value(repo, "rev-parse", "HEAD"),
        "git_branch": _git_value(repo, "branch", "--show-current"),
        "git_status": _git_value(repo, "status", "--short"),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": _package_version("numpy"),
        "scipy": _package_version("scipy"),
        "thread_environment": {
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
    }

    python = sys.executable
    gates: list[dict[str, object]] = []

    gates.append(
        _gate(
            "d4-focused-pytest",
            (
                python,
                "-m",
                "pytest",
                "tests/test_d4_langevin.py",
                "tests/test_d4_coupled_bath.py",
                "-q",
            ),
            repo,
            output_dir,
        )
    )
    gates.append(
        _gate(
            "full-pytest",
            (python, "-m", "pytest"),
            repo,
            output_dir,
        )
    )

    short_json = output_dir / "d4b-timestep.json"
    short_md = output_dir / "d4b-timestep.md"
    gates.append(
        _gate(
            "d4b-timestep-statistics",
            (
                python,
                "experiments/d4_coupled_bath.py",
                "--size",
                "20",
                "--temperature-K",
                "300",
                "--gamma-u-per-fs",
                "0.01",
                "--gamma-v-per-fs",
                "0.01",
                "--dt-values",
                "0.2",
                "0.1",
                "--final-time-fs",
                "2000",
                "--burn-in-fs",
                "500",
                "--seed",
                "20260903",
                "--zero-mode-policy",
                "project",
                "--sample-stride",
                "10",
                "--krylov-dimension",
                "6",
                "--output",
                str(short_json),
                "--markdown",
                str(short_md),
            ),
            repo,
            output_dir,
        )
    )

    long_json = output_dir / "d4b-stability-10ps.json"
    long_md = output_dir / "d4b-stability-10ps.md"
    gates.append(
        _gate(
            "d4b-stability-10ps",
            (
                python,
                "experiments/d4_coupled_bath.py",
                "--size",
                "20",
                "--temperature-K",
                "300",
                "--gamma-u-per-fs",
                "0.01",
                "--gamma-v-per-fs",
                "0.01",
                "--dt-values",
                "0.2",
                "--final-time-fs",
                "10000",
                "--burn-in-fs",
                "2000",
                "--seed",
                "20260903",
                "--zero-mode-policy",
                "project",
                "--sample-stride",
                "10",
                "--krylov-dimension",
                "6",
                "--output",
                str(long_json),
                "--markdown",
                str(long_md),
            ),
            repo,
            output_dir,
        )
    )

    failed = [gate["name"] for gate in gates if gate["return_code"] != 0]
    summary = {
        "metadata": metadata,
        "overall_status": "pass" if not failed else "fail",
        "failed_gates": failed,
        "gates": gates,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n=== D4B LOCAL VALIDATION SUMMARY ===")
    print(f"Commit: {metadata['git_commit']}")
    print(f"Branch: {metadata['git_branch']}")
    print(f"Result directory: {output_dir}")
    print(f"Overall: {summary['overall_status'].upper()}")
    if failed:
        print("Failed gates: " + ", ".join(str(item) for item in failed))
        return 1
    print("All D4b local validation gates completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
