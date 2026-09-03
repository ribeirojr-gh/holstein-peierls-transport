"""Run D5a diagnostics and repository regression gates locally.

The script is intended for use when hosted GitHub Actions are unavailable.  A
successful exit code means that the diagnostics executed and numerical
regressions passed; it does not assert that coherent Ehrenfest dynamics is
physically thermalized.  The D5a JSON artifact must be interpreted separately.
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


def _run_capture(command: Sequence[str], cwd: Path) -> tuple[int, str]:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
        check=False,
    )
    return completed.returncode, completed.stdout


def _git_value(repo: Path, *args: str) -> str:
    code, output = _run_capture(("git", *args), repo)
    return output.strip() if code == 0 else "unknown"


def _package_version(name: str) -> str:
    try:
        module = __import__(name)
        return str(module.__version__)
    except Exception:
        return "unavailable"


def _record(
    name: str,
    command: Sequence[str],
    repo: Path,
    output_dir: Path,
) -> dict[str, object]:
    print(f"\n=== {name} ===", flush=True)
    print(" ".join(command), flush=True)
    start = time.perf_counter()
    code, output = _run_capture(command, repo)
    elapsed = time.perf_counter() - start
    log = output_dir / f"{name}.log"
    log.write_text(output, encoding="utf-8")
    print(output, end="" if output.endswith("\n") else "\n", flush=True)
    print(f"--- {name}: {'PASS' if code == 0 else 'FAIL'} ({elapsed:.3f} s) ---", flush=True)
    return {
        "name": name,
        "command": list(command),
        "return_code": code,
        "status": "pass" if code == 0 else "fail",
        "elapsed_s": elapsed,
        "log": str(log.relative_to(repo)),
    }


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = repo / "d5a-local-validation" / timestamp
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
        "physical_interpretation_gate": (
            "none: D5a is diagnostic; runner PASS only means successful execution "
            "and preserved regressions"
        ),
    }

    python = sys.executable
    records: list[dict[str, object]] = []

    records.append(
        _record(
            "d5a-focused-pytest",
            (python, "-m", "pytest", "tests/test_d5_electronic_thermalization.py", "-q"),
            repo,
            output_dir,
        )
    )
    records.append(
        _record(
            "full-pytest",
            (python, "-m", "pytest"),
            repo,
            output_dir,
        )
    )

    d0a_json = output_dir / "d0a-frozen.json"
    records.append(
        _record(
            "d0a-frozen-propagators",
            (
                python,
                "experiments/d0a_frozen_propagators.py",
                "--nx", "4",
                "--ny", "4",
                "--dt-fs", "0.1",
                "--steps", "20",
                "--krylov-dimension", "8",
                "--output", str(d0a_json),
            ),
            repo,
            output_dir,
        )
    )

    for multiplicity in ("singlet", "triplet"):
        for seed in ("onsite", "bond_x", "bond_y"):
            name = f"s0-{multiplicity}-{seed}"
            artifact = output_dir / f"{multiplicity}-{seed}.json"
            records.append(
                _record(
                    name,
                    (
                        python,
                        "experiments/spin_adapted_isotropic_relaxation.py",
                        "--size", "4",
                        "--staggered-gap", "2.0",
                        "--multiplicity", multiplicity,
                        "--seed", seed,
                        "--require-converged",
                        "--output", str(artifact),
                    ),
                    repo,
                    output_dir,
                )
            )

    d5a_json = output_dir / "d5a-electronic-thermalization.json"
    d5a_md = output_dir / "d5a-electronic-thermalization.md"
    records.append(
        _record(
            "d5a-20x20-ensemble",
            (
                python,
                "experiments/d5a_electronic_thermalization.py",
                "--size", "20",
                "--temperature-K", "300.0",
                "--gamma-u-per-fs", "0.01",
                "--gamma-v-per-fs", "0.01",
                "--dt-fs", "0.2",
                "--final-time-fs", "10000.0",
                "--burn-in-fs", "2000.0",
                "--diagnostic-interval-fs", "100.0",
                "--seeds", "20260903", "20260904", "20260905", "20260906",
                "--zero-mode-policy", "project",
                "--krylov-dimension", "6",
                "--output", str(d5a_json),
                "--markdown", str(d5a_md),
            ),
            repo,
            output_dir,
        )
    )

    failed = [record["name"] for record in records if record["return_code"] != 0]
    summary = {
        "metadata": metadata,
        "overall_status": "pass" if not failed else "fail",
        "failed_gates": failed,
        "gates": records,
        "d5a_artifact": str(d5a_json.relative_to(repo)),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== D5A LOCAL VALIDATION SUMMARY ===")
    print(f"Commit: {metadata['git_commit']}")
    print(f"Branch: {metadata['git_branch']}")
    print(f"Result directory: {output_dir}")
    print(f"Overall execution/regression status: {summary['overall_status'].upper()}")
    print("Physical thermalization is intentionally not assigned a pass/fail threshold by this runner.")
    if failed:
        print("Failed gates: " + ", ".join(str(item) for item in failed))
        return 1
    print("All requested numerical gates passed; inspect the D5a JSON for the physical decision.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
