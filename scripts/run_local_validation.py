"""Run the repository validation gates locally without GitHub Actions.

This utility reproduces the lightweight CI gates on a user workstation and
adds the focused D4a Langevin tests.  It is intended for periods when hosted
GitHub Actions minutes are unavailable or when local hardware is preferred for
longer numerical validation runs.

The runner records:
- git commit and branch information;
- Python/platform information;
- NumPy and SciPy versions;
- stdout/stderr and return code for every validation gate;
- the JSON artifacts produced by the existing experiment drivers; and
- a machine-readable summary.json file.

No numerical tolerances are changed by this script.
"""

from __future__ import annotations

import argparse
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
    """Run one command and return ``(return_code, combined_output)``."""
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


def _command_record(
    *,
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
    log_path = output_dir / f"{name}.log"
    log_path.write_text(output, encoding="utf-8")
    print(output, end="" if output.endswith("\n") else "\n", flush=True)
    print(
        f"--- {name}: {'PASS' if code == 0 else 'FAIL'} "
        f"({elapsed:.3f} s) ---",
        flush=True,
    )
    return {
        "name": name,
        "command": list(command),
        "return_code": code,
        "status": "pass" if code == 0 else "fail",
        "elapsed_s": elapsed,
        "log": str(log_path.relative_to(repo)),
    }


def _artifact_path(output_dir: Path, filename: str) -> str:
    return str((output_dir / filename).resolve())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run D4a and repository CI-equivalent validation locally."
    )
    parser.add_argument(
        "--output-root",
        default="local-validation",
        help="Directory under the repository where timestamped results are stored.",
    )
    parser.add_argument(
        "--skip-full-pytest",
        action="store_true",
        help="Run only the focused D4a tests plus numerical CI gates.",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = repo / args.output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)

    # Match the single-thread numerical environment used for hosted benchmarks.
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    metadata: dict[str, object] = {
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

    records: list[dict[str, object]] = []
    python = sys.executable

    records.append(
        _command_record(
            name="d4a-focused-pytest",
            command=(python, "-m", "pytest", "tests/test_d4_langevin.py", "-q"),
            repo=repo,
            output_dir=output_dir,
        )
    )

    if not args.skip_full_pytest:
        records.append(
            _command_record(
                name="full-pytest",
                command=(python, "-m", "pytest"),
                repo=repo,
                output_dir=output_dir,
            )
        )

    d0a_json = output_dir / "d0a-frozen.json"
    records.append(
        _command_record(
            name="d0a-frozen-propagators",
            command=(
                python,
                "experiments/d0a_frozen_propagators.py",
                "--nx",
                "4",
                "--ny",
                "4",
                "--dt-fs",
                "0.1",
                "--steps",
                "20",
                "--krylov-dimension",
                "8",
                "--output",
                str(d0a_json),
            ),
            repo=repo,
            output_dir=output_dir,
        )
    )

    for multiplicity in ("singlet", "triplet"):
        for seed in ("onsite", "bond_x", "bond_y"):
            name = f"s0-{multiplicity}-{seed}"
            artifact = output_dir / f"{multiplicity}-{seed}.json"
            records.append(
                _command_record(
                    name=name,
                    command=(
                        python,
                        "experiments/spin_adapted_isotropic_relaxation.py",
                        "--size",
                        "4",
                        "--staggered-gap",
                        "2.0",
                        "--multiplicity",
                        multiplicity,
                        "--seed",
                        seed,
                        "--require-converged",
                        "--output",
                        str(artifact),
                    ),
                    repo=repo,
                    output_dir=output_dir,
                )
            )

    failed = [record["name"] for record in records if record["return_code"] != 0]
    summary = {
        "metadata": metadata,
        "overall_status": "pass" if not failed else "fail",
        "failed_gates": failed,
        "gates": records,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== LOCAL VALIDATION SUMMARY ===")
    print(f"Commit: {metadata['git_commit']}")
    print(f"Branch: {metadata['git_branch']}")
    print(f"Result directory: {output_dir}")
    print(f"Overall: {summary['overall_status'].upper()}")
    if failed:
        print("Failed gates: " + ", ".join(str(item) for item in failed))
        return 1
    print("All requested local validation gates passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
