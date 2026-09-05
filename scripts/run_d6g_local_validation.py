"""Run final D6 pair-dynamics closure locally without GitHub Actions."""

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
    output_dir = root / "d6g-local-validation" / stamp
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
            "D6g final zero-field finite-temperature pair closure: bipolaron BM, "
            "exciton DP, t_d=100 fs numerical control; includes repository D0a/S0 gates"
        ),
    }

    python = sys.executable
    focused = [
        "tests/test_d6_pair_frozen.py",
        "tests/test_d6_pair_coupled.py",
        "tests/test_d6_pair_thermal.py",
        "tests/test_d6_pair_thermalization.py",
        "tests/test_d6_pair_decoherence.py",
        "tests/test_d6_pair_idc_sensitivity.py",
    ]
    gates: list[dict] = []
    gates.append(
        _run("d6-focused-pytest", [python, "-m", "pytest", *focused, "-q"], output_dir)
    )
    gates.append(_run("full-pytest", [python, "-m", "pytest"], output_dir))

    artifact = output_dir / "d6g-pair-final-closure.json"
    markdown = output_dir / "d6g-pair-final-closure.md"
    gates.append(
        _run(
            "d6g-pair-final-closure-benchmark",
            [
                python,
                "experiments/d6g_pair_final_closure.py",
                "--size", "4",
                "--temperature-K", "300",
                "--gamma-u-per-fs", "0.01",
                "--gamma-v-per-fs", "0.01",
                "--dt-fs", "0.2",
                "--final-time-fs", "10000",
                "--burn-in-fs", "2000",
                "--decoherence-interval-fs", "100",
                "--seeds", "20260905", "20260906", "20260907", "20260908",
                "--krylov-dimension", "8",
                "--output", str(artifact),
                "--markdown", str(markdown),
            ],
            output_dir,
        )
    )

    d0a = output_dir / "d0a-frozen.json"
    gates.append(
        _run(
            "d0a-frozen-propagators",
            [
                python,
                "experiments/d0a_frozen_propagators.py",
                "--nx", "4", "--ny", "4",
                "--dt-fs", "0.1", "--steps", "20",
                "--krylov-dimension", "8",
                "--output", str(d0a),
            ],
            output_dir,
        )
    )

    for multiplicity in ("singlet", "triplet"):
        for seed in ("onsite", "bond_x", "bond_y"):
            name = f"s0-{multiplicity}-{seed}"
            output = output_dir / f"{multiplicity}-{seed}.json"
            gates.append(
                _run(
                    name,
                    [
                        python,
                        "experiments/spin_adapted_isotropic_relaxation.py",
                        "--size", "4",
                        "--staggered-gap", "2.0",
                        "--multiplicity", multiplicity,
                        "--seed", seed,
                        "--require-converged",
                        "--output", str(output),
                    ],
                    output_dir,
                )
            )

    failed = [gate["name"] for gate in gates if gate["status"] != "pass"]
    closure_pass = False
    closure_checks = None
    if artifact.exists():
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        closure_pass = bool(payload.get("closure_pass", False))
        closure_checks = payload.get("closure_checks")
        if not closure_pass and "d6g-pair-final-closure-benchmark" not in failed:
            failed.append("d6g-physical-closure")

    summary = {
        "metadata": metadata,
        "overall_status": "pass" if not failed else "fail",
        "failed_gates": failed,
        "gates": gates,
        "d6g_artifact": str(artifact.relative_to(root)) if artifact.exists() else None,
        "d6g_closure_pass": closure_pass,
        "d6g_closure_checks": closure_checks,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n=== D6G FINAL LOCAL VALIDATION SUMMARY ===")
    print(f"Output directory: {output_dir}")
    print(f"Overall status: {summary['overall_status'].upper()}")
    if failed:
        print("Failed gates: " + ", ".join(failed))
        raise SystemExit(1)
    print("All D6 final pair-dynamics and repository closure gates passed.")


if __name__ == "__main__":
    main()
