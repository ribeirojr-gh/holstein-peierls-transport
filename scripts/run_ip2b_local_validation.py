#!/usr/bin/env python3
"""Self-contained local fallback for the complete IP2b production ensemble."""

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


def _run(name: str, command: list[str], directory: Path) -> dict:
    print(f"\n=== {name} ===", flush=True)
    started = perf_counter()
    proc = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    elapsed = float(perf_counter() - started)
    log = directory / f"{name}.log"
    log.write_text(proc.stdout, encoding="utf-8")
    print(proc.stdout, end="", flush=True)
    return {
        "name": name,
        "command": command,
        "return_code": int(proc.returncode),
        "status": "pass" if proc.returncode == 0 else "fail",
        "elapsed_s": elapsed,
        "log": str(log),
    }


def _git(args: list[str]) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path("ip2b-local-validation") / stamp
    members = root / "members"
    aggregate = root / "aggregate"
    members.mkdir(parents=True, exist_ok=True)
    aggregate.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    gates = []

    gates.append(
        _run(
            "ip2b-pycompile",
            [
                py,
                "-m",
                "py_compile",
                "src/holstein_peierls/dynamics/ip2b_ensemble.py",
                "experiments/ip2b_member_counterfactual.py",
                "experiments/ip2b_aggregate.py",
            ],
            root,
        )
    )
    gates.append(
        _run(
            "ip2b-focused-pytest",
            [
                py,
                "-m",
                "pytest",
                "tests/test_ip2b_ensemble.py",
                "tests/test_ip1s_direction_reversal.py",
                "tests/test_ip2a_ensemble_preparation.py",
                "tests/test_d3_field_driven.py",
                "tests/test_numerical_validation.py",
                "-q",
            ],
            root,
        )
    )
    gates.append(_run("full-pytest", [py, "-m", "pytest"], root))

    recurrence_json = root / "ip2b-recurrence.json"
    recurrence_md = root / "ip2b-recurrence.md"
    gates.append(
        _run(
            "ip2b-recurrence-preflight",
            [
                py,
                "experiments/ip1p_phonon_recurrence_audit.py",
                "--nx",
                "40",
                "--ny",
                "40",
                "--lattice-spacing-A",
                "3.0",
                "--gamma-v-per-fs",
                "0.0",
                "--event-half-window-fs",
                "500.0",
                "--planned-final-time-fs",
                "6000.0",
                "--output",
                str(recurrence_json),
                "--markdown",
                str(recurrence_md),
            ],
            root,
        )
    )

    if all(gate["status"] == "pass" for gate in gates):
        for member_id in range(32):
            gate = _run(
                f"ip2b-member-{member_id:02d}",
                [
                    py,
                    "experiments/ip2b_member_counterfactual.py",
                    "--member-id",
                    str(member_id),
                    "--output",
                    str(members / f"member-{member_id:02d}.json"),
                    "--arrays",
                    str(members / f"member-{member_id:02d}.npz"),
                ],
                root,
            )
            gates.append(gate)
            if gate["status"] != "pass":
                break

    if all(gate["status"] == "pass" for gate in gates):
        gates.append(
            _run(
                "ip2b-aggregate",
                [
                    py,
                    "experiments/ip2b_aggregate.py",
                    "--input-dir",
                    str(members),
                    "--output",
                    str(aggregate / "ip2b-aggregate.json"),
                    "--markdown",
                    str(aggregate / "ip2b-aggregate.md"),
                    "--arrays",
                    str(aggregate / "ip2b-aggregate.npz"),
                ],
                root,
            )
        )

    failed = [gate["name"] for gate in gates if gate["status"] != "pass"]
    aggregate_payload = None
    aggregate_json = aggregate / "ip2b-aggregate.json"
    if aggregate_json.exists():
        aggregate_payload = json.loads(aggregate_json.read_text(encoding="utf-8"))
    summary = {
        "metadata": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git(["rev-parse", "HEAD"]),
            "git_branch": _git(["branch", "--show-current"]),
            "git_status": _git(["status", "--porcelain"]),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "thread_environment": {
                key: os.environ.get(key, "unset")
                for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
            },
        },
        "execution_integrity": "pass" if not failed and aggregate_payload else "fail",
        "failed_gates": failed,
        "physical_primary_pass": (
            None
            if aggregate_payload is None
            else bool(aggregate_payload["summary"]["primary_ip2b_pass"])
        ),
        "gates": gates,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nIP2b local artifact: {root}")
    print(f"Execution integrity: {summary['execution_integrity'].upper()}")
    print(f"Physical primary pass: {summary['physical_primary_pass']}")
    if summary["execution_integrity"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
