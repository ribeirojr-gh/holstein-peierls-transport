#!/usr/bin/env python3
"""Run or resume a manifest-defined S3 bipolaron campaign locally."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy

from holstein_peierls.s3_campaign import expand_tasks, summarize_campaign, task_id


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_output(*arguments: str) -> str:
    process = subprocess.run(
        ["git", *arguments], capture_output=True, text=True, check=False
    )
    return process.stdout.strip() if process.returncode == 0 else "unavailable"


def _branch_command(task: dict, manifest: dict, output: Path) -> list[str]:
    model = manifest["model"]
    numerical = manifest["numerical"]
    return [
        sys.executable,
        "experiments/bipolaron_branch_benchmark_seeded.py",
        "--size", str(task["size"]),
        "--u", str(task["U_eV"]),
        "--v1", str(task["V1_eV"]),
        "--branch", str(task["branch"]),
        "--j0x", str(model["Jx_eV"]),
        "--j0y", str(model["Jy_eV"]),
        "--k1", str(model["K1_eV_per_A2"]),
        "--k2", str(model["K2_eV_per_A2"]),
        "--alpha-intra", str(task["alpha_intra_eV_per_A"]),
        "--alpha-x", str(task["alpha_x_eV_per_A"]),
        "--alpha-y", str(task["alpha_y_eV_per_A"]),
        "--max-iterations", str(numerical["max_iterations"]),
        "--output", str(output),
    ]


def _decorate_record(record: dict, task: dict) -> dict:
    decorated = dict(record)
    decorated["coupling_scale"] = float(task["coupling_scale"])
    decorated["task_id"] = task_id(task)
    return decorated


def _validate_record_matches_task(record: dict, task: dict) -> None:
    expected = {
        "size": int(task["size"]),
        "U_eV": float(task["U_eV"]),
        "V1_eV": float(task["V1_eV"]),
        "branch": str(task["branch"]),
        "alpha_intra_eV_per_A": float(task["alpha_intra_eV_per_A"]),
        "alpha_x_eV_per_A": float(task["alpha_x_eV_per_A"]),
        "alpha_y_eV_per_A": float(task["alpha_y_eV_per_A"]),
    }
    for key, value in expected.items():
        actual = record.get(key)
        if isinstance(value, float):
            matches = actual is not None and abs(float(actual) - value) <= 1.0e-14
        else:
            matches = actual == value
        if not matches:
            raise SystemExit(
                f"checkpoint {task_id(task)} has mismatched {key}: "
                f"expected {value!r}, found {actual!r}"
            )


def _write_point_csv(path: Path, points: list[dict]) -> None:
    if not points:
        return
    keys: list[str] = []
    for point in points:
        for key in point:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(points)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--limit", type=int, help="Run at most this many pending tasks")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = expand_tasks(manifest)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    if args.run_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = Path("s3-local-runs") / manifest["campaign_id"] / stamp
    else:
        run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    branch_root = run_dir / "branches"
    branch_root.mkdir(exist_ok=True)

    frozen_manifest = run_dir / "manifest.json"
    if frozen_manifest.exists():
        if json.loads(frozen_manifest.read_text(encoding="utf-8")) != manifest:
            raise SystemExit("run directory contains a different frozen manifest")
    else:
        _write_json(frozen_manifest, manifest)

    provenance = {
        "created_or_resumed_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _manifest_sha256(manifest_path),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "executable": sys.executable,
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_branch": _git_output("branch", "--show-current"),
        "git_status": _git_output("status", "--short"),
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }
    _write_json(run_dir / "provenance.json", provenance)

    completed_now = 0
    failures: list[dict] = []
    for index, task in enumerate(tasks, start=1):
        identifier = task_id(task)
        task_dir = branch_root / identifier
        result_path = task_dir / "branch_result.json"
        if result_path.exists():
            record = json.loads(result_path.read_text(encoding="utf-8"))
            _validate_record_matches_task(record, task)
            decorated = _decorate_record(record, task)
            if decorated != record:
                _write_json(result_path, decorated)
            print(f"[{index}/{len(tasks)}] resume {identifier}", flush=True)
            continue
        if args.limit is not None and completed_now >= args.limit:
            break
        task_dir.mkdir(exist_ok=True)
        command = _branch_command(task, manifest, task_dir)
        print(f"[{index}/{len(tasks)}] run {identifier}", flush=True)
        process = subprocess.run(command, capture_output=True, text=True, check=False)
        (task_dir / "stdout.log").write_text(process.stdout, encoding="utf-8")
        (task_dir / "stderr.log").write_text(process.stderr, encoding="utf-8")
        if process.returncode != 0 or not result_path.exists():
            failures.append(
                {"task_id": identifier, "return_code": process.returncode, "command": command}
            )
            print(f"  FAIL return_code={process.returncode}", flush=True)
            continue
        record = json.loads(result_path.read_text(encoding="utf-8"))
        _validate_record_matches_task(record, task)
        _write_json(result_path, _decorate_record(record, task))
        completed_now += 1

    records = []
    for result_path in sorted(branch_root.glob("*/branch_result.json")):
        records.append(json.loads(result_path.read_text(encoding="utf-8")))
    summary = summarize_campaign(manifest, records)
    summary["provenance"] = provenance
    summary["failed_tasks"] = failures
    _write_json(run_dir / "summary.json", summary)
    _write_point_csv(run_dir / "points.csv", summary["points"])
    print(json.dumps({
        "run_dir": str(run_dir),
        "expected": summary["expected_branch_records"],
        "completed": summary["completed_branch_records"],
        "complete": summary["complete"],
        "failed_now": len(failures),
    }, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
