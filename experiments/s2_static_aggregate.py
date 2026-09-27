#!/usr/bin/env python3
"""Aggregate S2 static-sector regression artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from holstein_peierls.s2_static_regression import summarize_s2


def _load_one(pattern: str, root: Path) -> dict:
    files = sorted(root.rglob(pattern))
    if len(files) != 1:
        raise RuntimeError(f"expected exactly one {pattern}, found {len(files)}")
    return json.loads(files[0].read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--full-pytest-passed", action="store_true")
    args = parser.parse_args()

    polaron = _load_one("s2-polaron.json", args.input_dir)
    bipolaron = _load_one("branch_result.json", args.input_dir)
    exciton = _load_one("s2-exciton.json", args.input_dir)
    spin_files = sorted(args.input_dir.rglob("s2-spin-*.json"))
    if len(spin_files) != 4:
        raise RuntimeError(f"expected four S2 spin JSON files, found {len(spin_files)}")
    spin_records = [json.loads(path.read_text(encoding="utf-8")) for path in spin_files]

    summary = summarize_s2(
        polaron=polaron,
        bipolaron=bipolaron,
        exciton=exciton,
        spin_records=spin_records,
        full_pytest_passed=args.full_pytest_passed,
    )
    payload = {
        "scope": "S2 unified static-sector regression",
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# S2 unified static-sector regression",
        "",
        f"Primary classification: {'PASS' if summary['s2_pass'] else 'FAIL'}",
        "",
        "## Sector status",
        "",
        f"- one polaron: {'PASS' if summary['one_polaron']['pass'] else 'FAIL'}",
        f"- correlated bipolaron: {'PASS' if summary['bipolaron']['pass'] else 'FAIL'}",
        f"- spin-blind exciton: {'PASS' if summary['spin_blind_exciton']['pass'] else 'FAIL'}",
        f"- spin-adapted anchors: {'PASS' if summary['spin_adapted']['pass'] else 'FAIL'}",
        "",
        "## Global gates",
        "",
    ]
    for name, passed in summary["global_gates"].items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    lines.extend([
        "",
        "S2 is an integration regression, not a new phase diagram. "
        "The distinguishable electron-hole sector remains spin blind, and Paper-1 "
        "spin production continues to use the complete deterministic S1R root ensemble.",
    ])
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
