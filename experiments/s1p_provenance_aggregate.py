#!/usr/bin/env python3
"""Aggregate S1P provenance/environment records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from holstein_peierls.spin_adapted.s1p_provenance import summarize_s1p


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--markdown", type=Path, required=True)
    args = p.parse_args()

    files = sorted(args.input_dir.rglob("s1p-*.json"))
    records = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    summary = summarize_s1p(records)
    payload = {"scope": "S1P provenance/environment audit", "record_count": len(records), "summary": summary}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# S1P provenance/environment audit",
        "",
        f"- 32 records present: {summary['all_32_records_present']}",
        f"- all branches converged: {summary['all_branches_converged']}",
        f"- historical reference reproduced: {summary['historical_reference_reproduced_within_1e-8_eV']}",
        f"- provenance classification: {summary['provenance_classification']}",
        "",
        "## Factor sensitivity",
        "",
    ]
    for factor, item in summary["factor_sensitivity"].items():
        lines.append(
            f"- {factor}: root_selecting={item['root_selecting']}; "
            f"count={item['root_change_count']}; "
            f"max |dE|={item['maximum_absolute_energy_difference_eV']} eV"
        )
    lines.extend([
        "",
        "S1 remains formally failed. S1P diagnoses numerical provenance only and does not promote an optimizer or a physical singlet-triplet splitting.",
    ])
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
