#!/usr/bin/env python3
"""Aggregate the 12 preregistered S1 structural-optimizer branch records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from holstein_peierls.spin_adapted.rprop_bridge import summarize_s1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    files = sorted(args.input_dir.rglob("s1-*.json"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    summary = summarize_s1(records)

    payload = {
        "scope": "S1 spin-adapted RPROP bridge aggregate",
        "record_count": len(records),
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    comparisons = summary["promoted_comparisons"]
    lines = [
        "# S1 spin-adapted RPROP bridge",
        "",
        f"Primary classification: {'PASS' if summary['s1_pass'] else 'FAIL'}",
        "",
        "## Locked gates",
        "",
    ]
    for name, passed in summary["gates"].items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    lines.extend(
        [
            "",
            "## Promoted stationary comparisons",
            "",
            f"- singlet |E_RPROP-E_preconditioned|: {comparisons.get('singlet_energy_abs_difference_eV')}",
            f"- triplet |E_RPROP-E_preconditioned|: {comparisons.get('triplet_energy_abs_difference_eV')}",
            f"- |Delta_ST_RPROP-Delta_ST_preconditioned|: {comparisons.get('singlet_triplet_gap_abs_difference_eV')}",
            f"- RPROP singlet-triplet gap (eV): {comparisons.get('rprop_singlet_minus_triplet_eV')}",
            f"- preconditioned singlet-triplet gap (eV): {comparisons.get('preconditioned_singlet_minus_triplet_eV')}",
            "",
            "## Interpretation",
            "",
            "This is a numerical optimizer-bridge validation on the canonical gapped 4x4 control. The checkerboard gap and U/V values are regression controls, not material parameters. Cross-optimizer differences are not physical uncertainties.",
        ]
    )
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {args.output}")
    print(f"Wrote {args.markdown}")


if __name__ == "__main__":
    main()
