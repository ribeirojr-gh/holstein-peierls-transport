#!/usr/bin/env python3
"""Aggregate all 32 IP2b member artifacts under the locked finite-design rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from holstein_peierls.dynamics.ip2b_ensemble import summarize_ip2b_members


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    args = parser.parse_args()

    files = sorted(args.input_dir.rglob("member-*.json"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    summary = summarize_ip2b_members(records)

    by_id = {int(record["member_id"]): record for record in records}
    ordered = [by_id[i] for i in range(32)]
    d = np.full(32, np.nan, dtype=np.float64)
    dctrl = np.full(32, np.nan, dtype=np.float64)
    first010 = np.full(32, np.nan, dtype=np.float64)
    first025 = np.full(32, np.nan, dtype=np.float64)
    valid = np.zeros(32, dtype=np.bool_)
    pair_id = np.zeros(32, dtype=np.int64)
    pair_sign = np.zeros(32, dtype=np.int64)

    for record in ordered:
        i = int(record["member_id"])
        pair_id[i] = int(record["pair_id"])
        pair_sign[i] = int(record["pair_sign"])
        valid[i] = bool(record["valid_member"])
        if valid[i]:
            d[i] = float(record["primary"]["D_i"])
            dctrl[i] = float(record["primary"]["D_i_ctrl"])
            if record["primary"]["first_l1_ge_0p10_offset_fs"] is not None:
                first010[i] = float(record["primary"]["first_l1_ge_0p10_offset_fs"])
            if record["primary"]["first_l1_ge_0p25_offset_fs"] is not None:
                first025[i] = float(record["primary"]["first_l1_ge_0p25_offset_fs"])

    payload = {
        "scope": "IP2b complete 32-member paired counterfactual finite-design aggregation",
        "member_file_count": len(files),
        "member_ids": [int(record["member_id"]) for record in ordered],
        "summary": summary,
        "execution_integrity": {
            "all_32_member_json_files_present": len(files) == 32,
            "all_member_ids_exactly_0_to_31": [int(record["member_id"]) for record in ordered]
            == list(range(32)),
            "all_member_payloads_have_explicit_validity": all(
                "valid_member" in record and "rejection_reasons" in record for record in ordered
            ),
        },
        "interpretation_guard": {
            "workflow_success_is_not_physical_primary_pass": True,
            "physical_primary_result_is_summary_primary_ip2b_pass": True,
            "deterministic_grid_not_population_sample": True,
        },
    }
    if not all(payload["execution_integrity"].values()):
        raise RuntimeError("IP2b aggregate execution-integrity gate failed")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(
        args.arrays,
        member_id=np.arange(32, dtype=np.int64),
        pair_id=pair_id,
        pair_sign=pair_sign,
        valid_member=valid,
        D_i=d,
        D_i_ctrl=dctrl,
        first_l1_ge_0p10_offset_fs=first010,
        first_l1_ge_0p25_offset_fs=first025,
    )

    primary = summary["primary_distribution"]
    gates = summary["primary_gates"]
    lines = [
        "# IP2b complete deterministic paired ensemble",
        "",
        f"Execution integrity: PASS",
        f"Primary physical classification: {'PASS' if summary['primary_ip2b_pass'] else 'FAIL'}",
        "",
        f"- valid members: {summary['valid_member_count']}/32",
        f"- median D_i: {primary['median_D_i']}",
        f"- Q1/Q3 D_i: {primary['q1_D_i']} / {primary['q3_D_i']}",
        f"- fraction D_i >= 0.25: {primary['D_ge_0p25_fraction']}",
        f"- median intervention/control ratio: {primary['median_control_separation_ratio']}",
        "",
        "## Locked primary gates",
        "",
    ]
    for name, value in gates.items():
        lines.append(f"- {name}: {'PASS' if value else 'FAIL'}")
    lines.extend(
        [
            "",
            "## Secondary event summaries",
            "",
            f"- topology-history disagreement fraction: {summary['secondary_events']['history_disagreement_fraction']}",
            f"- native direct-recross fraction: {summary['secondary_events']['native_direct_recross_fraction']}",
            f"- reversed direct-recross fraction: {summary['secondary_events']['reversed_direct_recross_fraction']}",
            "",
            "This is a finite deterministic design. No p-value, population confidence interval, hopping probability, rate, mobility, diffusion coefficient or activation energy is inferred.",
        ]
    )
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {args.output}")
    print(f"Wrote {args.markdown}")
    print(f"Wrote {args.arrays}")


if __name__ == "__main__":
    main()
