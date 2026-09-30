#!/usr/bin/env python3
"""Compare frozen S3 production and finite-size summaries point by point."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


PointKey = tuple[float, float, float]


def _key(point: dict[str, Any]) -> PointKey:
    return (
        float(point["coupling_scale"]),
        float(point["U_eV"]),
        float(point["V1_eV"]),
    )


def _indexed(summary: dict[str, Any]) -> dict[PointKey, dict[str, Any]]:
    result: dict[PointKey, dict[str, Any]] = {}
    for point in summary["points"]:
        key = _key(point)
        if key in result:
            raise ValueError(f"duplicate S3 point in summary: {key}")
        result[key] = point
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def compare_summaries(
    production: dict[str, Any], finite_size: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if production.get("stage") != "production":
        raise ValueError("first summary must be an S3 production summary")
    if finite_size.get("stage") != "finite_size":
        raise ValueError("second summary must be an S3 finite_size summary")
    p20 = _indexed(production)
    p40 = _indexed(finite_size)
    all_keys = sorted(set(p20) | set(p40))
    point_rows: list[dict[str, Any]] = []

    for key in all_keys:
        low = p20.get(key)
        high = p40.get(key)
        row: dict[str, Any] = {
            "coupling_scale": key[0],
            "U_eV": key[1],
            "V1_eV": key[2],
            "present_at_20x20": low is not None,
            "present_at_40x40": high is not None,
        }
        if low is None or high is None:
            row["finite_size_status"] = (
                "not_selected_for_40x40" if low is not None else "not_in_20x20_reference_grid"
            )
            point_rows.append(row)
            continue

        topology20 = low.get("observable_topology")
        topology40 = high.get("observable_topology")
        class20 = low.get("classification")
        class40 = high.get("classification")
        converged = bool(low.get("all_branches_converged")) and bool(
            high.get("all_branches_converged")
        )
        peierls = bool(low.get("linear_peierls_gate")) and bool(
            high.get("linear_peierls_gate")
        )
        same_topology = topology20 == topology40
        same_classification = class20 == class40
        status = (
            "incomplete_or_unconverged"
            if not converged
            else "outside_linear_peierls_gate"
            if not peierls
            else "topology_changed_with_size"
            if not same_topology
            else "stable_at_sampled_point"
        )
        binding20 = low.get("binding_vs_separated_eV")
        binding40 = high.get("binding_vs_separated_eV")
        row.update(
            {
                "topology_20x20": topology20,
                "topology_40x40": topology40,
                "classification_20x20": class20,
                "classification_40x40": class40,
                "same_observable_topology": same_topology,
                "same_classification": same_classification,
                "all_required_branches_converged_both_sizes": converged,
                "linear_peierls_gate_both_sizes": peierls,
                "binding_20x20_eV": binding20,
                "binding_40x40_eV": binding40,
                "binding_change_40_minus_20_meV": (
                    None
                    if binding20 is None or binding40 is None
                    else 1000.0 * (float(binding40) - float(binding20))
                ),
                "required_branches_20x20": len(low.get("required_branches", [])) or 5,
                "required_branches_40x40": len(high.get("required_branches", [])),
                "finite_size_status": status,
            }
        )
        point_rows.append(row)

    by_slice: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for row in point_rows:
        if row.get("present_at_20x20") and row.get("present_at_40x40"):
            by_slice.setdefault((row["coupling_scale"], row["U_eV"]), []).append(row)
    bracket_rows: list[dict[str, Any]] = []
    for (coupling, hubbard_u), rows in sorted(by_slice.items()):
        ordered = sorted(rows, key=lambda row: row["V1_eV"])
        for lower, upper in zip(ordered, ordered[1:]):
            topology20_pair = (lower["topology_20x20"], upper["topology_20x20"])
            topology40_pair = (lower["topology_40x40"], upper["topology_40x40"])
            if topology20_pair[0] == topology20_pair[1] and topology40_pair[0] == topology40_pair[1]:
                continue
            same_bracket = topology20_pair == topology40_pair
            eligible = all(
                endpoint["finite_size_status"] == "stable_at_sampled_point"
                for endpoint in (lower, upper)
            )
            bracket_rows.append(
                {
                    "coupling_scale": coupling,
                    "U_eV": hubbard_u,
                    "lower_V1_eV": lower["V1_eV"],
                    "upper_V1_eV": upper["V1_eV"],
                    "topology_transition_20x20": " -> ".join(topology20_pair),
                    "topology_transition_40x40": " -> ".join(topology40_pair),
                    "same_transition_bracket_at_both_sizes": same_bracket,
                    "both_endpoints_stable_at_sampled_points": eligible,
                    "finite_size_status": (
                        "same_sampled_transition_bracket"
                        if same_bracket and eligible
                        else "transition_bracket_not_promotable_due_to_endpoint_gate"
                        if same_bracket
                        else "transition_bracket_changed_with_size"
                    ),
                    "interpretation_limit": "transition is bracketed by these samples; its location is not resolved within the interval",
                }
            )

    report = {
        "schema_version": 1,
        "production_campaign": production.get("campaign_id"),
        "finite_size_campaign": finite_size.get("campaign_id"),
        "point_count_compared": sum(
            row["present_at_20x20"] and row["present_at_40x40"] for row in point_rows
        ),
        "production_points_not_selected_for_40x40": sum(
            row["finite_size_status"] == "not_selected_for_40x40" for row in point_rows
        ),
        "40x40_points_without_20x20_reference": sum(
            row["finite_size_status"] == "not_in_20x20_reference_grid" for row in point_rows
        ),
        "point_count_stable_at_sampled_point": sum(
            row["finite_size_status"] == "stable_at_sampled_point" for row in point_rows
        ),
        "point_count_topology_changed": sum(
            row["finite_size_status"] == "topology_changed_with_size" for row in point_rows
        ),
        "transition_bracket_count": len(bracket_rows),
        "transition_brackets_matching": sum(
            row["finite_size_status"] == "same_sampled_transition_bracket"
            for row in bracket_rows
        ),
        "point_status_semantics": "stable_at_sampled_point requires complete converged point data at both sizes, both selected minima inside the linear-Peierls gate, and matching observable topology; it does not claim a globally converged phase boundary",
        "boundary_status_semantics": "same_sampled_transition_bracket means adjacent sampled topologies change in the same order at both sizes; no interpolation or exact boundary location is implied",
        "points": point_rows,
        "transition_brackets": bracket_rows,
    }
    return report, point_rows, bracket_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-summary", type=Path, required=True)
    parser.add_argument("--finite-size-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    production = json.loads(args.production_summary.read_text(encoding="utf-8"))
    finite_size = json.loads(args.finite_size_summary.read_text(encoding="utf-8"))
    report, points, brackets = compare_summaries(production, finite_size)
    report["inputs"] = {
        "production_summary_sha256": hashlib.sha256(
            args.production_summary.read_bytes()
        ).hexdigest(),
        "finite_size_summary_sha256": hashlib.sha256(
            args.finite_size_summary.read_bytes()
        ).hexdigest(),
        "production_solver_commit": production.get("provenance", {}).get("git_commit"),
        "finite_size_solver_commit": finite_size.get("provenance", {}).get("git_commit"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "finite_size_comparison.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(args.output_dir / "finite_size_points.csv", points)
    _write_csv(args.output_dir / "transition_brackets.csv", brackets)
    print(
        json.dumps(
            {
                "points": report["point_count_compared"],
                "stable_points": report["point_count_stable_at_sampled_point"],
                "transition_brackets": report["transition_bracket_count"],
                "matching_brackets": report["transition_brackets_matching"],
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
