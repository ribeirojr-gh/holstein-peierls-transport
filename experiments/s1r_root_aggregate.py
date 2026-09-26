#!/usr/bin/env python3
"""Aggregate the complete preregistered S1R deterministic root manifold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from holstein_peierls.spin_adapted.s1r_root_manifold import summarize_s1r


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument(
        "--full-pytest-passed",
        action="store_true",
        help="assert that the same workflow commit passed the full repository suite",
    )
    args = parser.parse_args()

    files = sorted(args.input_dir.rglob("s1r-*.json"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    summary = summarize_s1r(
        records,
        full_pytest_passed=bool(args.full_pytest_passed),
    )
    payload = {
        "scope": "S1R deterministic spin-adapted singlet root-manifold aggregate",
        "branch_json_count": len(files),
        "summary": summary,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    singlet_energy = np.full((2, 3, 16), np.nan, dtype=np.float64)
    singlet_valid = np.zeros((2, 3, 16), dtype=np.bool_)
    triplet_energy = np.full((2, 3), np.nan, dtype=np.float64)
    triplet_valid = np.zeros((2, 3), dtype=np.bool_)
    optimizer_index = {"preconditioned": 0, "rprop": 1}
    seed_index = {"onsite": 0, "bond_x": 1, "bond_y": 2}

    for record in records:
        oi = optimizer_index[str(record["structural_optimizer"])]
        si = seed_index[str(record["structural_seed"])]
        completed = bool(record.get("completed", False))
        converged = bool(record.get("converged", False))
        if str(record["multiplicity"]) == "singlet":
            ri = int(record["root_seed_id"])
            if completed and "total_referenced_energy_eV" in record:
                singlet_energy[oi, si, ri] = float(
                    record["total_referenced_energy_eV"]
                )
            singlet_valid[oi, si, ri] = completed and converged
        else:
            if completed and "total_referenced_energy_eV" in record:
                triplet_energy[oi, si] = float(
                    record["total_referenced_energy_eV"]
                )
            triplet_valid[oi, si] = completed and converged

    np.savez_compressed(
        args.arrays,
        singlet_energy_eV=singlet_energy,
        singlet_converged=singlet_valid,
        triplet_energy_eV=triplet_energy,
        triplet_converged=triplet_valid,
    )

    lines = [
        "# S1R deterministic singlet root-manifold audit",
        "",
        f"Primary classification: {'PASS' if summary['s1r_pass'] else 'FAIL'}",
        f"Branch records: {len(files)}/102",
        "",
        "## Locked gates",
        "",
    ]
    for name, passed in summary["gates"].items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")

    lines.extend(["", "## Promoted states", ""])
    for optimizer in ("preconditioned", "rprop"):
        singlet = summary["promoted_singlet"][optimizer]
        triplet = summary["promoted_triplet"][optimizer]
        lines.append(f"- {optimizer} singlet: {singlet}")
        lines.append(f"- {optimizer} triplet: {triplet}")

    lines.extend(
        [
            "",
            f"- promoted singlet |dE|: {summary['promoted_singlet_energy_abs_difference_eV']} eV",
            f"- promoted triplet |dE|: {summary['promoted_triplet_energy_abs_difference_eV']} eV",
            "",
            "## Converged singlet energy clusters",
            "",
        ]
    )
    for optimizer in ("preconditioned", "rprop"):
        lines.append(f"### {optimizer}")
        for cluster in summary["singlet_energy_clusters"][optimizer]:
            lines.append(
                f"- cluster {cluster['cluster_index']}: "
                f"{cluster['minimum_energy_eV']}..{cluster['maximum_energy_eV']} eV; "
                f"count={cluster['count']}"
            )

    lines.extend(
        [
            "",
            "## Interpretation guard",
            "",
            "The 1e-5 eV scale is a numerical root/optimizer-equivalence criterion, not a physical uncertainty. Auxiliary 1e-8 eV seed potentials generate starting orbitals only and are absent from physical energies and forces.",
        ]
    )
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"Wrote {args.output}")
    print(f"Wrote {args.markdown}")
    print(f"Wrote {args.arrays}")


if __name__ == "__main__":
    main()
