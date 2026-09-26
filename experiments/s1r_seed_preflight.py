#!/usr/bin/env python3
"""S1R deterministic root-seed uniqueness preflight on all initial geometries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from holstein_peierls.spin_adapted import (
    IsotropicControlParameters,
    deterministic_root_seed_orbitals,
    isotropic_relaxation_seed,
)


STRUCTURAL_SEEDS = ("onsite", "bond_x", "bond_y")
ROOT_SEED_IDS = tuple(range(16))
MINIMUM_ALLOWED_GAP_EV = 1.0e-12


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    parameters = IsotropicControlParameters().to_polaron_parameters(nx=4, ny=4)
    records: list[dict] = []
    all_pass = True

    for structural_seed in STRUCTURAL_SEEDS:
        lattice = isotropic_relaxation_seed(
            parameters, structural_seed, amplitude=1.0e-3
        )
        for root_seed_id in ROOT_SEED_IDS:
            orbitals, minimum_gap = deterministic_root_seed_orbitals(
                lattice,
                parameters,
                staggered_gap=2.0,
                root_seed_id=root_seed_id,
                seed_amplitude_eV=1.0e-8,
                minimum_auxiliary_gap_eV=MINIMUM_ALLOWED_GAP_EV,
            )
            orthogonality_error = float(
                np.max(
                    np.abs(
                        orbitals.T @ orbitals
                        - np.eye(parameters.n_sites, dtype=np.float64)
                    )
                )
            )
            passed = bool(
                minimum_gap > MINIMUM_ALLOWED_GAP_EV
                and orthogonality_error < 2.0e-12
                and np.all(np.isfinite(orbitals))
            )
            all_pass = all_pass and passed
            records.append(
                {
                    "structural_seed": structural_seed,
                    "root_seed_id": root_seed_id,
                    "minimum_auxiliary_eigenvalue_gap_eV": float(minimum_gap),
                    "maximum_orthogonality_error": orthogonality_error,
                    "pass": passed,
                }
            )

    payload = {
        "scope": "S1R deterministic root-seed preflight",
        "fixed_seed_amplitude_eV": 1.0e-8,
        "minimum_allowed_auxiliary_gap_eV": MINIMUM_ALLOWED_GAP_EV,
        "record_count": len(records),
        "all_48_initial_seed_checks_pass": bool(all_pass and len(records) == 48),
        "minimum_observed_auxiliary_gap_eV": float(
            min(record["minimum_auxiliary_eigenvalue_gap_eV"] for record in records)
        ),
        "maximum_observed_orthogonality_error": float(
            max(record["maximum_orthogonality_error"] for record in records)
        ),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# S1R deterministic root-seed preflight",
        "",
        f"- fixed auxiliary amplitude: 1e-8 eV",
        f"- required minimum auxiliary gap: > {MINIMUM_ALLOWED_GAP_EV} eV",
        f"- checks: {len(records)}",
        f"- minimum observed gap: {payload['minimum_observed_auxiliary_gap_eV']} eV",
        f"- maximum orthogonality error: {payload['maximum_observed_orthogonality_error']}",
        f"- overall: {'PASS' if payload['all_48_initial_seed_checks_pass'] else 'FAIL'}",
    ]
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))

    if not payload["all_48_initial_seed_checks_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
