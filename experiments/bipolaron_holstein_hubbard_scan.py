"""Exploratory adiabatic Holstein-Hubbard bipolaron scan.

This script is a research diagnostic, not a publication benchmark.  It scans the
on-site repulsion U for a finite periodic lattice, relaxes both onsite and
well-separated seeds, and compares the lowest two-particle energy with twice the
relaxed single-polaron energy for the same Holstein-only parameter set.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from holstein_peierls.parameters import StaticPolaronParameters
from holstein_peierls.polaron import solve_static_polaron
from holstein_peierls.two_particle.bipolaron import relax_static_bipolaron
from holstein_peierls.two_particle.parameters import BipolaronParameters


def _center_position(nx: int, ny: int) -> int:
    return (ny // 2) * nx + (nx // 2) + 1


def _single_polaron_energy(base: StaticPolaronParameters) -> float:
    holstein_only = replace(
        base,
        alpha_interx=0.0,
        alpha_intery=0.0,
        convergence_criterion=1.0e-7,
        max_iterations=1200,
    )
    result = solve_static_polaron(
        holstein_only,
        solver="sparse",
        gradient_mode="optimized",
        legacy_convergence=False,
    )
    return float(result.total_energy)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=6)
    parser.add_argument("--ny", type=int, default=6)
    parser.add_argument(
        "--u-values",
        type=float,
        nargs="+",
        default=[0.0, 0.2, 0.4, 0.55, 0.7, 1.0],
    )
    parser.add_argument("--output", type=Path, default=Path("bipolaron-scan"))
    args = parser.parse_args()

    position = _center_position(args.nx, args.ny)
    single_parameters = StaticPolaronParameters(
        nx=args.nx,
        ny=args.ny,
        polaron_position=position,
    )
    e_polaron = _single_polaron_energy(single_parameters)

    base = BipolaronParameters.from_polaron_parameters(single_parameters)
    base = replace(
        base,
        pair_position=position,
        max_iterations=700,
        convergence_criterion=1.0e-7,
        eigensolver_tolerance=2.0e-10,
    )

    rows: list[dict[str, float | int | str | bool]] = []
    for hubbard_u in args.u_values:
        p = replace(base, hubbard_u=float(hubbard_u))
        candidates = {}
        for initialization in ("onsite", "separated"):
            result = relax_static_bipolaron(p, initialization=initialization)
            candidates[initialization] = result

        best_name = min(candidates, key=lambda name: candidates[name].energy.total)
        best = candidates[best_name]
        binding = 2.0 * e_polaron - best.energy.total
        energy_gap = (
            candidates["separated"].energy.total
            - candidates["onsite"].energy.total
        )

        row = {
            "nx": args.nx,
            "ny": args.ny,
            "U_eV": float(hubbard_u),
            "single_polaron_total_energy_eV": e_polaron,
            "twice_single_polaron_energy_eV": 2.0 * e_polaron,
            "onsite_seed_total_energy_eV": candidates["onsite"].energy.total,
            "separated_seed_total_energy_eV": candidates["separated"].energy.total,
            "separated_minus_onsite_eV": energy_gap,
            "lowest_seed": best_name,
            "lowest_two_particle_energy_eV": best.energy.total,
            "binding_vs_2polarons_eV": binding,
            "onsite_pair_probability": best.onsite_pair_probability,
            "iterations": best.diagnostics.iterations,
            "converged": best.diagnostics.converged,
            "atomic_limit_binding_eV": p.atomic_holstein_pairing_scale - p.hubbard_u,
        }
        rows.append(row)
        print(
            f"U={hubbard_u:5.2f} eV  seed={best_name:9s}  "
            f"E2={best.energy.total:+.9f} eV  "
            f"Ebind={binding:+.6f} eV  P0={best.onsite_pair_probability:.4f}"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "holstein_hubbard_scan.csv"
    json_path = args.output / "holstein_hubbard_scan.json"

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2) + "\n")

    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
