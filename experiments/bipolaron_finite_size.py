"""Finite-size diagnostic for weakly bound Holstein-Hubbard bipolaron states.

The purpose of this experiment is to distinguish a genuine intersite/extended
bound state from a finite-periodic-cell artifact.  It evaluates selected U
values for several square lattices and records pair-separation observables.
Results are exploratory until size convergence is demonstrated.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

from holstein_peierls.parameters import StaticPolaronParameters
from holstein_peierls.polaron import solve_static_polaron
from holstein_peierls.two_particle.bipolaron import relax_static_bipolaron
from holstein_peierls.two_particle.observables import pair_observables
from holstein_peierls.two_particle.parameters import BipolaronParameters


def center_position(size: int) -> int:
    return (size // 2) * size + (size // 2) + 1


def single_polaron_energy(size: int) -> float:
    p = StaticPolaronParameters(
        nx=size,
        ny=size,
        polaron_position=center_position(size),
        alpha_interx=0.0,
        alpha_intery=0.0,
        max_iterations=1200,
        convergence_criterion=1.0e-7,
    )
    result = solve_static_polaron(
        p,
        solver="sparse",
        gradient_mode="optimized",
        legacy_convergence=False,
    )
    return float(result.total_energy)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[6, 8, 10])
    parser.add_argument("--u-values", nargs="+", type=float, default=[0.55, 0.70, 1.00])
    parser.add_argument("--output", type=Path, default=Path("bipolaron-finite-size"))
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    radial_records: list[dict[str, object]] = []

    for size in args.sizes:
        e1 = single_polaron_energy(size)
        single = StaticPolaronParameters(
            nx=size,
            ny=size,
            polaron_position=center_position(size),
        )
        base = BipolaronParameters.from_polaron_parameters(single)
        base = replace(
            base,
            pair_position=center_position(size),
            max_iterations=800,
            convergence_criterion=1.0e-7,
            eigensolver_tolerance=2.0e-10,
        )

        for hubbard_u in args.u_values:
            p = replace(base, hubbard_u=float(hubbard_u))
            candidates = {
                seed: relax_static_bipolaron(p, initialization=seed)
                for seed in ("onsite", "separated")
            }
            best_seed = min(candidates, key=lambda seed: candidates[seed].energy.total)
            best = candidates[best_seed]
            obs = pair_observables(best.ground_state, p)
            binding = 2.0 * e1 - best.energy.total

            rows.append(
                {
                    "size": size,
                    "n_sites": size * size,
                    "U_eV": float(hubbard_u),
                    "single_polaron_energy_eV": e1,
                    "two_polaron_threshold_eV": 2.0 * e1,
                    "best_seed": best_seed,
                    "two_particle_energy_eV": best.energy.total,
                    "binding_vs_2polarons_eV": binding,
                    "P_onsite": obs.onsite_probability,
                    "P_nearest_neighbour": obs.nearest_neighbour_probability,
                    "mean_separation_sites": obs.mean_separation,
                    "rms_separation_sites": obs.rms_separation,
                    "one_body_ipr": obs.one_body_ipr,
                    "iterations": best.diagnostics.iterations,
                    "converged": best.diagnostics.converged,
                }
            )
            for r2, probability in obs.radial_probability_by_r2.items():
                radial_records.append(
                    {
                        "size": size,
                        "U_eV": float(hubbard_u),
                        "best_seed": best_seed,
                        "r2": r2,
                        "probability": probability,
                    }
                )

            print(
                f"L={size:2d} U={hubbard_u:4.2f} seed={best_seed:9s} "
                f"Ebind={binding:+.8f} eV  P0={obs.onsite_probability:.5f} "
                f"PNN={obs.nearest_neighbour_probability:.5f} "
                f"<r>={obs.mean_separation:.4f}"
            )

    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "finite_size_summary.csv"
    radial_path = args.output / "finite_size_radial.csv"
    json_path = args.output / "finite_size.json"

    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with radial_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(radial_records[0].keys()))
        writer.writeheader()
        writer.writerows(radial_records)

    json_path.write_text(
        json.dumps({"summary": rows, "radial": radial_records}, indent=2) + "\n"
    )
    print(f"Summary: {summary_path}")
    print(f"Radial: {radial_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
