"""Compare reference and optimized full 20x20 static-polaron relaxations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from holstein_peierls.parameters import StaticPolaronParameters
from holstein_peierls.polaron import PolaronResult, solve_static_polaron


CONFIGURATIONS = (
    ("reference_dense_full", "dense_full", "reference", True),
    ("optimized_dense_lowest", "dense_lowest", "optimized", True),
    ("optimized_sparse", "sparse", "optimized", True),
    ("optimized_sparse_modern_stop", "sparse", "optimized", False),
)


def _run(
    parameters: StaticPolaronParameters,
    solver: str,
    gradient_mode: str,
    legacy_convergence: bool,
) -> tuple[PolaronResult, float]:
    start = perf_counter()
    result = solve_static_polaron(
        parameters,
        solver=solver,
        gradient_mode=gradient_mode,
        legacy_convergence=legacy_convergence,
    )
    return result, perf_counter() - start


def _summary(
    name: str,
    solver: str,
    gradient_mode: str,
    legacy_convergence: bool,
    result: PolaronResult,
    elapsed: float,
) -> dict[str, object]:
    return {
        "name": name,
        "solver": solver,
        "gradient_mode": gradient_mode,
        "legacy_convergence": legacy_convergence,
        "elapsed_seconds": elapsed,
        "iterations": result.diagnostics.iterations,
        "total_energy_eV": result.total_energy,
        "formation_energy_eV": result.formation_energy,
        "electronic_energy_eV": result.electronic_energy,
        "ipr": result.ipr,
        "legacy_ipr": result.legacy_ipr,
        "maximum_charge_density": float(np.max(result.charge_density)),
        "charge_sum": float(np.sum(result.charge_density)),
        "converged_u": result.diagnostics.converged_u,
        "converged_vx": result.diagnostics.converged_vx,
        "converged_vy": result.diagnostics.converged_vy,
    }


def _comparison(reference: PolaronResult, candidate: PolaronResult) -> dict[str, float]:
    return {
        "absolute_total_energy_difference_eV": abs(
            candidate.total_energy - reference.total_energy
        ),
        "absolute_formation_energy_difference_eV": abs(
            candidate.formation_energy - reference.formation_energy
        ),
        "maximum_charge_density_difference": float(
            np.max(np.abs(candidate.charge_density - reference.charge_density))
        ),
        "maximum_u_difference": float(np.max(np.abs(candidate.state.u - reference.state.u))),
        "maximum_vx_difference": float(
            np.max(np.abs(candidate.state.vx - reference.state.vx))
        ),
        "maximum_vy_difference": float(
            np.max(np.abs(candidate.state.vy - reference.state.vy))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parameters",
        type=Path,
        default=Path("examples/static_polaron/parameters1.inc"),
    )
    parser.add_argument("--output", type=Path, default=Path("validation-results"))
    args = parser.parse_args()

    parameters = StaticPolaronParameters.from_legacy_include(args.parameters)
    results: dict[str, PolaronResult] = {}
    summaries: list[dict[str, object]] = []

    for name, solver, gradient_mode, legacy_convergence in CONFIGURATIONS:
        result, elapsed = _run(
            parameters,
            solver=solver,
            gradient_mode=gradient_mode,
            legacy_convergence=legacy_convergence,
        )
        results[name] = result
        summaries.append(
            _summary(
                name,
                solver,
                gradient_mode,
                legacy_convergence,
                result,
                elapsed,
            )
        )
        print(
            f"{name}: iterations={result.diagnostics.iterations}, "
            f"E_form={result.formation_energy:.15e} eV, "
            f"elapsed={elapsed:.3f} s"
        )

    reference = results["reference_dense_full"]
    comparisons = {
        name: _comparison(reference, result)
        for name, result in results.items()
        if name != "reference_dense_full"
    }

    expected_legacy_formation_energy = 0.6354590577418
    legacy_error = abs(reference.formation_energy - expected_legacy_formation_energy)
    if legacy_error > 5.0e-10:
        raise RuntimeError(
            "20x20 legacy reference changed unexpectedly: "
            f"|dE|={legacy_error:.3e} eV"
        )

    for summary in summaries:
        if abs(float(summary["charge_sum"]) - 1.0) > 1.0e-10:
            raise RuntimeError(f"charge normalization failed for {summary['name']}")

    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "parameters": str(args.parameters),
        "expected_legacy_formation_energy_eV": expected_legacy_formation_energy,
        "runs": summaries,
        "comparisons_to_reference": comparisons,
    }
    output_path = args.output / "static_20x20.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"validation JSON: {output_path}")


if __name__ == "__main__":
    main()
