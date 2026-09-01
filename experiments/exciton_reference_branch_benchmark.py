"""Benchmark the generic static exciton against the validated polaron reference.

This is a model/control calculation, not a material parameterization.  By default
it uses the same generic Holstein-Peierls values that have been used to validate
the polaron framework, with equal electron/hole one-particle parameters and a
0.525 eV onsite electron-hole attraction as a numerical control scale.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from holstein_peierls.exciton import (
    DEFAULT_EXCITON_BRANCHES,
    ExcitonParameters,
    binding_energy,
    exciton_observables,
    relax_static_exciton,
)
from holstein_peierls.parameters import StaticPolaronParameters
from holstein_peierls.polaron import solve_static_polaron


def _center_position(nx: int, ny: int) -> int:
    """Return a deterministic one-based site near the cell center."""
    return (ny // 2) * nx + (nx // 2) + 1


def _reference_polaron_parameters(nx: int, ny: int, max_iterations: int) -> StaticPolaronParameters:
    return StaticPolaronParameters(
        nx=nx,
        ny=ny,
        k1=16.51,
        k2=0.51,
        j0x=0.100,
        j0y=0.015,
        alpha_intra=3.0,
        alpha_interx=0.4,
        alpha_intery=0.4,
        polaron_position=_center_position(nx, ny),
        max_iterations=max_iterations,
        convergence_criterion=1.0e-8,
    )


def _reference_exciton_parameters(
    polaron: StaticPolaronParameters,
    onsite_attraction: float,
) -> ExcitonParameters:
    return ExcitonParameters.from_reference_polaron(
        polaron,
        onsite_attraction=onsite_attraction,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=6)
    parser.add_argument("--onsite-attraction", type=float, default=0.525)
    parser.add_argument("--max-iterations", type=int, default=1600)
    parser.add_argument(
        "--output", type=Path, default=Path("exciton-reference-benchmark.json")
    )
    args = parser.parse_args()
    if args.size < 3:
        raise ValueError("benchmark size must be at least 3")

    polaron_parameters = _reference_polaron_parameters(
        args.size, args.size, args.max_iterations
    )
    exciton_parameters = _reference_exciton_parameters(
        polaron_parameters, args.onsite_attraction
    )

    start = perf_counter()
    polaron = solve_static_polaron(
        polaron_parameters,
        solver="sparse",
        gradient_mode="optimized",
        legacy_convergence=False,
    )
    polaron_elapsed = perf_counter() - start

    outcomes: list[dict[str, object]] = []
    for mode in DEFAULT_EXCITON_BRANCHES:
        branch_start = perf_counter()
        result = relax_static_exciton(exciton_parameters, initialization=mode)
        elapsed = perf_counter() - branch_start
        observables = exciton_observables(result.ground_state, exciton_parameters)
        outcomes.append(
            {
                "mode": mode,
                "converged": result.diagnostics.converged,
                "iterations": result.diagnostics.iterations,
                "elapsed_seconds": elapsed,
                "electronic_energy_eV": result.energy.electronic,
                "lattice_energy_eV": result.energy.lattice,
                "total_energy_eV": result.energy.total,
                "final_max_update_A": result.diagnostics.final_max_update,
                "final_max_gradient_eV_per_A": result.diagnostics.final_max_gradient,
                "onsite_probability": observables.onsite_probability,
                "electron_ipr": observables.electron_ipr,
                "hole_ipr": observables.hole_ipr,
                "mean_eh_separation_sites": observables.mean_separation_sites,
                "rms_eh_separation_sites": observables.rms_separation_sites,
                "binding_energy_eV": binding_energy(
                    result.energy.total,
                    polaron.total_energy,
                    polaron.total_energy,
                ),
            }
        )
        print(
            f"{mode:10s} converged={result.diagnostics.converged} "
            f"E={result.energy.total:.12f} eV "
            f"P0={observables.onsite_probability:.6f} "
            f"<r>={observables.mean_separation_sites:.6f} sites"
        )

    converged = [item for item in outcomes if bool(item["converged"])]
    if not converged:
        raise RuntimeError("no exciton branch converged in the reference benchmark")
    best = min(converged, key=lambda item: float(item["total_energy_eV"]))

    payload = {
        "model_status": "generic_reference_control_not_material_fit",
        "size": [args.size, args.size],
        "parameters": {
            "j0x_eV": polaron_parameters.j0x,
            "j0y_eV": polaron_parameters.j0y,
            "alpha_intra_eV_per_A": polaron_parameters.alpha_intra,
            "alpha_interx_eV_per_A": polaron_parameters.alpha_interx,
            "alpha_intery_eV_per_A": polaron_parameters.alpha_intery,
            "k1_eV_per_A2": polaron_parameters.k1,
            "k2_eV_per_A2": polaron_parameters.k2,
            "onsite_eh_attraction_eV": args.onsite_attraction,
        },
        "polaron_reference": {
            "total_energy_eV": polaron.total_energy,
            "formation_energy_eV": polaron.formation_energy,
            "ipr": polaron.ipr,
            "iterations": polaron.diagnostics.iterations,
            "converged_u": polaron.diagnostics.converged_u,
            "converged_vx": polaron.diagnostics.converged_vx,
            "converged_vy": polaron.diagnostics.converged_vy,
            "elapsed_seconds": polaron_elapsed,
        },
        "dissociation_reference_eV": 2.0 * polaron.total_energy,
        "branches": outcomes,
        "best_branch": best["mode"],
        "best_total_energy_eV": best["total_energy_eV"],
        "best_binding_energy_eV": best["binding_energy_eV"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
