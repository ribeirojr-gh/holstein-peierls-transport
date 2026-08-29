"""Command-line interface for the static polaron solver."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import __version__
from .io import read_legacy_lattice, write_legacy_lattice, write_run_metadata, write_static_fields
from .parameters import StaticPolaronParameters
from .polaron import solve_static_polaron


def main() -> None:
    parser = argparse.ArgumentParser(description="Relax a 2D Holstein-Peierls polaron")
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("run-static"))
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--solver", choices=("dense_full", "dense_lowest", "sparse"), default="dense_lowest")
    parser.add_argument(
        "--gradient",
        choices=("optimized", "reference"),
        default="optimized",
        help="Use the O(N) optimized gradient or the historical density-matrix path.",
    )
    parser.add_argument("--legacy-convergence", action="store_true")
    parser.add_argument("--no-legacy-seed", action="store_true")
    args = parser.parse_args()

    parameters = StaticPolaronParameters.from_legacy_include(args.parameters)
    initial_state = None
    if args.input_dir is not None:
        initial_state = read_legacy_lattice(args.input_dir, parameters, prefix="in")
    elif parameters.readinput.lower() == "y":
        parser.error("parameters1.inc requests input lattice files; provide --input-dir")

    result = solve_static_polaron(
        parameters,
        initial_state=initial_state,
        solver=args.solver,
        gradient_mode=args.gradient,
        legacy_convergence=args.legacy_convergence,
        apply_legacy_seed=not args.no_legacy_seed,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    write_legacy_lattice(args.output, result.state, prefix="out")
    write_static_fields(args.output, result.state, result.charge_density)
    (args.output / "pfe.dat").write_text(f"{result.formation_energy:.17e}\n")
    write_run_metadata(args.output, {
        "program": "holstein-peierls-transport",
        "version": __version__,
        "solver": args.solver,
        "gradient_mode": args.gradient,
        "legacy_convergence": args.legacy_convergence,
        "parameters": parameters.to_dict(),
        "results": {
            "total_energy_eV": result.total_energy,
            "electronic_energy_eV": result.electronic_energy,
            "polaron_formation_energy_eV": result.formation_energy,
            "ipr": result.ipr,
            "legacy_ipr": result.legacy_ipr,
            "iterations": result.diagnostics.iterations,
            "converged_u": result.diagnostics.converged_u,
            "converged_vx": result.diagnostics.converged_vx,
            "converged_vy": result.diagnostics.converged_vy,
        },
    })

    print(f"iterations: {result.diagnostics.iterations}")
    print(f"total energy: {result.total_energy:.12e} eV")
    print(f"formation energy: {result.formation_energy:.12e} eV")
    print(f"IPR: {result.ipr:.12e}")
    print(f"legacy IPR: {result.legacy_ipr:.12e}")


if __name__ == "__main__":
    main()
