#!/usr/bin/env python3
"""Run one preregistered S1R deterministic root-manifold branch."""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy

from holstein_peierls.spin_adapted import (
    IsotropicControlParameters,
    SpinMultiplicity,
    density_density_control_interaction,
    deterministic_root_seed_orbitals,
    isotropic_relaxation_seed,
    relax_isotropic_spin_branch,
)


def _write_failure(
    output: Path,
    *,
    optimizer: str,
    multiplicity: str,
    structural_seed: str,
    root_seed_id: int,
    error: Exception,
) -> None:
    payload = {
        "scope": "S1R deterministic root-manifold branch",
        "structural_optimizer": optimizer,
        "multiplicity": multiplicity,
        "structural_seed": structural_seed,
        "root_seed_id": int(root_seed_id),
        "completed": False,
        "converged": False,
        "execution_error": f"{type(error).__name__}: {error}",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "thread_environment": {
                key: os.environ.get(key)
                for key in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            },
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--structural-optimizer",
        choices=("preconditioned", "rprop"),
        required=True,
    )
    parser.add_argument(
        "--multiplicity",
        choices=("singlet", "triplet"),
        required=True,
    )
    parser.add_argument(
        "--structural-seed",
        choices=("onsite", "bond_x", "bond_y"),
        required=True,
    )
    parser.add_argument("--root-seed-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    args = parser.parse_args()

    if args.multiplicity == "singlet":
        if not 0 <= args.root_seed_id < 16:
            raise ValueError("singlet S1R root-seed ID must be in 0..15")
    elif args.root_seed_id != 0:
        raise ValueError("triplet S1R control uses root-seed ID 0 only")

    try:
        control = IsotropicControlParameters()
        parameters = control.to_polaron_parameters(
            nx=4,
            ny=4,
            max_iterations=2000,
        )
        interaction = density_density_control_interaction(
            parameters,
            onsite_u=0.525,
            nearest_neighbor_v=0.08,
        )
        initial_lattice = isotropic_relaxation_seed(
            parameters,
            args.structural_seed,
            amplitude=1.0e-3,
        )
        _, initial_auxiliary_gap = deterministic_root_seed_orbitals(
            initial_lattice,
            parameters,
            staggered_gap=2.0,
            root_seed_id=args.root_seed_id,
            seed_amplitude_eV=1.0e-8,
            minimum_auxiliary_gap_eV=1.0e-12,
        )

        started = perf_counter()
        branch = relax_isotropic_spin_branch(
            parameters,
            interaction,
            multiplicity=SpinMultiplicity(args.multiplicity),
            seed=args.structural_seed,
            seed_amplitude=1.0e-3,
            orbital_gradient_tolerance=1.0e-8,
            orbital_max_iterations=800,
            gradient_convergence_criterion=1.0e-6,
            staggered_gap=2.0,
            structural_optimizer=args.structural_optimizer,
            root_seed_id=args.root_seed_id,
            root_seed_amplitude_eV=1.0e-8,
        )
        elapsed = perf_counter() - started
        result = branch.result

        excitation_density = np.asarray(
            result.state.excitation_density, dtype=np.float64
        ).reshape(4, 4)
        excitation_rdm = np.asarray(
            np.real_if_close(result.state.excitation_rdm),
            dtype=np.float64,
        )
        u = np.asarray(result.lattice.u, dtype=np.float64)
        vx = np.asarray(result.lattice.vx, dtype=np.float64)
        vy = np.asarray(result.lattice.vy, dtype=np.float64)

        payload = {
            "scope": "S1R deterministic root-manifold branch",
            "structural_optimizer": args.structural_optimizer,
            "multiplicity": args.multiplicity,
            "structural_seed": args.structural_seed,
            "root_seed_id": int(args.root_seed_id),
            "completed": True,
            "initial_auxiliary_gap_eV": float(initial_auxiliary_gap),
            "auxiliary_seed_amplitude_eV": 1.0e-8,
            "converged": bool(result.diagnostics.converged),
            "iterations": int(result.diagnostics.iterations),
            "final_max_update_A": float(result.diagnostics.final_max_update),
            "final_max_gradient_eV_per_A": float(
                result.diagnostics.final_max_gradient
            ),
            "neutral_orbitals_converged": bool(
                result.diagnostics.neutral_orbitals_converged
            ),
            "excited_orbitals_converged": bool(
                result.diagnostics.excited_orbitals_converged
            ),
            "neutral_orbital_max_gradient": float(
                result.state.neutral.diagnostics.final_max_gradient
            ),
            "excited_orbital_max_gradient": float(
                result.state.excited.diagnostics.final_max_gradient
            ),
            "electronic_excitation_eV": float(
                result.energy.electronic_excitation
            ),
            "lattice_energy_eV": float(result.energy.lattice),
            "total_referenced_energy_eV": float(result.energy.total),
            "particle_number_change": float(
                result.state.particle_number_change
            ),
            "excitation_density_l2": float(
                np.linalg.norm(excitation_density)
            ),
            "excitation_rdm_frobenius": float(
                np.linalg.norm(excitation_rdm)
            ),
            "positive_excitation_charge": float(
                np.sum(np.clip(excitation_density, 0.0, None))
            ),
            "negative_excitation_charge": float(
                -np.sum(np.clip(excitation_density, None, 0.0))
            ),
            "max_abs_u_A": float(np.max(np.abs(u))),
            "max_abs_vx_A": float(np.max(np.abs(vx))),
            "max_abs_vy_A": float(np.max(np.abs(vy))),
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "thread_environment": {
                    key: os.environ.get(key)
                    for key in (
                        "OPENBLAS_NUM_THREADS",
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                    )
                },
            },
            "elapsed_seconds": float(elapsed),
            "excitation_density": excitation_density.tolist(),
            "excitation_rdm": excitation_rdm.tolist(),
            "lattice_u_A": u.tolist(),
            "lattice_vx_A": vx.tolist(),
            "lattice_vy_A": vy.tolist(),
        }

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        np.savez_compressed(
            args.arrays,
            excitation_density=excitation_density,
            excitation_rdm=excitation_rdm,
            lattice_u_A=u,
            lattice_vx_A=vx,
            lattice_vy_A=vy,
        )
        print(
            json.dumps(
                {
                    key: value
                    for key, value in payload.items()
                    if key
                    not in {
                        "excitation_density",
                        "excitation_rdm",
                        "lattice_u_A",
                        "lattice_vx_A",
                        "lattice_vy_A",
                    }
                },
                indent=2,
            )
        )
    except Exception as error:
        _write_failure(
            args.output,
            optimizer=args.structural_optimizer,
            multiplicity=args.multiplicity,
            structural_seed=args.structural_seed,
            root_seed_id=args.root_seed_id,
            error=error,
        )
        raise


if __name__ == "__main__":
    main()
