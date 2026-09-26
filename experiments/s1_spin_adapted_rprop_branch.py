#!/usr/bin/env python3
"""Run one S1 spin-adapted stationary branch for the RPROP bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from holstein_peierls.spin_adapted import (
    IsotropicControlParameters,
    SpinMultiplicity,
    density_density_control_interaction,
    relax_isotropic_spin_branch,
)


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
        "--seed",
        choices=("onsite", "bond_x", "bond_y"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

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

    started = perf_counter()
    branch = relax_isotropic_spin_branch(
        parameters,
        interaction,
        multiplicity=SpinMultiplicity(args.multiplicity),
        seed=args.seed,
        seed_amplitude=1.0e-3,
        orbital_gradient_tolerance=1.0e-8,
        orbital_max_iterations=800,
        gradient_convergence_criterion=1.0e-6,
        staggered_gap=2.0,
        structural_optimizer=args.structural_optimizer,
    )
    elapsed = perf_counter() - started

    result = branch.result
    delta_density = np.asarray(result.state.excitation_density, dtype=np.float64)
    lattice_vector = np.concatenate(
        [
            np.asarray(result.lattice.u, dtype=np.float64).reshape(-1),
            np.asarray(result.lattice.vx, dtype=np.float64).reshape(-1),
            np.asarray(result.lattice.vy, dtype=np.float64).reshape(-1),
        ]
    )
    payload = {
        "scope": "S1 spin-adapted structural-optimizer bridge",
        "structural_optimizer": args.structural_optimizer,
        "multiplicity": branch.multiplicity.value,
        "seed": branch.seed.value,
        "control": {
            "size": 4,
            "Jx_eV": control.j1,
            "Jy_eV": control.j2,
            "alpha_intra_eV_per_A": control.alpha1,
            "alpha_interx_eV_per_A": control.alpha2,
            "alpha_intery_eV_per_A": control.alpha2,
            "K1_eV_per_A2": control.k1,
            "K2_eV_per_A2": control.k2,
            "onsite_U_eV": 0.525,
            "nearest_neighbor_V_eV": 0.08,
            "staggered_gap_eV": 2.0,
            "seed_amplitude_A": 1.0e-3,
            "lattice_max_iterations": 2000,
            "orbital_gradient_tolerance": 1.0e-8,
            "orbital_max_iterations": 800,
            "structural_gradient_tolerance_eV_per_A": 1.0e-6,
            "structural_update_tolerance_A": parameters.convergence_criterion,
        },
        "converged": bool(result.diagnostics.converged),
        "iterations": int(result.diagnostics.iterations),
        "final_max_update_A": float(result.diagnostics.final_max_update),
        "final_max_gradient_eV_per_A": float(result.diagnostics.final_max_gradient),
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
        "electronic_excitation_eV": float(result.energy.electronic_excitation),
        "lattice_energy_eV": float(result.energy.lattice),
        "total_referenced_energy_eV": float(result.energy.total),
        "particle_number_change": float(result.state.particle_number_change),
        "excitation_density_l2": float(np.linalg.norm(delta_density)),
        "positive_excitation_charge": float(
            np.sum(np.clip(delta_density, 0.0, None))
        ),
        "negative_excitation_charge": float(
            -np.sum(np.clip(delta_density, None, 0.0))
        ),
        "max_abs_u_A": float(np.max(np.abs(result.lattice.u))),
        "max_abs_vx_A": float(np.max(np.abs(result.lattice.vx))),
        "max_abs_vy_A": float(np.max(np.abs(result.lattice.vy))),
        "max_abs_vx_row_mean_A": float(
            np.max(np.abs(np.mean(result.lattice.vx, axis=1)))
        ),
        "max_abs_vy_column_mean_A": float(
            np.max(np.abs(np.mean(result.lattice.vy, axis=0)))
        ),
        "lattice_vector_A": lattice_vector.tolist(),
        "elapsed_seconds": float(elapsed),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "lattice_vector_A"}, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
