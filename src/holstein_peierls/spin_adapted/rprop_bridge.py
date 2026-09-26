"""Locked S1 aggregation for the spin-adapted RPROP bridge."""

from __future__ import annotations

import math

import numpy as np


OPTIMIZERS = ("preconditioned", "rprop")
MULTIPLICITIES = ("singlet", "triplet")
SEEDS = ("onsite", "bond_x", "bond_y")
ENERGY_TOLERANCE_EV = 1.0e-5
GAP_TOLERANCE_EV = 1.0e-5
PARTICLE_NUMBER_TOLERANCE = 1.0e-10


def _expected_keys() -> set[tuple[str, str, str]]:
    return {
        (optimizer, multiplicity, seed)
        for optimizer in OPTIMIZERS
        for multiplicity in MULTIPLICITIES
        for seed in SEEDS
    }


def summarize_s1(records: list[dict]) -> dict:
    """Apply the preregistered S1 stationary-state equivalence criteria."""
    if len(records) != 12:
        raise ValueError("S1 requires exactly 12 optimizer/spin/seed records")

    keys = [
        (
            str(record["structural_optimizer"]),
            str(record["multiplicity"]),
            str(record["seed"]),
        )
        for record in records
    ]
    if set(keys) != _expected_keys() or len(set(keys)) != 12:
        raise ValueError("S1 records must cover the exact 2x2x3 design")

    by_optimizer: dict[str, dict[str, list[dict]]] = {
        optimizer: {spin: [] for spin in MULTIPLICITIES}
        for optimizer in OPTIMIZERS
    }
    for record in records:
        by_optimizer[str(record["structural_optimizer"])][
            str(record["multiplicity"])
        ].append(record)

    all_converged = {
        optimizer: all(
            bool(record["converged"])
            for spin in MULTIPLICITIES
            for record in by_optimizer[optimizer][spin]
        )
        for optimizer in OPTIMIZERS
    }

    best: dict[str, dict[str, dict]] = {
        optimizer: {} for optimizer in OPTIMIZERS
    }
    for optimizer in OPTIMIZERS:
        for spin in MULTIPLICITIES:
            converged = [
                record
                for record in by_optimizer[optimizer][spin]
                if bool(record["converged"])
            ]
            if converged:
                best[optimizer][spin] = min(
                    converged,
                    key=lambda record: float(record["total_referenced_energy_eV"]),
                )

    complete_best = all(
        spin in best[optimizer]
        for optimizer in OPTIMIZERS
        for spin in MULTIPLICITIES
    )

    comparisons = {}
    if complete_best:
        es_r = float(best["rprop"]["singlet"]["total_referenced_energy_eV"])
        et_r = float(best["rprop"]["triplet"]["total_referenced_energy_eV"])
        es_p = float(
            best["preconditioned"]["singlet"]["total_referenced_energy_eV"]
        )
        et_p = float(
            best["preconditioned"]["triplet"]["total_referenced_energy_eV"]
        )
        gap_r = es_r - et_r
        gap_p = es_p - et_p
        comparisons = {
            "rprop_singlet_energy_eV": es_r,
            "preconditioned_singlet_energy_eV": es_p,
            "singlet_energy_abs_difference_eV": abs(es_r - es_p),
            "rprop_triplet_energy_eV": et_r,
            "preconditioned_triplet_energy_eV": et_p,
            "triplet_energy_abs_difference_eV": abs(et_r - et_p),
            "rprop_singlet_minus_triplet_eV": gap_r,
            "preconditioned_singlet_minus_triplet_eV": gap_p,
            "singlet_triplet_gap_abs_difference_eV": abs(gap_r - gap_p),
        }

        for spin in MULTIPLICITIES:
            r = best["rprop"][spin]
            p = best["preconditioned"][spin]
            r_lattice = np.asarray(r["lattice_vector_A"], dtype=np.float64)
            p_lattice = np.asarray(p["lattice_vector_A"], dtype=np.float64)
            delta = r_lattice - p_lattice
            comparisons[f"{spin}_promoted_lattice_max_abs_difference_A"] = float(
                np.max(np.abs(delta))
            )
            comparisons[f"{spin}_promoted_lattice_rms_difference_A"] = float(
                np.sqrt(np.mean(delta * delta))
            )

    rprop_particle_ok = all(
        abs(float(record["particle_number_change"])) < PARTICLE_NUMBER_TOLERANCE
        for spin in MULTIPLICITIES
        for record in by_optimizer["rprop"][spin]
    )
    finite = all(
        all(
            math.isfinite(float(record[field]))
            for field in (
                "total_referenced_energy_eV",
                "electronic_excitation_eV",
                "lattice_energy_eV",
                "final_max_update_A",
                "final_max_gradient_eV_per_A",
                "particle_number_change",
            )
        )
        and np.all(np.isfinite(np.asarray(record["lattice_vector_A"], dtype=float)))
        for record in records
    )

    gates = {
        "all_six_rprop_branches_converged": bool(all_converged["rprop"]),
        "all_six_preconditioned_branches_converged": bool(
            all_converged["preconditioned"]
        ),
        "promoted_singlet_energy_agrees_within_1e-5_eV": bool(
            complete_best
            and comparisons["singlet_energy_abs_difference_eV"]
            <= ENERGY_TOLERANCE_EV
        ),
        "promoted_triplet_energy_agrees_within_1e-5_eV": bool(
            complete_best
            and comparisons["triplet_energy_abs_difference_eV"]
            <= ENERGY_TOLERANCE_EV
        ),
        "promoted_spin_gap_agrees_within_1e-5_eV": bool(
            complete_best
            and comparisons["singlet_triplet_gap_abs_difference_eV"]
            <= GAP_TOLERANCE_EV
        ),
        "all_rprop_particle_number_changes_below_1e-10": bool(rprop_particle_ok),
        "all_records_finite": bool(finite),
    }

    branch_differences = []
    index = {key: record for key, record in zip(keys, records, strict=True)}
    for spin in MULTIPLICITIES:
        for seed in SEEDS:
            rp = index[("rprop", spin, seed)]
            pc = index[("preconditioned", spin, seed)]
            branch_differences.append(
                {
                    "multiplicity": spin,
                    "seed": seed,
                    "rprop_energy_eV": float(rp["total_referenced_energy_eV"]),
                    "preconditioned_energy_eV": float(
                        pc["total_referenced_energy_eV"]
                    ),
                    "absolute_energy_difference_eV": abs(
                        float(rp["total_referenced_energy_eV"])
                        - float(pc["total_referenced_energy_eV"])
                    ),
                    "rprop_iterations": int(rp["iterations"]),
                    "preconditioned_iterations": int(pc["iterations"]),
                }
            )

    return {
        "design_complete": True,
        "best_by_optimizer_and_spin": {
            optimizer: {
                spin: (
                    None
                    if spin not in best[optimizer]
                    else {
                        key: value
                        for key, value in best[optimizer][spin].items()
                        if key != "lattice_vector_A"
                    }
                )
                for spin in MULTIPLICITIES
            }
            for optimizer in OPTIMIZERS
        },
        "promoted_comparisons": comparisons,
        "branch_differences": branch_differences,
        "gates": gates,
        "s1_pass": bool(all(gates.values())),
        "interpretation_guard": {
            "cross_optimizer_tolerance_is_numerical_not_physical_uncertainty": True,
            "checkerboard_gap_is_validation_control_not_material_gap": True,
            "spin_adapted_model_is_not_spin_blind_pair_solver": True,
        },
    }
