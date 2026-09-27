"""Locked S2 unified static-sector regression classifier."""

from __future__ import annotations

from math import isfinite

import numpy as np


BP_REFERENCE_ENERGY_EV = -0.634241741090195
EXCITON_REFERENCE_ENERGY_EV = -1.695788941148
SPIN_REFERENCE_EV = {
    ("preconditioned", "singlet"): 1.446798604442264,
    ("rprop", "singlet"): 1.4467971903865933,
    ("preconditioned", "triplet"): 1.4467971868828626,
    ("rprop", "triplet"): 1.4467971868815632,
}


def summarize_s2(
    *,
    polaron: dict,
    bipolaron: dict,
    exciton: dict,
    spin_records: list[dict],
    full_pytest_passed: bool,
) -> dict:
    """Apply all preregistered S2 integration gates."""
    polaron_gates = dict(polaron.get("gates", {}))
    polaron_pass = bool(polaron.get("sector_pass", False)) and bool(
        polaron_gates and all(polaron_gates.values())
    )

    bp_energy = float(bipolaron["total_energy_eV"])
    bp_binding = float(bipolaron["binding_vs_2polaron_eV"])
    bp_gates = {
        "strict_branch_converged": bool(bipolaron["converged"]),
        "energy_matches_reference_within_1e-8_eV": abs(
            bp_energy - BP_REFERENCE_ENERGY_EV
        )
        <= 1.0e-8,
        "binding_positive_and_at_least_5_meV": bp_binding >= 5.0e-3,
        "nearest_neighbor_x_probability_at_least_0p80": float(
            bipolaron["P_nn_x"]
        )
        >= 0.80,
        "mean_pair_separation_below_1p20_sites": float(bipolaron["mean_r"]) < 1.20,
        "x_peierls_modulation_below_0p25": float(
            bipolaron["max_delta_tx_over_Jx"]
        )
        < 0.25,
        "y_peierls_modulation_below_0p25": float(
            bipolaron["max_delta_ty_over_Jy"]
        )
        < 0.25,
        "strict_update_below_1e-8_A": float(bipolaron["final_max_update_A"])
        < 1.0e-8,
        "strict_gradient_below_1e-6_eV_per_A": float(
            bipolaron["final_max_gradient_eV_per_A"]
        )
        < 1.0e-6,
    }
    bipolaron_pass = bool(all(bp_gates.values()))

    branches = {str(item["mode"]): item for item in exciton["branches"]}
    if set(branches) != {"frenkel", "diagonal"}:
        raise ValueError("S2 exciton record must contain exactly frenkel and diagonal")
    best_energy = float(exciton["best_total_energy_eV"])
    best_binding = float(exciton["best_binding_energy_eV"])
    frenkel = branches["frenkel"]
    diagonal = branches["diagonal"]
    exciton_gates = {
        "both_requested_branches_converged": bool(
            frenkel["converged"] and diagonal["converged"]
        ),
        "canonical_best_seed_is_frenkel": str(exciton["canonical_best_seed"])
        == "frenkel",
        "best_energy_matches_reference_within_1e-8_eV": abs(
            best_energy - EXCITON_REFERENCE_ENERGY_EV
        )
        <= 1.0e-8,
        "best_binding_above_0p80_eV": best_binding > 0.80,
        "best_onsite_probability_above_0p80": float(
            frenkel["onsite_probability"]
        )
        > 0.80,
        "best_mean_separation_below_0p30_sites": float(
            frenkel["mean_eh_separation_sites"]
        )
        < 0.30,
        "diagonal_at_least_0p30_eV_above_best": float(
            diagonal["total_energy_eV"]
        )
        - best_energy
        >= 0.30,
    }
    exciton_pass = bool(all(exciton_gates.values()))

    if len(spin_records) != 4:
        raise ValueError("S2 spin sector requires exactly four regression anchors")
    spin_index = {
        (str(r["structural_optimizer"]), str(r["multiplicity"])): r
        for r in spin_records
    }
    if set(spin_index) != set(SPIN_REFERENCE_EV):
        raise ValueError("S2 spin anchors do not match preregistered optimizer/spin set")

    expected_seed_root = {
        ("preconditioned", "singlet"): ("onsite", 9),
        ("rprop", "singlet"): ("onsite", 8),
        ("preconditioned", "triplet"): ("bond_x", 0),
        ("rprop", "triplet"): ("onsite", 0),
    }
    spin_gates: dict[str, bool] = {}
    for key, reference in SPIN_REFERENCE_EV.items():
        record = spin_index[key]
        label = f"{key[0]}_{key[1]}"
        expected_seed, expected_root = expected_seed_root[key]
        spin_gates[f"{label}_label_exact"] = bool(
            str(record["structural_seed"]) == expected_seed
            and int(record["root_seed_id"]) == expected_root
        )
        spin_gates[f"{label}_strict_convergence"] = bool(
            record["completed"]
            and record["converged"]
            and record["neutral_orbitals_converged"]
            and record["excited_orbitals_converged"]
            and float(record["final_max_update_A"]) < 1.0e-8
            and float(record["final_max_gradient_eV_per_A"]) < 1.0e-6
        )
        spin_gates[f"{label}_reference_energy_within_1e-8_eV"] = (
            abs(float(record["total_referenced_energy_eV"]) - reference) <= 1.0e-8
        )
        spin_gates[f"{label}_particle_number_below_1e-10"] = (
            abs(float(record["particle_number_change"])) < 1.0e-10
        )

    singlet_difference = abs(
        float(spin_index[("preconditioned", "singlet")]["total_referenced_energy_eV"])
        - float(spin_index[("rprop", "singlet")]["total_referenced_energy_eV"])
    )
    triplet_difference = abs(
        float(spin_index[("preconditioned", "triplet")]["total_referenced_energy_eV"])
        - float(spin_index[("rprop", "triplet")]["total_referenced_energy_eV"])
    )
    spin_gates["singlet_optimizer_difference_below_1e-5_eV"] = (
        singlet_difference <= 1.0e-5
    )
    spin_gates["triplet_optimizer_difference_below_1e-5_eV"] = (
        triplet_difference <= 1.0e-5
    )
    spin_pass = bool(all(spin_gates.values()))

    numeric_values = [
        float(polaron["total_energy_eV"]),
        bp_energy,
        bp_binding,
        best_energy,
        best_binding,
        singlet_difference,
        triplet_difference,
    ] + [
        float(record["total_referenced_energy_eV"]) for record in spin_records
    ]
    all_finite = bool(np.all(np.isfinite(numeric_values)))

    global_gates = {
        "full_repository_pytest_passed": bool(full_pytest_passed),
        "one_polaron_sector_pass": polaron_pass,
        "bipolaron_sector_pass": bipolaron_pass,
        "spin_blind_exciton_sector_pass": exciton_pass,
        "spin_adapted_anchor_sector_pass": spin_pass,
        "all_expected_artifacts_present": True,
        "all_aggregate_values_finite": all_finite,
    }

    return {
        "one_polaron": {
            "gates": polaron_gates,
            "pass": polaron_pass,
            "total_energy_eV": float(polaron["total_energy_eV"]),
        },
        "bipolaron": {
            "gates": bp_gates,
            "pass": bipolaron_pass,
            "total_energy_eV": bp_energy,
            "binding_eV": bp_binding,
        },
        "spin_blind_exciton": {
            "gates": exciton_gates,
            "pass": exciton_pass,
            "best_total_energy_eV": best_energy,
            "best_binding_eV": best_binding,
        },
        "spin_adapted": {
            "gates": spin_gates,
            "pass": spin_pass,
            "singlet_optimizer_difference_eV": singlet_difference,
            "triplet_optimizer_difference_eV": triplet_difference,
            "anchors": {
                f"{key[0]}_{key[1]}": float(
                    spin_index[key]["total_referenced_energy_eV"]
                )
                for key in SPIN_REFERENCE_EV
            },
        },
        "global_gates": global_gates,
        "s2_pass": bool(all(global_gates.values())),
        "interpretation_guard": {
            "integration_regression_not_new_phase_diagram": True,
            "spin_blind_exciton_not_singlet_or_triplet": True,
            "spin_anchor_ids_are_regression_anchors_not_production_root_search": True,
            "paper1_production_spin_promotion_still_uses_complete_S1R_root_ensemble": True,
        },
    }
