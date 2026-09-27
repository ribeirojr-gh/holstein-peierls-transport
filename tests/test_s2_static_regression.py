import copy

from holstein_peierls.s2_static_regression import summarize_s2


def _fixtures():
    polaron = {
        "total_energy_eV": -0.405459057756,
        "sector_pass": True,
        "gates": {
            "all_coordinate_families_converged": True,
            "final_gradient_below_1e-6_eV_per_A": True,
            "charge_density_normalized_within_1e-12": True,
            "energy_matches_reference_within_1e-8_eV": True,
            "ipr_positive_and_finite": True,
        },
    }
    bipolaron = {
        "total_energy_eV": -0.634241741090195,
        "binding_vs_2polaron_eV": 0.0068,
        "P_nn_x": 0.86,
        "mean_r": 1.05,
        "max_delta_tx_over_Jx": 0.16,
        "max_delta_ty_over_Jy": 0.14,
        "final_max_update_A": 9e-9,
        "final_max_gradient_eV_per_A": 3e-8,
        "converged": True,
    }
    exciton = {
        "canonical_best_seed": "frenkel",
        "best_total_energy_eV": -1.695788941148,
        "best_binding_energy_eV": 0.884,
        "branches": [
            {
                "mode": "frenkel",
                "converged": True,
                "total_energy_eV": -1.695788941148,
                "onsite_probability": 0.86,
                "mean_eh_separation_sites": 0.14,
            },
            {
                "mode": "diagonal",
                "converged": True,
                "total_energy_eV": -1.2825,
                "onsite_probability": 0.39,
                "mean_eh_separation_sites": 0.69,
            },
        ],
    }
    spin = [
        {
            "structural_optimizer": "preconditioned",
            "multiplicity": "singlet",
            "structural_seed": "onsite",
            "root_seed_id": 9,
            "completed": True,
            "converged": True,
            "neutral_orbitals_converged": True,
            "excited_orbitals_converged": True,
            "total_referenced_energy_eV": 1.446798604442264,
            "particle_number_change": 0.0,
            "final_max_update_A": 9e-9,
            "final_max_gradient_eV_per_A": 3e-7,
        },
        {
            "structural_optimizer": "rprop",
            "multiplicity": "singlet",
            "structural_seed": "onsite",
            "root_seed_id": 8,
            "completed": True,
            "converged": True,
            "neutral_orbitals_converged": True,
            "excited_orbitals_converged": True,
            "total_referenced_energy_eV": 1.4467971903865933,
            "particle_number_change": 0.0,
            "final_max_update_A": 9e-9,
            "final_max_gradient_eV_per_A": 5e-7,
        },
        {
            "structural_optimizer": "preconditioned",
            "multiplicity": "triplet",
            "structural_seed": "bond_x",
            "root_seed_id": 0,
            "completed": True,
            "converged": True,
            "neutral_orbitals_converged": True,
            "excited_orbitals_converged": True,
            "total_referenced_energy_eV": 1.4467971868828626,
            "particle_number_change": 0.0,
            "final_max_update_A": 9e-9,
            "final_max_gradient_eV_per_A": 3e-7,
        },
        {
            "structural_optimizer": "rprop",
            "multiplicity": "triplet",
            "structural_seed": "onsite",
            "root_seed_id": 0,
            "completed": True,
            "converged": True,
            "neutral_orbitals_converged": True,
            "excited_orbitals_converged": True,
            "total_referenced_energy_eV": 1.4467971868815632,
            "particle_number_change": 0.0,
            "final_max_update_A": 9e-9,
            "final_max_gradient_eV_per_A": 3e-7,
        },
    ]
    return polaron, bipolaron, exciton, spin


def test_s2_passes_complete_reference_fixture():
    polaron, bp, exciton, spin = _fixtures()
    result = summarize_s2(
        polaron=polaron,
        bipolaron=bp,
        exciton=exciton,
        spin_records=spin,
        full_pytest_passed=True,
    )
    assert result["s2_pass"]
    assert all(result["global_gates"].values())


def test_s2_fails_if_bipolaron_binding_sign_regresses():
    polaron, bp, exciton, spin = _fixtures()
    bp["binding_vs_2polaron_eV"] = -0.001
    result = summarize_s2(
        polaron=polaron,
        bipolaron=bp,
        exciton=exciton,
        spin_records=spin,
        full_pytest_passed=True,
    )
    assert not result["bipolaron"]["pass"]
    assert not result["s2_pass"]


def test_s2_fails_if_exciton_is_relabelled_by_wrong_minimum():
    polaron, bp, exciton, spin = _fixtures()
    exciton["canonical_best_seed"] = "diagonal"
    result = summarize_s2(
        polaron=polaron,
        bipolaron=bp,
        exciton=exciton,
        spin_records=spin,
        full_pytest_passed=True,
    )
    assert not result["spin_blind_exciton"]["pass"]


def test_s2_requires_exact_spin_anchor_labels():
    polaron, bp, exciton, spin = _fixtures()
    bad = copy.deepcopy(spin)
    bad[0]["root_seed_id"] = 8
    result = summarize_s2(
        polaron=polaron,
        bipolaron=bp,
        exciton=exciton,
        spin_records=bad,
        full_pytest_passed=True,
    )
    assert not result["spin_adapted"]["pass"]


def test_s2_full_pytest_is_a_locked_global_gate():
    polaron, bp, exciton, spin = _fixtures()
    result = summarize_s2(
        polaron=polaron,
        bipolaron=bp,
        exciton=exciton,
        spin_records=spin,
        full_pytest_passed=False,
    )
    assert not result["global_gates"]["full_repository_pytest_passed"]
    assert not result["s2_pass"]
