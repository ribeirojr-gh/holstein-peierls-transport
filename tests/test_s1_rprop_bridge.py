import copy

import pytest

from holstein_peierls.spin_adapted.rprop_bridge import summarize_s1


def _record(optimizer, spin, seed, energy, *, converged=True, particle=0.0):
    base = 1.0 if spin == "singlet" else 0.9988
    lattice = [0.0] * 48
    if seed == "onsite":
        lattice[0] = 1.0e-3
    return {
        "structural_optimizer": optimizer,
        "multiplicity": spin,
        "seed": seed,
        "converged": converged,
        "iterations": 100,
        "final_max_update_A": 1.0e-9,
        "final_max_gradient_eV_per_A": 1.0e-7,
        "particle_number_change": particle,
        "total_referenced_energy_eV": energy,
        "electronic_excitation_eV": base,
        "lattice_energy_eV": energy - base,
        "lattice_vector_A": lattice,
    }


def _complete_records(delta=0.0):
    records = []
    for optimizer in ("preconditioned", "rprop"):
        shift = delta if optimizer == "rprop" else 0.0
        for spin, base in (("singlet", 1.4480), ("triplet", 1.4468)):
            for i, seed in enumerate(("onsite", "bond_x", "bond_y")):
                records.append(
                    _record(
                        optimizer,
                        spin,
                        seed,
                        base + 0.01 * i + shift,
                    )
                )
    return records


def test_s1_passes_when_promoted_states_agree():
    result = summarize_s1(_complete_records(delta=2.0e-6))
    assert result["s1_pass"]
    assert all(result["gates"].values())


def test_s1_fails_when_promoted_energy_exceeds_tolerance():
    result = summarize_s1(_complete_records(delta=2.0e-5))
    assert not result["gates"]["promoted_singlet_energy_agrees_within_1e-5_eV"]
    assert not result["gates"]["promoted_triplet_energy_agrees_within_1e-5_eV"]
    assert not result["s1_pass"]


def test_s1_fails_if_any_rprop_branch_is_unconverged():
    records = _complete_records()
    item = next(
        record
        for record in records
        if record["structural_optimizer"] == "rprop"
        and record["multiplicity"] == "singlet"
        and record["seed"] == "bond_y"
    )
    item["converged"] = False
    result = summarize_s1(records)
    assert not result["gates"]["all_six_rprop_branches_converged"]
    assert not result["s1_pass"]


def test_s1_fails_particle_number_gate():
    records = _complete_records()
    item = next(
        record
        for record in records
        if record["structural_optimizer"] == "rprop"
    )
    item["particle_number_change"] = 2.0e-10
    result = summarize_s1(records)
    assert not result["gates"]["all_rprop_particle_number_changes_below_1e-10"]


def test_s1_requires_exact_design():
    records = _complete_records()
    with pytest.raises(ValueError):
        summarize_s1(records[:-1])

    bad = copy.deepcopy(records)
    bad[-1]["seed"] = "onsite"
    with pytest.raises(ValueError):
        summarize_s1(bad)
