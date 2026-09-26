import copy

import numpy as np

from holstein_peierls.spin_adapted.s1r_root_manifold import summarize_s1r


def _environment():
    return {
        "python": "3.12.14",
        "numpy": "2.5.3",
        "scipy": "1.18.1",
        "thread_environment": {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
    }


def _arrays(offset=0.0):
    density = np.zeros((4, 4), dtype=float)
    density[0, 0] = 1.0 + offset
    density[0, 1] = -1.0 - offset
    rdm = np.zeros((16, 16), dtype=float)
    np.fill_diagonal(rdm, density.reshape(-1))
    u = 0.01 * density
    vx = np.zeros((4, 4), dtype=float)
    vy = np.zeros((4, 4), dtype=float)
    return density, rdm, u, vx, vy


def _record(optimizer, multiplicity, seed, root_id, energy, *, converged=True):
    density, rdm, u, vx, vy = _arrays()
    return {
        "structural_optimizer": optimizer,
        "multiplicity": multiplicity,
        "structural_seed": seed,
        "root_seed_id": root_id,
        "completed": True,
        "initial_auxiliary_gap_eV": 2.0e-10,
        "converged": converged,
        "neutral_orbitals_converged": True,
        "excited_orbitals_converged": True,
        "final_max_update_A": 1.0e-9,
        "final_max_gradient_eV_per_A": 1.0e-7,
        "total_referenced_energy_eV": energy,
        "electronic_excitation_eV": energy - 0.1,
        "lattice_energy_eV": 0.1,
        "particle_number_change": 0.0,
        "environment": _environment(),
        "excitation_density": density.tolist(),
        "excitation_rdm": rdm.tolist(),
        "lattice_u_A": u.tolist(),
        "lattice_vx_A": vx.tolist(),
        "lattice_vy_A": vy.tolist(),
    }


def _complete_records(rprop_shift=2.0e-6):
    records = []
    for optimizer in ("preconditioned", "rprop"):
        shift = rprop_shift if optimizer == "rprop" else 0.0
        for seed_index, seed in enumerate(("onsite", "bond_x", "bond_y")):
            for root_id in range(16):
                energy = 1.40 + 1.0e-4 * root_id + 2.0e-3 * seed_index + shift
                records.append(
                    _record(optimizer, "singlet", seed, root_id, energy)
                )
            records.append(
                _record(
                    optimizer,
                    "triplet",
                    seed,
                    0,
                    1.39 + 2.0e-3 * seed_index + shift,
                )
            )
    return records


def test_s1r_complete_synthetic_design_passes_locked_gates():
    result = summarize_s1r(_complete_records(), full_pytest_passed=True)
    assert result["record_count"] == 102
    assert result["strict_valid_singlet_counts"] == {
        "preconditioned": 48,
        "rprop": 48,
    }
    assert result["strict_valid_triplet_counts"] == {
        "preconditioned": 3,
        "rprop": 3,
    }
    assert result["s1r_pass"]
    assert all(result["gates"].values())


def test_s1r_requires_at_least_45_singlets_per_optimizer():
    records = _complete_records()
    invalidated = 0
    for record in records:
        if (
            record["structural_optimizer"] == "rprop"
            and record["multiplicity"] == "singlet"
            and invalidated < 4
        ):
            record["converged"] = False
            invalidated += 1
    result = summarize_s1r(records, full_pytest_passed=True)
    assert result["strict_valid_singlet_counts"]["rprop"] == 44
    assert not result["gates"]["at_least_45_of_48_rprop_singlets_converged"]
    assert not result["s1r_pass"]


def test_s1r_does_not_relax_optimizer_energy_tolerance():
    result = summarize_s1r(
        _complete_records(rprop_shift=2.0e-5),
        full_pytest_passed=True,
    )
    assert not result["gates"]["promoted_singlet_minima_agree_within_1e-5_eV"]
    assert not result["gates"]["promoted_triplet_minima_agree_within_1e-5_eV"]
    assert not result["s1r_pass"]


def test_s1r_full_pytest_is_a_locked_gate():
    result = summarize_s1r(_complete_records(), full_pytest_passed=False)
    assert not result["gates"]["full_repository_pytest_passed"]
    assert not result["s1r_pass"]


def test_s1r_environment_must_be_exactly_pinned():
    records = _complete_records()
    records[0]["environment"]["numpy"] = "2.5.2"
    result = summarize_s1r(records, full_pytest_passed=True)
    assert not result["gates"]["production_environment_exactly_pinned"]
    assert not result["s1r_pass"]


def test_s1r_symmetry_alignment_recovers_translated_promoted_density_and_lattice():
    records = _complete_records(rprop_shift=0.0)
    pre = next(
        record
        for record in records
        if record["structural_optimizer"] == "preconditioned"
        and record["multiplicity"] == "singlet"
        and record["structural_seed"] == "onsite"
        and record["root_seed_id"] == 0
    )
    rp = next(
        record
        for record in records
        if record["structural_optimizer"] == "rprop"
        and record["multiplicity"] == "singlet"
        and record["structural_seed"] == "onsite"
        and record["root_seed_id"] == 0
    )
    density = np.asarray(pre["excitation_density"])
    u = np.asarray(pre["lattice_u_A"])
    rp["excitation_density"] = np.roll(density, shift=(1, 2), axis=(0, 1)).tolist()
    rp["lattice_u_A"] = np.roll(u, shift=(1, 2), axis=(0, 1)).tolist()
    rp["excitation_rdm"] = copy.deepcopy(pre["excitation_rdm"])

    result = summarize_s1r(records, full_pytest_passed=True)
    diag = result["promoted_signature_diagnostics"]
    assert diag["best_D4_translation_density_alignment"]["rms_difference"] < 1.0e-14
    assert diag["best_D4_translation_lattice_alignment"]["rms_difference_A"] < 1.0e-14


def test_s1r_requires_exact_102_record_design():
    records = _complete_records()
    try:
        summarize_s1r(records[:-1], full_pytest_passed=True)
    except ValueError:
        pass
    else:
        raise AssertionError("missing S1R record must fail the exact design")
