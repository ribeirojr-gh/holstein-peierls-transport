import copy

import pytest

from holstein_peierls.s3_campaign import (
    BRANCHES,
    classify_topology,
    expand_tasks,
    summarize_campaign,
    task_id,
    validate_manifest,
)


def manifest():
    return {
        "schema_version": 1,
        "campaign_id": "test-s3",
        "stage": "pilot",
        "lattice_sizes": [8],
        "model": {
            "boundary_conditions": "periodic",
            "Jx_eV": 0.0575,
            "Jy_eV": 0.0575,
            "K1_eV_per_A2": 16.51,
            "K2_eV_per_A2": 0.51,
            "base_alpha_intra_eV_per_A": 3.0,
            "base_alpha_x_eV_per_A": 0.1,
            "base_alpha_y_eV_per_A": 0.1,
        },
        "grid": {"U_eV": [1.0], "V1_eV": [0.016], "coupling_scale": [0.8, 1.0]},
        "branches": list(BRANCHES),
        "numerical": {"max_iterations": 1200},
        "classification": {
            "energy_tie_tolerance_eV": 1e-8,
            "robust_binding_threshold_eV": 0.005,
            "linear_peierls_ratio_max": 0.25,
        },
    }


def record(branch, energy, *, p0=0.01, pnnx=0.02, pnny=0.02, pdiag=0.90, mean_r=1.42):
    return {
        "size": 8,
        "coupling_scale": 1.0,
        "U_eV": 1.0,
        "V1_eV": 0.016,
        "branch": branch,
        "total_energy_eV": energy,
        "P_onsite": p0,
        "P_nn": pnnx + pnny,
        "P_nn_x": pnnx,
        "P_nn_y": pnny,
        "P_diagonal": pdiag,
        "mean_r": mean_r,
        "max_delta_tx_over_Jx": 0.10,
        "max_delta_ty_over_Jy": 0.10,
        "converged": True,
        "final_max_update_A": 1e-9,
        "final_max_gradient_eV_per_A": 1e-8,
    }


def test_manifest_expands_locked_branch_matrix():
    payload = manifest()
    validate_manifest(payload)
    tasks = expand_tasks(payload)
    assert len(tasks) == 10
    assert tasks[0]["alpha_intra_eV_per_A"] == pytest.approx(2.4)
    assert len({task_id(task) for task in tasks}) == len(tasks)


def test_manifest_rejects_incomplete_seed_ensemble():
    payload = manifest()
    payload["branches"] = payload["branches"][:-1]
    with pytest.raises(ValueError, match="complete locked S3 seed order"):
        validate_manifest(payload)


def test_topology_uses_observables_not_seed_label():
    item = record("onsite", -1.0)
    assert classify_topology(item) == "diagonal"


def test_aggregate_promotes_energy_minimum_and_preserves_seed_as_metadata():
    payload = manifest()
    payload["grid"]["coupling_scale"] = [1.0]
    records = [record(branch, -0.59) for branch in BRANCHES]
    for item in records:
        if item["branch"] == "onsite":
            item["total_energy_eV"] = -0.61
        elif item["branch"] == "separated":
            item["total_energy_eV"] = -0.60
    summary = summarize_campaign(payload, records)
    point = summary["points"][0]
    assert summary["complete"]
    assert point["selected_seed"] == "onsite"
    assert point["classification"] == "diagonal"
    assert point["binding_vs_separated_eV"] == pytest.approx(0.01)
    assert point["quantitative_at_this_size"]
    assert point["finite_size_status"] == "pending"


def test_aggregate_marks_subthreshold_binding_marginal_and_peierls_failure():
    payload = manifest()
    payload["grid"]["coupling_scale"] = [1.0]
    records = [record(branch, -0.599) for branch in BRANCHES]
    for item in records:
        if item["branch"] == "diagonal":
            item["total_energy_eV"] = -0.6005
            item["max_delta_tx_over_Jx"] = 0.30
        elif item["branch"] == "separated":
            item["total_energy_eV"] = -0.6000
    point = summarize_campaign(payload, records)["points"][0]
    assert point["classification"] == "marginal_diagonal"
    assert not point["linear_peierls_gate"]
    assert not point["quantitative_at_this_size"]


def test_aggregate_rejects_duplicate_records():
    payload = manifest()
    payload["grid"]["coupling_scale"] = [1.0]
    item = record("onsite", -0.6)
    with pytest.raises(ValueError, match="duplicate"):
        summarize_campaign(payload, [item, copy.deepcopy(item)])
