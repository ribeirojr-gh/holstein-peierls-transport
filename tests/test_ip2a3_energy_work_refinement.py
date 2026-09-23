import pytest

from holstein_peierls.dynamics.energy_work_refinement import (
    REFERENCE_LABELS,
    assess_timestep_refinement,
)


def _records(first, second):
    return [
        {"reference": label, "dt_fs": dt, "maximum_energy_work_residual_eV": residual}
        for label, residuals in zip(REFERENCE_LABELS, (first, second), strict=True)
        for dt, residual in zip((0.2, 0.1, 0.05), residuals, strict=True)
    ]


def test_selects_largest_tightened_dt_with_both_reference_passes():
    result = assess_timestep_refinement(
        _records((2.22e-6, 6.0e-7, 1.5e-7), (2.24e-6, 6.2e-7, 1.7e-7))
    )
    assert result["numerical_refinement_succeeded"]
    assert result["candidate_refined_dt_fs"] == 0.1
    assert result["ip2a2_selection_remains_failed"]


def test_selects_half_half_dt_only_when_half_dt_fails():
    result = assess_timestep_refinement(
        _records((2.3e-6, 2.1e-6, 1.0e-6), (2.4e-6, 2.2e-6, 1.2e-6))
    )
    assert result["candidate_refined_dt_fs"] == 0.05


def test_rejects_nonmonotonic_even_if_a_refined_dt_passes():
    result = assess_timestep_refinement(
        _records((2.3e-6, 1.0e-6, 1.2e-6), (2.3e-6, 1.0e-6, 5.0e-7))
    )
    assert not result["all_reference_residuals_decrease_with_dt"]
    assert not result["numerical_refinement_succeeded"]


def test_rejects_no_qualifying_refined_dt():
    result = assess_timestep_refinement(
        _records((3e-6, 2.7e-6, 2.4e-6), (3e-6, 2.7e-6, 2.4e-6))
    )
    assert result["candidate_refined_dt_fs"] is None


def test_rejects_missing_duplicate_and_unknown_trials():
    sample = _records((2.3e-6, 1e-6, 0.5e-6), (2.3e-6, 1e-6, 0.5e-6))
    with pytest.raises(ValueError):
        assess_timestep_refinement(sample[:5])
    duplicate = [dict(item) for item in sample]
    duplicate[5] = dict(duplicate[0])
    with pytest.raises(ValueError):
        assess_timestep_refinement(duplicate)
    bad = [dict(item) for item in sample]
    bad[0]["reference"] = "posthoc_selected"
    with pytest.raises(ValueError):
        assess_timestep_refinement(bad)
