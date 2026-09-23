import pytest

from holstein_peierls.dynamics.ensemble_calibration import select_production_energy


def _trial(*, stable=True, clean=True, early=False, valid_x=True):
    return {
        "numerically_stable": stable,
        "field_free_clean": clean,
        "early_driven_event": early,
        "valid_first_x_event": valid_x,
    }


def _candidate(energy, trials):
    return {"candidate_energy_eV": energy, "trials": trials}


def test_selects_smallest_qualifying_energy():
    records = [
        _candidate(1e-4, [_trial() for _ in range(8)]),
        _candidate(1e-5, [_trial(valid_x=i < 5) for i in range(8)]),
        _candidate(3e-5, [_trial(valid_x=i < 6) for i in range(8)]),
    ]
    result = select_production_energy(records)
    assert result["selection_succeeded"]
    assert result["selected_energy_eV"] == pytest.approx(3e-5)


def test_candidate_fails_for_any_early_event():
    trials = [_trial() for _ in range(8)]
    trials[2] = _trial(early=True)
    result = select_production_energy([_candidate(1e-5, trials)])
    assert not result["selection_succeeded"]
    assert result["candidate_evaluations"][0]["no_early_driven_events"] is False


def test_candidate_fails_for_dirty_field_free_control():
    trials = [_trial() for _ in range(8)]
    trials[0] = _trial(clean=False)
    result = select_production_energy([_candidate(1e-5, trials)])
    assert not result["selection_succeeded"]


def test_candidate_fails_for_numerical_instability():
    trials = [_trial() for _ in range(8)]
    trials[7] = _trial(stable=False)
    result = select_production_energy([_candidate(1e-5, trials)])
    assert not result["selection_succeeded"]


def test_exact_six_of_eight_valid_x_passes():
    trials = [_trial(valid_x=i < 6) for i in range(8)]
    result = select_production_energy([_candidate(1e-5, trials)])
    assert result["selection_succeeded"]


def test_five_of_eight_valid_x_fails():
    trials = [_trial(valid_x=i < 5) for i in range(8)]
    result = select_production_energy([_candidate(1e-5, trials)])
    assert not result["selection_succeeded"]


def test_wrong_trial_count_cannot_qualify():
    result = select_production_energy([_candidate(1e-5, [_trial() for _ in range(7)])])
    assert not result["selection_succeeded"]


def test_rejects_invalid_requirements():
    with pytest.raises(ValueError):
        select_production_energy([], pilot_count=0)
    with pytest.raises(ValueError):
        select_production_energy([], pilot_count=8, required_valid_x=9)
