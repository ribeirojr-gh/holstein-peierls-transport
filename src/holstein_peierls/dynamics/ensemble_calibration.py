"""Pre-intervention feasibility selection for IP2a-2."""

from __future__ import annotations


def select_production_energy(
    candidate_records: list[dict],
    *,
    pilot_count: int = 8,
    required_valid_x: int = 6,
) -> dict:
    """Select the smallest preregistered candidate satisfying IP2a-2 gates."""
    if pilot_count <= 0 or required_valid_x <= 0 or required_valid_x > pilot_count:
        raise ValueError("invalid pilot/valid-x requirements")

    evaluations = []
    for record in sorted(candidate_records, key=lambda item: float(item["candidate_energy_eV"])):
        trials = list(record.get("trials", []))
        if len(trials) != pilot_count:
            stable_all = False
            clean_all = False
            no_early_all = False
            valid_x_count = 0
        else:
            stable_all = all(bool(t.get("numerically_stable", False)) for t in trials)
            clean_all = all(bool(t.get("field_free_clean", False)) for t in trials)
            no_early_all = all(not bool(t.get("early_driven_event", False)) for t in trials)
            valid_x_count = sum(bool(t.get("valid_first_x_event", False)) for t in trials)

        qualifies = bool(
            len(trials) == pilot_count
            and stable_all
            and clean_all
            and no_early_all
            and valid_x_count >= required_valid_x
        )
        evaluations.append(
            {
                "candidate_energy_eV": float(record["candidate_energy_eV"]),
                "trial_count": len(trials),
                "all_trials_numerically_stable": stable_all,
                "all_field_free_controls_clean": clean_all,
                "no_early_driven_events": no_early_all,
                "valid_first_x_event_count": int(valid_x_count),
                "required_valid_first_x_event_count": int(required_valid_x),
                "qualifies": qualifies,
            }
        )

    qualifying = [item for item in evaluations if item["qualifies"]]
    selected = None if not qualifying else float(qualifying[0]["candidate_energy_eV"])
    return {
        "selected_energy_eV": selected,
        "selection_succeeded": selected is not None,
        "pilot_count": int(pilot_count),
        "required_valid_x": int(required_valid_x),
        "candidate_evaluations": evaluations,
    }
