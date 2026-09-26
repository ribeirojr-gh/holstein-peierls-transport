import copy

from holstein_peierls.spin_adapted.s1p_provenance import (
    HISTORICAL_REFERENCE_EV,
    summarize_s1p,
)


def _records():
    records = []
    for source in ("historical", "current"):
        for numpy_version in ("2.5.2", "2.5.3"):
            for thread in ("historical_unpinned", "single_thread"):
                for (multiplicity, seed), energy in HISTORICAL_REFERENCE_EV.items():
                    records.append(
                        {
                            "source_label": source,
                            "numpy_version_requested": numpy_version,
                            "thread_policy": thread,
                            "multiplicity": multiplicity,
                            "seed": seed,
                            "converged": True,
                            "total_referenced_energy_eV": energy,
                        }
                    )
    return records


def test_s1p_identifies_exact_historical_reproduction_without_sensitivity():
    result = summarize_s1p(_records())
    assert result["historical_reference_reproduced_within_1e-8_eV"]
    assert result["provenance_classification"] == "no_tested_factor_root_selecting"


def test_s1p_detects_numpy_root_selection():
    records = _records()
    for item in records:
        if (
            item["source_label"] == "current"
            and item["numpy_version_requested"] == "2.5.3"
            and item["thread_policy"] == "single_thread"
            and item["multiplicity"] == "singlet"
            and item["seed"] == "onsite"
        ):
            item["total_referenced_energy_eV"] -= 2.0e-5
    result = summarize_s1p(records)
    assert result["factor_sensitivity"]["numpy"]["root_selecting"]


def test_s1p_detects_thread_and_source_independently():
    records = _records()
    changed = copy.deepcopy(records)
    for item in changed:
        if (
            item["source_label"] == "historical"
            and item["numpy_version_requested"] == "2.5.2"
            and item["thread_policy"] == "single_thread"
            and item["multiplicity"] == "singlet"
            and item["seed"] == "bond_y"
        ):
            item["total_referenced_energy_eV"] += 3.0e-5
        if (
            item["source_label"] == "current"
            and item["numpy_version_requested"] == "2.5.3"
            and item["thread_policy"] == "historical_unpinned"
            and item["multiplicity"] == "singlet"
            and item["seed"] == "bond_x"
        ):
            item["total_referenced_energy_eV"] += 4.0e-5
    result = summarize_s1p(changed)
    assert result["factor_sensitivity"]["thread_policy"]["root_selecting"]
    assert result["factor_sensitivity"]["source_code"]["root_selecting"]


def test_s1p_requires_exact_matrix():
    records = _records()
    try:
        summarize_s1p(records[:-1])
    except ValueError:
        pass
    else:
        raise AssertionError("missing record should fail")
