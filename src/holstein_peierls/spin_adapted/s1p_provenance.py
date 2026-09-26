"""S1P provenance comparisons for the spin-adapted S0 root audit."""

from __future__ import annotations

from collections import defaultdict
from math import isfinite


ROOT_DIFFERENCE_EV = 1.0e-5
HISTORICAL_REPRODUCTION_EV = 1.0e-8
HISTORICAL_REFERENCE_EV = {
    ("singlet", "onsite"): 1.448028500905174,
    ("singlet", "bond_x"): 1.458848887171766,
    ("singlet", "bond_y"): 1.458734984906944,
    ("triplet", "onsite"): 1.446797186881587,
}


def _key(record: dict) -> tuple[str, str, str, str, str]:
    return (
        str(record["source_label"]),
        str(record["numpy_version_requested"]),
        str(record["thread_policy"]),
        str(record["multiplicity"]),
        str(record["seed"]),
    )


def summarize_s1p(records: list[dict]) -> dict:
    """Summarize the exact 2x2x2x4 S1P environment matrix."""
    if len(records) != 32:
        raise ValueError("S1P requires exactly 32 records")
    index = {_key(record): record for record in records}
    if len(index) != 32:
        raise ValueError("S1P record keys must be unique")

    expected = {
        (source, numpy_version, thread_policy, multiplicity, seed)
        for source in ("historical", "current")
        for numpy_version in ("2.5.2", "2.5.3")
        for thread_policy in ("historical_unpinned", "single_thread")
        for multiplicity, seed in (
            ("singlet", "onsite"),
            ("singlet", "bond_x"),
            ("singlet", "bond_y"),
            ("triplet", "onsite"),
        )
    }
    if set(index) != expected:
        raise ValueError("S1P records do not cover the preregistered matrix")

    all_converged = all(bool(record["converged"]) for record in records)
    all_finite = all(isfinite(float(record["total_referenced_energy_eV"])) for record in records)

    historical_differences = {}
    hist_ok = True
    for case, reference in HISTORICAL_REFERENCE_EV.items():
        rec = index[
            ("historical", "2.5.2", "historical_unpinned", case[0], case[1])
        ]
        delta = abs(float(rec["total_referenced_energy_eV"]) - reference)
        historical_differences[f"{case[0]}_{case[1]}"] = delta
        hist_ok = hist_ok and bool(rec["converged"]) and delta <= HISTORICAL_REPRODUCTION_EV

    factor_records: dict[str, list[dict]] = defaultdict(list)

    # NumPy comparisons: source/thread/case fixed.
    for source in ("historical", "current"):
        for thread in ("historical_unpinned", "single_thread"):
            for multiplicity, seed in HISTORICAL_REFERENCE_EV:
                a = index[(source, "2.5.2", thread, multiplicity, seed)]
                b = index[(source, "2.5.3", thread, multiplicity, seed)]
                factor_records["numpy"].append(
                    _comparison(a, b, "2.5.2", "2.5.3")
                )

    # Thread comparisons: source/numpy/case fixed.
    for source in ("historical", "current"):
        for numpy_version in ("2.5.2", "2.5.3"):
            for multiplicity, seed in HISTORICAL_REFERENCE_EV:
                a = index[(source, numpy_version, "historical_unpinned", multiplicity, seed)]
                b = index[(source, numpy_version, "single_thread", multiplicity, seed)]
                factor_records["thread_policy"].append(
                    _comparison(a, b, "historical_unpinned", "single_thread")
                )

    # Source comparisons: numpy/thread/case fixed.
    for numpy_version in ("2.5.2", "2.5.3"):
        for thread in ("historical_unpinned", "single_thread"):
            for multiplicity, seed in HISTORICAL_REFERENCE_EV:
                a = index[("historical", numpy_version, thread, multiplicity, seed)]
                b = index[("current", numpy_version, thread, multiplicity, seed)]
                factor_records["source_code"].append(
                    _comparison(a, b, "historical", "current")
                )

    factor_summary = {}
    for factor, comparisons in factor_records.items():
        root_changes = [
            item for item in comparisons
            if item["both_converged"] and item["absolute_energy_difference_eV"] > ROOT_DIFFERENCE_EV
        ]
        factor_summary[factor] = {
            "root_selecting": bool(root_changes),
            "root_change_count": len(root_changes),
            "maximum_absolute_energy_difference_eV": max(
                item["absolute_energy_difference_eV"] for item in comparisons
            ),
            "root_changes": root_changes,
            "all_comparisons": comparisons,
        }

    active = [
        factor for factor, summary in factor_summary.items()
        if summary["root_selecting"]
    ]
    classification = (
        "no_tested_factor_root_selecting"
        if not active
        else "+".join(active)
    )

    return {
        "all_32_records_present": True,
        "all_branches_converged": all_converged,
        "all_energies_finite": all_finite,
        "historical_reference_reproduced_within_1e-8_eV": bool(hist_ok),
        "historical_reference_absolute_differences_eV": historical_differences,
        "root_difference_threshold_eV": ROOT_DIFFERENCE_EV,
        "factor_sensitivity": factor_summary,
        "provenance_classification": classification,
        "interpretation_guard": {
            "s1_remains_failed": True,
            "environment_sensitivity_is_not_physical_uncertainty": True,
            "deterministic_root_enumeration_still_required": True,
        },
    }


def _comparison(a: dict, b: dict, a_label: str, b_label: str) -> dict:
    return {
        "multiplicity": str(a["multiplicity"]),
        "seed": str(a["seed"]),
        "source_label_a": str(a["source_label"]),
        "source_label_b": str(b["source_label"]),
        "numpy_a": str(a["numpy_version_requested"]),
        "numpy_b": str(b["numpy_version_requested"]),
        "thread_policy_a": str(a["thread_policy"]),
        "thread_policy_b": str(b["thread_policy"]),
        "factor_level_a": a_label,
        "factor_level_b": b_label,
        "energy_a_eV": float(a["total_referenced_energy_eV"]),
        "energy_b_eV": float(b["total_referenced_energy_eV"]),
        "absolute_energy_difference_eV": abs(
            float(a["total_referenced_energy_eV"])
            - float(b["total_referenced_energy_eV"])
        ),
        "both_converged": bool(a["converged"]) and bool(b["converged"]),
    }
