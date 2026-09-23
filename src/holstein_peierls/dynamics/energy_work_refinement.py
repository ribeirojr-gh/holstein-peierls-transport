"""Pure preregistered numerical classification of the IP2a-3 dt diagnostic."""

from __future__ import annotations

from math import isfinite, log2

DT_GRID_FS = (0.2, 0.1, 0.05)
REFERENCE_LABELS = ("unperturbed", "member_0_1e-5_eV")
CALIBRATION_BOUND_EV = 2.0e-6


def assess_timestep_refinement(
    records: list[dict],
    *,
    maximum_balance_eV: float = CALIBRATION_BOUND_EV,
) -> dict:
    """Classify work-balance refinement without reclassifying IP2a-2.

    Each record must contain reference, dt_fs, and maximum_energy_work_residual_eV.
    """
    bound = float(maximum_balance_eV)
    if not isfinite(bound) or bound <= 0.0:
        raise ValueError("maximum_balance_eV must be finite and positive")
    if len(records) != len(REFERENCE_LABELS) * len(DT_GRID_FS):
        raise ValueError("exactly six preregistered records are required")

    grid = {}
    for record in records:
        label = str(record["reference"])
        dt = float(record["dt_fs"])
        residual = float(record["maximum_energy_work_residual_eV"])
        if label not in REFERENCE_LABELS or dt not in DT_GRID_FS:
            raise ValueError("reference or dt differs from preregistration")
        if not isfinite(residual) or residual < 0.0:
            raise ValueError("energy-work residual must be finite and nonnegative")
        key = (label, dt)
        if key in grid:
            raise ValueError("duplicate reference/dt record")
        grid[key] = residual

    rows = []
    all_monotonic = True
    for label in REFERENCE_LABELS:
        values = [grid[(label, dt)] for dt in DT_GRID_FS]
        monotonic = values[0] > values[1] > values[2]
        all_monotonic = all_monotonic and monotonic
        factors = [
            None if values[i + 1] == 0.0 else float(values[i] / values[i + 1])
            for i in range(2)
        ]
        orders = [
            None if factor is None or factor <= 0.0 else float(log2(factor))
            for factor in factors
        ]
        rows.append(
            {
                "reference": label,
                "residuals_eV_by_decreasing_dt": values,
                "strictly_decreasing": bool(monotonic),
                "reduction_factors": factors,
                "effective_log2_orders": orders,
            }
        )

    viable_dt = [
        dt for dt in DT_GRID_FS[1:]
        if all(grid[(label, dt)] <= bound for label in REFERENCE_LABELS)
    ]
    selected = float(viable_dt[0]) if all_monotonic and viable_dt else None
    return {
        "ip2a2_selection_remains_failed": True,
        "original_balance_bound_eV": bound,
        "all_reference_residuals_decrease_with_dt": bool(all_monotonic),
        "refined_dt_values_meeting_bound_for_both": viable_dt,
        "candidate_refined_dt_fs": selected,
        "numerical_refinement_succeeded": selected is not None,
        "reference_diagnostics": rows,
    }
