import math

import pytest

from holstein_peierls.dimer_finite_difference import (
    FiniteDifferenceScanPlan,
    TransferIntegralSample,
    analyze_finite_difference_scan,
)


def test_duplicate_coordinate_step_sample_is_rejected() -> None:
    plan = FiniteDifferenceScanPlan(
        (0.0025, 0.005, 0.01),
        tuple(math.radians(value) for value in (0.25, 0.5, 1.0)),
    )
    perturbations = plan.perturbations()
    samples = [TransferIntegralSample(item, 0.03 * item.amount) for item in perturbations]
    samples.append(samples[0])
    with pytest.raises(ValueError, match="duplicate finite-difference sample"):
        analyze_finite_difference_scan(tuple(samples), plan)
