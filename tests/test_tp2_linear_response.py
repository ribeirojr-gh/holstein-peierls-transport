import numpy as np
import pytest

from holstein_peierls.dynamics.linear_response import (
    analyze_paired_field_ensemble,
    electron_mobility_from_odd_velocity,
    mean_confidence_interval,
    paired_velocity_components,
    seed_linear_response,
    through_origin_slope,
)


def test_paired_velocity_components_cancel_even_bias():
    odd_true = np.array([-1.0e-3, -2.0e-3, -4.0e-3])
    even_bias = np.array([3.0e-4, -2.0e-4, 5.0e-4])
    plus = even_bias + odd_true
    minus = even_bias - odd_true
    odd, even = paired_velocity_components(plus, minus)
    assert np.allclose(odd, odd_true)
    assert np.allclose(even, even_bias)


def test_electron_mobility_unit_conversion_and_sign():
    # mu = 1 cm^2/(V s), E = +1 mV/A -> v = -0.01 A/fs.
    mobility = electron_mobility_from_odd_velocity(-0.01, 1.0)
    assert float(mobility) == pytest.approx(1.0)
    mobility_negative = electron_mobility_from_odd_velocity(+0.01, 1.0)
    assert float(mobility_negative) == pytest.approx(-1.0)


def test_through_origin_slope_recovers_exact_linear_data():
    x = np.array([0.5, 1.0, 2.0])
    y = -3.25 * x
    slope, r2 = through_origin_slope(x, y)
    assert slope == pytest.approx(-3.25)
    assert r2 == pytest.approx(1.0)


def test_seed_linear_response_recovers_known_mobility():
    fields_mV = np.array([0.5, 1.0, 2.0])
    target_mu = 2.5
    fields_V = 1.0e-3 * fields_mV
    velocity = -(target_mu / 0.1) * fields_V
    response = seed_linear_response(fields_mV, velocity)
    assert response.electron_mobility_cm2_per_V_s == pytest.approx(target_mu)
    assert response.r_squared_origin == pytest.approx(1.0)


def test_mean_confidence_interval_collapses_for_identical_samples():
    result = mean_confidence_interval([2.0, 2.0, 2.0, 2.0])
    assert result.mean == pytest.approx(2.0)
    assert result.sample_std == pytest.approx(0.0)
    assert result.standard_error == pytest.approx(0.0)
    assert result.lower == pytest.approx(2.0)
    assert result.upper == pytest.approx(2.0)
    assert result.sample_count == 4


def test_ensemble_analysis_preserves_seed_independence_and_known_response():
    fields = np.array([0.5, 1.0, 2.0])
    seed_mu = np.array([1.0, 1.5, 2.0, 2.5])
    even_bias = np.array([2.0e-4, -1.0e-4, 3.0e-4, -2.0e-4])[:, None]
    fields_V = 1.0e-3 * fields[None, :]
    odd = -(seed_mu[:, None] / 0.1) * fields_V
    plus = even_bias + odd
    minus = even_bias - odd

    result = analyze_paired_field_ensemble(fields, plus, minus)

    assert result.mobility_ci.sample_count == 4
    assert result.mobility_ci.mean == pytest.approx(np.mean(seed_mu))
    assert result.ensemble_mean_mobility_cm2_per_V_s == pytest.approx(np.mean(seed_mu))
    assert result.ensemble_mean_r_squared_origin == pytest.approx(1.0)
    assert result.maximum_fractional_linearity_residual == pytest.approx(0.0, abs=1e-14)
    assert np.allclose(
        [response.electron_mobility_cm2_per_V_s for response in result.seed_responses],
        seed_mu,
    )
    assert np.allclose(np.mean(result.even_velocity_A_per_fs, axis=1), even_bias[:, 0])


def test_ensemble_analysis_rejects_duplicate_or_zero_fields():
    plus = np.zeros((2, 3))
    minus = np.zeros((2, 3))
    with pytest.raises(ValueError):
        analyze_paired_field_ensemble([0.5, 0.5, 1.0], plus, minus)
    with pytest.raises(ValueError):
        analyze_paired_field_ensemble([0.0, 0.5, 1.0], plus, minus)
