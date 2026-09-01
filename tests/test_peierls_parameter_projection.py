import numpy as np
import pytest

from holstein_peierls.dimer_finite_difference import (
    CoordinateDerivativeResult,
    DIMER_COORDINATES,
    DimerDerivativeResult,
)
from holstein_peierls.peierls_parameter_projection import (
    OrientedBondDerivative,
    assemble_g3_bond_mode_couplings,
    predict_linearized_transfer_integral_variance,
)


def _derivative(values: tuple[float, ...], *, passes: bool = True) -> DimerDerivativeResult:
    coordinates = []
    for coordinate, value in zip(DIMER_COORDINATES, values):
        coordinates.append(
            CoordinateDerivativeResult(
                coordinate_label=coordinate.label,
                derivative_ev_per_unit=value,
                unit=f"eV/{coordinate.unit}",
                step_sizes=(0.1, 0.2, 0.3),
                central_derivatives_ev_per_unit=(value, value, value),
                max_relative_spread=0.0 if passes else 1.0,
                fit_rms_ev_per_unit=0.0,
                passes_linearity=passes,
            )
        )
    return DimerDerivativeResult(tuple(coordinates))


def test_assemble_g3_couplings_requires_every_oriented_family_explicitly() -> None:
    expected = ("bond_a", "bond_b")
    only_a = (
        OrientedBondDerivative(
            "bond_a",
            _derivative((1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
            tuple(map(tuple, np.eye(3).tolist())),
        ),
    )
    with pytest.raises(ValueError, match="missing=.*bond_b"):
        assemble_g3_bond_mode_couplings(only_a, expected)


def test_assemble_g3_couplings_preserves_requested_family_order_and_frames() -> None:
    theta = np.deg2rad(90.0)
    axes = np.array(
        [[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]]
    )
    entries = (
        OrientedBondDerivative(
            "bond_b",
            _derivative((2.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
            tuple(map(tuple, axes.tolist())),
        ),
        OrientedBondDerivative(
            "bond_a",
            _derivative((1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
            tuple(map(tuple, np.eye(3).tolist())),
        ),
    )
    output = assemble_g3_bond_mode_couplings(entries, ("bond_a", "bond_b"))
    assert [label for label, _ in output] == ["bond_a", "bond_b"]
    assert np.allclose(output[0][1], (1.0, 0.0, 0.0, 0.0, 0.0, 0.0), atol=1.0e-15)
    assert np.allclose(output[1][1][:3], (0.0, 2.0, 0.0), atol=2.0e-15)


def test_assemble_g3_couplings_rejects_failed_linearity_gate() -> None:
    entry = OrientedBondDerivative(
        "bond",
        _derivative((1.0, 0.0, 0.0, 0.0, 0.0, 0.0), passes=False),
        tuple(map(tuple, np.eye(3).tolist())),
    )
    with pytest.raises(ValueError, match="linearity gate failed"):
        assemble_g3_bond_mode_couplings((entry,), ("bond",))


def test_covariance_projection_matches_direct_variance_formula() -> None:
    coupling = np.array([0.2, -0.1, 0.05, 0.03, -0.02, 0.01])
    diagonal_variances = np.array([0.04, 0.01, 0.09, 0.0025, 0.0036, 0.0049])
    covariance = np.diag(diagonal_variances)
    expected = float(np.sum(coupling**2 * diagonal_variances))
    assert np.isclose(
        predict_linearized_transfer_integral_variance(coupling, covariance),
        expected,
        rtol=0.0,
        atol=1.0e-16,
    )


def test_covariance_projection_rejects_non_psd_matrix() -> None:
    covariance = np.eye(6)
    covariance[0, 0] = -0.1
    with pytest.raises(ValueError, match="positive semidefinite"):
        predict_linearized_transfer_integral_variance(np.ones(6), covariance)
