import math

import numpy as np
import pytest

from holstein_peierls.dimer_finite_difference import (
    DIMER_COORDINATES,
    DIMER_COORDINATE_LABELS,
    DimerPerturbation,
    FiniteDifferenceScanPlan,
    RigidDimerReference,
    RigidMoleculeGeometry,
    TransferIntegralSample,
    analyze_finite_difference_scan,
    local_derivative_to_global_rigid_body_vector,
    prepare_scan_geometries,
)


def _molecule_b() -> RigidMoleculeGeometry:
    return RigidMoleculeGeometry(
        symbols=("C", "H"),
        coordinates_angstrom=((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        pivot_angstrom=(1.0, 0.0, 0.0),
    )


def _reference(axes: np.ndarray | None = None) -> RigidDimerReference:
    molecule_a = RigidMoleculeGeometry(
        symbols=("C",),
        coordinates_angstrom=((0.0, 0.0, 0.0),),
        pivot_angstrom=(0.0, 0.0, 0.0),
    )
    if axes is None:
        axes = np.eye(3)
    return RigidDimerReference(
        molecule_a=molecule_a,
        molecule_b=_molecule_b(),
        local_axes_global=tuple(map(tuple, axes.tolist())),
    )


def _plan() -> FiniteDifferenceScanPlan:
    return FiniteDifferenceScanPlan(
        translation_steps_angstrom=(0.0025, 0.005, 0.01),
        rotation_steps_radian=tuple(
            math.radians(value) for value in (0.25, 0.5, 1.0)
        ),
    )


def test_scan_plan_contains_three_central_pairs_for_all_six_dofs() -> None:
    plan = _plan()
    perturbations = plan.perturbations()
    assert len(perturbations) == 36
    assert DIMER_COORDINATE_LABELS == (
        "translation_long",
        "translation_short",
        "translation_normal",
        "rotation_long",
        "rotation_short",
        "rotation_normal",
    )
    for coordinate in DIMER_COORDINATES:
        selected = [
            item for item in perturbations if item.coordinate_label == coordinate.label
        ]
        assert len(selected) == 6
        magnitudes = sorted({abs(item.amount) for item in selected})
        assert len(magnitudes) == 3


def test_translation_moves_only_target_molecule_and_pivot() -> None:
    reference = _reference()
    coordinate = DIMER_COORDINATES[0]
    shifted = reference.perturb_target(coordinate, 0.25)
    assert np.array_equal(shifted.molecule_a.coordinates, reference.molecule_a.coordinates)
    assert np.allclose(
        shifted.molecule_b.coordinates,
        reference.molecule_b.coordinates + np.array([0.25, 0.0, 0.0]),
        rtol=0.0,
        atol=1.0e-15,
    )
    assert np.allclose(
        shifted.molecule_b.pivot,
        reference.molecule_b.pivot + np.array([0.25, 0.0, 0.0]),
        rtol=0.0,
        atol=1.0e-15,
    )


def test_rotation_uses_target_pivot_and_preserves_internal_distances() -> None:
    reference = _reference()
    coordinate = DIMER_COORDINATES[5]
    rotated = reference.perturb_target(coordinate, math.pi / 2.0)
    assert np.allclose(rotated.molecule_b.pivot, reference.molecule_b.pivot)
    assert np.allclose(
        rotated.molecule_b.coordinates[1],
        np.array([1.0, 1.0, 0.0]),
        rtol=0.0,
        atol=2.0e-15,
    )
    before = np.linalg.norm(
        reference.molecule_b.coordinates[1] - reference.molecule_b.coordinates[0]
    )
    after = np.linalg.norm(
        rotated.molecule_b.coordinates[1] - rotated.molecule_b.coordinates[0]
    )
    assert np.isclose(after, before, rtol=0.0, atol=2.0e-15)


def test_reference_rejects_non_orthonormal_or_left_handed_frames() -> None:
    with pytest.raises(ValueError, match="orthonormal"):
        _reference(np.diag([1.0, 1.0, 2.0]))
    with pytest.raises(ValueError, match="right-handed"):
        _reference(np.diag([1.0, 1.0, -1.0]))


def test_prepare_scan_geometries_matches_each_signed_perturbation() -> None:
    reference = _reference()
    prepared = prepare_scan_geometries(reference, _plan())
    assert len(prepared) == 36
    first_perturbation, first_geometry = prepared[0]
    assert first_perturbation.coordinate_label == "translation_long"
    assert first_perturbation.amount == -0.0025
    assert np.allclose(
        first_geometry.molecule_b.pivot,
        np.array([0.9975, 0.0, 0.0]),
        rtol=0.0,
        atol=1.0e-15,
    )


def test_multistep_central_difference_recovers_zero_step_six_dof_gradient() -> None:
    plan = _plan()
    exact = np.array([0.21, -0.08, 0.035, 0.012, -0.019, 0.007])
    cubic = np.array([12.0, -7.0, 4.0, 0.8, -0.5, 0.3])
    index = {label: position for position, label in enumerate(DIMER_COORDINATE_LABELS)}
    samples = []
    for perturbation in plan.perturbations():
        position = index[perturbation.coordinate_label]
        q = perturbation.amount
        value = 0.055 + exact[position] * q + cubic[position] * q**3
        samples.append(TransferIntegralSample(perturbation, value))

    result = analyze_finite_difference_scan(tuple(samples), plan)
    assert np.allclose(result.local_vector, exact, rtol=0.0, atol=2.0e-12)
    assert result.all_pass_linearity
    assert all(item.fit_rms_ev_per_unit < 1.0e-12 for item in result.coordinates)


def test_scan_rejects_missing_central_partner() -> None:
    plan = _plan()
    perturbations = plan.perturbations()
    samples = tuple(
        TransferIntegralSample(item, 0.01 * item.amount)
        for item in perturbations
        if not (
            item.coordinate_label == "translation_long"
            and item.amount == plan.translation_steps_angstrom[0]
        )
    )
    with pytest.raises(ValueError, match=r"missing \+/- finite-difference sample"):
        analyze_finite_difference_scan(samples, plan)


def test_linearity_gate_flags_step_dependent_derivative() -> None:
    plan = _plan()
    samples = []
    for perturbation in plan.perturbations():
        q = perturbation.amount
        if perturbation.coordinate_label == "translation_long":
            value = 0.1 * q + 5000.0 * q**3 + 2.0e7 * q**5
        else:
            value = 0.02 * q
        samples.append(TransferIntegralSample(perturbation, value))
    result = analyze_finite_difference_scan(
        tuple(samples), plan, maximum_relative_spread=0.05
    )
    first = result.coordinates[0]
    assert not first.passes_linearity
    assert first.max_relative_spread > 0.05


def test_local_derivative_transforms_translation_and_rotation_blocks_to_global_frame() -> None:
    theta = math.radians(30.0)
    axes = np.array(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    local = np.array([1.0, 2.0, 3.0, -4.0, 5.0, -6.0])
    global_vector = local_derivative_to_global_rigid_body_vector(local, axes)
    assert np.allclose(global_vector[:3], axes @ local[:3])
    assert np.allclose(global_vector[3:], axes @ local[3:])


def test_scan_plan_rejects_underresolved_or_duplicate_step_grids() -> None:
    with pytest.raises(ValueError, match="at least three"):
        FiniteDifferenceScanPlan((0.005, 0.01), (0.01, 0.02, 0.03))
    with pytest.raises(ValueError, match="unique"):
        FiniteDifferenceScanPlan((0.005, 0.005, 0.01), (0.01, 0.02, 0.03))


def test_perturbation_rejects_wrong_coordinate_unit() -> None:
    with pytest.raises(ValueError, match="unit"):
        DimerPerturbation("rotation_long", 0.1, "angstrom")
