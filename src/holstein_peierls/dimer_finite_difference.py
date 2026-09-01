"""Backend-independent finite-difference pipeline for rigid molecular dimers.

G5e separates geometry generation and numerical differentiation from the
choice of electronic-structure backend.  A backend only needs to evaluate a
signed transfer integral for each generated dimer geometry.  The resulting
samples can then be converted into bond-resolved Peierls derivatives with
explicit linearity diagnostics.

The six coordinates follow the molecular-dimer convention used in the
pentacene dynamical-disorder literature: translations along the long, short,
and normal axes, followed by rotations around those same axes.  Translations
are measured in angstrom and rotations in radians.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class DimerCoordinate:
    """One relative rigid-body coordinate in a dimer-local orthonormal frame."""

    label: str
    kind: str
    axis_index: int
    unit: str

    def __post_init__(self) -> None:
        if self.kind not in {"translation", "rotation"}:
            raise ValueError("kind must be 'translation' or 'rotation'")
        if self.axis_index not in {0, 1, 2}:
            raise ValueError("axis_index must be 0, 1, or 2")
        expected_unit = "angstrom" if self.kind == "translation" else "radian"
        if self.unit != expected_unit:
            raise ValueError(f"{self.kind} coordinate must use {expected_unit}")


DIMER_COORDINATES = (
    DimerCoordinate("translation_long", "translation", 0, "angstrom"),
    DimerCoordinate("translation_short", "translation", 1, "angstrom"),
    DimerCoordinate("translation_normal", "translation", 2, "angstrom"),
    DimerCoordinate("rotation_long", "rotation", 0, "radian"),
    DimerCoordinate("rotation_short", "rotation", 1, "radian"),
    DimerCoordinate("rotation_normal", "rotation", 2, "radian"),
)
DIMER_COORDINATE_LABELS = tuple(item.label for item in DIMER_COORDINATES)


@dataclass(frozen=True, slots=True)
class RigidMoleculeGeometry:
    """Atomic geometry plus an explicit rigid-body rotation pivot."""

    symbols: tuple[str, ...]
    coordinates_angstrom: tuple[tuple[float, float, float], ...]
    pivot_angstrom: tuple[float, float, float]

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.coordinates_angstrom, dtype=np.float64)
        pivot = np.asarray(self.pivot_angstrom, dtype=np.float64)
        if not self.symbols:
            raise ValueError("a rigid molecule must contain at least one atom")
        if coordinates.shape != (len(self.symbols), 3):
            raise ValueError("coordinates must have shape (n_atoms, 3)")
        if pivot.shape != (3,):
            raise ValueError("pivot must contain three Cartesian coordinates")
        if not np.all(np.isfinite(coordinates)) or not np.all(np.isfinite(pivot)):
            raise ValueError("molecular coordinates and pivot must be finite")

    @property
    def coordinates(self) -> FloatArray:
        return np.asarray(self.coordinates_angstrom, dtype=np.float64)

    @property
    def pivot(self) -> FloatArray:
        return np.asarray(self.pivot_angstrom, dtype=np.float64)

    def translated(self, displacement_angstrom: FloatArray) -> "RigidMoleculeGeometry":
        displacement = np.asarray(displacement_angstrom, dtype=np.float64)
        if displacement.shape != (3,) or not np.all(np.isfinite(displacement)):
            raise ValueError("translation must be a finite three-vector")
        coordinates = self.coordinates + displacement
        pivot = self.pivot + displacement
        return RigidMoleculeGeometry(
            symbols=self.symbols,
            coordinates_angstrom=tuple(map(tuple, coordinates.tolist())),
            pivot_angstrom=tuple(pivot.tolist()),
        )

    def rotated(self, axis: FloatArray, angle_radian: float) -> "RigidMoleculeGeometry":
        axis_values = np.asarray(axis, dtype=np.float64)
        if axis_values.shape != (3,) or not np.all(np.isfinite(axis_values)):
            raise ValueError("rotation axis must be a finite three-vector")
        norm = float(np.linalg.norm(axis_values))
        if norm <= 0.0:
            raise ValueError("rotation axis must be non-zero")
        if not np.isfinite(angle_radian):
            raise ValueError("rotation angle must be finite")
        unit_axis = axis_values / norm
        x, y, z = unit_axis
        cross = np.array(
            [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64
        )
        identity = np.eye(3, dtype=np.float64)
        rotation = (
            np.cos(angle_radian) * identity
            + (1.0 - np.cos(angle_radian)) * np.outer(unit_axis, unit_axis)
            + np.sin(angle_radian) * cross
        )
        shifted = self.coordinates - self.pivot
        coordinates = shifted @ rotation.T + self.pivot
        return RigidMoleculeGeometry(
            symbols=self.symbols,
            coordinates_angstrom=tuple(map(tuple, coordinates.tolist())),
            pivot_angstrom=self.pivot_angstrom,
        )


@dataclass(frozen=True, slots=True)
class RigidDimerReference:
    """Reference dimer and the local orthonormal frame used for finite differences.

    ``local_axes_global`` stores the long, short, and normal unit vectors as
    columns in global Cartesian coordinates.  The first molecule remains fixed;
    finite-difference perturbations are applied to the second molecule.
    """

    molecule_a: RigidMoleculeGeometry
    molecule_b: RigidMoleculeGeometry
    local_axes_global: tuple[tuple[float, float, float], ...]

    def __post_init__(self) -> None:
        axes = np.asarray(self.local_axes_global, dtype=np.float64)
        if axes.shape != (3, 3) or not np.all(np.isfinite(axes)):
            raise ValueError("local_axes_global must be a finite 3x3 matrix")
        if not np.allclose(axes.T @ axes, np.eye(3), rtol=0.0, atol=1.0e-12):
            raise ValueError("local axes must be orthonormal")
        if not np.isclose(np.linalg.det(axes), 1.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("local axes must form a right-handed frame")

    @property
    def axes(self) -> FloatArray:
        return np.asarray(self.local_axes_global, dtype=np.float64)

    def perturb_target(
        self, coordinate: DimerCoordinate, amount: float
    ) -> "RigidDimerReference":
        if not np.isfinite(amount):
            raise ValueError("perturbation amount must be finite")
        axis = self.axes[:, coordinate.axis_index]
        if coordinate.kind == "translation":
            target = self.molecule_b.translated(amount * axis)
        else:
            target = self.molecule_b.rotated(axis, amount)
        return RigidDimerReference(
            molecule_a=self.molecule_a,
            molecule_b=target,
            local_axes_global=self.local_axes_global,
        )


@dataclass(frozen=True, slots=True)
class DimerPerturbation:
    """One signed finite-difference perturbation."""

    coordinate_label: str
    amount: float
    unit: str

    def __post_init__(self) -> None:
        if self.coordinate_label not in DIMER_COORDINATE_LABELS:
            raise ValueError("unknown dimer coordinate")
        if self.amount == 0.0 or not np.isfinite(self.amount):
            raise ValueError("perturbation amount must be finite and non-zero")
        expected = next(
            item.unit for item in DIMER_COORDINATES if item.label == self.coordinate_label
        )
        if self.unit != expected:
            raise ValueError("perturbation unit is inconsistent with coordinate")


@dataclass(frozen=True, slots=True)
class FiniteDifferenceScanPlan:
    """Multi-step central-difference plan for all six rigid-dimer coordinates."""

    translation_steps_angstrom: tuple[float, ...]
    rotation_steps_radian: tuple[float, ...]

    def __post_init__(self) -> None:
        for name, steps in (
            ("translation", self.translation_steps_angstrom),
            ("rotation", self.rotation_steps_radian),
        ):
            values = np.asarray(steps, dtype=np.float64)
            if values.ndim != 1 or len(values) < 3:
                raise ValueError(f"{name} scan requires at least three step sizes")
            if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
                raise ValueError(f"{name} step sizes must be finite and positive")
            if len(set(float(value) for value in values)) != len(values):
                raise ValueError(f"{name} step sizes must be unique")

    def steps_for(self, coordinate: DimerCoordinate) -> tuple[float, ...]:
        return (
            self.translation_steps_angstrom
            if coordinate.kind == "translation"
            else self.rotation_steps_radian
        )

    def perturbations(self) -> tuple[DimerPerturbation, ...]:
        output: list[DimerPerturbation] = []
        for coordinate in DIMER_COORDINATES:
            for step in self.steps_for(coordinate):
                output.append(DimerPerturbation(coordinate.label, -step, coordinate.unit))
                output.append(DimerPerturbation(coordinate.label, step, coordinate.unit))
        return tuple(output)


@dataclass(frozen=True, slots=True)
class TransferIntegralSample:
    """Signed transfer integral evaluated for one perturbation."""

    perturbation: DimerPerturbation
    transfer_integral_ev: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.transfer_integral_ev):
            raise ValueError("transfer integral must be finite")


@dataclass(frozen=True, slots=True)
class CoordinateDerivativeResult:
    """Central-difference derivative and convergence diagnostics for one coordinate."""

    coordinate_label: str
    derivative_ev_per_unit: float
    unit: str
    step_sizes: tuple[float, ...]
    central_derivatives_ev_per_unit: tuple[float, ...]
    max_relative_spread: float
    fit_rms_ev_per_unit: float
    passes_linearity: bool


@dataclass(frozen=True, slots=True)
class DimerDerivativeResult:
    """Six-component local derivative with explicit finite-difference diagnostics."""

    coordinates: tuple[CoordinateDerivativeResult, ...]

    def __post_init__(self) -> None:
        labels = tuple(item.coordinate_label for item in self.coordinates)
        if labels != DIMER_COORDINATE_LABELS:
            raise ValueError("coordinate results must follow the canonical six-DOF order")

    @property
    def local_vector(self) -> FloatArray:
        return np.asarray(
            [item.derivative_ev_per_unit for item in self.coordinates],
            dtype=np.float64,
        )

    @property
    def all_pass_linearity(self) -> bool:
        return all(item.passes_linearity for item in self.coordinates)


def prepare_scan_geometries(
    reference: RigidDimerReference,
    plan: FiniteDifferenceScanPlan,
) -> tuple[tuple[DimerPerturbation, RigidDimerReference], ...]:
    """Generate all displaced dimer geometries required by ``plan``."""
    coordinate_map = {item.label: item for item in DIMER_COORDINATES}
    return tuple(
        (
            perturbation,
            reference.perturb_target(
                coordinate_map[perturbation.coordinate_label], perturbation.amount
            ),
        )
        for perturbation in plan.perturbations()
    )


def _coordinate_samples(
    samples: tuple[TransferIntegralSample, ...], coordinate: DimerCoordinate
) -> dict[float, float]:
    selected: dict[float, float] = {}
    for item in samples:
        if item.perturbation.coordinate_label != coordinate.label:
            continue
        amount = float(item.perturbation.amount)
        if amount in selected:
            raise ValueError(
                f"duplicate finite-difference sample for {coordinate.label} at {amount}"
            )
        selected[amount] = float(item.transfer_integral_ev)
    return selected


def analyze_finite_difference_scan(
    samples: tuple[TransferIntegralSample, ...],
    plan: FiniteDifferenceScanPlan,
    *,
    maximum_relative_spread: float = 0.10,
    derivative_scale_floor_ev_per_unit: float = 1.0e-3,
) -> DimerDerivativeResult:
    """Estimate the zero-step derivatives from multi-step central differences.

    For a smooth transfer-integral curve, the central derivative obeys
    ``d(h) = g + c h^2 + O(h^4)``.  G5e therefore fits the derivative against
    ``h^2`` and uses the intercept as the zero-step estimate.  The spread of the
    raw central derivatives provides a deliberately simple linearity gate.
    """
    if maximum_relative_spread <= 0.0 or not np.isfinite(maximum_relative_spread):
        raise ValueError("maximum_relative_spread must be finite and positive")
    if derivative_scale_floor_ev_per_unit <= 0.0 or not np.isfinite(
        derivative_scale_floor_ev_per_unit
    ):
        raise ValueError("derivative scale floor must be finite and positive")

    output: list[CoordinateDerivativeResult] = []
    for coordinate in DIMER_COORDINATES:
        evaluated = _coordinate_samples(samples, coordinate)
        steps = tuple(sorted(float(value) for value in plan.steps_for(coordinate)))
        derivatives: list[float] = []
        for step in steps:
            try:
                minus = evaluated[-step]
                plus = evaluated[step]
            except KeyError as exc:
                raise ValueError(
                    f"missing +/- finite-difference sample for {coordinate.label} at {step}"
                ) from exc
            derivatives.append((plus - minus) / (2.0 * step))

        h2 = np.square(np.asarray(steps, dtype=np.float64))
        derivative_values = np.asarray(derivatives, dtype=np.float64)
        design = np.column_stack((np.ones_like(h2), h2))
        coefficients, *_ = np.linalg.lstsq(design, derivative_values, rcond=None)
        intercept = float(coefficients[0])
        fitted = design @ coefficients
        fit_rms = float(np.sqrt(np.mean(np.square(derivative_values - fitted))))
        spread = float(np.max(np.abs(derivative_values - intercept)))
        scale = max(abs(intercept), derivative_scale_floor_ev_per_unit)
        relative_spread = spread / scale
        output.append(
            CoordinateDerivativeResult(
                coordinate_label=coordinate.label,
                derivative_ev_per_unit=intercept,
                unit=f"eV/{coordinate.unit}",
                step_sizes=steps,
                central_derivatives_ev_per_unit=tuple(derivatives),
                max_relative_spread=relative_spread,
                fit_rms_ev_per_unit=fit_rms,
                passes_linearity=relative_spread <= maximum_relative_spread,
            )
        )
    return DimerDerivativeResult(tuple(output))


def local_derivative_to_global_rigid_body_vector(
    local_derivative: FloatArray,
    local_axes_global: FloatArray,
) -> FloatArray:
    """Transform local translation/rotation derivatives into one global 6-vector.

    The first three entries are translation derivatives and the final three are
    infinitesimal rotation-pseudovector derivatives.  Both transform with the
    same proper-rotation matrix.  This does *not* establish symmetry relations
    between distinct oriented bond families; every promoted G3 family must have
    its own reference frame or an explicitly proven symmetry mapping.
    """
    derivative = np.asarray(local_derivative, dtype=np.float64)
    axes = np.asarray(local_axes_global, dtype=np.float64)
    if derivative.shape != (6,) or not np.all(np.isfinite(derivative)):
        raise ValueError("local derivative must be a finite six-vector")
    if axes.shape != (3, 3) or not np.allclose(
        axes.T @ axes, np.eye(3), rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("local axes must be an orthonormal 3x3 matrix")
    if not np.isclose(np.linalg.det(axes), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("local axes must form a right-handed frame")
    return np.concatenate((axes @ derivative[:3], axes @ derivative[3:]))
