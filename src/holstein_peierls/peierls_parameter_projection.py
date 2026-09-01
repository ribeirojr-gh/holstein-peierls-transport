"""Projection of explicit dimer derivatives into G3 coupling vectors.

The G5e finite-difference stage evaluates derivatives in a dimer-local frame.
G3, however, requires one coupling vector for each *oriented* bond family in a
single coordinate convention.  This module performs that local-to-global
projection and deliberately refuses to infer missing bond families by symmetry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .dimer_finite_difference import (
    DimerDerivativeResult,
    local_derivative_to_global_rigid_body_vector,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class OrientedBondDerivative:
    """One explicitly evaluated bond derivative and its local-to-global frame."""

    family_label: str
    derivative: DimerDerivativeResult
    local_axes_global: tuple[tuple[float, float, float], ...]

    def __post_init__(self) -> None:
        if not self.family_label:
            raise ValueError("family_label must be non-empty")
        axes = np.asarray(self.local_axes_global, dtype=np.float64)
        if axes.shape != (3, 3) or not np.all(np.isfinite(axes)):
            raise ValueError("local_axes_global must be a finite 3x3 matrix")
        if not np.allclose(axes.T @ axes, np.eye(3), rtol=0.0, atol=1.0e-12):
            raise ValueError("local axes must be orthonormal")
        if not np.isclose(np.linalg.det(axes), 1.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("local axes must form a right-handed frame")

    @property
    def global_vector(self) -> FloatArray:
        return local_derivative_to_global_rigid_body_vector(
            self.derivative.local_vector,
            np.asarray(self.local_axes_global, dtype=np.float64),
        )


def assemble_g3_bond_mode_couplings(
    derivatives: tuple[OrientedBondDerivative, ...],
    expected_family_labels: tuple[str, ...],
    *,
    require_linearity: bool = True,
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    """Return explicit G3-compatible six-mode coupling vectors.

    No family is generated from another family by inversion, translation, or
    basis-label arguments.  Such a reduction can be added later only through an
    explicit, independently tested symmetry map.
    """
    labels = [item.family_label for item in derivatives]
    if len(labels) != len(set(labels)):
        raise ValueError("oriented bond derivative labels must be unique")
    if len(expected_family_labels) != len(set(expected_family_labels)):
        raise ValueError("expected family labels must be unique")
    if set(labels) != set(expected_family_labels):
        missing = sorted(set(expected_family_labels) - set(labels))
        extra = sorted(set(labels) - set(expected_family_labels))
        raise ValueError(
            f"explicit derivative set does not match bond families; missing={missing}, extra={extra}"
        )

    by_label = {item.family_label: item for item in derivatives}
    output: list[tuple[str, tuple[float, ...]]] = []
    for label in expected_family_labels:
        item = by_label[label]
        if require_linearity and not item.derivative.all_pass_linearity:
            raise ValueError(f"finite-difference linearity gate failed for {label}")
        vector = item.global_vector
        output.append((label, tuple(float(value) for value in vector)))
    return tuple(output)


def predict_linearized_transfer_integral_variance(
    coupling_vector: FloatArray,
    relative_coordinate_covariance: FloatArray,
) -> float:
    """Return ``g.T @ Cov(Delta q) @ g`` for a six-mode rigid-body model.

    The covariance must use the same coordinate units as the coupling vector:
    global translations in angstrom followed by infinitesimal rotations in
    radians.  This is the later G5f bridge to the 295 K Neef disorder data.
    """
    coupling = np.asarray(coupling_vector, dtype=np.float64)
    covariance = np.asarray(relative_coordinate_covariance, dtype=np.float64)
    if coupling.shape != (6,) or not np.all(np.isfinite(coupling)):
        raise ValueError("coupling_vector must be a finite six-vector")
    if covariance.shape != (6, 6) or not np.all(np.isfinite(covariance)):
        raise ValueError("relative-coordinate covariance must be a finite 6x6 matrix")
    if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1.0e-12):
        raise ValueError("relative-coordinate covariance must be symmetric")
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(covariance)))
    if minimum_eigenvalue < -1.0e-12:
        raise ValueError("relative-coordinate covariance must be positive semidefinite")
    variance = float(coupling @ covariance @ coupling)
    return max(0.0, variance)
