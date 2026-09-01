"""Pentacene-specific G5e finite-difference and backend provenance contract.

This module intentionally contains no FHI-aims input writer.  It records the
published electronic-coupling protocol and the numerical scan grid used by the
backend-independent G5e pipeline.  A future backend adapter must reproduce the
published H2n@DA FO-DFT settings rather than guessing code-specific keywords.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..dimer_finite_difference import FiniteDifferenceScanPlan


@dataclass(frozen=True, slots=True)
class FoDftReferenceProtocol:
    """Traceable electronic-coupling protocol used to evaluate dimer scans."""

    source_doi: str
    implementation: str
    method_variant: str
    exchange_correlation: str
    basis_tier: str
    integration_grids: str
    electronic_level_convergence_ev: float
    include_vdw_correction: bool
    coupling_definition: str


NEEF_2024_FODFT_PROTOCOL = FoDftReferenceProtocol(
    source_doi="10.48550/arXiv.2412.06030",
    implementation="FHI-aims",
    method_variant="H2n@DA fragment-orbital DFT",
    exchange_correlation="PBE",
    basis_tier="tier2",
    integration_grids="tight",
    electronic_level_convergence_ev=1.0e-6,
    include_vdw_correction=False,
    coupling_definition=(
        "Kohn-Sham matrix element between the highest occupied molecular orbitals "
        "of donor and acceptor fragments"
    ),
)


# The smallest translation step matches the 0.0025 A finite-displacement scale
# independently used in the peer-reviewed full-BZ EPC validation of Gnoli et al.
# The two larger translation steps are G5e convergence checks.  The rotational
# grid is a numerical starting proposal, not a published pentacene parameter;
# it must pass the same h->0 stability gate before any derivative is promoted.
PENTACENE_G5E_TRANSLATION_STEPS_ANGSTROM = (0.0025, 0.005, 0.01)
PENTACENE_G5E_ROTATION_STEPS_DEGREE = (0.25, 0.5, 1.0)


PENTACENE_G5E_ORIENTED_BOND_FAMILIES = (
    "a_AA",
    "a_BB",
    "diag_plus_AB_forward",
    "diag_plus_AB_backward",
    "diag_minus_AB_forward",
    "diag_minus_AB_backward",
)


PENTACENE_G5E_G3_MODE_UNITS = (
    "angstrom",
    "angstrom",
    "angstrom",
    "radian",
    "radian",
    "radian",
)


def pentacene_g5e_scan_plan() -> FiniteDifferenceScanPlan:
    """Return the initial three-scale central-difference plan for each dimer."""
    return FiniteDifferenceScanPlan(
        translation_steps_angstrom=PENTACENE_G5E_TRANSLATION_STEPS_ANGSTROM,
        rotation_steps_radian=tuple(
            math.radians(value) for value in PENTACENE_G5E_ROTATION_STEPS_DEGREE
        ),
    )


def pentacene_g5e_promotion_requirements() -> tuple[str, ...]:
    """Return the non-negotiable gates before a derivative can enter G3."""
    return (
        "signed_transfer_integral_with_continuous_orbital_gauge",
        "fixed_reference_dimer_geometry_and_explicit_rigid_body_pivots",
        "explicit_right_handed_long_short_normal_frame_for_each_oriented_bond",
        "central_difference_samples_at_three_or_more_step_sizes",
        "zero_step_derivative_stable_under_step_size_reduction",
        "all_six_rigid_body_coordinates_evaluated_per_promoted_oriented_bond",
        "no_silent_symmetry_copy_between_distinct_oriented_bond_families",
        "local_derivative_transformed_to_global_translation_rotation_coordinates",
        "same_electronic_structure_protocol_for_all_scan_points",
        "later_room_temperature_covariance_validation_against_neef_sigma_t",
    )


def pentacene_g5e_displaced_calculation_count_per_oriented_bond() -> int:
    """Return the number of non-equilibrium transfer-integral evaluations per bond."""
    return len(pentacene_g5e_scan_plan().perturbations())


def pentacene_g5e_full_oriented_scan_calculation_count() -> int:
    """Return the conservative count when all six oriented families are evaluated."""
    return (
        len(PENTACENE_G5E_ORIENTED_BOND_FAMILIES)
        * pentacene_g5e_displaced_calculation_count_per_oriented_bond()
    )
