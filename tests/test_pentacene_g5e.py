import math

from holstein_peierls.materials.pentacene import (
    pentacene_293k_candidate_transport_lattice,
)
from holstein_peierls.materials.pentacene_g5e import (
    NEEF_2024_FODFT_PROTOCOL,
    PENTACENE_G5E_G3_MODE_UNITS,
    PENTACENE_G5E_ORIENTED_BOND_FAMILIES,
    PENTACENE_G5E_ROTATION_STEPS_DEGREE,
    PENTACENE_G5E_TRANSLATION_STEPS_ANGSTROM,
    pentacene_g5e_displaced_calculation_count_per_oriented_bond,
    pentacene_g5e_full_oriented_scan_calculation_count,
    pentacene_g5e_promotion_requirements,
    pentacene_g5e_scan_plan,
)


def test_pentacene_g5e_oriented_families_match_material_graph_exactly() -> None:
    lattice = pentacene_293k_candidate_transport_lattice(2, 2)
    assert PENTACENE_G5E_ORIENTED_BOND_FAMILIES == tuple(
        family.label for family in lattice.bonds
    )


def test_pentacene_g5e_initial_scan_has_three_scales_and_216_displacements() -> None:
    plan = pentacene_g5e_scan_plan()
    assert plan.translation_steps_angstrom == (0.0025, 0.005, 0.01)
    assert PENTACENE_G5E_TRANSLATION_STEPS_ANGSTROM[0] == 0.0025
    assert PENTACENE_G5E_ROTATION_STEPS_DEGREE == (0.25, 0.5, 1.0)
    assert tuple(round(math.degrees(value), 12) for value in plan.rotation_steps_radian) == (
        0.25,
        0.5,
        1.0,
    )
    assert pentacene_g5e_displaced_calculation_count_per_oriented_bond() == 36
    assert pentacene_g5e_full_oriented_scan_calculation_count() == 216


def test_neef_fodft_reference_protocol_is_fully_traceable() -> None:
    protocol = NEEF_2024_FODFT_PROTOCOL
    assert protocol.implementation == "FHI-aims"
    assert protocol.method_variant == "H2n@DA fragment-orbital DFT"
    assert protocol.exchange_correlation == "PBE"
    assert protocol.basis_tier == "tier2"
    assert protocol.integration_grids == "tight"
    assert protocol.electronic_level_convergence_ev == 1.0e-6
    assert protocol.include_vdw_correction is False
    assert protocol.source_doi == "10.48550/arXiv.2412.06030"
    assert "highest occupied molecular orbitals" in protocol.coupling_definition


def test_g5e_promotion_contract_forbids_silent_symmetry_copy() -> None:
    requirements = pentacene_g5e_promotion_requirements()
    assert "no_silent_symmetry_copy_between_distinct_oriented_bond_families" in requirements
    assert "signed_transfer_integral_with_continuous_orbital_gauge" in requirements
    assert "later_room_temperature_covariance_validation_against_neef_sigma_t" in requirements
    assert PENTACENE_G5E_G3_MODE_UNITS == (
        "angstrom",
        "angstrom",
        "angstrom",
        "radian",
        "radian",
        "radian",
    )
