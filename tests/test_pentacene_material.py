import math

import numpy as np

from holstein_peierls.materials.pentacene import (
    PENTACENE_293K,
    STEHR_2011_HOLE_COUPLINGS,
    UNRESOLVED_PENTACENE_FIELDS,
    pentacene_293k_candidate_transport_lattice,
    pentacene_293k_projected_lattice,
)
from holstein_peierls.molecular_hamiltonian import bond_displacement_angstrom


def test_293k_projected_cell_matches_mattheus_ab_metrics() -> None:
    lattice = pentacene_293k_projected_lattice(2, 2)
    a1 = np.asarray(lattice.a1_angstrom)
    a2 = np.asarray(lattice.a2_angstrom)
    assert np.isclose(np.linalg.norm(a1), 6.266, rtol=0.0, atol=1e-14)
    assert np.isclose(np.linalg.norm(a2), 7.775, rtol=0.0, atol=1e-14)
    angle = math.degrees(
        math.acos(float(np.dot(a1, a2) / (np.linalg.norm(a1) * np.linalg.norm(a2))))
    )
    assert np.isclose(angle, 84.684, rtol=0.0, atol=1e-12)


def test_293k_projected_basis_uses_reported_inversion_centers() -> None:
    lattice = pentacene_293k_projected_lattice(2, 2)
    assert lattice.basis[0].position_fractional == (0.0, 0.0)
    assert lattice.basis[1].position_fractional == (0.5, 0.5)
    positions = lattice.physical_positions_angstrom()
    expected_b = 0.5 * (
        np.asarray(lattice.a1_angstrom) + np.asarray(lattice.a2_angstrom)
    )
    assert np.allclose(positions[1], expected_b, rtol=0.0, atol=1e-15)


def test_candidate_transport_graph_has_expected_four_geometric_families() -> None:
    lattice = pentacene_293k_candidate_transport_lattice(3, 3)
    assert [family.label for family in lattice.bonds] == [
        "a_AA",
        "a_BB",
        "diag_plus_AB",
        "diag_minus_AB",
    ]
    assert len(lattice.generated_bonds()) == 4 * 9

    displacements = {
        family.label: bond_displacement_angstrom(lattice, family)
        for family in lattice.bonds
    }
    a1 = np.asarray(lattice.a1_angstrom)
    a2 = np.asarray(lattice.a2_angstrom)
    assert np.allclose(displacements["a_AA"], a1)
    assert np.allclose(displacements["a_BB"], a1)
    assert np.allclose(displacements["diag_plus_AB"], 0.5 * (a1 + a2))
    assert np.allclose(displacements["diag_minus_AB"], 0.5 * (-a1 + a2))


def test_stehr_couplings_remain_unsigned_evidence_not_model_hoppings() -> None:
    assert [item.magnitude_mev for item in STEHR_2011_HOLE_COUPLINGS] == [
        90.69,
        55.05,
        39.68,
        36.62,
    ]
    assert all(not item.signed_value_known for item in STEHR_2011_HOLE_COUPLINGS)
    assert {item.candidate_geometry_group for item in STEHR_2011_HOLE_COUPLINGS} == {
        "diagonal_AB",
        "a_axis_same_basis",
    }


def test_material_record_keeps_blocking_parameters_explicitly_unresolved() -> None:
    required = {
        "signed_hopping_assignment",
        "bond_resolved_peierls_derivatives",
        "effective_bond_stiffness_matrices",
        "screened_onsite_hubbard_u",
        "screened_short_range_contact_interactions",
        "long_range_dielectric_convention",
    }
    assert required.issubset(set(UNRESOLVED_PENTACENE_FIELDS))
    assert PENTACENE_293K.source_doi == "10.1107/S010827010100703X"
    assert PENTACENE_293K.ccdc_id == "170186"
