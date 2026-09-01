import math

import numpy as np

from holstein_peierls.materials.pentacene import (
    DEWIJS_2003_HOMO_ONSITE_DIFFERENCE_EV,
    DEWIJS_2003_HOMO_SIGNED_HOPPINGS_EV,
    NEEF_2024_ARPES_HOLE_HOPPINGS,
    NEEF_2024_MD_HOLE_HOPPING_STATISTICS,
    NEEF_2024_MD_TEMPERATURE_K,
    NEEF_2024_MEAN_TRANSLATIONAL_FLUCTUATION_ANGSTROM,
    NEEF_2024_PEER_REVIEWED,
    NEEF_2024_SOURCE_DOI,
    PENTACENE_293K,
    PENTACENE_293K_CROSS_SOURCE_HOLE_HOPPINGS_EV,
    PENTACENE_293K_CROSS_SOURCE_STATUS,
    PENTACENE_90K,
    STEHR_2011_HOLE_COUPLINGS,
    UNRESOLVED_PENTACENE_FIELDS,
    pentacene_293k_candidate_transport_lattice,
    pentacene_293k_cross_source_hole_model,
    pentacene_293k_projected_lattice,
    pentacene_90k_dewijs_homo_model,
    pentacene_room_temperature_neef_arpes_homo_model,
)
from holstein_peierls.molecular_hamiltonian import (
    MolecularTightBindingModel,
    bloch_hamiltonian,
    bond_displacement_angstrom,
    reciprocal_vectors_per_angstrom,
    sampled_bloch_spectrum,
)


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


def test_candidate_transport_graph_has_complete_herringbone_connectivity() -> None:
    lattice = pentacene_293k_candidate_transport_lattice(3, 3)
    assert [family.label for family in lattice.bonds] == [
        "a_AA",
        "a_BB",
        "diag_plus_AB_forward",
        "diag_plus_AB_backward",
        "diag_minus_AB_forward",
        "diag_minus_AB_backward",
    ]
    bonds = lattice.generated_bonds()
    assert len(bonds) == 6 * 9

    degree = np.zeros(lattice.n_sites, dtype=np.int64)
    for bond in bonds:
        degree[bond.source] += 1
        degree[bond.target] += 1
    assert np.array_equal(degree, np.full(lattice.n_sites, 6, dtype=np.int64))

    displacements = {
        family.label: bond_displacement_angstrom(lattice, family)
        for family in lattice.bonds
    }
    a1 = np.asarray(lattice.a1_angstrom)
    a2 = np.asarray(lattice.a2_angstrom)
    assert np.allclose(displacements["a_AA"], a1)
    assert np.allclose(displacements["a_BB"], a1)
    assert np.allclose(
        displacements["diag_plus_AB_forward"], 0.5 * (a1 + a2)
    )
    assert np.allclose(
        displacements["diag_plus_AB_backward"], -0.5 * (a1 + a2)
    )
    assert np.allclose(
        displacements["diag_minus_AB_forward"], 0.5 * (a1 - a2)
    )
    assert np.allclose(
        displacements["diag_minus_AB_backward"], -0.5 * (a1 - a2)
    )


def test_stehr_293k_magnitudes_have_directional_mapping_but_remain_unsigned() -> None:
    assert [item.label for item in STEHR_2011_HOLE_COUPLINGS] == [
        "V1",
        "V2",
        "V3",
        "V4",
    ]
    assert [item.magnitude_mev for item in STEHR_2011_HOLE_COUPLINGS] == [
        90.69,
        55.05,
        39.68,
        36.62,
    ]
    assert [item.direction_miller for item in STEHR_2011_HOLE_COUPLINGS[:2]] == [
        (1, -1, 0),
        (1, 1, 0),
    ]
    assert [item.candidate_geometry_group for item in STEHR_2011_HOLE_COUPLINGS] == [
        "diag_minus_AB",
        "diag_plus_AB",
        "a_axis_same_basis",
        "a_axis_same_basis",
    ]
    assert all(not item.signed_value_known for item in STEHR_2011_HOLE_COUPLINGS)


def test_dewijs_90k_signed_reference_reproduces_reported_homo_bandwidth() -> None:
    model = pentacene_90k_dewijs_homo_model(8, 8)
    spectrum = sampled_bloch_spectrum(model)
    bandwidth = float(spectrum[-1] - spectrum[0])

    assert np.isclose(bandwidth, 0.5894980915999644, rtol=0.0, atol=2e-14)
    assert abs(bandwidth - 0.6) < 0.02
    assert DEWIJS_2003_HOMO_SIGNED_HOPPINGS_EV == (
        ("a_same_basis", 0.031),
        ("diag_plus_AB", -0.056),
        ("diag_minus_AB", 0.091),
    )
    assert DEWIJS_2003_HOMO_ONSITE_DIFFERENCE_EV == 0.042
    assert PENTACENE_90K.temperature_k == 90.0
    assert PENTACENE_90K.ccdc_id == "170187"


def test_dewijs_ab_sign_flip_is_a_basis_gauge_transformation() -> None:
    reference = pentacene_90k_dewijs_homo_model(4, 4)
    flipped = MolecularTightBindingModel(
        lattice=reference.lattice,
        bond_transfer_integrals_ev=tuple(
            (label, -value if label.startswith("diag_") else value)
            for label, value in reference.bond_transfer_integrals_ev
        ),
        onsite_energies_ev=reference.onsite_energies_ev,
    )
    reciprocal = reciprocal_vectors_per_angstrom(reference.lattice)
    kpoints = (
        np.zeros(2),
        0.23 * reciprocal[:, 0] + 0.37 * reciprocal[:, 1],
        0.5 * (reciprocal[:, 0] + reciprocal[:, 1]),
    )
    for k in kpoints:
        expected = np.linalg.eigvalsh(bloch_hamiltonian(reference, k))
        candidate = np.linalg.eigvalsh(bloch_hamiltonian(flipped, k))
        assert np.allclose(candidate, expected, rtol=0.0, atol=2e-15)


def test_293k_cross_source_candidate_uses_stehr_magnitudes_and_dewijs_sign_pattern() -> None:
    model = pentacene_293k_cross_source_hole_model(6, 6)
    transfer = model.transfer_integrals
    assert PENTACENE_293K_CROSS_SOURCE_HOLE_HOPPINGS_EV == (
        ("a_AA", 0.03968),
        ("a_BB", 0.03662),
        ("diag_plus_AB_forward", -0.05505),
        ("diag_plus_AB_backward", -0.05505),
        ("diag_minus_AB_forward", 0.09069),
        ("diag_minus_AB_backward", 0.09069),
    )
    assert np.isclose(abs(transfer["diag_minus_AB_forward"]), 0.09069)
    assert np.isclose(abs(transfer["diag_plus_AB_forward"]), 0.05505)
    assert transfer["diag_plus_AB_forward"] < 0.0
    assert transfer["diag_minus_AB_forward"] > 0.0
    assert "candidate only" in PENTACENE_293K_CROSS_SOURCE_STATUS

    spectrum = sampled_bloch_spectrum(model)
    bandwidth = float(spectrum[-1] - spectrum[0])
    assert np.isclose(bandwidth, 0.5829921234459348, rtol=0.0, atol=3e-14)
    assert 0.58 < bandwidth < 0.59


def test_293k_a_basis_label_swap_leaves_equal_onsite_band_spectrum_invariant() -> None:
    reference = pentacene_293k_cross_source_hole_model(4, 4)
    swapped = MolecularTightBindingModel(
        lattice=reference.lattice,
        bond_transfer_integrals_ev=tuple(
            (
                label,
                reference.transfer_integrals[
                    "a_BB" if label == "a_AA" else "a_AA"
                ],
            )
            if label in {"a_AA", "a_BB"}
            else (label, value)
            for label, value in reference.bond_transfer_integrals_ev
        ),
        onsite_energies_ev=reference.onsite_energies_ev,
    )
    reciprocal = reciprocal_vectors_per_angstrom(reference.lattice)
    for k in (
        0.17 * reciprocal[:, 0] + 0.41 * reciprocal[:, 1],
        0.5 * reciprocal[:, 0],
        0.5 * (reciprocal[:, 0] + reciprocal[:, 1]),
    ):
        expected = np.linalg.eigvalsh(bloch_hamiltonian(reference, k))
        candidate = np.linalg.eigvalsh(bloch_hamiltonian(swapped, k))
        assert np.allclose(candidate, expected, rtol=0.0, atol=2e-15)


def test_neef_arpes_reference_is_signed_single_source_room_temperature_evidence() -> None:
    assert NEEF_2024_SOURCE_DOI == "10.48550/arXiv.2412.06030"
    assert NEEF_2024_PEER_REVIEWED is False
    assert [(item.label, item.value_mev, item.uncertainty_mev) for item in NEEF_2024_ARPES_HOLE_HOPPINGS] == [
        ("t_a", 35.0, 10.0),
        ("t_plus", 55.0, 5.0),
        ("t_minus", -70.0, 5.0),
    ]
    assert [item.temperature_description for item in NEEF_2024_ARPES_HOLE_HOPPINGS] == [
        "room temperature",
        "room temperature",
        "room temperature",
    ]
    assert all("ARPES" in item.evidence_type for item in NEEF_2024_ARPES_HOLE_HOPPINGS)
    assert all(not item.peer_reviewed for item in NEEF_2024_ARPES_HOLE_HOPPINGS)


def test_neef_room_temperature_model_matches_published_analytic_herringbone_hamiltonian() -> None:
    model = pentacene_room_temperature_neef_arpes_homo_model(5, 4)
    reciprocal = reciprocal_vectors_per_angstrom(model.lattice)
    k = 0.231 * reciprocal[:, 0] + 0.347 * reciprocal[:, 1]
    candidate = bloch_hamiltonian(model, k)

    a = np.asarray(model.lattice.a1_angstrom)
    b = np.asarray(model.lattice.a2_angstrom)
    ka = float(np.dot(k, a))
    kb = float(np.dot(k, b))
    h0 = 2.0 * 0.035 * np.cos(ka)
    h1 = (
        2.0 * 0.055 * np.cos(0.5 * (ka + kb))
        + 2.0 * (-0.070) * np.cos(0.5 * (ka - kb))
    )
    expected = np.array(((h0, h1), (h1, h0)), dtype=np.complex128)
    assert np.allclose(candidate, expected, rtol=0.0, atol=2e-15)


def test_neef_and_dewijs_share_the_same_frustrated_sign_topology_up_to_basis_gauge() -> None:
    dewijs_ta = dict(DEWIJS_2003_HOMO_SIGNED_HOPPINGS_EV)["a_same_basis"]
    dewijs_plus = -dict(DEWIJS_2003_HOMO_SIGNED_HOPPINGS_EV)["diag_plus_AB"]
    dewijs_minus = -dict(DEWIJS_2003_HOMO_SIGNED_HOPPINGS_EV)["diag_minus_AB"]
    neef = {item.label: item.value_mev for item in NEEF_2024_ARPES_HOLE_HOPPINGS}

    assert dewijs_ta > 0.0 and dewijs_plus > 0.0 and dewijs_minus < 0.0
    assert neef["t_a"] > 0.0 and neef["t_plus"] > 0.0 and neef["t_minus"] < 0.0
    assert dewijs_ta * dewijs_plus * dewijs_minus < 0.0
    assert neef["t_a"] * neef["t_plus"] * neef["t_minus"] < 0.0


def test_neef_295k_md_statistics_are_stored_as_disorder_evidence_not_peierls_parameters() -> None:
    assert NEEF_2024_MD_TEMPERATURE_K == 295.0
    assert [(item.label, item.mean_mev, item.standard_deviation_mev) for item in NEEF_2024_MD_HOLE_HOPPING_STATISTICS] == [
        ("t_a", 32.0, 12.0),
        ("t_plus", 39.5, 18.0),
        ("t_minus", -78.8, 18.4),
    ]
    assert all(item.temperature_k == 295.0 for item in NEEF_2024_MD_HOLE_HOPPING_STATISTICS)
    assert all("FO-DFT" in item.method for item in NEEF_2024_MD_HOLE_HOPPING_STATISTICS)
    assert np.isclose(NEEF_2024_MEAN_TRANSLATIONAL_FLUCTUATION_ANGSTROM, 0.21)

    arpes_magnitudes = {item.label: abs(item.value_mev) for item in NEEF_2024_ARPES_HOLE_HOPPINGS}
    relative_disorder = {
        item.label: item.standard_deviation_mev / arpes_magnitudes[item.label]
        for item in NEEF_2024_MD_HOLE_HOPPING_STATISTICS
    }
    assert np.isclose(relative_disorder["t_a"], 12.0 / 35.0)
    assert np.isclose(relative_disorder["t_plus"], 18.0 / 55.0)
    assert np.isclose(relative_disorder["t_minus"], 18.4 / 70.0)


def test_material_record_keeps_remaining_blocking_parameters_explicitly_unresolved() -> None:
    required = {
        "peer_reviewed_temperature_matched_signed_hopping_parameterization",
        "bond_resolved_peierls_derivatives",
        "effective_bond_stiffness_matrices",
        "screened_onsite_hubbard_u",
        "screened_short_range_contact_interactions",
        "long_range_dielectric_convention",
    }
    assert required.issubset(set(UNRESOLVED_PENTACENE_FIELDS))
    assert "single_source_signed_hopping_parameterization" not in UNRESOLVED_PENTACENE_FIELDS
    assert PENTACENE_293K.source_doi == "10.1107/S010827010100703X"
    assert PENTACENE_293K.ccdc_id == "170186"
