import math

import numpy as np

from holstein_peierls.materials.pentacene import (
    DEWIJS_2003_HOMO_ONSITE_DIFFERENCE_EV,
    DEWIJS_2003_HOMO_SIGNED_HOPPINGS_EV,
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

    # The three-parameter signed fit gives 0.589498 eV for the complete two-band
    # HOMO complex, consistent with the approximately 0.6 eV width reported by
    # de Wijs et al. The even 8x8 reciprocal mesh contains the M point where both
    # extrema occur for this fitted model.
    assert np.isclose(bandwidth, 0.5894980915999644, rtol=0.0, atol=2e-14)
    assert abs(bandwidth - 0.6) < 0.02
    assert DEWIJS_2003_HOMO_SIGNED_HOPPINGS_EV == (
        ("a_same_basis", 0.031),
        ("diag_plus_AB", -0.056),
        ("diag_minus_AB", 0.091),
    )
    assert DEWIJS_2003_HOMO_ONSITE_DIFFERENCE_EV == 0.042
    assert PENTACENE_90K.temperature_k == 90.0


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
    assert np.isclose(bandwidth, 0.5846727523889548, rtol=0.0, atol=3e-14)


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


def test_material_record_keeps_remaining_blocking_parameters_explicitly_unresolved() -> None:
    required = {
        "single_source_signed_hopping_parameterization",
        "bond_resolved_peierls_derivatives",
        "effective_bond_stiffness_matrices",
        "screened_onsite_hubbard_u",
        "screened_short_range_contact_interactions",
        "long_range_dielectric_convention",
    }
    assert required.issubset(set(UNRESOLVED_PENTACENE_FIELDS))
    assert PENTACENE_293K.source_doi == "10.1107/S010827010100703X"
    assert PENTACENE_293K.ccdc_id == "170186"
