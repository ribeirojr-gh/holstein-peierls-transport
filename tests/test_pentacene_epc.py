import pytest

from holstein_peierls.materials.pentacene_epc import (
    GNOLI_2025_DATASET_DOI,
    GNOLI_2025_EPC_FINITE_DISPLACEMENT_ANGSTROM,
    GNOLI_2025_LT_DOMINANT_EPC,
    GNOLI_2025_REPRESENTATION,
    GNOLI_2025_SOURCE_DOI,
    ModalEpcEvidence,
    g5d_projection_requirements,
)


def test_gnoli_lt_dominant_modal_epc_values_are_traceable() -> None:
    assert GNOLI_2025_SOURCE_DOI == "10.1021/acs.jpcc.5c04906"
    assert GNOLI_2025_DATASET_DOI == "10.5281/zenodo.17368135"
    assert GNOLI_2025_EPC_FINITE_DISPLACEMENT_ANGSTROM == 0.0025
    assert [item.q_point_label for item in GNOLI_2025_LT_DOMINANT_EPC] == ["U", "X"]
    assert [item.frequency_cm_inverse for item in GNOLI_2025_LT_DOMINANT_EPC] == [
        26.7,
        39.2,
    ]
    assert [item.epc_ev_per_angstrom for item in GNOLI_2025_LT_DOMINANT_EPC] == [
        0.27,
        0.24,
    ]


def test_lt_dominant_modes_are_long_axis_translation_dominated() -> None:
    assert [item.dominant_coordinate_fraction for item in GNOLI_2025_LT_DOMINANT_EPC] == [
        0.67,
        0.46,
    ]
    assert all(
        item.dominant_coordinate == "translation_long_inertia_axis"
        for item in GNOLI_2025_LT_DOMINANT_EPC
    )


def test_modal_epc_is_explicitly_not_a_direct_g3_bond_derivative() -> None:
    assert GNOLI_2025_REPRESENTATION == "normal_mode_bandwidth_deformation_potential"
    assert all(not item.direct_g3_compatible for item in GNOLI_2025_LT_DOMINANT_EPC)
    requirements = g5d_projection_requirements()
    assert "phonon_eigenvector_in_molecular_local_coordinates" in requirements
    assert "bond_resolved_signed_transfer_integral_response" in requirements
    assert "consistent_elastic_or_phonon_hessian_representation" in requirements


def test_modal_evidence_rejects_unphysical_values_or_false_direct_promotion() -> None:
    with pytest.raises(ValueError, match="frequency"):
        ModalEpcEvidence("LT", "U", 0.0, 0.27, "TL", 0.67)
    with pytest.raises(ValueError, match="non-negative"):
        ModalEpcEvidence("LT", "U", 26.7, -0.01, "TL", 0.67)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ModalEpcEvidence("LT", "U", 26.7, 0.27, "TL", 1.2)
    with pytest.raises(ValueError, match="direct-G3-compatible"):
        ModalEpcEvidence("LT", "U", 26.7, 0.27, "TL", 0.67, direct_g3_compatible=True)


def test_strongest_lt_records_are_non_gamma_full_bz_evidence() -> None:
    assert all(item.q_point_label != "Gamma" for item in GNOLI_2025_LT_DOMINANT_EPC)
