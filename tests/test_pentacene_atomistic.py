import math

import numpy as np
import pytest

from holstein_peierls.materials.pentacene import (
    pentacene_293k_candidate_transport_lattice,
)
from holstein_peierls.materials.pentacene_atomistic import (
    MATT_HEUS_2001_293K_CCDC,
    MATT_HEUS_2001_CIF_PATH,
    MATT_HEUS_2001_COORDINATE_STATUS,
    MATT_HEUS_2001_SOURCE_DOI,
    PENTACENE_293K_ATOMISTIC_DIMER_FAMILIES,
    PENTACENE_293K_CELL_3D,
    pentacene_293k_atomistic_basis_a,
    pentacene_293k_atomistic_basis_b,
    pentacene_293k_atomistic_dimer,
    pentacene_293k_atomistic_dimer_set,
)
from holstein_peierls.molecular_hamiltonian import bond_displacement_angstrom


def test_atomistic_source_provenance_and_cell_volume_match_mattheus_293k() -> None:
    assert MATT_HEUS_2001_SOURCE_DOI == "10.1107/S010827010100703X"
    assert MATT_HEUS_2001_CIF_PATH == "sk1477sup1.cif"
    assert MATT_HEUS_2001_293K_CCDC == "170186"
    assert "rounded fractional coordinates" in MATT_HEUS_2001_COORDINATE_STATUS
    assert np.isclose(PENTACENE_293K_CELL_3D.volume_angstrom3, 685.15, atol=0.01)


def test_each_reconstructed_basis_molecule_is_complete_c22h14_and_inversion_symmetric() -> None:
    for molecule in (
        pentacene_293k_atomistic_basis_a(),
        pentacene_293k_atomistic_basis_b(),
    ):
        symbols = molecule.rigid_geometry.symbols
        assert len(symbols) == 36
        assert symbols.count("C") == 22
        assert symbols.count("H") == 14
        fractional = np.asarray(molecule.fractional_coordinates)
        center = np.asarray(molecule.inversion_center_fractional)
        half = len(fractional) // 2
        assert np.allclose(
            fractional[:half] + fractional[half:],
            np.broadcast_to(2.0 * center, (half, 3)),
            rtol=0.0,
            atol=1.0e-15,
        )


def test_crystallographic_inversion_center_is_exact_center_of_mass_for_reconstructed_pairs() -> None:
    masses = {"C": 12.011, "H": 1.008}
    for molecule in (
        pentacene_293k_atomistic_basis_a(),
        pentacene_293k_atomistic_basis_b(),
    ):
        weights = np.asarray([masses[symbol] for symbol in molecule.rigid_geometry.symbols])
        coordinates = molecule.rigid_geometry.coordinates
        center_of_mass = np.average(coordinates, axis=0, weights=weights)
        assert np.allclose(
            center_of_mass,
            molecule.rigid_geometry.pivot,
            rtol=0.0,
            atol=3.0e-14,
        )


def test_principal_axes_are_right_handed_and_ordered_long_short_normal() -> None:
    for molecule in (
        pentacene_293k_atomistic_basis_a(),
        pentacene_293k_atomistic_basis_b(),
    ):
        axes = molecule.axes
        moments = np.asarray(molecule.principal_moments_u_angstrom2)
        assert np.allclose(axes.T @ axes, np.eye(3), rtol=0.0, atol=2.0e-14)
        assert np.isclose(np.linalg.det(axes), 1.0, rtol=0.0, atol=2.0e-14)
        assert np.all(np.diff(moments) > 0.0)


def test_long_axis_angles_to_c_star_reproduce_published_heart_line_angles() -> None:
    # In the conventional triclinic embedding used here, a and b span xy, so c*
    # is parallel to global z.  The principal long axis is the first column.
    angles = []
    for molecule in (
        pentacene_293k_atomistic_basis_a(),
        pentacene_293k_atomistic_basis_b(),
    ):
        cosine = abs(float(molecule.axes[2, 0]))
        angles.append(math.degrees(math.acos(cosine)))
    # Mattheus et al. report 25.17(2) and 24.39(2) degrees.  The first value is
    # within ~0.05 degree because we use the printed rounded coordinates and a
    # mass-weighted principal axis rather than their graphical heart-line.
    assert np.allclose(sorted(angles), sorted((25.17, 24.39)), rtol=0.0, atol=0.08)


def test_atomistic_dimer_families_match_g5b_graph_order_exactly() -> None:
    lattice = pentacene_293k_candidate_transport_lattice(2, 2)
    assert PENTACENE_293K_ATOMISTIC_DIMER_FAMILIES == tuple(
        family.label for family in lattice.bonds
    )
    dimers = pentacene_293k_atomistic_dimer_set()
    assert tuple(item.family_label for item in dimers) == PENTACENE_293K_ATOMISTIC_DIMER_FAMILIES


@pytest.mark.parametrize("family_label", PENTACENE_293K_ATOMISTIC_DIMER_FAMILIES)
def test_atomistic_dimer_centroid_displacement_matches_g5b_2d_bond_geometry(
    family_label: str,
) -> None:
    lattice = pentacene_293k_candidate_transport_lattice(2, 2)
    family = next(item for item in lattice.bonds if item.label == family_label)
    expected_2d = bond_displacement_angstrom(lattice, family)
    atomistic = pentacene_293k_atomistic_dimer(family_label)
    actual = atomistic.centroid_displacement_angstrom
    assert np.allclose(actual[:2], expected_2d, rtol=0.0, atol=2.0e-14)
    assert abs(actual[2]) < 2.0e-14


def test_each_atomistic_dimer_uses_source_molecule_principal_frame() -> None:
    basis = {
        "A": pentacene_293k_atomistic_basis_a(),
        "B": pentacene_293k_atomistic_basis_b(),
    }
    for dimer in pentacene_293k_atomistic_dimer_set():
        expected = basis[dimer.source_basis].axes
        assert np.allclose(dimer.reference.axes, expected, rtol=0.0, atol=2.0e-14)
        assert np.isclose(np.linalg.det(dimer.reference.axes), 1.0, atol=2.0e-14)


def test_unknown_atomistic_dimer_family_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown pentacene atomistic dimer family"):
        pentacene_293k_atomistic_dimer("third_dimension")
