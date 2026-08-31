import numpy as np
import pytest

from holstein_peierls.molecular_lattice import (
    BondFamily,
    MolecularBasisSite,
    PeriodicMolecularLattice2D,
    rectangular_legacy_lattice,
)
from holstein_peierls.two_particle.interaction import minimum_image_distances_angstrom
from holstein_peierls.two_particle.parameters import BipolaronParameters


def test_geometry_validation_rejects_degenerate_or_ambiguous_inputs() -> None:
    basis = (MolecularBasisSite("A", (0.0, 0.0)),)
    with pytest.raises(ValueError, match="positive"):
        PeriodicMolecularLattice2D(0, 2, (1.0, 0.0), (0.0, 1.0), basis)
    with pytest.raises(ValueError, match="linearly independent"):
        PeriodicMolecularLattice2D(2, 2, (1.0, 0.0), (2.0, 0.0), basis)
    with pytest.raises(ValueError, match="unique"):
        PeriodicMolecularLattice2D(
            2,
            2,
            (1.0, 0.0),
            (0.0, 1.0),
            (
                MolecularBasisSite("A", (0.0, 0.0)),
                MolecularBasisSite("A", (0.5, 0.5)),
            ),
        )


def test_site_index_roundtrip_preserves_legacy_cell_ordering() -> None:
    lattice = PeriodicMolecularLattice2D(
        n1=3,
        n2=2,
        a1_angstrom=(5.0, 0.0),
        a2_angstrom=(1.0, 4.0),
        basis=(
            MolecularBasisSite("A", (0.0, 0.0)),
            MolecularBasisSite("B", (0.5, 0.5)),
        ),
    )
    for index in range(lattice.n_sites):
        cell_y, cell_x, basis = lattice.site_coordinates(index)
        assert lattice.site_index(cell_y, cell_x, basis) == index
    assert lattice.site_index(0, 0, 0) == 0
    assert lattice.site_index(0, 0, 1) == 1
    assert lattice.site_index(0, 1, 0) == 2
    assert lattice.site_index(1, 0, 0) == 6


def test_rectangular_positions_and_bond_graph_match_legacy_topology() -> None:
    lattice = rectangular_legacy_lattice(3, 3, 6.0, 8.0)
    expected_positions = np.array(
        [
            (0.0, 0.0),
            (6.0, 0.0),
            (12.0, 0.0),
            (0.0, 8.0),
            (6.0, 8.0),
            (12.0, 8.0),
            (0.0, 16.0),
            (6.0, 16.0),
            (12.0, 16.0),
        ]
    )
    assert np.array_equal(lattice.physical_positions_angstrom(), expected_positions)

    bonds = lattice.generated_bonds()
    assert len(bonds) == 18
    assert sum(bond.family == "x" for bond in bonds) == 9
    assert sum(bond.family == "y" for bond in bonds) == 9
    edges = {(bond.source, bond.target, bond.family) for bond in bonds}
    assert (0, 1, "x") in edges
    assert (0, 2, "x") in edges
    assert (0, 3, "y") in edges
    assert (0, 6, "y") in edges


def test_small_periodic_cell_collapses_same_family_duplicate_edges() -> None:
    lattice = rectangular_legacy_lattice(2, 3, 6.0, 8.0)
    bonds = lattice.generated_bonds()
    assert sum(bond.family == "x" for bond in bonds) == 3
    assert sum(bond.family == "y" for bond in bonds) == 6


def test_legacy_rectangular_distance_matrix_is_bitwise_identical_to_v0_6() -> None:
    nx, ny = 7, 6
    ax, ay = 6.266, 7.775
    lattice = rectangular_legacy_lattice(nx, ny, ax, ay)
    reference_parameters = BipolaronParameters(
        nx=nx,
        ny=ny,
        pair_position=1,
        long_range_coulomb=True,
        lattice_spacing_x_angstrom=ax,
        lattice_spacing_y_angstrom=ay,
        relative_permittivity=4.0,
    )
    reference = minimum_image_distances_angstrom(reference_parameters)
    candidate = lattice.minimum_image_distances_angstrom()
    assert np.array_equal(candidate, reference)


def test_nonorthogonal_minimum_image_uses_supercell_metric() -> None:
    lattice = PeriodicMolecularLattice2D(
        n1=2,
        n2=2,
        a1_angstrom=(2.0, 0.0),
        a2_angstrom=(1.0, 2.0),
        basis=(MolecularBasisSite("A", (0.0, 0.0)),),
    )
    source = lattice.site_index(0, 0, 0)
    diagonal = lattice.site_index(1, 1, 0)
    displacement = lattice.minimum_image_displacement_angstrom(source, diagonal)
    assert np.isclose(np.dot(displacement, displacement), 5.0)
    assert np.isclose(
        lattice.minimum_image_distances_angstrom()[source, diagonal], np.sqrt(5.0)
    )


def test_two_basis_positions_and_periodic_bond_families_are_deterministic() -> None:
    lattice = PeriodicMolecularLattice2D(
        n1=2,
        n2=2,
        a1_angstrom=(5.0, 0.0),
        a2_angstrom=(1.0, 4.0),
        basis=(
            MolecularBasisSite("A", (0.0, 0.0)),
            MolecularBasisSite("B", (0.5, 0.5)),
        ),
        bonds=(
            BondFamily("AB0", 0, 1, (0, 0)),
            BondFamily("AB1", 0, 1, (-1, 0)),
        ),
    )
    positions = lattice.physical_positions_angstrom()
    assert np.array_equal(positions[0], np.array((0.0, 0.0)))
    assert np.array_equal(positions[1], np.array((3.0, 2.0)))
    assert np.array_equal(positions[2], np.array((5.0, 0.0)))
    bonds = lattice.generated_bonds()
    assert len(bonds) == 8
    assert [bond.family for bond in bonds[:4]] == ["AB0"] * 4
    assert [bond.family for bond in bonds[4:]] == ["AB1"] * 4
