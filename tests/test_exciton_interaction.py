import numpy as np
import pytest

from holstein_peierls.exciton import ExcitonParameters
from holstein_peierls.exciton.interaction import (
    electron_hole_interaction_matrix,
    minimum_image_offsets,
)


def test_local_and_nearest_neighbor_attractions_are_negative() -> None:
    parameters = ExcitonParameters(
        nx=4,
        ny=4,
        onsite_attraction=0.5,
        nearest_neighbor_attraction=0.2,
    )
    interaction = electron_hole_interaction_matrix(parameters)
    dx, dy = minimum_image_offsets(parameters)
    nearest = ((dx == 1) & (dy == 0)) | ((dx == 0) & (dy == 1))

    assert np.allclose(np.diag(interaction), -0.5)
    assert np.allclose(interaction[nearest], -0.2)
    other = ~(nearest | np.eye(parameters.n_sites, dtype=bool))
    assert np.allclose(interaction[other], 0.0)


def test_long_range_attraction_has_expected_inverse_distance_scaling() -> None:
    parameters = ExcitonParameters(
        nx=4,
        ny=4,
        long_range_coulomb=True,
        lattice_spacing_x_angstrom=7.0,
        lattice_spacing_y_angstrom=7.0,
        relative_permittivity=10.0,
    )
    interaction = electron_hole_interaction_matrix(parameters)
    # Site 0 to +x and to the first diagonal.
    cardinal = interaction[0, 1]
    diagonal = interaction[0, 5]
    assert cardinal < 0.0
    assert diagonal < 0.0
    assert diagonal / cardinal == pytest.approx(1.0 / np.sqrt(2.0), rel=1.0e-13)
    assert interaction[0, 0] == 0.0


def test_short_range_shell_attraction_replaces_continuum() -> None:
    parameters = ExcitonParameters(
        nx=6,
        ny=6,
        long_range_coulomb=True,
        lattice_spacing_x_angstrom=7.0,
        lattice_spacing_y_angstrom=7.0,
        relative_permittivity=10.0,
        short_range_shell_attractions=((1, 1, 0.123),),
    )
    interaction = electron_hole_interaction_matrix(parameters)
    dx, dy = minimum_image_offsets(parameters)
    diagonal = (dx == 1) & (dy == 1)
    assert np.allclose(interaction[diagonal], -0.123)


def test_exciton_parameter_validation_rejects_ambiguous_cardinal_replacement() -> None:
    with pytest.raises(ValueError, match="cannot coexist"):
        ExcitonParameters(
            nx=6,
            ny=6,
            long_range_coulomb=True,
            lattice_spacing_x_angstrom=7.0,
            lattice_spacing_y_angstrom=7.0,
            relative_permittivity=10.0,
            nearest_neighbor_attraction=0.1,
            short_range_shell_attractions=((1, 0, 0.2),),
        )
