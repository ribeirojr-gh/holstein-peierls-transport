import numpy as np
import pytest

from holstein_peierls.two_particle.interaction import (
    COULOMB_PREFACTOR_EV_ANGSTROM,
    interaction_expectation,
    minimum_image_distances_angstrom,
    pair_interaction_matrix,
)
from holstein_peierls.two_particle.parameters import BipolaronParameters


def _symmetric_pair(n: int, first: int, second: int) -> np.ndarray:
    psi = np.zeros((n, n), dtype=float)
    if first == second:
        psi[first, first] = 1.0
    else:
        amplitude = 1.0 / np.sqrt(2.0)
        psi[first, second] = amplitude
        psi[second, first] = amplitude
    return psi


def _long_range_parameters(**overrides: object) -> BipolaronParameters:
    values: dict[str, object] = {
        "nx": 4,
        "ny": 4,
        "pair_position": 6,
        "hubbard_u": 0.70,
        "long_range_coulomb": True,
        "lattice_spacing_x_angstrom": 6.0,
        "lattice_spacing_y_angstrom": 8.0,
        "relative_permittivity": 3.0,
    }
    values.update(overrides)
    return BipolaronParameters(**values)


def test_long_range_coulomb_requires_explicit_physical_parameters() -> None:
    with pytest.raises(ValueError, match="requires explicit positive values"):
        BipolaronParameters(long_range_coulomb=True)

    with pytest.raises(ValueError, match="must be positive"):
        BipolaronParameters(
            long_range_coulomb=True,
            lattice_spacing_x_angstrom=6.0,
            lattice_spacing_y_angstrom=8.0,
            relative_permittivity=0.0,
        )


def test_minimum_image_distances_use_anisotropic_physical_spacings() -> None:
    p = _long_range_parameters()
    distance = minimum_image_distances_angstrom(p)

    assert np.isclose(distance[0, 1], 6.0)
    assert np.isclose(distance[0, 3], 6.0)  # periodic x boundary
    assert np.isclose(distance[0, 4], 8.0)
    assert np.isclose(distance[0, 12], 8.0)  # periodic y boundary
    assert np.isclose(distance[0, 5], 10.0)  # diagonal 3-4-5 geometry scaled by 2
    assert np.isclose(distance[0, 0], 0.0)


def test_continuum_coulomb_tail_matches_analytic_pair_values() -> None:
    p = _long_range_parameters()
    interaction = pair_interaction_matrix(p)
    prefactor = COULOMB_PREFACTOR_EV_ANGSTROM / p.relative_permittivity

    assert np.isclose(interaction[0, 0], p.hubbard_u)
    assert np.isclose(interaction[0, 1], prefactor / 6.0)
    assert np.isclose(interaction[0, 4], prefactor / 8.0)
    assert np.isclose(interaction[0, 5], prefactor / 10.0)
    assert np.isclose(interaction[0, 3], interaction[0, 1])
    assert np.isclose(interaction[0, 12], interaction[0, 4])


def test_nonzero_v1_overrides_nearest_coulomb_without_double_counting() -> None:
    p = _long_range_parameters(nearest_neighbor_v=0.11)
    interaction = pair_interaction_matrix(p)
    prefactor = COULOMB_PREFACTOR_EV_ANGSTROM / p.relative_permittivity

    assert np.isclose(interaction[0, 1], 0.11)
    assert np.isclose(interaction[0, 4], 0.11)
    assert np.isclose(interaction[0, 5], prefactor / 10.0)
    assert np.isclose(interaction[0, 0], p.hubbard_u)


def test_long_range_interaction_expectation_uses_full_tail() -> None:
    p = _long_range_parameters(hubbard_u=0.65)
    n = p.n_sites
    onsite = _symmetric_pair(n, 0, 0)
    diagonal = _symmetric_pair(n, 0, 5)
    far = _symmetric_pair(n, 0, 10)

    prefactor = COULOMB_PREFACTOR_EV_ANGSTROM / p.relative_permittivity
    assert np.isclose(interaction_expectation(onsite, p), 0.65)
    assert np.isclose(interaction_expectation(diagonal, p), prefactor / 10.0)

    # On a 4x4 torus, site 10 is two cells away in both directions.
    far_distance = np.sqrt((2.0 * 6.0) ** 2 + (2.0 * 8.0) ** 2)
    assert np.isclose(interaction_expectation(far, p), prefactor / far_distance)


def test_long_range_disabled_preserves_extended_hubbard_semantics() -> None:
    p = BipolaronParameters(
        nx=4,
        ny=4,
        pair_position=6,
        hubbard_u=0.70,
        nearest_neighbor_v=0.11,
    )
    interaction = pair_interaction_matrix(p)

    assert np.isclose(interaction[0, 0], 0.70)
    assert np.isclose(interaction[0, 1], 0.11)
    assert np.isclose(interaction[0, 4], 0.11)
    assert np.isclose(interaction[0, 5], 0.0)


def test_isotropic_geometry_is_rotation_degenerate() -> None:
    p = _long_range_parameters(
        lattice_spacing_x_angstrom=7.0,
        lattice_spacing_y_angstrom=7.0,
        nearest_neighbor_v=0.0,
    )
    interaction = pair_interaction_matrix(p)

    assert np.isclose(interaction[0, 1], interaction[0, 4])
    assert np.isclose(interaction[0, 5], interaction[0, 7])
