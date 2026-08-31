import numpy as np
import pytest

from holstein_peierls.two_particle.interaction import (
    COULOMB_PREFACTOR_EV_ANGSTROM,
    pair_interaction_matrix,
)
from holstein_peierls.two_particle.parameters import BipolaronParameters


def _parameters(**overrides: object) -> BipolaronParameters:
    values: dict[str, object] = {
        "nx": 6,
        "ny": 6,
        "pair_position": 8,
        "hubbard_u": 0.70,
        "long_range_coulomb": True,
        "lattice_spacing_x_angstrom": 6.0,
        "lattice_spacing_y_angstrom": 8.0,
        "relative_permittivity": 4.0,
    }
    values.update(overrides)
    return BipolaronParameters(**values)


def test_shell_overrides_require_long_range_coulomb() -> None:
    with pytest.raises(ValueError, match="require long_range_coulomb=True"):
        BipolaronParameters(short_range_shell_overrides=((1, 1, 0.10),))


def test_shell_override_validation_rejects_ambiguous_or_invalid_shells() -> None:
    with pytest.raises(ValueError, match="onsite shell"):
        _parameters(short_range_shell_overrides=((0, 0, 0.10),))

    with pytest.raises(ValueError, match="must be non-negative"):
        _parameters(short_range_shell_overrides=((-1, 0, 0.10),))

    with pytest.raises(ValueError, match="must be non-negative"):
        _parameters(short_range_shell_overrides=((1, 1, -0.10),))

    with pytest.raises(ValueError, match="duplicate"):
        _parameters(
            short_range_shell_overrides=((1, 1, 0.10), (1, 1, 0.20))
        )

    with pytest.raises(ValueError, match="outside the minimum-image range"):
        _parameters(short_range_shell_overrides=((4, 0, 0.10),))

    with pytest.raises(ValueError, match="cannot be combined"):
        _parameters(
            nearest_neighbor_v=0.11,
            short_range_shell_overrides=((1, 0, 0.20),),
        )


def test_explicit_shell_overrides_replace_only_selected_continuum_shells() -> None:
    p = _parameters(
        short_range_shell_overrides=(
            (1, 0, 0.21),
            (0, 1, 0.17),
            (1, 1, 0.09),
        )
    )
    interaction = pair_interaction_matrix(p)
    prefactor = COULOMB_PREFACTOR_EV_ANGSTROM / p.relative_permittivity

    # Explicit short-range shells.
    assert np.isclose(interaction[0, 1], 0.21)
    assert np.isclose(interaction[0, 6], 0.17)
    assert np.isclose(interaction[0, 7], 0.09)

    # Periodic symmetry of the same shells.
    assert np.isclose(interaction[0, 5], 0.21)
    assert np.isclose(interaction[0, 30], 0.17)
    assert np.isclose(interaction[0, 35], 0.09)

    # A shell that was not replaced retains the continuum tail.
    assert np.isclose(interaction[0, 2], prefactor / 12.0)
    assert np.isclose(interaction[0, 12], prefactor / 16.0)
    assert np.isclose(interaction[0, 0], p.hubbard_u)


def test_legacy_v1_can_be_combined_with_additional_diagonal_override() -> None:
    p = _parameters(
        nearest_neighbor_v=0.11,
        short_range_shell_overrides=((1, 1, 0.07),),
    )
    interaction = pair_interaction_matrix(p)
    prefactor = COULOMB_PREFACTOR_EV_ANGSTROM / p.relative_permittivity

    assert np.isclose(interaction[0, 1], 0.11)
    assert np.isclose(interaction[0, 6], 0.11)
    assert np.isclose(interaction[0, 7], 0.07)
    assert np.isclose(interaction[0, 2], prefactor / 12.0)


def test_zero_shell_override_is_an_explicit_screened_replacement() -> None:
    p = _parameters(short_range_shell_overrides=((1, 1, 0.0),))
    interaction = pair_interaction_matrix(p)

    assert np.isclose(interaction[0, 7], 0.0)
    assert interaction[0, 1] > 0.0


def test_empty_shell_overrides_preserve_v0_5_long_range_semantics() -> None:
    baseline = _parameters(nearest_neighbor_v=0.11)
    explicit_empty = _parameters(
        nearest_neighbor_v=0.11,
        short_range_shell_overrides=(),
    )

    assert np.array_equal(
        pair_interaction_matrix(baseline),
        pair_interaction_matrix(explicit_empty),
    )
