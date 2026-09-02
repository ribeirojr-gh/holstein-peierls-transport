import numpy as np
import pytest

from holstein_peierls.spin_adapted import (
    IsotropicControlParameters,
    IsotropicRelaxationSeed,
    half_filled_n_closed,
    isotropic_relaxation_seed,
)


def test_half_filled_n_closed_matches_one_pi_electron_per_site() -> None:
    assert half_filled_n_closed(4) == 1
    assert half_filled_n_closed(16) == 7
    assert half_filled_n_closed(400) == 199
    with pytest.raises(ValueError, match="even N"):
        half_filled_n_closed(9)


def test_onsite_seed_has_zero_uniform_holstein_component() -> None:
    parameters = IsotropicControlParameters().to_polaron_parameters(nx=6, ny=6)
    seed = isotropic_relaxation_seed(
        parameters,
        IsotropicRelaxationSeed.ONSITE,
        amplitude=2.0e-3,
    )
    assert np.isclose(np.sum(seed.u), 0.0, atol=1.0e-15)
    assert np.max(np.abs(seed.u)) == pytest.approx(2.0e-3)
    assert np.count_nonzero(seed.vx) == 0
    assert np.count_nonzero(seed.vy) == 0


def test_x_and_y_bond_seeds_are_exact_isotropic_partners() -> None:
    parameters = IsotropicControlParameters().to_polaron_parameters(nx=6, ny=6)
    x_seed = isotropic_relaxation_seed(parameters, "bond_x", amplitude=1.5e-3)
    y_seed = isotropic_relaxation_seed(parameters, "bond_y", amplitude=1.5e-3)

    assert np.count_nonzero(x_seed.vx) == 2
    assert np.count_nonzero(x_seed.vy) == 0
    assert np.count_nonzero(y_seed.vx) == 0
    assert np.count_nonzero(y_seed.vy) == 2
    np.testing.assert_allclose(
        np.sort(x_seed.vx[x_seed.vx != 0.0]),
        np.sort(y_seed.vy[y_seed.vy != 0.0]),
        atol=0.0,
    )
    assert np.isclose(np.sum(x_seed.vx), 0.0, atol=1.0e-15)
    assert np.isclose(np.sum(y_seed.vy), 0.0, atol=1.0e-15)
