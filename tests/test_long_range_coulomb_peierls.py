import numpy as np
import pytest

from holstein_peierls.lattice import LatticeState
from holstein_peierls.two_particle.parameters import BipolaronParameters
from holstein_peierls.two_particle.peierls import (
    energy_gradient,
    solve_holstein_peierls_ground_state,
    total_energy,
)


@pytest.mark.parametrize(
    "shell_overrides",
    [
        (),
        ((1, 1, 0.09),),
    ],
)
def test_structural_gradient_remains_correct_with_frozen_long_range_coulomb(
    shell_overrides: tuple[tuple[int, int, float], ...],
) -> None:
    p = BipolaronParameters(
        nx=3,
        ny=3,
        pair_position=5,
        hubbard_u=0.70,
        long_range_coulomb=True,
        lattice_spacing_x_angstrom=6.5,
        lattice_spacing_y_angstrom=7.5,
        relative_permittivity=4.0,
        short_range_shell_overrides=shell_overrides,
        k2=0.73,
        j0x=0.10,
        j0y=0.04,
        alpha_interx=0.10,
        alpha_intery=0.07,
        eigensolver_tolerance=1.0e-12,
    )
    rng = np.random.default_rng(404)
    state = LatticeState(
        u=rng.normal(scale=0.02, size=(3, 3)),
        vx=rng.normal(scale=0.01, size=(3, 3)),
        vy=rng.normal(scale=0.01, size=(3, 3)),
    )
    state.u[1, 1] -= 0.10

    ground = solve_holstein_peierls_ground_state(state, p)
    gradient, _ = energy_gradient(state, p, ground_state=ground)
    epsilon = 2.0e-6

    for field, index, analytical in (
        ("u", (1, 1), gradient.u[1, 1]),
        ("vx", (1, 1), gradient.vx[1, 1]),
        ("vy", (1, 1), gradient.vy[1, 1]),
    ):
        plus = state.copy()
        minus = state.copy()
        getattr(plus, field)[index] += epsilon
        getattr(minus, field)[index] -= epsilon
        ep, _ = total_energy(plus, p)
        em, _ = total_energy(minus, p)
        numerical = (ep.total - em.total) / (2.0 * epsilon)
        assert np.isclose(analytical, numerical, rtol=5.0e-5, atol=5.0e-7)
