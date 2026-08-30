from dataclasses import replace

import numpy as np
import pytest

from holstein_peierls.lattice import LatticeState
from holstein_peierls.two_particle.bipolaron import (
    expectation_energy,
    solve_bipolaron_ground_state,
)
from holstein_peierls.two_particle.interaction import (
    COULOMB_PREFACTOR_EV_ANGSTROM,
    interaction_expectation,
    minimum_image_distances_angstrom,
    pair_interaction_matrix,
)
from holstein_peierls.two_particle.observables import pair_observables
from holstein_peierls.two_particle.parameters import BipolaronParameters
from holstein_peierls.two_particle.peierls import (
    energy_gradient,
    solve_holstein_peierls_ground_state,
    total_energy,
)


def _symmetric_pair(n: int, first: int, second: int) -> np.ndarray:
    psi = np.zeros((n, n), dtype=float)
    if first == second:
        psi[first, first] = 1.0
    else:
        amplitude = 1.0 / np.sqrt(2.0)
        psi[first, second] = amplitude
        psi[second, first] = amplitude
    return psi


def test_extended_hubbard_matrix_uses_periodic_nearest_neighbours() -> None:
    p = BipolaronParameters(
        nx=4,
        ny=4,
        pair_position=6,
        hubbard_u=0.71,
        nearest_neighbor_v=0.13,
    )
    interaction = pair_interaction_matrix(p)

    assert np.isclose(interaction[0, 0], 0.71)
    assert np.isclose(interaction[0, 1], 0.13)
    assert np.isclose(interaction[0, 3], 0.13)  # periodic x boundary
    assert np.isclose(interaction[0, 4], 0.13)
    assert np.isclose(interaction[0, 12], 0.13)  # periodic y boundary
    assert np.isclose(interaction[0, 5], 0.0)


def test_interaction_expectation_distinguishes_onsite_nn_and_far_pairs() -> None:
    p = BipolaronParameters(
        nx=4,
        ny=4,
        pair_position=6,
        hubbard_u=0.60,
        nearest_neighbor_v=0.08,
        j0x=0.0,
        j0y=0.0,
        alpha_intra=0.0,
    )
    n = p.n_sites
    onsite = _symmetric_pair(n, 5, 5)
    nearest = _symmetric_pair(n, 5, 6)
    separated = _symmetric_pair(n, 5, 15)

    assert np.isclose(interaction_expectation(onsite, p), p.hubbard_u)
    assert np.isclose(interaction_expectation(nearest, p), p.nearest_neighbor_v)
    assert np.isclose(interaction_expectation(separated, p), 0.0)

    u = np.zeros((p.ny, p.nx))
    assert np.isclose(expectation_energy(onsite, u, p).total, p.hubbard_u)
    assert np.isclose(expectation_energy(nearest, u, p).total, p.nearest_neighbor_v)
    assert np.isclose(expectation_energy(separated, u, p).total, 0.0)


def test_ground_state_hellmann_feynman_derivative_with_respect_to_v1() -> None:
    base = BipolaronParameters(
        nx=3,
        ny=3,
        pair_position=5,
        hubbard_u=0.55,
        nearest_neighbor_v=0.07,
        j0x=0.10,
        j0y=0.04,
        eigensolver_tolerance=1.0e-12,
    )
    u = np.zeros((3, 3))
    u[1, 1] = -0.11
    u[1, 2] = -0.07

    ground = solve_bipolaron_ground_state(u, base)
    obs = pair_observables(ground, base)

    epsilon = 2.0e-6
    plus = solve_bipolaron_ground_state(
        u, replace(base, nearest_neighbor_v=base.nearest_neighbor_v + epsilon)
    )
    minus = solve_bipolaron_ground_state(
        u, replace(base, nearest_neighbor_v=base.nearest_neighbor_v - epsilon)
    )
    numerical = (plus.energy - minus.energy) / (2.0 * epsilon)

    assert np.isclose(
        numerical,
        obs.nearest_neighbour_probability,
        rtol=2.0e-5,
        atol=2.0e-7,
    )


def test_structural_gradient_remains_correct_with_finite_v1() -> None:
    p = BipolaronParameters(
        nx=3,
        ny=3,
        pair_position=5,
        hubbard_u=0.55,
        nearest_neighbor_v=0.06,
        k2=0.73,
        j0x=0.10,
        j0y=0.04,
        alpha_interx=0.10,
        alpha_intery=0.07,
        eigensolver_tolerance=1.0e-12,
    )
    rng = np.random.default_rng(202)
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
        assert np.isclose(analytical, numerical, rtol=4.0e-5, atol=4.0e-7)


def test_fully_isotropic_x_y_pair_seeds_are_rotation_degenerate() -> None:
    p = BipolaronParameters(
        nx=4,
        ny=4,
        pair_position=6,
        hubbard_u=0.525,
        nearest_neighbor_v=0.030,
        j0x=0.0575,
        j0y=0.0575,
        alpha_interx=0.10,
        alpha_intery=0.10,
        eigensolver_tolerance=1.0e-12,
    )
    displacement = -p.alpha_intra / p.k1

    ux = np.zeros((p.ny, p.nx), dtype=float)
    uy = np.zeros_like(ux)
    cy, cx = divmod(p.pair_index, p.nx)
    ux[cy, cx] = displacement
    ux[cy, (cx + 1) % p.nx] = displacement
    uy[cy, cx] = displacement
    uy[(cy + 1) % p.ny, cx] = displacement

    zero = np.zeros_like(ux)
    state_x = LatticeState(u=ux, vx=zero.copy(), vy=zero.copy())
    state_y = LatticeState(u=uy, vx=zero.copy(), vy=zero.copy())

    ground_x = solve_holstein_peierls_ground_state(state_x, p)
    ground_y = solve_holstein_peierls_ground_state(state_y, p)
    obs_x = pair_observables(ground_x, p)
    obs_y = pair_observables(ground_y, p)

    assert np.isclose(ground_x.energy, ground_y.energy, atol=3.0e-12)
    assert np.isclose(
        obs_x.nearest_neighbour_x_probability,
        obs_y.nearest_neighbour_y_probability,
        atol=3.0e-12,
    )
    assert np.isclose(obs_x.onsite_probability, obs_y.onsite_probability, atol=3.0e-12)


def _long_range_parameters(**overrides: object) -> BipolaronParameters:
    values: dict[str, object] = {
        "nx": 4,
        "ny": 4,
        "pair_position": 6,
        "hubbard_u": 0.62,
        "long_range_coulomb": True,
        "lattice_spacing_x_angstrom": 6.0,
        "lattice_spacing_y_angstrom": 8.0,
        "relative_permittivity": 3.0,
    }
    values.update(overrides)
    return BipolaronParameters(**values)


def test_long_range_mode_requires_explicit_positive_geometry_and_screening() -> None:
    with pytest.raises(ValueError, match="requires explicit positive values"):
        BipolaronParameters(long_range_coulomb=True)

    with pytest.raises(ValueError, match="must be positive"):
        BipolaronParameters(
            long_range_coulomb=True,
            lattice_spacing_x_angstrom=6.0,
            lattice_spacing_y_angstrom=8.0,
            relative_permittivity=0.0,
        )


def test_minimum_image_physical_distances_are_periodic_and_anisotropic() -> None:
    p = _long_range_parameters()
    distance = minimum_image_distances_angstrom(p)

    assert np.isclose(distance[0, 0], 0.0)
    assert np.isclose(distance[0, 1], 6.0)
    assert np.isclose(distance[0, 3], 6.0)  # periodic x image
    assert np.isclose(distance[0, 4], 8.0)
    assert np.isclose(distance[0, 12], 8.0)  # periodic y image
    assert np.isclose(distance[0, 5], 10.0)  # sqrt(6^2 + 8^2)


def test_screened_coulomb_matrix_matches_analytic_offsite_values() -> None:
    p = _long_range_parameters()
    interaction = pair_interaction_matrix(p)
    prefactor = COULOMB_PREFACTOR_EV_ANGSTROM / p.relative_permittivity

    assert np.isclose(interaction[0, 0], p.hubbard_u)
    assert np.isclose(interaction[0, 1], prefactor / 6.0)
    assert np.isclose(interaction[0, 4], prefactor / 8.0)
    assert np.isclose(interaction[0, 5], prefactor / 10.0)
    assert np.isclose(interaction[0, 3], interaction[0, 1])
    assert np.isclose(interaction[0, 12], interaction[0, 4])
    assert np.allclose(interaction, interaction.T)


def test_explicit_v1_replaces_long_range_nearest_neighbour_value() -> None:
    p = _long_range_parameters(nearest_neighbor_v=0.19)
    interaction = pair_interaction_matrix(p)
    prefactor = COULOMB_PREFACTOR_EV_ANGSTROM / p.relative_permittivity

    assert np.isclose(interaction[0, 1], 0.19)
    assert np.isclose(interaction[0, 3], 0.19)
    assert np.isclose(interaction[0, 4], 0.19)
    assert np.isclose(interaction[0, 12], 0.19)
    assert np.isclose(interaction[0, 5], prefactor / 10.0)
    assert np.isclose(interaction[0, 0], p.hubbard_u)


def test_long_range_coulomb_expectation_for_localized_pair_is_exact() -> None:
    p = _long_range_parameters(j0x=0.0, j0y=0.0, alpha_intra=0.0)
    n = p.n_sites
    onsite = _symmetric_pair(n, 0, 0)
    x_neighbour = _symmetric_pair(n, 0, 1)
    diagonal = _symmetric_pair(n, 0, 5)
    prefactor = COULOMB_PREFACTOR_EV_ANGSTROM / p.relative_permittivity

    assert np.isclose(interaction_expectation(onsite, p), p.hubbard_u)
    assert np.isclose(interaction_expectation(x_neighbour, p), prefactor / 6.0)
    assert np.isclose(interaction_expectation(diagonal, p), prefactor / 10.0)


def test_structural_gradient_remains_correct_with_fixed_long_range_coulomb() -> None:
    p = _long_range_parameters(
        nx=3,
        ny=3,
        pair_position=5,
        lattice_spacing_x_angstrom=6.2,
        lattice_spacing_y_angstrom=7.4,
        relative_permittivity=4.0,
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
        assert np.isclose(analytical, numerical, rtol=6.0e-5, atol=6.0e-7)
