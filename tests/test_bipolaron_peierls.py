from dataclasses import replace

import numpy as np

from holstein_peierls.hamiltonian import build_sparse_hamiltonian
from holstein_peierls.lattice import LatticeState
from holstein_peierls.parameters import StaticPolaronParameters
from holstein_peierls.two_particle.bipolaron import solve_bipolaron_ground_state
from holstein_peierls.two_particle.parameters import BipolaronParameters
from holstein_peierls.two_particle.peierls import (
    build_one_particle_hamiltonian,
    energy_gradient,
    relax_static_holstein_peierls_bipolaron,
    solve_holstein_peierls_ground_state,
    total_energy,
)


def test_peierls_one_particle_hamiltonian_matches_validated_single_polaron() -> None:
    single = StaticPolaronParameters(
        nx=3,
        ny=3,
        polaron_position=5,
        k1=16.51,
        k2=0.73,
        j0x=0.10,
        j0y=0.04,
        alpha_intra=2.1,
        alpha_interx=0.17,
        alpha_intery=0.11,
    )
    pair = BipolaronParameters.from_polaron_parameters(single, hubbard_u=0.23)
    rng = np.random.default_rng(12)
    state = LatticeState(
        u=rng.normal(scale=0.03, size=(3, 3)),
        vx=rng.normal(scale=0.02, size=(3, 3)),
        vy=rng.normal(scale=0.02, size=(3, 3)),
    )

    reference = build_sparse_hamiltonian(state, single).toarray()
    two_particle_one_body = build_one_particle_hamiltonian(state, pair).toarray()
    assert np.allclose(two_particle_one_body, reference, atol=1e-15)


def test_zero_peierls_distortion_recovers_holstein_hubbard_electronic_problem() -> None:
    p = BipolaronParameters(
        nx=3,
        ny=3,
        pair_position=5,
        hubbard_u=0.31,
        alpha_interx=0.4,
        alpha_intery=0.4,
    )
    rng = np.random.default_rng(22)
    u = rng.normal(scale=0.02, size=(3, 3))
    u[1, 1] -= 0.12
    state = LatticeState(u=u, vx=np.zeros_like(u), vy=np.zeros_like(u))

    reference = solve_bipolaron_ground_state(u, p)
    peierls = solve_holstein_peierls_ground_state(state, p)
    assert np.isclose(peierls.energy, reference.energy, atol=2e-12)
    assert peierls.exchange_symmetry_error < 2e-14


def test_full_holstein_peierls_gradient_matches_finite_difference() -> None:
    p = BipolaronParameters(
        nx=3,
        ny=3,
        pair_position=5,
        hubbard_u=0.18,
        k2=0.73,
        j0x=0.10,
        j0y=0.04,
        alpha_interx=0.17,
        alpha_intery=0.11,
    )
    rng = np.random.default_rng(41)
    state = LatticeState(
        u=rng.normal(scale=0.02, size=(3, 3)),
        vx=rng.normal(scale=0.01, size=(3, 3)),
        vy=rng.normal(scale=0.01, size=(3, 3)),
    )
    state.u[1, 1] -= 0.10

    ground = solve_holstein_peierls_ground_state(state, p)
    gradient, _ = energy_gradient(state, p, ground_state=ground)
    epsilon = 2.0e-6

    checks = (
        ("u", (1, 1), gradient.u[1, 1]),
        ("vx", (1, 1), gradient.vx[1, 1]),
        ("vy", (1, 1), gradient.vy[1, 1]),
    )
    for field, index, analytical in checks:
        plus = state.copy()
        minus = state.copy()
        getattr(plus, field)[index] += epsilon
        getattr(minus, field)[index] -= epsilon
        ep, _ = total_energy(plus, p)
        em, _ = total_energy(minus, p)
        numerical = (ep.total - em.total) / (2.0 * epsilon)
        assert np.isclose(analytical, numerical, rtol=3e-5, atol=3e-7)


def test_peierls_relaxation_reports_stationary_residual_gradient() -> None:
    p = BipolaronParameters(
        nx=3,
        ny=3,
        pair_position=5,
        hubbard_u=0.30,
        k2=1.0,
        alpha_interx=0.08,
        alpha_intery=0.05,
        max_iterations=800,
        convergence_criterion=1.0e-6,
        gradient_convergence_criterion=2.0e-5,
        eigensolver_tolerance=1.0e-12,
    )
    result = relax_static_holstein_peierls_bipolaron(p, initialization="onsite")
    gradient, _ = energy_gradient(
        result.lattice,
        p,
        ground_state=result.ground_state,
    )

    assert result.diagnostics.converged
    assert np.isclose(
        result.diagnostics.final_max_gradient,
        gradient.maximum_absolute_component,
        atol=1e-12,
    )
    assert gradient.maximum_absolute_component < p.gradient_convergence_criterion
    assert result.diagnostics.final_max_update < p.convergence_criterion


def test_peierls_relaxation_reduces_to_holstein_when_couplings_are_zero() -> None:
    base = BipolaronParameters(
        nx=3,
        ny=3,
        pair_position=5,
        hubbard_u=0.30,
        max_iterations=800,
        convergence_criterion=1.0e-6,
        gradient_convergence_criterion=2.0e-5,
        eigensolver_tolerance=1.0e-12,
    )
    p = replace(base, alpha_interx=0.0, alpha_intery=0.0)
    result = relax_static_holstein_peierls_bipolaron(p, initialization="onsite")

    assert result.diagnostics.converged
    assert np.max(np.abs(result.vx)) == 0.0
    assert np.max(np.abs(result.vy)) == 0.0
