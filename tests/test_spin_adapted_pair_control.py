import numpy as np

from holstein_peierls.exciton.solver import (
    lattice_energy,
    solve_exciton_ground_state,
)
from holstein_peierls.lattice import LatticeState
from holstein_peierls.spin_adapted import (
    ExchangeControl,
    IsotropicControlParameters,
    SpinMultiplicity,
    singlet_triplet_gap,
    solve_spin_adapted_pair,
)
from holstein_peierls.spin_adapted.pair_control import spin_adapted_pair_gradient


def _zero_lattice(ny: int, nx: int) -> LatticeState:
    shape = (ny, nx)
    return LatticeState(
        u=np.zeros(shape),
        vx=np.zeros(shape),
        vy=np.zeros(shape),
    )


def test_zero_exchange_recovers_spin_blind_pair_reference() -> None:
    parameters = IsotropicControlParameters().to_exciton_parameters(
        nx=3,
        ny=3,
        onsite_attraction=0.4,
    )
    lattice = _zero_lattice(3, 3)
    reference = solve_exciton_ground_state(lattice, parameters)

    for multiplicity in (SpinMultiplicity.SINGLET, SpinMultiplicity.TRIPLET):
        state = solve_spin_adapted_pair(
            lattice,
            parameters,
            ExchangeControl(),
            multiplicity,
        )
        assert np.isclose(state.energy, reference.energy, atol=1.0e-11)
        np.testing.assert_allclose(
            np.square(state.wavefunction),
            np.square(reference.wavefunction),
            atol=1.0e-9,
        )


def test_atomic_exchange_control_has_exact_two_k_singlet_triplet_gap() -> None:
    parameters = IsotropicControlParameters().to_exciton_parameters(
        nx=1,
        ny=1,
        electron_j0x=0.0,
        electron_j0y=0.0,
        hole_j0x=0.0,
        hole_j0y=0.0,
        electron_alpha_intra=0.0,
        electron_alpha_interx=0.0,
        electron_alpha_intery=0.0,
        hole_alpha_intra=0.0,
        hole_alpha_interx=0.0,
        hole_alpha_intery=0.0,
        onsite_attraction=0.5,
    )
    lattice = _zero_lattice(1, 1)
    exchange = ExchangeControl(onsite=0.12)

    singlet = solve_spin_adapted_pair(
        lattice, parameters, exchange, SpinMultiplicity.SINGLET
    )
    triplet = solve_spin_adapted_pair(
        lattice, parameters, exchange, SpinMultiplicity.TRIPLET
    )
    assert np.isclose(singlet.energy, -0.5 + 0.12)
    assert np.isclose(triplet.energy, -0.5 - 0.12)
    assert np.isclose(
        singlet_triplet_gap(lattice, parameters, exchange),
        0.24,
        atol=1.0e-12,
    )


def test_spin_adapted_pair_structural_gradient_matches_finite_difference() -> None:
    control = IsotropicControlParameters(alpha1=0.35, alpha2=0.35)
    parameters = control.to_exciton_parameters(
        nx=3,
        ny=3,
        onsite_attraction=0.35,
        eigensolver_tolerance=1.0e-12,
    )
    lattice = _zero_lattice(3, 3)
    lattice.u[1, 1] = -0.03
    lattice.vx[1, 1] = 0.02
    lattice.vy[1, 1] = -0.01
    exchange = ExchangeControl(onsite=0.05, nearest_neighbor=0.015)

    state = solve_spin_adapted_pair(
        lattice, parameters, exchange, SpinMultiplicity.SINGLET
    )
    gradient, _ = spin_adapted_pair_gradient(
        lattice,
        parameters,
        exchange,
        SpinMultiplicity.SINGLET,
        ground_state=state,
    )

    def total_energy(test_lattice: LatticeState) -> float:
        electronic = solve_spin_adapted_pair(
            test_lattice,
            parameters,
            exchange,
            SpinMultiplicity.SINGLET,
            initial_state=state.wavefunction,
        ).energy
        return electronic + lattice_energy(test_lattice, parameters)

    step = 1.0e-6
    plus = lattice.copy()
    minus = lattice.copy()
    plus.u[1, 1] += step
    minus.u[1, 1] -= step
    fd_u = (total_energy(plus) - total_energy(minus)) / (2.0 * step)
    assert np.isclose(gradient.u[1, 1], fd_u, rtol=2.0e-5, atol=2.0e-7)

    plus = lattice.copy()
    minus = lattice.copy()
    plus.vx[1, 1] += step
    minus.vx[1, 1] -= step
    fd_vx = (total_energy(plus) - total_energy(minus)) / (2.0 * step)
    assert np.isclose(gradient.vx[1, 1], fd_vx, rtol=2.0e-5, atol=2.0e-7)
