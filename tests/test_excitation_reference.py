import numpy as np

from holstein_peierls.lattice import LatticeState
from holstein_peierls.spin_adapted import (
    IsotropicControlParameters,
    SpinMultiplicity,
    density_density_control_interaction,
    referenced_excitation_energy,
    referenced_excitation_gradient,
    solve_referenced_excitation,
)


def _distorted_lattice() -> LatticeState:
    u = np.asarray(
        (
            (0.008, -0.011, 0.004),
            (-0.006, 0.013, -0.003),
            (0.002, -0.005, 0.009),
        )
    )
    vx = np.asarray(
        (
            (0.004, -0.003, 0.001),
            (-0.002, 0.005, -0.004),
            (0.003, -0.001, 0.002),
        )
    )
    vy = np.asarray(
        (
            (-0.003, 0.002, 0.004),
            (0.001, -0.004, 0.003),
            (0.005, -0.002, -0.001),
        )
    )
    return LatticeState(u=u.copy(), vx=vx.copy(), vy=vy.copy())


def _problem():
    # Canonical coding control requested for S0: J1=J2=current J1 and
    # alpha1=alpha2=current alpha1.
    parameters = IsotropicControlParameters().to_polaron_parameters(
        nx=3,
        ny=3,
        eigensolver_tolerance=1.0e-12,
    )
    interaction = density_density_control_interaction(
        parameters,
        onsite_u=0.525,
        nearest_neighbor_v=0.08,
    )
    return parameters, interaction


def test_neutral_excitation_rdm_has_zero_trace_for_both_spin_sectors() -> None:
    parameters, interaction = _problem()
    lattice = _distorted_lattice()
    for multiplicity in (SpinMultiplicity.SINGLET, SpinMultiplicity.TRIPLET):
        state = solve_referenced_excitation(
            lattice,
            parameters,
            interaction,
            n_closed=1,
            multiplicity=multiplicity,
            orbital_gradient_tolerance=2.0e-9,
        )
        assert state.neutral.diagnostics.converged
        assert state.excited.diagnostics.converged
        assert abs(state.particle_number_change) < 2.0e-12
        assert np.isclose(np.trace(state.neutral_rdm), 4.0, atol=2.0e-12)
        assert np.isclose(np.trace(state.excited_rdm), 4.0, atol=2.0e-12)


def test_exchange_off_referenced_singlet_and_triplet_are_degenerate() -> None:
    parameters = IsotropicControlParameters().to_polaron_parameters(nx=3, ny=3)
    lattice = _distorted_lattice()
    interaction = np.zeros((parameters.n_sites, parameters.n_sites))
    singlet = solve_referenced_excitation(
        lattice,
        parameters,
        interaction,
        n_closed=1,
        multiplicity=SpinMultiplicity.SINGLET,
        orbital_gradient_tolerance=2.0e-9,
    )
    triplet = solve_referenced_excitation(
        lattice,
        parameters,
        interaction,
        n_closed=1,
        multiplicity=SpinMultiplicity.TRIPLET,
        orbital_gradient_tolerance=2.0e-9,
    )
    assert np.isclose(
        singlet.electronic_excitation_energy,
        triplet.electronic_excitation_energy,
        atol=2.0e-10,
    )


def test_referenced_structural_gradient_matches_reoptimized_finite_difference() -> None:
    parameters, interaction = _problem()
    lattice = _distorted_lattice()
    central = solve_referenced_excitation(
        lattice,
        parameters,
        interaction,
        n_closed=1,
        multiplicity=SpinMultiplicity.SINGLET,
        orbital_gradient_tolerance=5.0e-10,
        orbital_max_iterations=800,
    )
    assert central.neutral.diagnostics.converged
    assert central.excited.diagnostics.converged
    gradient = referenced_excitation_gradient(lattice, parameters, central)

    def total_energy(test_lattice: LatticeState) -> float:
        solved = solve_referenced_excitation(
            test_lattice,
            parameters,
            interaction,
            n_closed=1,
            multiplicity=SpinMultiplicity.SINGLET,
            initial_neutral_orbitals=central.neutral.orbitals,
            initial_excited_orbitals=central.excited.orbitals,
            orbital_gradient_tolerance=5.0e-10,
            orbital_max_iterations=800,
        )
        assert solved.neutral.diagnostics.converged
        assert solved.excited.diagnostics.converged
        return referenced_excitation_energy(test_lattice, parameters, solved).total

    step = 2.0e-6
    for field, analytic in (
        ("u", gradient.u[1, 1]),
        ("vx", gradient.vx[1, 1]),
        ("vy", gradient.vy[1, 1]),
    ):
        plus = lattice.copy()
        minus = lattice.copy()
        getattr(plus, field)[1, 1] += step
        getattr(minus, field)[1, 1] -= step
        finite_difference = (total_energy(plus) - total_energy(minus)) / (2.0 * step)
        assert np.isclose(analytic, finite_difference, rtol=3.0e-5, atol=3.0e-7)
