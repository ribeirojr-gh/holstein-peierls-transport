from __future__ import annotations

import numpy as np
import pytest

from holstein_peierls.dynamics.field import (
    UniformElectricField2D,
    build_dense_field_hamiltonian,
)
from holstein_peierls.dynamics.spin_adapted_rkmk import rkmk4_projector_step
from holstein_peierls.dynamics.spin_adapted_time_dependent import (
    compare_time_dependent_projectors,
    integrate_dop853_time_dependent_projectors,
    integrate_rkmk4_time_dependent_projectors,
    rkmk4_time_dependent_projector_step,
)
from holstein_peierls.dynamics.time_dependent import integrate_dop853_time_dependent
from holstein_peierls.lattice import LatticeState
from holstein_peierls.spin_adapted.excitation_reference import (
    density_density_control_interaction,
    excited_state_definition,
    reference_shell_sizes,
)
from holstein_peierls.spin_adapted.open_shell import CLOSED_SHELL_SINGLET
from holstein_peierls.spin_adapted.orbital_optimization import (
    optimize_open_shell_orbitals,
)
from holstein_peierls.spin_adapted.relaxation_control import (
    half_filled_n_closed,
    isotropic_staggered_site_energies,
)
from holstein_peierls.spin_adapted.isotropic import IsotropicControlParameters
from holstein_peierls.spin_adapted.spin import SpinMultiplicity


def _ring_hamiltonian(time_fs: float) -> np.ndarray:
    n = 4
    hopping = 0.1
    hbar_ev_fs = 0.6582119569
    phase = -(3.0 * 0.2 / hbar_ev_fs) * time_fs
    matrix = np.zeros((n, n), dtype=np.complex128)
    for site in range(n):
        neighbour = (site + 1) % n
        forward = -hopping * np.exp(1.0j * phase)
        matrix[site, neighbour] += forward
        matrix[neighbour, site] += np.conj(forward)
    matrix += np.diag(np.asarray((-0.03, 0.01, 0.02, 0.0)))
    return matrix


def _rank_one_projector() -> tuple[np.ndarray, ...]:
    state = np.asarray((1.0, 0.2j, -0.1, 0.05j), dtype=np.complex128)
    state /= np.linalg.norm(state)
    return (np.outer(state, state.conj()),)


def test_time_dependent_rkmk_reduces_to_d0b_for_constant_one_body() -> None:
    matrix = _ring_hamiltonian(0.0)
    projectors = _rank_one_projector()
    interaction = np.zeros_like(matrix.real)
    dt = 0.07
    frozen = rkmk4_projector_step(
        matrix,
        interaction,
        projectors,
        CLOSED_SHELL_SINGLET,
        dt,
    )
    dynamic = rkmk4_time_dependent_projector_step(
        lambda _time: matrix,
        interaction,
        projectors,
        CLOSED_SHELL_SINGLET,
        0.31,
        dt,
    )
    assert np.allclose(dynamic[0], frozen[0], rtol=0.0, atol=2.0e-15)


def test_closed_shell_projector_matches_linear_tdse_under_field() -> None:
    projectors = _rank_one_projector()
    interaction = np.zeros((4, 4), dtype=np.float64)
    # Any normalized vector spanning the rank-one projector is a valid orbital.
    eigenvalues, eigenvectors = np.linalg.eigh(projectors[0])
    state = eigenvectors[:, int(np.argmax(eigenvalues))]
    final_time = 1.0

    linear = integrate_dop853_time_dependent(
        _ring_hamiltonian,
        state,
        final_time_fs=final_time,
        rtol=2.0e-13,
        atol=2.0e-15,
        max_step_fs=0.003,
    )
    reference_projector = np.outer(linear.state, linear.state.conj())
    candidate = integrate_rkmk4_time_dependent_projectors(
        _ring_hamiltonian,
        interaction,
        projectors,
        CLOSED_SHELL_SINGLET,
        dt_fs=0.02,
        steps=50,
    )
    metrics = compare_time_dependent_projectors(
        (reference_projector,),
        candidate.projectors,
        CLOSED_SHELL_SINGLET,
    )
    assert metrics.projector_distance < 2.0e-10
    assert metrics.constraints.maximum_idempotency_error < 2.0e-14
    assert metrics.constraints.maximum_hermiticity_error < 2.0e-14


def test_time_dependent_rkmk_has_fourth_order_global_convergence() -> None:
    projectors = _rank_one_projector()
    interaction = np.zeros((4, 4), dtype=np.float64)
    reference = integrate_dop853_time_dependent_projectors(
        _ring_hamiltonian,
        interaction,
        projectors,
        CLOSED_SHELL_SINGLET,
        final_time_fs=2.0,
        rtol=2.0e-13,
        atol=2.0e-15,
        max_step_fs=0.004,
    )
    assert reference.success
    errors: list[float] = []
    for dt in (0.2, 0.1, 0.05):
        result = integrate_rkmk4_time_dependent_projectors(
            _ring_hamiltonian,
            interaction,
            projectors,
            CLOSED_SHELL_SINGLET,
            dt_fs=dt,
            steps=int(round(2.0 / dt)),
        )
        errors.append(
            compare_time_dependent_projectors(
                reference.projectors,
                result.projectors,
                CLOSED_SHELL_SINGLET,
            ).projector_distance
        )
    assert np.log2(errors[0] / errors[1]) > 3.6
    assert np.log2(errors[1] / errors[2]) > 3.6


def _s0_field_problem(multiplicity: SpinMultiplicity):
    control = IsotropicControlParameters()
    parameters = control.to_polaron_parameters(nx=4, ny=4)
    lattice = LatticeState.zeros(4, 4)
    field = UniformElectricField2D.from_millivolt_per_angstrom(2.0, 0.0)
    staggered = isotropic_staggered_site_energies(parameters, 2.0)

    def one_body_at(time_fs: float) -> np.ndarray:
        return np.asarray(
            build_dense_field_hamiltonian(lattice, parameters, field, time_fs)
            + np.diag(staggered.ravel(order="C")),
            dtype=np.complex128,
        )

    interaction = density_density_control_interaction(
        parameters,
        onsite_u=0.525,
        nearest_neighbor_v=0.08,
    )
    definition = excited_state_definition(multiplicity)
    n_closed = half_filled_n_closed(parameters.n_sites)
    _, shell_sizes = reference_shell_sizes(n_closed, multiplicity)
    initial_one_body = np.asarray(one_body_at(0.0).real, dtype=np.float64)
    _, orbitals = np.linalg.eigh(initial_one_body)
    optimized = optimize_open_shell_orbitals(
        initial_one_body,
        interaction,
        orbitals,
        shell_sizes,
        definition,
        gradient_tolerance=1.0e-8,
        max_iterations=800,
    )
    assert optimized.diagnostics.converged
    projectors = tuple(
        np.asarray(projector, dtype=np.complex128)
        for projector in optimized.projectors
    )
    return one_body_at, interaction, projectors, definition


@pytest.mark.parametrize(
    "multiplicity",
    (SpinMultiplicity.SINGLET, SpinMultiplicity.TRIPLET),
)
def test_s0_spin_branches_agree_with_tight_dop853_under_legacy_scale_field(
    multiplicity: SpinMultiplicity,
) -> None:
    one_body_at, interaction, projectors, definition = _s0_field_problem(multiplicity)
    final_time = 0.4
    reference = integrate_dop853_time_dependent_projectors(
        one_body_at,
        interaction,
        projectors,
        definition,
        final_time_fs=final_time,
        rtol=2.0e-11,
        atol=2.0e-13,
        max_step_fs=0.005,
    )
    candidate = integrate_rkmk4_time_dependent_projectors(
        one_body_at,
        interaction,
        projectors,
        definition,
        dt_fs=0.01,
        steps=40,
    )
    metrics = compare_time_dependent_projectors(
        reference.projectors,
        candidate.projectors,
        definition,
    )
    assert reference.success
    assert metrics.projector_distance < 1.0e-8
    assert metrics.rdm_distance < 1.0e-8
    assert metrics.constraints.maximum_idempotency_error < 2.0e-13
    assert metrics.constraints.maximum_mutual_orthogonality_error < 2.0e-13
