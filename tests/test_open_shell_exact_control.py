import numpy as np

from holstein_peierls.spin_adapted import (
    HIGH_SPIN_TRIPLET,
    OPEN_SHELL_SINGLET,
    general_open_shell_energy,
)
from holstein_peierls.spin_adapted.exact_control import (
    exact_density_density_hamiltonian,
    expectation_value,
    high_spin_triplet_state,
    open_shell_singlet_state,
)
from holstein_peierls.spin_adapted.open_shell import projector_from_orbitals


def _orthogonal_orbitals() -> np.ndarray:
    raw = np.asarray(
        (
            (1.0, 0.3, -0.2, 0.1),
            (0.2, 1.0, 0.4, -0.1),
            (-0.3, 0.1, 1.0, 0.2),
            (0.1, -0.2, 0.2, 1.0),
        )
    )
    q, _ = np.linalg.qr(raw)
    return q


def test_open_shell_singlet_energy_matches_direct_many_body_expectation() -> None:
    orbitals = _orthogonal_orbitals()
    one_body = np.asarray(
        (
            (-0.35, -0.08, 0.01, -0.02),
            (-0.08, -0.12, -0.06, 0.00),
            (0.01, -0.06, 0.18, -0.07),
            (-0.02, 0.00, -0.07, 0.31),
        )
    )
    interaction = np.asarray(
        (
            (0.75, 0.22, 0.11, 0.18),
            (0.22, 0.68, 0.19, 0.10),
            (0.11, 0.19, 0.72, 0.21),
            (0.18, 0.10, 0.21, 0.70),
        )
    )
    p_closed = projector_from_orbitals(orbitals[:, [0]])
    p_v = projector_from_orbitals(orbitals[:, [1]])
    p_c = projector_from_orbitals(orbitals[:, [2]])
    functional = general_open_shell_energy(
        one_body,
        interaction,
        (p_closed, p_v, p_c),
        OPEN_SHELL_SINGLET,
    )

    h_exact, basis = exact_density_density_hamiltonian(
        one_body, interaction, n_electrons=4
    )
    state = open_shell_singlet_state(orbitals[:, :3], n_closed=1, basis=basis)
    exact = expectation_value(state, h_exact)
    assert np.isclose(functional, exact, atol=2.0e-12)


def test_high_spin_triplet_energy_matches_direct_many_body_expectation() -> None:
    orbitals = _orthogonal_orbitals()
    one_body = np.asarray(
        (
            (-0.35, -0.08, 0.01, -0.02),
            (-0.08, -0.12, -0.06, 0.00),
            (0.01, -0.06, 0.18, -0.07),
            (-0.02, 0.00, -0.07, 0.31),
        )
    )
    interaction = np.asarray(
        (
            (0.75, 0.22, 0.11, 0.18),
            (0.22, 0.68, 0.19, 0.10),
            (0.11, 0.19, 0.72, 0.21),
            (0.18, 0.10, 0.21, 0.70),
        )
    )
    p_closed = projector_from_orbitals(orbitals[:, [0]])
    p_open = projector_from_orbitals(orbitals[:, [1, 2]])
    functional = general_open_shell_energy(
        one_body,
        interaction,
        (p_closed, p_open),
        HIGH_SPIN_TRIPLET,
    )

    h_exact, basis = exact_density_density_hamiltonian(
        one_body, interaction, n_electrons=4
    )
    state = high_spin_triplet_state(orbitals[:, :3], n_closed=1, basis=basis)
    exact = expectation_value(state, h_exact)
    assert np.isclose(functional, exact, atol=2.0e-12)
