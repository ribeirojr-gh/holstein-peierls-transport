import numpy as np

from holstein_peierls.hamiltonian import build_dense_hamiltonian
from holstein_peierls.lattice import LatticeState
from holstein_peierls.parameters import StaticPolaronParameters
from holstein_peierls.translation_barrier import (
    frozen_translation_profile,
    interpolate_lattice_states,
    physical_energy_decomposition,
    translate_lattice_state,
    translated_site_index,
)


def _random_lattice(ny=4, nx=5, seed=12):
    rng = np.random.default_rng(seed)
    return LatticeState(
        rng.normal(scale=0.02, size=(ny, nx)),
        rng.normal(scale=0.02, size=(ny, nx)),
        rng.normal(scale=0.02, size=(ny, nx)),
    )


def _translated_vector(vector, ny, nx, direction):
    array = np.asarray(vector).reshape((ny, nx), order="C")
    shifts = {"+x": (0, 1), "-x": (0, -1), "+y": (1, 0), "-y": (-1, 0)}
    return np.roll(array, shift=shifts[direction], axis=(0, 1)).reshape(-1, order="C")


def test_translated_site_index_wraps_periodically():
    parameters = StaticPolaronParameters(nx=5, ny=4, polaron_position=5)
    assert translated_site_index(4, parameters, "+x") == 0
    assert translated_site_index(0, parameters, "-x") == 4
    assert translated_site_index(19, parameters, "+y") == 4
    assert translated_site_index(0, parameters, "-y") == 15


def test_interpolation_recovers_exact_endpoints():
    start = _random_lattice()
    end = translate_lattice_state(start, "+x")
    image0 = interpolate_lattice_states(start, end, 0.0)
    image1 = interpolate_lattice_states(start, end, 1.0)
    for name in ("u", "vx", "vy"):
        assert np.array_equal(getattr(image0, name), getattr(start, name))
        assert np.array_equal(getattr(image1, name), getattr(end, name))


def test_energy_decomposition_matches_direct_hamiltonian_expectation():
    parameters = StaticPolaronParameters(nx=5, ny=4, polaron_position=8)
    lattice = _random_lattice()
    rng = np.random.default_rng(55)
    psi = rng.normal(size=parameters.n_sites) + 1j * rng.normal(size=parameters.n_sites)
    psi /= np.linalg.norm(psi)

    decomposition = physical_energy_decomposition(lattice, parameters, psi)
    hamiltonian = build_dense_hamiltonian(lattice, parameters)
    electronic = float(np.vdot(psi, hamiltonian @ psi).real)
    dx = np.roll(lattice.vx, -1, axis=1) - lattice.vx
    dy = np.roll(lattice.vy, -1, axis=0) - lattice.vy
    lattice_energy = (
        0.5 * parameters.k1 * float(np.sum(lattice.u**2))
        + 0.5 * parameters.k2 * float(np.sum(dx**2) + np.sum(dy**2))
    )

    assert abs(decomposition.electronic - electronic) < 1.0e-12
    assert abs(decomposition.lattice - lattice_energy) < 1.0e-12
    assert abs(decomposition.total - (electronic + lattice_energy)) < 1.0e-12


def test_energy_decomposition_is_translation_invariant():
    parameters = StaticPolaronParameters(nx=5, ny=4, polaron_position=8)
    lattice = _random_lattice()
    rng = np.random.default_rng(77)
    psi = rng.normal(size=parameters.n_sites)
    psi /= np.linalg.norm(psi)

    original = physical_energy_decomposition(lattice, parameters, psi)
    translated_lattice = translate_lattice_state(lattice, "+x")
    translated_psi = _translated_vector(psi, parameters.ny, parameters.nx, "+x")
    shifted = physical_energy_decomposition(translated_lattice, parameters, translated_psi)

    assert abs(original.total - shifted.total) < 1.0e-12
    assert abs(original.intramolecular_elastic - shifted.intramolecular_elastic) < 1.0e-12
    assert abs(original.intermolecular_elastic_x - shifted.intermolecular_elastic_x) < 1.0e-12
    assert abs(original.intermolecular_elastic_y - shifted.intermolecular_elastic_y) < 1.0e-12
    assert abs(original.holstein_coupling - shifted.holstein_coupling) < 1.0e-12
    assert abs(original.bare_transfer_x - shifted.bare_transfer_x) < 1.0e-12
    assert abs(original.bare_transfer_y - shifted.bare_transfer_y) < 1.0e-12
    assert abs(original.peierls_coupling_x - shifted.peierls_coupling_x) < 1.0e-12
    assert abs(original.peierls_coupling_y - shifted.peierls_coupling_y) < 1.0e-12


def test_frozen_profile_has_equivalent_translated_endpoints():
    parameters = StaticPolaronParameters(nx=4, ny=4, polaron_position=7)
    lattice = _random_lattice(ny=4, nx=4, seed=99)
    profile = frozen_translation_profile(
        lattice,
        parameters,
        direction="+x",
        image_count=3,
        solver="dense_lowest",
    )
    assert len(profile.images) == 3
    assert profile.target_site == translated_site_index(parameters.polaron_index, parameters, "+x")
    assert profile.endpoint_energy_mismatch_eV < 1.0e-11
    assert profile.maximum_decomposition_error_eV < 1.0e-11
    assert max(image.norm_error for image in profile.images) < 1.0e-12
    assert profile.barrier_eV >= -1.0e-12


def test_frozen_profile_requires_odd_image_count():
    parameters = StaticPolaronParameters(nx=4, ny=4, polaron_position=7)
    lattice = _random_lattice(ny=4, nx=4)
    try:
        frozen_translation_profile(lattice, parameters, image_count=4)
    except ValueError as exc:
        assert "odd" in str(exc)
    else:
        raise AssertionError("expected ValueError")
