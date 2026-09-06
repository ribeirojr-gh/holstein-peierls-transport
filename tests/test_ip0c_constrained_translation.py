import numpy as np

from holstein_peierls.lattice import LatticeState
from holstein_peierls.parameters import StaticPolaronParameters
from holstein_peierls.polaron import solve_static_polaron
from holstein_peierls.translation_barrier import (
    interpolate_lattice_states,
    translate_lattice_state,
)
from holstein_peierls.translation_relaxation import (
    constrained_translation_profile,
    flatten_lattice_state,
    project_orthogonal_translation_subspace,
    reaction_coordinate_fraction,
    relax_constrained_translation_image,
    translation_direction_vector,
    unflatten_lattice_state,
)


def _parameters() -> StaticPolaronParameters:
    return StaticPolaronParameters(nx=4, ny=4, polaron_position=11)


def test_flatten_unflatten_roundtrip():
    rng = np.random.default_rng(7)
    state = LatticeState(
        rng.normal(size=(3, 4)),
        rng.normal(size=(3, 4)),
        rng.normal(size=(3, 4)),
    )
    rebuilt = unflatten_lattice_state(flatten_lattice_state(state), state.shape)
    assert np.array_equal(rebuilt.u, state.u)
    assert np.array_equal(rebuilt.vx, state.vx)
    assert np.array_equal(rebuilt.vy, state.vy)


def test_projector_removes_translation_and_gauge_components():
    rng = np.random.default_rng(11)
    start = LatticeState(
        rng.normal(size=(4, 4)),
        rng.normal(size=(4, 4)),
        rng.normal(size=(4, 4)),
    )
    end = translate_lattice_state(start, "+x")
    direction = translation_direction_vector(start, end)
    vector = rng.normal(size=direction.size)
    projected = project_orthogonal_translation_subspace(vector, direction, start.shape)
    n = 16
    assert abs(float(np.dot(projected, direction))) < 1.0e-10
    assert abs(float(np.mean(projected[n : 2 * n]))) < 1.0e-14
    assert abs(float(np.mean(projected[2 * n :]))) < 1.0e-14


def test_linear_interpolation_has_requested_reaction_coordinate():
    parameters = _parameters()
    relaxed = solve_static_polaron(parameters, solver="dense_lowest").state
    end = translate_lattice_state(relaxed, "+x")
    for fraction in (0.0, 0.2, 0.5, 0.9, 1.0):
        state = interpolate_lattice_states(relaxed, end, fraction)
        measured = reaction_coordinate_fraction(state, relaxed, end)
        assert abs(measured - fraction) < 1.0e-12


def test_constrained_relaxation_preserves_coordinate_and_lowers_energy():
    parameters = _parameters()
    relaxed = solve_static_polaron(parameters, solver="dense_lowest").state
    end = translate_lattice_state(relaxed, "+x")
    image = relax_constrained_translation_image(
        relaxed,
        end,
        parameters,
        0.5,
        solver="dense_lowest",
        max_iterations=120,
        gradient_tolerance_eV_per_A=1.0e-6,
    )
    assert image.diagnostics.reaction_coordinate_error < 1.0e-10
    assert image.energy_eV <= image.frozen_energy_eV + 1.0e-10
    assert image.diagnostics.projected_gradient_max_eV_per_A < 2.0e-5


def test_endpoint_images_are_translation_equivalent():
    parameters = _parameters()
    static = solve_static_polaron(parameters, solver="dense_lowest")
    profile = constrained_translation_profile(
        static.state,
        parameters,
        direction="+x",
        image_count=3,
        solver="dense_lowest",
        max_iterations=120,
    )
    assert profile.endpoint_energy_mismatch_eV < 1.0e-10
    assert profile.images[0].fraction == 0.0
    assert profile.images[-1].fraction == 1.0
    assert profile.barrier_eV >= -1.0e-10


def test_relaxed_profile_never_exceeds_its_frozen_images():
    parameters = _parameters()
    static = solve_static_polaron(parameters, solver="dense_lowest")
    profile = constrained_translation_profile(
        static.state,
        parameters,
        direction="+y",
        image_count=5,
        solver="dense_lowest",
        max_iterations=150,
    )
    for image in profile.images:
        assert image.energy_eV <= image.frozen_energy_eV + 1.0e-9
        assert image.diagnostics.reaction_coordinate_error < 1.0e-10
