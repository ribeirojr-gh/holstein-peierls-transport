import numpy as np
import pytest

from holstein_peierls.dynamics.ehrenfest import LatticeVelocity
from holstein_peierls.dynamics.mode_memory import frozen_surface_equilibrium
from holstein_peierls.dynamics.field_release import HeldPeierlsPhase2D
from holstein_peierls.dynamics.field import UniformElectricField2D
from holstein_peierls.dynamics.traveling_current_attribution import (
    decompose_vx_current,
    projection_attribution,
    row_cosine_similarity,
    squared_norm_ratio,
    traveling_vx_fields,
)
from holstein_peierls.lattice import LatticeState
from holstein_peierls.parameters import StaticPolaronParameters


def test_traveling_fields_reconstruct_coordinate_and_velocity():
    parameters = StaticPolaronParameters(nx=8, ny=4)
    rng = np.random.default_rng(7)
    equilibrium = rng.normal(scale=0.01, size=(4, 8))
    coordinate = equilibrium + rng.normal(scale=0.02, size=(4, 8))
    velocity = rng.normal(scale=1.0e-4, size=(4, 8))
    fields = traveling_vx_fields(coordinate, velocity, equilibrium, parameters)
    displacement = (
        fields.plus_displacement
        + fields.minus_displacement
        + fields.special_displacement
    )
    reconstructed_velocity = (
        fields.plus_velocity
        + fields.minus_velocity
        + fields.special_velocity
    )
    assert np.max(np.abs(displacement - (coordinate - equilibrium))) < 1.0e-12
    assert np.max(np.abs(reconstructed_velocity - velocity)) < 1.0e-12


def test_current_decomposition_closes_exactly():
    parameters = StaticPolaronParameters(nx=8, ny=4)
    rng = np.random.default_rng(11)
    equilibrium = rng.normal(scale=0.01, size=(4, 8))
    coordinate = equilibrium + rng.normal(scale=0.02, size=(4, 8))
    velocity = rng.normal(scale=1.0e-4, size=(4, 8))
    currents = decompose_vx_current(
        coordinate,
        velocity,
        equilibrium,
        parameters,
        carrier_dx_sites=-1,
    )
    reconstructed = (
        currents["retrograde"]
        + currents["comoving"]
        + currents["special"]
        + currents["cross"]
    )
    assert np.max(np.abs(reconstructed - currents["full"])) < 1.0e-14


def test_direction_mapping_reverses_with_carrier_direction():
    parameters = StaticPolaronParameters(nx=8, ny=4)
    rng = np.random.default_rng(13)
    equilibrium = np.zeros((4, 8))
    coordinate = rng.normal(scale=0.02, size=(4, 8))
    velocity = rng.normal(scale=1.0e-4, size=(4, 8))
    minus_carrier = decompose_vx_current(
        coordinate, velocity, equilibrium, parameters, carrier_dx_sites=-1
    )
    plus_carrier = decompose_vx_current(
        coordinate, velocity, equilibrium, parameters, carrier_dx_sites=1
    )
    assert np.allclose(minus_carrier["retrograde"], plus_carrier["comoving"])
    assert np.allclose(minus_carrier["comoving"], plus_carrier["retrograde"])


def test_projection_attribution_is_additive_for_exact_sum():
    full = np.array([[1.0, 2.0], [3.0, -1.0]])
    first = 0.7 * full
    second = 0.2 * full
    third = 0.1 * full
    values = [projection_attribution(x, full) for x in (first, second, third)]
    assert sum(values) == pytest.approx(1.0)
    assert squared_norm_ratio(first, full) == pytest.approx(0.49)
    cosine = row_cosine_similarity(first, full)
    assert np.allclose(cosine, 1.0)
