"""Tests for the IP0d full-charge-cloud translation constraint."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from holstein_peierls.electronic import solve_ground_state
from holstein_peierls.energy import lattice_energy
from holstein_peierls.parameters import StaticPolaronParameters
from holstein_peierls.polaron import solve_static_polaron
from holstein_peierls.translation_barrier import (
    interpolate_lattice_states,
    translate_lattice_state,
)
from holstein_peierls.translation_charge_constraint import (
    charge_constrained_frozen_profile,
    solve_charge_constrained_state,
    translation_charge_order_parameter,
)


def _relaxed_case(*, ratio: float = 0.5):
    """Return a small, localized test fixture with distinguishable endpoints.

    A 4x4 fully isotropic control is deliberately *not* used as the default
    fixture.  At this size the relaxed isotropic ground state can be essentially
    translation invariant, so a charge-cloud order parameter between one-site
    translated copies is undefined.  The production IP0d benchmark uses 20x20,
    where the isotropic polaron is localized enough for the endpoint charge
    clouds to be distinguishable.
    """
    parameters = StaticPolaronParameters(nx=4, ny=4, polaron_position=11)
    parameters = replace(parameters, j0y=parameters.j0x * ratio)
    result = solve_static_polaron(
        parameters,
        solver="dense_lowest",
        gradient_mode="optimized",
        legacy_convergence=False,
    )
    assert result.diagnostics.converged
    return parameters, result.state


def test_translation_charge_order_parameter_has_unit_endpoint_coordinates() -> None:
    parameters, state = _relaxed_case(ratio=0.5)
    order = translation_charge_order_parameter(
        state, parameters, "+x", solver="dense_lowest"
    )
    assert np.isclose(order.start_expectation - order.offset, 1.0, atol=1.0e-11)
    assert np.isclose(order.end_expectation - order.offset, -1.0, atol=1.0e-11)
    assert np.isclose(order.start_expectation - order.end_expectation, 2.0, atol=1.0e-11)


def test_translation_charge_order_parameter_rejects_indistinguishable_endpoints() -> None:
    parameters, state = _relaxed_case(ratio=1.0)
    with pytest.raises(
        ValueError, match="translated endpoint charge densities are not distinguishable"
    ):
        translation_charge_order_parameter(
            state, parameters, "+x", solver="dense_lowest"
        )


def test_natural_endpoint_constraint_needs_zero_bias() -> None:
    parameters, state = _relaxed_case(ratio=0.15)
    order = translation_charge_order_parameter(
        state, parameters, "+x", solver="dense_lowest"
    )
    constrained = solve_charge_constrained_state(
        state,
        parameters,
        order,
        order.start_expectation,
        solver="dense_lowest",
    )
    ground = solve_ground_state(state, parameters, solver="dense_lowest")
    assert abs(constrained.bias_lambda_eV) < 1.0e-9
    assert constrained.constraint_residual < 1.0e-10
    assert np.isclose(constrained.physical_energy_eV, ground.energy, atol=1.0e-10)


def test_midpoint_constraint_hits_full_charge_order_target() -> None:
    parameters, state = _relaxed_case(ratio=0.5)
    translated = translate_lattice_state(state, "+x")
    midpoint = interpolate_lattice_states(state, translated, 0.5)
    order = translation_charge_order_parameter(
        state, parameters, "+x", solver="dense_lowest"
    )
    target = order.target(0.5)
    constrained = solve_charge_constrained_state(
        midpoint,
        parameters,
        order,
        target,
        solver="dense_lowest",
    )
    assert constrained.constraint_residual < 1.0e-9
    assert abs(order.centered_coordinate(constrained.expectation)) < 1.0e-9
    assert constrained.norm_error < 1.0e-12


def test_constrained_physical_energy_is_not_below_adiabatic_ground_state() -> None:
    parameters, state = _relaxed_case(ratio=0.15)
    translated = translate_lattice_state(state, "+y")
    lattice = interpolate_lattice_states(state, translated, 0.5)
    order = translation_charge_order_parameter(
        state, parameters, "+y", solver="dense_lowest"
    )
    constrained = solve_charge_constrained_state(
        lattice,
        parameters,
        order,
        order.target(0.5),
        solver="dense_lowest",
    )
    ground = solve_ground_state(lattice, parameters, solver="dense_lowest")
    assert constrained.physical_energy_eV >= ground.energy - 1.0e-10


def test_biased_eigenvalue_identity_removes_constraint_bias() -> None:
    parameters, state = _relaxed_case(ratio=0.5)
    translated = translate_lattice_state(state, "+x")
    lattice = interpolate_lattice_states(state, translated, 0.25)
    order = translation_charge_order_parameter(
        state, parameters, "+x", solver="dense_lowest"
    )
    constrained = solve_charge_constrained_state(
        lattice,
        parameters,
        order,
        order.target(0.25),
        solver="dense_lowest",
    )
    expected = (
        constrained.physical_energy_eV
        + constrained.bias_lambda_eV * constrained.expectation
    )
    assert np.isclose(constrained.biased_eigenvalue_eV, expected, atol=1.0e-10)
    assert constrained.biased_energy_identity_error_eV < 1.0e-10


def test_charge_constrained_profile_preserves_endpoints_and_variational_bound() -> None:
    parameters, state = _relaxed_case(ratio=0.5)
    profile = charge_constrained_frozen_profile(
        state,
        parameters,
        direction="+y",
        image_count=5,
        solver="dense_lowest",
    )
    assert profile.endpoint_energy_mismatch_eV < 1.0e-9
    assert profile.maximum_constraint_residual < 1.0e-8
    assert profile.maximum_norm_error < 1.0e-12
    assert profile.maximum_biased_identity_error_eV < 1.0e-9
    assert profile.minimum_constraint_energy_penalty_eV >= -1.0e-9
    assert profile.barrier_eV >= 0.0
    coordinates = np.asarray(
        [image.centered_coordinate for image in profile.images], dtype=np.float64
    )
    assert np.allclose(coordinates, np.linspace(1.0, -1.0, 5), atol=1.0e-8)

    # Endpoint totals include exactly the same lattice energy as the physical
    # unconstrained relaxed endpoints; no Lagrange-multiplier energy is retained.
    translated = translate_lattice_state(state, "+y")
    for lattice, image in ((state, profile.images[0]), (translated, profile.images[-1])):
        ground = solve_ground_state(lattice, parameters, solver="dense_lowest")
        intra, inter = lattice_energy(lattice, parameters)
        assert np.isclose(
            image.physical_total_energy_eV,
            intra + inter + ground.energy,
            atol=1.0e-9,
        )
