import pytest

from holstein_peierls.exciton import (
    ExcitonParameters,
    binding_energy,
    exciton_observables,
    relax_static_exciton,
)


def test_reference_frenkel_exciton_relaxes_to_stationary_state() -> None:
    parameters = ExcitonParameters(
        nx=4,
        ny=4,
        onsite_attraction=0.525,
        max_iterations=1200,
    )
    result = relax_static_exciton(parameters, initialization="frenkel")
    observables = exciton_observables(result.ground_state, parameters)

    assert result.diagnostics.converged
    assert result.diagnostics.final_max_update < parameters.convergence_criterion
    assert result.diagnostics.final_max_gradient < parameters.gradient_convergence_criterion
    assert result.ground_state.onsite_probability > 0.0
    assert 0.0 < observables.electron_ipr <= 1.0
    assert 0.0 < observables.hole_ipr <= 1.0
    assert result.energy.total < 0.0


def test_binding_energy_convention_is_positive_for_bound_state() -> None:
    assert binding_energy(-1.3, -0.5, -0.6) == pytest.approx(0.2)
