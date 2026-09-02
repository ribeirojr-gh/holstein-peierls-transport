import numpy as np
import pytest

from holstein_peierls.exciton import ExcitonParameters, energy_gradient
from holstein_peierls.exciton.solver import total_energy
from holstein_peierls.lattice import LatticeState


def _central_difference(
    state: LatticeState,
    parameters: ExcitonParameters,
    field: str,
    index: tuple[int, int],
    step: float = 1.0e-6,
) -> float:
    plus = state.copy()
    minus = state.copy()
    getattr(plus, field)[index] += step
    getattr(minus, field)[index] -= step
    e_plus, _ = total_energy(plus, parameters)
    e_minus, _ = total_energy(minus, parameters)
    return (e_plus.total - e_minus.total) / (2.0 * step)


@pytest.mark.parametrize("field", ["u", "vx", "vy"])
def test_exciton_lattice_gradient_matches_finite_difference(field: str) -> None:
    parameters = ExcitonParameters(
        nx=3,
        ny=3,
        exciton_position=5,
        electron_j0x=0.09,
        electron_j0y=0.025,
        hole_j0x=0.06,
        hole_j0y=0.04,
        electron_alpha_intra=2.7,
        hole_alpha_intra=1.9,
        electron_alpha_interx=0.35,
        electron_alpha_intery=0.22,
        hole_alpha_interx=0.18,
        hole_alpha_intery=0.31,
        onsite_attraction=0.30,
    )
    rng = np.random.default_rng(17)
    state = LatticeState(
        u=2.0e-3 * rng.standard_normal((3, 3)),
        vx=2.0e-3 * rng.standard_normal((3, 3)),
        vy=2.0e-3 * rng.standard_normal((3, 3)),
    )
    gradient, _ = energy_gradient(state, parameters)
    index = (1, 1)
    analytic = getattr(gradient, field)[index]
    numeric = _central_difference(state, parameters, field, index)
    assert analytic == pytest.approx(numeric, abs=2.0e-6)
