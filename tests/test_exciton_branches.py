import pytest

from holstein_peierls.exciton import ExcitonParameters, relax_exciton_branches


def test_branch_scan_requires_convergence_and_handles_ties_deterministically() -> None:
    parameters = ExcitonParameters(
        nx=3,
        ny=3,
        exciton_position=5,
        electron_j0x=1.0e-4,
        electron_j0y=1.0e-4,
        hole_j0x=1.0e-4,
        hole_j0y=1.0e-4,
        electron_alpha_intra=0.0,
        electron_alpha_interx=0.0,
        electron_alpha_intery=0.0,
        hole_alpha_intra=0.0,
        hole_alpha_interx=0.0,
        hole_alpha_intery=0.0,
        onsite_attraction=0.525,
        max_iterations=4,
    )
    scan = relax_exciton_branches(parameters, modes=("frenkel", "ct_x", "separated"))
    minimum = min(item.result.energy.total for item in scan.outcomes)

    assert scan.all_converged
    assert scan.best_result.diagnostics.converged
    assert scan.best_result.energy.total == pytest.approx(minimum, abs=1.0e-10)
    assert scan.best_mode == scan.degenerate_best_modes[0]


def test_exactly_degenerate_seed_results_use_requested_order() -> None:
    parameters = ExcitonParameters(
        nx=3,
        ny=3,
        exciton_position=5,
        electron_j0x=1.0e-4,
        electron_j0y=1.0e-4,
        hole_j0x=1.0e-4,
        hole_j0y=1.0e-4,
        electron_alpha_intra=0.0,
        electron_alpha_interx=0.0,
        electron_alpha_intery=0.0,
        hole_alpha_intra=0.0,
        hole_alpha_interx=0.0,
        hole_alpha_intery=0.0,
        onsite_attraction=0.525,
        max_iterations=4,
    )
    scan = relax_exciton_branches(
        parameters,
        modes=("ct_x", "frenkel"),
        energy_tie_tolerance_ev=1.0e-8,
    )
    assert scan.best_mode == "ct_x"
    assert scan.degenerate_best_modes == ("ct_x", "frenkel")
