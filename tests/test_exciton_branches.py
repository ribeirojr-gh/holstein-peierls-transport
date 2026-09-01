from holstein_peierls.exciton import ExcitonParameters, relax_exciton_branches


def test_branch_scan_requires_convergence_and_selects_lowest_converged_state() -> None:
    parameters = ExcitonParameters(
        nx=3,
        ny=3,
        electron_j0x=0.0,
        electron_j0y=0.0,
        hole_j0x=0.0,
        hole_j0y=0.0,
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
    assert scan.all_converged
    assert scan.best_result.diagnostics.converged
    assert scan.best_result.energy.total == min(
        item.result.energy.total for item in scan.outcomes
    )
