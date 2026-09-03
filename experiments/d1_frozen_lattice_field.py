"""Reproducible D1 frozen-lattice electric-field benchmark.

The benchmark has two deliberately separate sectors.

1. Linear one-polaron D1a: a validated 20x20 static polaron is relaxed first,
   then its lattice is frozen and a uniform electric field is switched on at
   t=0.  The electronic state is propagated with tightened DOP853, RK4 and
   fourth-order commutator-free Magnus + Lanczos.
2. Spin-adapted D1b: the 4x4 checkerboard-gapped S0 singlet and triplet controls
   are optimized at zero field, then propagated under the same field with
   tightened DOP853 and fourth-order RKMK.

The default 2 mV/angstrom field is inherited from the scale used by the archived
legacy dynamics examples.  It is a numerical/legacy-scale control, not a
pentacene material prediction.  Classical lattice coordinates remain frozen in
all timed propagations.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from holstein_peierls.dynamics.field import (
    UniformElectricField2D,
    build_dense_field_hamiltonian,
    build_sparse_field_hamiltonian,
)
from holstein_peierls.dynamics.spin_adapted import projector_constraints
from holstein_peierls.dynamics.spin_adapted_time_dependent import (
    compare_time_dependent_projectors,
    integrate_dop853_time_dependent_projectors,
    integrate_rkmk4_time_dependent_projectors,
)
from holstein_peierls.dynamics.time_dependent import (
    compare_time_dependent_states,
    integrate_cfm4_lanczos,
    integrate_dop853_time_dependent,
    integrate_rk4_time_dependent,
)
from holstein_peierls.electronic import solve_ground_state
from holstein_peierls.lattice import LatticeState
from holstein_peierls.parameters import StaticPolaronParameters
from holstein_peierls.polaron import solve_static_polaron
from holstein_peierls.spin_adapted.excitation_reference import (
    density_density_control_interaction,
    excited_state_definition,
    reference_shell_sizes,
)
from holstein_peierls.spin_adapted.isotropic import IsotropicControlParameters
from holstein_peierls.spin_adapted.orbital_optimization import optimize_open_shell_orbitals
from holstein_peierls.spin_adapted.relaxation_control import (
    half_filled_n_closed,
    isotropic_staggered_site_energies,
)
from holstein_peierls.spin_adapted.spin import SpinMultiplicity


def _linear_benchmark(
    *,
    size: int,
    field_mv_per_angstrom: float,
    final_time_fs: float,
    dt_values: tuple[float, ...],
    krylov_dimensions: tuple[int, ...],
) -> dict[str, object]:
    parameters = StaticPolaronParameters(
        nx=size,
        ny=size,
        polaron_position=(size // 2) * size + (size // 2) + 1,
    )

    relaxation_start = perf_counter()
    polaron = solve_static_polaron(
        parameters,
        solver="sparse",
        gradient_mode="optimized",
        legacy_convergence=False,
    )
    relaxation_elapsed = perf_counter() - relaxation_start
    if not polaron.diagnostics.converged:
        raise RuntimeError("20x20 static-polaron preparation did not converge")

    ground = solve_ground_state(polaron.state, parameters, solver="sparse")
    initial_state = np.asarray(ground.wavefunction, dtype=np.complex128)
    field = UniformElectricField2D.from_millivolt_per_angstrom(
        field_mv_per_angstrom,
        0.0,
        ax_angstrom=3.0,
        ay_angstrom=3.0,
    )

    def hamiltonian_at(time_fs: float):
        return build_sparse_field_hamiltonian(
            polaron.state,
            parameters,
            field,
            time_fs,
        )

    reference = integrate_dop853_time_dependent(
        hamiltonian_at,
        initial_state,
        final_time_fs=final_time_fs,
        rtol=2.0e-12,
        atol=2.0e-14,
        max_step_fs=min(0.02, min(dt_values)),
    )
    if not reference.success:
        raise RuntimeError(f"linear DOP853 reference failed: {reference.message}")

    records: list[dict[str, object]] = []
    moderate = integrate_dop853_time_dependent(
        hamiltonian_at,
        initial_state,
        final_time_fs=final_time_fs,
        rtol=1.0e-8,
        atol=1.0e-10,
        max_step_fs=max(dt_values),
    )
    records.append(
        {
            "method": "dop853_rtol1e-8",
            "elapsed_seconds": moderate.elapsed_seconds,
            "hamiltonian_evaluations": moderate.hamiltonian_evaluations,
            "hamiltonian_applications": moderate.hamiltonian_applications,
            "metrics": asdict(compare_time_dependent_states(reference.state, moderate.state)),
        }
    )

    for dt_fs in dt_values:
        steps = int(round(final_time_fs / dt_fs))
        if not np.isclose(steps * dt_fs, final_time_fs, rtol=0.0, atol=1.0e-12):
            raise ValueError("linear final time must be an integer multiple of every dt")

        rk4 = integrate_rk4_time_dependent(
            hamiltonian_at,
            initial_state,
            dt_fs=dt_fs,
            steps=steps,
        )
        records.append(
            {
                "method": f"rk4_dt{dt_fs:g}",
                "elapsed_seconds": rk4.elapsed_seconds,
                "hamiltonian_evaluations": rk4.hamiltonian_evaluations,
                "hamiltonian_applications": rk4.hamiltonian_applications,
                "metrics": asdict(compare_time_dependent_states(reference.state, rk4.state)),
            }
        )

        for krylov_dimension in krylov_dimensions:
            cfm4 = integrate_cfm4_lanczos(
                hamiltonian_at,
                initial_state,
                dt_fs=dt_fs,
                steps=steps,
                krylov_dimension=krylov_dimension,
            )
            records.append(
                {
                    "method": f"cfm4_m{krylov_dimension}_dt{dt_fs:g}",
                    "elapsed_seconds": cfm4.elapsed_seconds,
                    "hamiltonian_evaluations": cfm4.hamiltonian_evaluations,
                    "hamiltonian_applications": cfm4.hamiltonian_applications,
                    "metrics": asdict(
                        compare_time_dependent_states(reference.state, cfm4.state)
                    ),
                }
            )

    phase_x, phase_y = field.phases(final_time_fs)
    return {
        "lattice_size": [size, size],
        "n_sites": parameters.n_sites,
        "static_polaron_preparation_seconds": float(relaxation_elapsed),
        "static_polaron_iterations": polaron.diagnostics.iterations,
        "static_polaron_energy_eV": polaron.total_energy,
        "initial_electronic_energy_eV": ground.energy,
        "field_mV_per_A": field_mv_per_angstrom,
        "field_angle_radians": 0.0,
        "final_phase_x_radians": phase_x,
        "final_phase_y_radians": phase_y,
        "final_time_fs": final_time_fs,
        "reference": {
            "elapsed_seconds": reference.elapsed_seconds,
            "hamiltonian_evaluations": reference.hamiltonian_evaluations,
            "hamiltonian_applications": reference.hamiltonian_applications,
            "accepted_steps": reference.accepted_steps,
        },
        "records": records,
    }


def _spin_problem(
    multiplicity: SpinMultiplicity,
    *,
    field_mv_per_angstrom: float,
):
    control = IsotropicControlParameters()
    parameters = control.to_polaron_parameters(nx=4, ny=4)
    lattice = LatticeState.zeros(4, 4)
    field = UniformElectricField2D.from_millivolt_per_angstrom(
        field_mv_per_angstrom,
        0.0,
        ax_angstrom=3.0,
        ay_angstrom=3.0,
    )
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
    zero_field_one_body = np.asarray(one_body_at(0.0).real, dtype=np.float64)
    _, orbitals = np.linalg.eigh(zero_field_one_body)
    optimized = optimize_open_shell_orbitals(
        zero_field_one_body,
        interaction,
        orbitals,
        shell_sizes,
        definition,
        gradient_tolerance=1.0e-8,
        max_iterations=800,
    )
    if not optimized.diagnostics.converged:
        raise RuntimeError(
            f"{multiplicity.value} D1 S0 preparation did not converge: "
            f"gradient={optimized.diagnostics.final_max_gradient:.3e}"
        )
    projectors = tuple(
        np.asarray(projector, dtype=np.complex128)
        for projector in optimized.projectors
    )
    return one_body_at, interaction, projectors, definition, optimized


def _spin_benchmark(
    *,
    field_mv_per_angstrom: float,
    final_time_fs: float,
    dt_values: tuple[float, ...],
) -> dict[str, object]:
    branches: list[dict[str, object]] = []
    for multiplicity in (SpinMultiplicity.SINGLET, SpinMultiplicity.TRIPLET):
        one_body_at, interaction, initial, definition, optimized = _spin_problem(
            multiplicity,
            field_mv_per_angstrom=field_mv_per_angstrom,
        )
        reference = integrate_dop853_time_dependent_projectors(
            one_body_at,
            interaction,
            initial,
            definition,
            final_time_fs=final_time_fs,
            rtol=2.0e-11,
            atol=2.0e-13,
            max_step_fs=min(0.005, min(dt_values)),
        )
        if not reference.success:
            raise RuntimeError(
                f"{multiplicity.value} D1b DOP853 failed: {reference.message}"
            )

        records: list[dict[str, object]] = []
        moderate = integrate_dop853_time_dependent_projectors(
            one_body_at,
            interaction,
            initial,
            definition,
            final_time_fs=final_time_fs,
            rtol=1.0e-8,
            atol=1.0e-10,
            max_step_fs=max(dt_values),
        )
        records.append(
            {
                "method": "dop853_rtol1e-8",
                "elapsed_seconds": moderate.elapsed_seconds,
                "rhs_evaluations": moderate.rhs_evaluations,
                "metrics": asdict(
                    compare_time_dependent_projectors(
                        reference.projectors,
                        moderate.projectors,
                        definition,
                    )
                ),
            }
        )

        for dt_fs in dt_values:
            steps = int(round(final_time_fs / dt_fs))
            if not np.isclose(steps * dt_fs, final_time_fs, rtol=0.0, atol=1.0e-12):
                raise ValueError("spin final time must be an integer multiple of every dt")
            result = integrate_rkmk4_time_dependent_projectors(
                one_body_at,
                interaction,
                initial,
                definition,
                dt_fs=dt_fs,
                steps=steps,
            )
            records.append(
                {
                    "method": f"rkmk4_dt{dt_fs:g}",
                    "elapsed_seconds": result.elapsed_seconds,
                    "rhs_evaluations": result.rhs_evaluations,
                    "metrics": asdict(
                        compare_time_dependent_projectors(
                            reference.projectors,
                            result.projectors,
                            definition,
                        )
                    ),
                }
            )

        branches.append(
            {
                "multiplicity": multiplicity.value,
                "initial_energy_eV": optimized.energy,
                "initial_orbital_gradient": optimized.diagnostics.final_max_gradient,
                "reference": {
                    "elapsed_seconds": reference.elapsed_seconds,
                    "rhs_evaluations": reference.rhs_evaluations,
                    "accepted_steps": reference.accepted_steps,
                    "constraints": asdict(projector_constraints(reference.projectors, definition)),
                },
                "records": records,
            }
        )
    return {
        "lattice_size": [4, 4],
        "field_mV_per_A": field_mv_per_angstrom,
        "staggered_gap_eV": 2.0,
        "onsite_U_eV": 0.525,
        "nearest_neighbor_V_eV": 0.08,
        "final_time_fs": final_time_fs,
        "branches": branches,
    }


def _markdown(payload: dict[str, object]) -> str:
    linear = payload["linear_polaron"]
    spin = payload["spin_adapted"]
    lines = [
        "# D1 frozen-lattice electric-field benchmark",
        "",
        "## Linear 20x20 polaron",
        "",
        "| method | elapsed [ms] | H evaluations | H applications | phase-aligned error | norm error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for record in linear["records"]:
        metrics = record["metrics"]
        lines.append(
            "| {method} | {elapsed:.6g} | {evaluations} | {applications} | {error:.3e} | {norm:.3e} |".format(
                method=record["method"],
                elapsed=1.0e3 * record["elapsed_seconds"],
                evaluations=record["hamiltonian_evaluations"],
                applications=record["hamiltonian_applications"],
                error=metrics["phase_aligned_state_error"],
                norm=metrics["norm_error"],
            )
        )

    lines.extend(
        [
            "",
            "## Spin-adapted 4x4 control",
            "",
            "| multiplicity | method | elapsed [ms] | RHS evals | projector error | RDM error | idempotency | orthogonality |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for branch in spin["branches"]:
        for record in branch["records"]:
            metrics = record["metrics"]
            constraints = metrics["constraints"]
            lines.append(
                "| {multiplicity} | {method} | {elapsed:.6g} | {rhs} | {projector:.3e} | {rdm:.3e} | {idempotency:.3e} | {orthogonality:.3e} |".format(
                    multiplicity=branch["multiplicity"],
                    method=record["method"],
                    elapsed=1.0e3 * record["elapsed_seconds"],
                    rhs=record["rhs_evaluations"],
                    projector=metrics["projector_distance"],
                    rdm=metrics["rdm_distance"],
                    idempotency=constraints["maximum_idempotency_error"],
                    orthogonality=constraints["maximum_mutual_orthogonality_error"],
                )
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linear-size", type=int, default=20)
    parser.add_argument("--field-mv-per-a", type=float, default=2.0)
    parser.add_argument("--linear-final-time-fs", type=float, default=20.0)
    parser.add_argument("--linear-dt-values", nargs="+", type=float, default=[0.2, 0.1])
    parser.add_argument("--krylov-dimensions", nargs="+", type=int, default=[6, 8])
    parser.add_argument("--spin-final-time-fs", type=float, default=1.0)
    parser.add_argument("--spin-dt-values", nargs="+", type=float, default=[0.04, 0.02, 0.01])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    linear_dt = tuple(sorted(set(args.linear_dt_values), reverse=True))
    spin_dt = tuple(sorted(set(args.spin_dt_values), reverse=True))
    krylov = tuple(sorted(set(args.krylov_dimensions)))
    if min(linear_dt) <= 0.0 or min(spin_dt) <= 0.0:
        raise ValueError("all time steps must be positive")
    if min(krylov) <= 0:
        raise ValueError("Krylov dimensions must be positive")

    payload = {
        "benchmark_note": (
            "Single-thread numerical control. The 2 mV/A field is a legacy-scale "
            "control, not a material prediction. Timings are runner-specific."
        ),
        "linear_polaron": _linear_benchmark(
            size=args.linear_size,
            field_mv_per_angstrom=args.field_mv_per_a,
            final_time_fs=args.linear_final_time_fs,
            dt_values=linear_dt,
            krylov_dimensions=krylov,
        ),
        "spin_adapted": _spin_benchmark(
            field_mv_per_angstrom=args.field_mv_per_a,
            final_time_fs=args.spin_final_time_fs,
            dt_values=spin_dt,
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    text = _markdown(payload)
    args.markdown.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
