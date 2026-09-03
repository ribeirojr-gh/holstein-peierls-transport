"""D0b frozen-geometry benchmark for the spin-adapted projector equations.

The benchmark starts from the same 4x4 checkerboard-gapped isotropic control used
by S0. For each multiplicity, the state-specific open-shell orbitals are first
converged at the frozen geometry. A small allowed occupied-virtual orbital
rotation then moves the electronic state away from the stationary solution
without changing the lattice. The resulting nonlinear electronic motion is
compared between:

- a tight adaptive DOP853 reference;
- a looser adaptive DOP853 control;
- fixed-step RK4;
- a structure-preserving predictor exponential midpoint control; and
- fourth-order Runge-Kutta-Munthe-Kaas (RKMK4), which evolves on the
  anti-Hermitian Lie algebra and preserves the projector manifold by
  construction.

No lattice motion, field, thermostat, or post-step projector repair is used.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import expm

from holstein_peierls.dynamics.spin_adapted import (
    compare_projector_states,
    integrate_dop853_projectors,
    integrate_predictor_exponential_midpoint,
    integrate_rk4_projectors,
    projector_constraints,
    projector_energy,
)
from holstein_peierls.dynamics.spin_adapted_rkmk import integrate_rkmk4_projectors
from holstein_peierls.hamiltonian import build_dense_hamiltonian
from holstein_peierls.lattice import LatticeState
from holstein_peierls.spin_adapted.excitation_reference import (
    density_density_control_interaction,
    excited_state_definition,
    reference_shell_sizes,
)
from holstein_peierls.spin_adapted.isotropic import IsotropicControlParameters
from holstein_peierls.spin_adapted.orbital_optimization import (
    optimize_open_shell_orbitals,
    shell_projectors_from_complete_orbitals,
)
from holstein_peierls.spin_adapted.relaxation_control import (
    half_filled_n_closed,
    isotropic_staggered_site_energies,
)
from holstein_peierls.spin_adapted.spin import SpinMultiplicity


def _problem(
    *,
    size: int,
    staggered_gap: float,
    onsite_u: float,
    nearest_neighbor_v: float,
) -> tuple[np.ndarray, np.ndarray, object]:
    control = IsotropicControlParameters()
    parameters = control.to_polaron_parameters(
        nx=size,
        ny=size,
        polaron_position=(size // 2) * size + (size // 2) + 1,
    )
    lattice = LatticeState.zeros(size, size)
    one_body = build_dense_hamiltonian(lattice, parameters).astype(np.float64)
    site_energies = isotropic_staggered_site_energies(parameters, staggered_gap)
    one_body = np.asarray(
        one_body + np.diag(site_energies.ravel(order="C")),
        dtype=np.float64,
    )
    interaction = density_density_control_interaction(
        parameters,
        onsite_u=onsite_u,
        nearest_neighbor_v=nearest_neighbor_v,
    )
    return one_body, interaction, parameters


def _kicked_projectors(
    orbitals: np.ndarray,
    shell_sizes: tuple[int, ...],
    definition: object,
    *,
    angle: float,
) -> tuple[np.ndarray, ...]:
    occupied_columns = sum(shell_sizes)
    if occupied_columns >= orbitals.shape[1]:
        raise ValueError("benchmark requires at least one virtual orbital")
    if angle <= 0.0:
        raise ValueError("kick angle must be positive")

    generator = np.zeros((orbitals.shape[1], orbitals.shape[1]), dtype=np.float64)
    occupied = occupied_columns - 1
    virtual = occupied_columns
    generator[occupied, virtual] = +angle
    generator[virtual, occupied] = -angle
    rotation = expm(generator)
    kicked = np.asarray(orbitals @ rotation, dtype=np.float64)
    return tuple(
        np.asarray(projector, dtype=np.complex128)
        for projector in shell_projectors_from_complete_orbitals(
            kicked,
            shell_sizes,
            definition,
        )
    )


def _record(
    *,
    name: str,
    result: object,
    one_body: np.ndarray,
    interaction: np.ndarray,
    initial: tuple[np.ndarray, ...],
    reference: tuple[np.ndarray, ...],
    definition: object,
    final_time_fs: float,
) -> dict[str, object]:
    metrics = compare_projector_states(
        one_body,
        interaction,
        initial,
        reference,
        result.projectors,
        definition,
    )
    return {
        "method": name,
        "final_time_fs": final_time_fs,
        "elapsed_seconds": result.elapsed_seconds,
        "rhs_evaluations": result.rhs_evaluations,
        "accepted_steps": result.accepted_steps,
        "success": result.success,
        "message": result.message,
        "metrics": asdict(metrics),
    }


def _run_multiplicity(
    *,
    multiplicity: SpinMultiplicity,
    one_body: np.ndarray,
    interaction: np.ndarray,
    parameters: object,
    kick_angle: float,
    final_time_fs: float,
    dt_values: tuple[float, ...],
) -> dict[str, object]:
    definition = excited_state_definition(multiplicity)
    n_closed = half_filled_n_closed(parameters.n_sites)
    _, shell_sizes = reference_shell_sizes(n_closed, multiplicity)
    _, initial_orbitals = np.linalg.eigh(one_body)

    optimization_start = perf_counter()
    optimized = optimize_open_shell_orbitals(
        one_body,
        interaction,
        np.asarray(initial_orbitals, dtype=np.float64),
        shell_sizes,
        definition,
        gradient_tolerance=1.0e-8,
        max_iterations=800,
    )
    optimization_elapsed = perf_counter() - optimization_start
    if not optimized.diagnostics.converged:
        raise RuntimeError(
            f"{multiplicity.value} S0 electronic control did not converge: "
            f"gradient={optimized.diagnostics.final_max_gradient:.3e}"
        )

    initial = _kicked_projectors(
        optimized.orbitals,
        shell_sizes,
        definition,
        angle=kick_angle,
    )
    initial_constraints = projector_constraints(initial, definition)
    initial_energy = projector_energy(one_body, interaction, initial, definition)

    reference = integrate_dop853_projectors(
        one_body,
        interaction,
        initial,
        definition,
        final_time_fs=final_time_fs,
        rtol=2.0e-12,
        atol=2.0e-14,
        max_step_fs=0.01,
    )
    if not reference.success:
        raise RuntimeError(
            f"tight DOP853 reference failed for {multiplicity.value}: "
            f"{reference.message}"
        )

    records: list[dict[str, object]] = []
    moderate = integrate_dop853_projectors(
        one_body,
        interaction,
        initial,
        definition,
        final_time_fs=final_time_fs,
        rtol=1.0e-8,
        atol=1.0e-10,
        max_step_fs=0.05,
    )
    records.append(
        _record(
            name="dop853_rtol1e-8",
            result=moderate,
            one_body=one_body,
            interaction=interaction,
            initial=initial,
            reference=reference.projectors,
            definition=definition,
            final_time_fs=final_time_fs,
        )
    )

    for dt_fs in dt_values:
        steps_float = final_time_fs / dt_fs
        steps = int(round(steps_float))
        if not np.isclose(steps * dt_fs, final_time_fs, rtol=0.0, atol=1.0e-12):
            raise ValueError("final time must be an integer multiple of every dt")

        rk4 = integrate_rk4_projectors(
            one_body,
            interaction,
            initial,
            definition,
            dt_fs=dt_fs,
            steps=steps,
        )
        records.append(
            _record(
                name=f"rk4_dt{dt_fs:g}",
                result=rk4,
                one_body=one_body,
                interaction=interaction,
                initial=initial,
                reference=reference.projectors,
                definition=definition,
                final_time_fs=final_time_fs,
            )
        )

        midpoint = integrate_predictor_exponential_midpoint(
            one_body,
            interaction,
            initial,
            definition,
            dt_fs=dt_fs,
            steps=steps,
        )
        records.append(
            _record(
                name=f"exp_midpoint_dt{dt_fs:g}",
                result=midpoint,
                one_body=one_body,
                interaction=interaction,
                initial=initial,
                reference=reference.projectors,
                definition=definition,
                final_time_fs=final_time_fs,
            )
        )

        rkmk4 = integrate_rkmk4_projectors(
            one_body,
            interaction,
            initial,
            definition,
            dt_fs=dt_fs,
            steps=steps,
        )
        records.append(
            _record(
                name=f"rkmk4_dt{dt_fs:g}",
                result=rkmk4,
                one_body=one_body,
                interaction=interaction,
                initial=initial,
                reference=reference.projectors,
                definition=definition,
                final_time_fs=final_time_fs,
            )
        )

    reference_constraints = projector_constraints(reference.projectors, definition)
    reference_energy = projector_energy(
        one_body,
        interaction,
        reference.projectors,
        definition,
    )
    return {
        "multiplicity": multiplicity.value,
        "shell_sizes": list(shell_sizes),
        "optimization": {
            "elapsed_seconds": float(optimization_elapsed),
            "iterations": optimized.diagnostics.iterations,
            "final_energy": optimized.diagnostics.final_energy,
            "final_max_gradient": optimized.diagnostics.final_max_gradient,
            "accepted_steps": optimized.diagnostics.accepted_steps,
            "rejected_steps": optimized.diagnostics.rejected_steps,
        },
        "kick_angle_radians": kick_angle,
        "initial_energy": initial_energy,
        "initial_constraints": asdict(initial_constraints),
        "reference": {
            "elapsed_seconds": reference.elapsed_seconds,
            "rhs_evaluations": reference.rhs_evaluations,
            "accepted_steps": reference.accepted_steps,
            "initial_energy": initial_energy,
            "final_energy": reference_energy,
            "energy_drift": abs(reference_energy - initial_energy),
            "constraints": asdict(reference_constraints),
        },
        "records": records,
    }


def _markdown(payload: dict[str, object]) -> str:
    lines = [
        "# D0b frozen-geometry spin-adapted projector benchmark",
        "",
        f"Lattice: `{payload['size']} x {payload['size']}`; final time: `{payload['final_time_fs']} fs`.",
        "",
        "| multiplicity | method | elapsed [ms] | RHS evals | projector error | RDM error | energy drift [eV] | idempotency | orthogonality |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for branch in payload["branches"]:
        multiplicity = branch["multiplicity"]
        for record in branch["records"]:
            metrics = record["metrics"]
            constraints = metrics["constraints"]
            lines.append(
                "| {mult} | {method} | {elapsed:.6g} | {rhs} | {proj:.3e} | {rdm:.3e} | {energy:.3e} | {idem:.3e} | {orth:.3e} |".format(
                    mult=multiplicity,
                    method=record["method"],
                    elapsed=1.0e3 * record["elapsed_seconds"],
                    rhs=record["rhs_evaluations"],
                    proj=metrics["projector_distance"],
                    rdm=metrics["rdm_distance"],
                    energy=metrics["energy_drift"],
                    idem=constraints["maximum_idempotency_error"],
                    orth=constraints["maximum_mutual_orthogonality_error"],
                )
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--staggered-gap", type=float, default=2.0)
    parser.add_argument("--onsite-u", type=float, default=0.525)
    parser.add_argument("--nearest-neighbor-v", type=float, default=0.08)
    parser.add_argument("--kick-angle", type=float, default=0.05)
    parser.add_argument("--final-time-fs", type=float, default=0.4)
    parser.add_argument("--dt-values", nargs="+", type=float, default=[0.04, 0.02, 0.01])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    if args.size < 2 or args.size % 2 != 0:
        raise ValueError("D0b benchmark size must be even and at least two")
    if args.final_time_fs <= 0.0:
        raise ValueError("final time must be positive")
    dt_values = tuple(sorted(set(args.dt_values), reverse=True))
    if not dt_values or min(dt_values) <= 0.0:
        raise ValueError("time steps must be positive")

    one_body, interaction, parameters = _problem(
        size=args.size,
        staggered_gap=args.staggered_gap,
        onsite_u=args.onsite_u,
        nearest_neighbor_v=args.nearest_neighbor_v,
    )
    branches = [
        _run_multiplicity(
            multiplicity=multiplicity,
            one_body=one_body,
            interaction=interaction,
            parameters=parameters,
            kick_angle=args.kick_angle,
            final_time_fs=args.final_time_fs,
            dt_values=dt_values,
        )
        for multiplicity in (SpinMultiplicity.SINGLET, SpinMultiplicity.TRIPLET)
    ]
    payload = {
        "size": args.size,
        "staggered_gap_ev": args.staggered_gap,
        "onsite_u_ev": args.onsite_u,
        "nearest_neighbor_v_ev": args.nearest_neighbor_v,
        "kick_angle_radians": args.kick_angle,
        "final_time_fs": args.final_time_fs,
        "dt_values_fs": list(dt_values),
        "branches": branches,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    text = _markdown(payload)
    args.markdown.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
