"""Benchmark fully relaxed spin-adapted excitations on the isotropic S0 control.

This experiment is intentionally material agnostic.  It uses the canonical
coding control

    J1 = J2 = 0.100 eV
    alpha1 = alpha2 = 3.0 eV/A

with the historical K1/K2 values and a simple density-density validation
interaction.  The neutral pi system is half filled.  Multiple small structural
seeds are used to detect symmetry-related or metastable minima rather than to
supply physical material information.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from holstein_peierls.spin_adapted import (
    IsotropicControlParameters,
    IsotropicRelaxationSeed,
    SpinMultiplicity,
    density_density_control_interaction,
    half_filled_n_closed,
    relax_isotropic_spin_branch,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--onsite-u", type=float, default=0.525)
    parser.add_argument("--nearest-v", type=float, default=0.08)
    parser.add_argument("--seed-amplitude", type=float, default=1.0e-3)
    parser.add_argument("--orbital-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--orbital-max-iterations", type=int, default=800)
    parser.add_argument("--lattice-max-iterations", type=int, default=1200)
    parser.add_argument("--gradient-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _branch_summary(branch) -> dict[str, object]:
    result = branch.result
    delta_gamma = result.state.excitation_rdm
    delta_density = result.state.excitation_density
    positive_charge = float(np.sum(np.clip(delta_density, 0.0, None)))
    negative_charge = float(-np.sum(np.clip(delta_density, None, 0.0)))
    return {
        "multiplicity": branch.multiplicity.value,
        "seed": branch.seed.value,
        "converged": result.diagnostics.converged,
        "iterations": result.diagnostics.iterations,
        "final_max_update_A": result.diagnostics.final_max_update,
        "final_max_gradient_eV_per_A": result.diagnostics.final_max_gradient,
        "neutral_orbitals_converged": result.diagnostics.neutral_orbitals_converged,
        "excited_orbitals_converged": result.diagnostics.excited_orbitals_converged,
        "neutral_orbital_max_gradient": result.state.neutral.diagnostics.final_max_gradient,
        "excited_orbital_max_gradient": result.state.excited.diagnostics.final_max_gradient,
        "electronic_excitation_eV": result.energy.electronic_excitation,
        "lattice_energy_eV": result.energy.lattice,
        "total_referenced_energy_eV": result.energy.total,
        "particle_number_change": result.state.particle_number_change,
        "positive_excitation_charge": positive_charge,
        "negative_excitation_charge": negative_charge,
        "excitation_density_l2": float(np.sqrt(np.sum(delta_density**2))),
        "excitation_rdm_frobenius": float(np.linalg.norm(delta_gamma)),
        "max_abs_u_A": float(np.max(np.abs(result.lattice.u))),
        "max_abs_vx_A": float(np.max(np.abs(result.lattice.vx))),
        "max_abs_vy_A": float(np.max(np.abs(result.lattice.vy))),
    }


def main() -> None:
    args = _parser().parse_args()
    if args.size % 2 != 0:
        raise SystemExit("--size must be even for the half-filled closed-shell reference")

    control = IsotropicControlParameters()
    parameters = control.to_polaron_parameters(
        nx=args.size,
        ny=args.size,
        max_iterations=args.lattice_max_iterations,
    )
    interaction = density_density_control_interaction(
        parameters,
        onsite_u=args.onsite_u,
        nearest_neighbor_v=args.nearest_v,
    )

    summaries: list[dict[str, object]] = []
    for multiplicity in (SpinMultiplicity.SINGLET, SpinMultiplicity.TRIPLET):
        for seed in IsotropicRelaxationSeed:
            branch = relax_isotropic_spin_branch(
                parameters,
                interaction,
                multiplicity=multiplicity,
                seed=seed,
                seed_amplitude=args.seed_amplitude,
                orbital_gradient_tolerance=args.orbital_tolerance,
                orbital_max_iterations=args.orbital_max_iterations,
                gradient_convergence_criterion=args.gradient_tolerance,
            )
            summaries.append(_branch_summary(branch))

    by_spin: dict[str, list[dict[str, object]]] = {}
    for item in summaries:
        by_spin.setdefault(str(item["multiplicity"]), []).append(item)

    best: dict[str, dict[str, object]] = {}
    for spin, items in by_spin.items():
        converged = [item for item in items if bool(item["converged"])]
        if converged:
            best[spin] = min(
                converged,
                key=lambda item: float(item["total_referenced_energy_eV"]),
            )

    singlet_energy = (
        float(best["singlet"]["total_referenced_energy_eV"])
        if "singlet" in best
        else None
    )
    triplet_energy = (
        float(best["triplet"]["total_referenced_energy_eV"])
        if "triplet" in best
        else None
    )
    gap = (
        singlet_energy - triplet_energy
        if singlet_energy is not None and triplet_energy is not None
        else None
    )

    payload = {
        "control": {
            "size": args.size,
            "n_sites": parameters.n_sites,
            "n_closed": half_filled_n_closed(parameters.n_sites),
            "J1_eV": control.j1,
            "J2_eV": control.j2,
            "alpha1_eV_per_A": control.alpha1,
            "alpha2_eV_per_A": control.alpha2,
            "K1_eV_per_A2": control.k1,
            "K2_eV_per_A2": control.k2,
            "onsite_U_eV": args.onsite_u,
            "nearest_neighbor_V_eV": args.nearest_v,
            "seed_amplitude_A": args.seed_amplitude,
        },
        "branches": summaries,
        "best_by_spin": best,
        "singlet_minus_triplet_gap_eV": gap,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
