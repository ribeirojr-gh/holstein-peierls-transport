"""Run the deterministic D0a frozen-H electronic propagator benchmark.

The benchmark uses the historical one-polaron parameter scales but an explicitly
synthetic deterministic frozen lattice distortion.  It is a numerical control,
not a material-specific trajectory and not a coupled electron-lattice dynamics
calculation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from holstein_peierls.dynamics import benchmark_frozen_hamiltonian
from holstein_peierls.hamiltonian import build_dense_hamiltonian, build_sparse_hamiltonian
from holstein_peierls.lattice import LatticeState
from holstein_peierls.parameters import StaticPolaronParameters


def deterministic_frozen_lattice(nx: int, ny: int) -> LatticeState:
    """Return a smooth periodic non-symmetric lattice control in angstrom."""
    if nx < 3 or ny < 3:
        raise ValueError("D0a benchmark requires nx, ny >= 3")
    y, x = np.indices((ny, nx), dtype=np.float64)
    phase_x = 2.0 * np.pi * x / nx
    phase_y = 2.0 * np.pi * y / ny
    u = 0.012 * np.cos(phase_x) + 0.008 * np.sin(phase_y + 0.37)
    vx = 0.016 * np.sin(phase_x + 0.41) * np.cos(phase_y)
    vy = 0.014 * np.cos(phase_x - 0.23) * np.sin(phase_y + 0.19)
    return LatticeState(
        u=np.asarray(u, dtype=np.float64),
        vx=np.asarray(vx, dtype=np.float64),
        vy=np.asarray(vy, dtype=np.float64),
    )


def deterministic_initial_state(nx: int, ny: int) -> np.ndarray:
    """Return a localized complex wavepacket that is not an H eigenstate."""
    y, x = np.indices((ny, nx), dtype=np.float64)
    center_x = 0.37 * nx
    center_y = 0.61 * ny
    dx_raw = np.abs(x - center_x)
    dy_raw = np.abs(y - center_y)
    dx = np.minimum(dx_raw, nx - dx_raw)
    dy = np.minimum(dy_raw, ny - dy_raw)
    sigma = max(1.0, 0.18 * min(nx, ny))
    amplitude = np.exp(-(dx * dx + 1.3 * dy * dy) / (2.0 * sigma * sigma))
    phase = np.exp(1.0j * (0.71 * x + 0.43 * y + 0.013 * x * y))
    state = (amplitude * phase).ravel(order="C").astype(np.complex128)
    state /= np.linalg.norm(state)
    return state


def run(args: argparse.Namespace) -> dict[str, object]:
    center = (args.ny // 2) * args.nx + (args.nx // 2) + 1
    parameters = StaticPolaronParameters(
        nx=args.nx,
        ny=args.ny,
        polaron_position=center,
    )
    lattice = deterministic_frozen_lattice(args.nx, args.ny)
    initial_state = deterministic_initial_state(args.nx, args.ny)
    dense = build_dense_hamiltonian(lattice, parameters)
    sparse = build_sparse_hamiltonian(lattice, parameters)
    suite = benchmark_frozen_hamiltonian(
        dense,
        sparse,
        initial_state,
        dt_fs=args.dt_fs,
        steps=args.steps,
        krylov_dimension=args.krylov_dimension,
    )
    return {
        "checkpoint": "D0a",
        "scope": "frozen one-particle Holstein-Peierls Hamiltonian",
        "parameter_status": "historical-scale numerical control; not a material fit",
        "lattice": {"nx": args.nx, "ny": args.ny, "n_sites": args.nx * args.ny},
        "parameters": {
            "j0x_ev": parameters.j0x,
            "j0y_ev": parameters.j0y,
            "alpha_intra_ev_per_angstrom": parameters.alpha_intra,
            "alpha_interx_ev_per_angstrom": parameters.alpha_interx,
            "alpha_intery_ev_per_angstrom": parameters.alpha_intery,
        },
        "benchmark": suite.to_dict(),
    }


def _print_summary(payload: dict[str, object]) -> None:
    benchmark = payload["benchmark"]
    assert isinstance(benchmark, dict)
    records = benchmark["records"]
    assert isinstance(records, list)
    print(
        "method                       time/step [s]   Hpsi   eig   solve   "
        "norm error       fidelity error   phase error      energy error [eV]"
    )
    for item in records:
        metrics = item["metrics"]
        print(
            f"{item['method']:<28s} "
            f"{item['seconds_per_step']:>12.5e} "
            f"{item['hamiltonian_applications']:>6d} "
            f"{item['eigendecompositions']:>5d} "
            f"{item['linear_solves']:>7d} "
            f"{metrics['norm_error']:>14.5e} "
            f"{1.0 - metrics['fidelity']:>14.5e} "
            f"{metrics['phase_aligned_state_error']:>14.5e} "
            f"{metrics['energy_error']:>18.5e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=8)
    parser.add_argument("--ny", type=int, default=8)
    parser.add_argument("--dt-fs", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--krylov-dimension", type=int, default=12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = run(args)
    _print_summary(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
