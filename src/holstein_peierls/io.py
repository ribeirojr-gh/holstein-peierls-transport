"""Input/output helpers compatible with the archived static solver."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .lattice import LatticeState, flatten_site_array, reshape_site_array
from .parameters import StaticPolaronParameters


def read_legacy_lattice(
    directory: str | Path, parameters: StaticPolaronParameters, *, prefix: str = "in"
) -> LatticeState:
    """Read ``inU.dat``, ``inVX.dat`` and ``inVY.dat``-style files."""
    directory = Path(directory)
    arrays = []
    for suffix in ("U", "VX", "VY"):
        data = np.loadtxt(directory / f"{prefix}{suffix}.dat", dtype=np.float64)
        if data.size != parameters.n_sites:
            raise ValueError(
                f"{prefix}{suffix}.dat has {data.size} values, expected {parameters.n_sites}"
            )
        arrays.append(reshape_site_array(data, parameters.ny, parameters.nx))
    return LatticeState(*arrays)


def write_legacy_lattice(
    directory: str | Path, state: LatticeState, *, prefix: str = "out"
) -> None:
    """Write lattice vectors in the one-value-per-line legacy format."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for suffix, array in (("U", state.u), ("VX", state.vx), ("VY", state.vy)):
        np.savetxt(
            directory / f"{prefix}{suffix}.dat",
            flatten_site_array(array),
            fmt="%.17e",
        )


def write_static_fields(
    directory: str | Path,
    state: LatticeState,
    charge_density: np.ndarray,
) -> None:
    """Write human-readable static fields using explicit x/y coordinates."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    ny, nx = state.shape
    q = np.asarray(charge_density, dtype=np.float64).reshape((ny, nx))
    bond_x = np.roll(state.vx, -1, axis=1) - state.vx
    bond_y = np.roll(state.vy, -1, axis=0) - state.vy

    for filename, field in (
        ("charge_density.dat", q),
        ("intramolecular_displacement.dat", state.u),
        ("intermolecular_bond_x.dat", bond_x),
        ("intermolecular_bond_y.dat", bond_y),
    ):
        with (directory / filename).open("w", encoding="utf-8") as handle:
            handle.write("# x y value\n")
            for y in range(ny):
                for x in range(nx):
                    handle.write(f"{x + 1:d} {y + 1:d} {field[y, x]:.17e}\n")
                handle.write("\n")


def write_run_metadata(directory: str | Path, metadata: dict[str, object]) -> None:
    """Store machine-readable run metadata."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "run.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
