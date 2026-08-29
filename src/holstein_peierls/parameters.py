"""Input parameters for the static Holstein-Peierls polaron calculation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any


@dataclass(frozen=True, slots=True)
class StaticPolaronParameters:
    """Parameters used by the legacy ``rprop.f90`` program.

    Units are intentionally kept identical to the original implementation:
    energies in eV, displacements in angstrom, and force constants in eV/A^2.
    The masses and derived frequencies are retained for provenance even though
    they are not required by the static minimization.
    """

    nx: int = 20
    ny: int = 20
    readinput: str = "n"
    k1: float = 16.51
    k2: float = 0.51
    j0x: float = 0.100
    j0y: float = 0.015
    alpha_intra: float = 3.0
    alpha_interx: float = 0.4
    alpha_intery: float = 0.4
    m1: float = 7.5e10
    m2: float = 1.5e11
    polaron_position: int = 205
    max_iterations: int = 2000
    update_start: float = 1.0e-3
    update_max: float = 1.0e-2
    update_min: float = 2.0**-52
    acceleration_factor: float = 1.2
    deceleration_factor: float = 0.5
    convergence_criterion: float = 1.0e-8

    @property
    def n_sites(self) -> int:
        return self.nx * self.ny

    @property
    def polaron_index(self) -> int:
        index = self.polaron_position - 1
        if not 0 <= index < self.n_sites:
            raise ValueError(
                f"polaron_position={self.polaron_position} is outside a "
                f"{self.nx}x{self.ny} lattice"
            )
        return index

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_legacy_include(cls, path: str | Path) -> "StaticPolaronParameters":
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        assignments: dict[str, str] = {}
        for raw_line in text.splitlines():
            line = raw_line.split("!", 1)[0].strip()
            if not line or "=" not in line:
                continue
            line = re.sub(r"^parameter\s+", "", line, flags=re.IGNORECASE)
            for part in line.split(","):
                if "=" not in part:
                    continue
                key, value = part.split("=", 1)
                assignments[key.strip().lower()] = value.strip().rstrip("&").strip()

        def parse_number(name: str, *, integer: bool = False) -> int | float:
            value = assignments[name]
            value = re.sub(r"[dD]([+-]?\d+)", r"e\1", value)
            if value.lower().startswith("2.0**"):
                exponent = float(value.split("**", 1)[1])
                parsed: float = 2.0**exponent
            else:
                parsed = float(value)
            return int(parsed) if integer else parsed

        def parse_string(name: str) -> str:
            return assignments[name].strip().strip("'\"")

        return cls(
            nx=parse_number("nx", integer=True),
            ny=parse_number("ny", integer=True),
            readinput=parse_string("readinput"),
            k1=parse_number("k1"),
            k2=parse_number("k2"),
            j0x=parse_number("j0x"),
            j0y=parse_number("j0y"),
            alpha_intra=parse_number("alpha_intra"),
            alpha_interx=parse_number("alpha_interx"),
            alpha_intery=parse_number("alpha_intery"),
            m1=parse_number("m1"),
            m2=parse_number("m2"),
            polaron_position=parse_number("polaron_position", integer=True),
            max_iterations=parse_number("iteration", integer=True),
            update_start=parse_number("update_start"),
            update_max=parse_number("update_max"),
            update_min=parse_number("update_min"),
            acceleration_factor=parse_number("acc_val"),
            deceleration_factor=parse_number("dec_val"),
            convergence_criterion=parse_number("convergence_criterion"),
        )
