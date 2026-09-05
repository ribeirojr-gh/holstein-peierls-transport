"""Finite-temperature D6 pair trajectories with fixed-interval IDC events.

Between collapse events the trajectory is exactly the validated D6c
BAOAB/CF4-Lanczos pair dynamics.  At each event the pair state is collapsed in
the correct instantaneous physical sector by :mod:`pair_decoherence`.

The zero-field generalized energy balance is

    Delta E_matter ~= Q_lattice_bath + Q_electronic_environment.

The two stochastic processes use independent caller-owned NumPy generators.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.random import Generator

from .electronic_decoherence import IDCScheme
from .langevin import LangevinBath
from .pair_coupled import (
    MovingPairHamiltonianFactory,
    PairCoupledState,
    PairLatticeMasses,
    PairParameters,
)
from .pair_decoherence import (
    PairInstantaneousDecoherenceEvent,
    apply_pair_instantaneous_decoherence,
)
from .pair_frozen import PairSector
from .pair_thermal import (
    ZeroModePolicy,
    pair_coupled_baoab_step,
)


@dataclass(frozen=True, slots=True)
class PairIDCPropagationResult:
    """Final state and accumulated external energy exchanges for D6e."""

    state: PairCoupledState
    lattice_bath_heat_eV: float
    electronic_environment_exchange_eV: float
    elapsed_seconds: float
    steps: int
    idc_events: int
    hamiltonian_evaluations: int
    hamiltonian_applications: int
    scheme: IDCScheme
    decoherence_interval_fs: float


def decoherence_interval_steps(decoherence_interval_fs: float, dt_fs: float) -> int:
    interval = float(decoherence_interval_fs)
    dt = float(dt_fs)
    if not np.isfinite(interval) or not np.isfinite(dt) or interval <= 0.0 or dt <= 0.0:
        raise ValueError("decoherence interval and dt must be finite and positive")
    steps = int(round(interval / dt))
    if steps <= 0 or not np.isclose(steps * dt, interval, rtol=0.0, atol=1.0e-12):
        raise ValueError("decoherence_interval_fs must be an integer multiple of dt_fs")
    return steps


def apply_idc_to_pair_state(
    state: PairCoupledState,
    sector: PairSector,
    parameters: PairParameters,
    temperature_K: float,
    scheme: IDCScheme,
    rng: Generator,
    *,
    factory: MovingPairHamiltonianFactory | None = None,
) -> tuple[PairCoupledState, PairInstantaneousDecoherenceEvent]:
    """Apply one IDC event while holding lattice coordinates/velocities fixed."""
    event = apply_pair_instantaneous_decoherence(
        state.lattice,
        sector,
        parameters,
        state.electronic_state,
        temperature_K,
        scheme,
        rng,
        factory=factory,
    )
    collapsed = PairCoupledState(
        state.lattice.copy(),
        state.velocity.copy(),
        event.electronic_state.copy(),
    )
    return collapsed, event


def integrate_pair_coupled_baoab_idc(
    initial_state: PairCoupledState,
    sector: PairSector,
    parameters: PairParameters,
    masses: PairLatticeMasses,
    bath: LangevinBath,
    lattice_rng: Generator,
    decoherence_rng: Generator,
    *,
    dt_fs: float,
    steps: int,
    decoherence_interval_fs: float,
    scheme: IDCScheme,
    zero_mode_policy: ZeroModePolicy = "project",
    krylov_dimension: int = 8,
    initial_time_fs: float = 0.0,
) -> PairIDCPropagationResult:
    """Integrate a finite-temperature pair trajectory with periodic IDC events."""
    if not isinstance(lattice_rng, Generator) or not isinstance(decoherence_rng, Generator):
        raise TypeError("lattice_rng and decoherence_rng must be numpy Generators")
    if steps <= 0:
        raise ValueError("steps must be positive")
    event_stride = decoherence_interval_steps(decoherence_interval_fs, dt_fs)
    factory = MovingPairHamiltonianFactory(sector, parameters)
    current = PairCoupledState(
        initial_state.lattice.copy(),
        initial_state.velocity.copy(),
        np.asarray(initial_state.electronic_state, dtype=np.complex128).copy(),
    )
    lattice_heat = 0.0
    electronic_exchange = 0.0
    events = 0
    h_eval = 0
    h_apply = 0
    time_fs = float(initial_time_fs)
    start = perf_counter()
    for step in range(steps):
        thermal = pair_coupled_baoab_step(
            current,
            sector,
            parameters,
            masses,
            bath,
            lattice_rng,
            time_fs,
            dt_fs,
            zero_mode_policy=zero_mode_policy,
            krylov_dimension=krylov_dimension,
            factory=factory,
        )
        current = thermal.state
        lattice_heat += thermal.bath_heat_eV
        h_eval += thermal.hamiltonian_evaluations
        h_apply += thermal.hamiltonian_applications
        time_fs += float(dt_fs)
        if (step + 1) % event_stride == 0:
            current, event = apply_idc_to_pair_state(
                current,
                sector,
                parameters,
                bath.temperature_K,
                scheme,
                decoherence_rng,
                factory=factory,
            )
            electronic_exchange += event.electronic_environment_exchange_eV
            events += 1
    return PairIDCPropagationResult(
        state=current,
        lattice_bath_heat_eV=float(lattice_heat),
        electronic_environment_exchange_eV=float(electronic_exchange),
        elapsed_seconds=float(perf_counter() - start),
        steps=int(steps),
        idc_events=int(events),
        hamiltonian_evaluations=int(h_eval),
        hamiltonian_applications=int(h_apply),
        scheme=scheme,
        decoherence_interval_fs=float(decoherence_interval_fs),
    )
