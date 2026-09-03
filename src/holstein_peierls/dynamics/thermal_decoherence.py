"""D5b coupling of D4 thermal Ehrenfest dynamics to instantaneous decoherence.

This module intentionally starts with the zero-field one-carrier problem. The
validated D4 BAOAB/CF4-Lanczos step advances the coherent quantum-classical
trajectory between fixed-interval IDC events. At an event the electronic state
is collapsed in the instantaneous adiabatic basis using one of the D5b schemes
from :mod:`electronic_decoherence`.

Two independent caller-owned random streams are required:

- ``lattice_rng`` drives the D4 Ornstein-Uhlenbeck lattice bath;
- ``decoherence_rng`` selects the electronic collapse state.

This separation prevents a change of decoherence scheme from shifting the raw
random-number sequence used by the lattice thermostat.

The total matter-energy bookkeeping for zero external field is

    Delta E_matter ~= Q_lattice + Q_electronic_environment,

where the first term is the kinetic-energy exchange in D4 O steps and the second
is the sum of exact electronic-energy jumps at IDC events. The residual is only
an integration diagnostic; a thermostatted trajectory does not conserve matter
energy by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.random import Generator

from ..parameters import StaticPolaronParameters
from .coupled import CoupledEhrenfestState
from .electronic_decoherence import (
    IDCScheme,
    InstantaneousDecoherenceEvent,
    apply_instantaneous_decoherence,
)
from .langevin import LangevinBath
from .thermal import (
    ElectronicMethod,
    ZeroModePolicy,
    coupled_baoab_step,
)


@dataclass(frozen=True, slots=True)
class ThermostattedIDCPropagationResult:
    """Final D5b state and explicit energy exchanges."""

    state: CoupledEhrenfestState
    lattice_bath_heat_eV: float
    electronic_environment_exchange_eV: float
    elapsed_seconds: float
    steps: int
    decoherence_events: int
    decoherence_diagonalizations: int
    hamiltonian_evaluations: int
    hamiltonian_applications: int
    scheme: IDCScheme
    decoherence_interval_fs: float
    zero_mode_policy: ZeroModePolicy


def decoherence_interval_steps(
    decoherence_interval_fs: float,
    dt_fs: float,
    *,
    tolerance: float = 1.0e-10,
) -> int:
    """Return the integer step count between fixed IDC events.

    D5b initially requires a fixed interval commensurate with the propagation
    timestep so that event timing is exact and auditable. Random/Poisson event
    schedules are a later model extension.
    """
    interval = float(decoherence_interval_fs)
    dt = float(dt_fs)
    if not np.isfinite(interval) or interval <= 0.0:
        raise ValueError("decoherence_interval_fs must be finite and positive")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_fs must be finite and positive")
    ratio = interval / dt
    steps = int(round(ratio))
    if steps <= 0 or abs(ratio - steps) > tolerance * max(1.0, abs(ratio)):
        raise ValueError("decoherence interval must be an integer multiple of dt_fs")
    return steps


def apply_idc_to_coupled_state(
    state: CoupledEhrenfestState,
    parameters: StaticPolaronParameters,
    temperature_K: float,
    scheme: IDCScheme,
    rng: Generator,
) -> tuple[CoupledEhrenfestState, InstantaneousDecoherenceEvent]:
    """Apply one IDC event while leaving lattice coordinates/velocities fixed."""
    event = apply_instantaneous_decoherence(
        state.lattice,
        parameters,
        state.electronic_state,
        temperature_K,
        scheme,
        rng,
    )
    collapsed = CoupledEhrenfestState(
        state.lattice.copy(),
        state.velocity.copy(),
        event.electronic_state.copy(),
    )
    return collapsed, event


def integrate_coupled_baoab_idc(
    initial_state: CoupledEhrenfestState,
    parameters: StaticPolaronParameters,
    bath: LangevinBath,
    lattice_rng: Generator,
    decoherence_rng: Generator,
    *,
    dt_fs: float,
    steps: int,
    decoherence_interval_fs: float,
    scheme: IDCScheme,
    initial_time_fs: float = 0.0,
    zero_mode_policy: ZeroModePolicy = "project",
    electronic_method: ElectronicMethod = "cfm4_lanczos",
    krylov_dimension: int = 6,
) -> ThermostattedIDCPropagationResult:
    """Integrate zero-field finite-temperature D5b dynamics with fixed IDC events."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not isinstance(lattice_rng, Generator) or not isinstance(decoherence_rng, Generator):
        raise TypeError("lattice_rng and decoherence_rng must be numpy.random.Generator instances")
    event_stride = decoherence_interval_steps(decoherence_interval_fs, dt_fs)
    current = CoupledEhrenfestState(
        initial_state.lattice.copy(),
        initial_state.velocity.copy(),
        np.asarray(initial_state.electronic_state, dtype=np.complex128).copy(),
    )
    time_fs = float(initial_time_fs)
    if not np.isfinite(time_fs):
        raise ValueError("initial_time_fs must be finite")

    lattice_heat = 0.0
    electronic_exchange = 0.0
    events = 0
    h_evaluations = 0
    h_applications = 0
    start = perf_counter()

    for step in range(steps):
        thermal = coupled_baoab_step(
            current,
            parameters,
            bath,
            lattice_rng,
            time_fs,
            dt_fs,
            zero_mode_policy=zero_mode_policy,
            electronic_method=electronic_method,
            krylov_dimension=krylov_dimension,
        )
        current = thermal.state
        lattice_heat += thermal.bath_heat_eV
        h_evaluations += thermal.hamiltonian_evaluations
        h_applications += thermal.hamiltonian_applications
        time_fs += float(dt_fs)

        if (step + 1) % event_stride == 0:
            current, event = apply_idc_to_coupled_state(
                current,
                parameters,
                bath.temperature_K,
                scheme,
                decoherence_rng,
            )
            electronic_exchange += event.electronic_environment_exchange_eV
            events += 1

    elapsed = perf_counter() - start
    return ThermostattedIDCPropagationResult(
        state=current,
        lattice_bath_heat_eV=float(lattice_heat),
        electronic_environment_exchange_eV=float(electronic_exchange),
        elapsed_seconds=float(elapsed),
        steps=int(steps),
        decoherence_events=int(events),
        decoherence_diagonalizations=int(events),
        hamiltonian_evaluations=int(h_evaluations),
        hamiltonian_applications=int(h_applications),
        scheme=scheme,
        decoherence_interval_fs=float(decoherence_interval_fs),
        zero_mode_policy=zero_mode_policy,
    )
