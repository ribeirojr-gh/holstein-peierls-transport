"""Driven finite-temperature one-polaron dynamics with field-aware IDC.

This TP0 module composes three previously validated components without changing
their individual physics:

- D3 uniform-electric-field Peierls phases and external-work accounting;
- D4 BAOAB lattice thermostatting; and
- D5 instantaneous decoherence (IDC) probabilities.

The important new requirement is that an IDC event under a finite field is
performed in the eigensystem of the **instantaneous field-dependent Hamiltonian
at the event time**.  The zero-field D5 Hamiltonian must not be reused after the
field has been switched on.

The explicit matter-energy bookkeeping is

    Delta E_matter ~= Q_lattice + Q_electronic + W_field.

`Q_electronic` is the exact before/after electronic-energy jump caused by the
collapse at fixed lattice coordinates, velocities, field and time.  No hidden
velocity rescaling or post-step normalization is performed.
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
    adiabatic_populations,
    idc_collapse_probabilities,
)
from .field import UniformElectricField2D, build_dense_field_hamiltonian
from .langevin import LangevinBath
from .thermal import (
    ElectronicMethod,
    ZeroModePolicy,
    coupled_baoab_step,
)
from .thermal_decoherence import decoherence_interval_steps


@dataclass(frozen=True, slots=True)
class DrivenThermostattedIDCPropagationResult:
    """Final TP0 state and all explicit environment/field energy exchanges."""

    state: CoupledEhrenfestState
    lattice_bath_heat_eV: float
    electronic_environment_exchange_eV: float
    field_work_eV: float
    elapsed_seconds: float
    steps: int
    decoherence_events: int
    decoherence_diagonalizations: int
    hamiltonian_evaluations: int
    hamiltonian_applications: int
    scheme: IDCScheme
    decoherence_interval_fs: float
    zero_mode_policy: ZeroModePolicy


def apply_field_instantaneous_decoherence(
    state: CoupledEhrenfestState,
    parameters: StaticPolaronParameters,
    field: UniformElectricField2D,
    time_fs: float,
    temperature_K: float,
    scheme: IDCScheme,
    rng: Generator,
) -> InstantaneousDecoherenceEvent:
    """Collapse in the instantaneous field-dependent adiabatic eigenbasis.

    The event time is part of the Hamiltonian because the uniform field is
    represented through a time-dependent vector potential.  The returned
    electronic-environment exchange is the exact energy discontinuity at fixed
    classical variables and fixed event time.
    """
    if not isinstance(rng, Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    time = float(time_fs)
    temperature = float(temperature_K)
    if not np.isfinite(time):
        raise ValueError("time_fs must be finite")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature_K must be positive and finite")
    state.lattice.validate()
    if state.lattice.shape != (parameters.ny, parameters.nx):
        raise ValueError("lattice shape does not match parameters")

    hamiltonian = build_dense_field_hamiltonian(
        state.lattice,
        parameters,
        field,
        time,
    )
    energies, vectors = np.linalg.eigh(hamiltonian)
    population = adiabatic_populations(vectors, state.electronic_state)
    probabilities = idc_collapse_probabilities(
        energies,
        population,
        temperature,
        scheme,
    )
    selected = int(rng.choice(energies.size, p=probabilities))
    collapsed = np.asarray(vectors[:, selected], dtype=np.complex128).copy()

    before = float(np.dot(population, energies))
    after = float(energies[selected])
    return InstantaneousDecoherenceEvent(
        electronic_state=collapsed,
        selected_state_index=selected,
        adiabatic_energies_eV=np.asarray(energies, dtype=np.float64),
        adiabatic_populations_before=np.asarray(population, dtype=np.float64),
        collapse_probabilities=np.asarray(probabilities, dtype=np.float64),
        electronic_energy_before_eV=before,
        electronic_energy_after_eV=after,
        electronic_environment_exchange_eV=float(after - before),
        scheme=scheme,
    )


def apply_field_idc_to_coupled_state(
    state: CoupledEhrenfestState,
    parameters: StaticPolaronParameters,
    field: UniformElectricField2D,
    time_fs: float,
    temperature_K: float,
    scheme: IDCScheme,
    rng: Generator,
) -> tuple[CoupledEhrenfestState, InstantaneousDecoherenceEvent]:
    """Apply one field-aware IDC event without changing classical variables."""
    event = apply_field_instantaneous_decoherence(
        state,
        parameters,
        field,
        time_fs,
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


def integrate_coupled_baoab_idc_field(
    initial_state: CoupledEhrenfestState,
    parameters: StaticPolaronParameters,
    bath: LangevinBath,
    field: UniformElectricField2D,
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
) -> DrivenThermostattedIDCPropagationResult:
    """Integrate finite-T field-driven dynamics with fixed-interval IDC events."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not isinstance(lattice_rng, Generator) or not isinstance(
        decoherence_rng, Generator
    ):
        raise TypeError(
            "lattice_rng and decoherence_rng must be numpy.random.Generator instances"
        )
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
    field_work = 0.0
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
            field=field,
            zero_mode_policy=zero_mode_policy,
            electronic_method=electronic_method,
            krylov_dimension=krylov_dimension,
        )
        current = thermal.state
        lattice_heat += thermal.bath_heat_eV
        field_work += thermal.field_work_eV
        h_evaluations += thermal.hamiltonian_evaluations
        h_applications += thermal.hamiltonian_applications
        time_fs += float(dt_fs)

        if (step + 1) % event_stride == 0:
            current, event = apply_field_idc_to_coupled_state(
                current,
                parameters,
                field,
                time_fs,
                bath.temperature_K,
                scheme,
                decoherence_rng,
            )
            electronic_exchange += event.electronic_environment_exchange_eV
            events += 1

    elapsed = perf_counter() - start
    return DrivenThermostattedIDCPropagationResult(
        state=current,
        lattice_bath_heat_eV=float(lattice_heat),
        electronic_environment_exchange_eV=float(electronic_exchange),
        field_work_eV=float(field_work),
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
