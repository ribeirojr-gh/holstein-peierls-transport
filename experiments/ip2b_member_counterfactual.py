#!/usr/bin/env python3
"""IP2b one-member production paired counterfactual calculation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

import ip1l_interhop_wake_memory as base
import ip1n_gauge_continuous_field_release as ip1n
import ip2a4_refined_complete_calibration as calibration
from holstein_peierls.dynamics.coupled import CoupledEhrenfestState
from holstein_peierls.dynamics.direction_reversal import (
    direction_reversal_fourier_errors,
    reverse_non_special_vx_velocity,
    vx_velocity_kinetic_energy_eV,
)
from holstein_peierls.dynamics.driven import (
    coupled_field_verlet_step,
    field_dynamic_total_energy,
)
from holstein_peierls.dynamics.ehrenfest import LatticeVelocity
from holstein_peierls.dynamics.ensemble_preparation import peierls_velocity_member
from holstein_peierls.dynamics.field import (
    UniformElectricField2D,
    build_sparse_field_hamiltonian,
)
from holstein_peierls.dynamics.field_release import (
    HeldPeierlsPhase2D,
    trailing_outward_matrix,
)
from holstein_peierls.dynamics.ip2b_ensemble import (
    first_x_event_within_window,
    is_direct_recross,
    x_event_topology_history,
)
from holstein_peierls.dynamics.mode_memory import (
    frozen_surface_equilibrium,
    harmonic_wave_numbers,
    modal_energy_arrays,
    real_space_excitation_energies,
    traveling_vx_energy_split,
)
from holstein_peierls.dynamics.numerical_validation import (
    extensive_energy_balance_passes,
    size_scaled_energy_balance_tolerance_eV,
)
from holstein_peierls.dynamics.single_hop_memory import continuation_neighbor
from holstein_peierls.dynamics.thermal import zero_mode_means
from holstein_peierls.dynamics.traveling_current_attribution import (
    decompose_vx_current,
    projection_attribution,
)


SIZE = 40
PREPARATION_ENERGY_EV = 1.0e-5
FIELD_MV_PER_A = 10.0
DT_FS = 0.10
SEARCH_TIME_FS = 4000.0
FIELD_FREE_CONTROL_FS = 200.0
EARLY_EVENT_CUTOFF_FS = 200.0
CONTINUATION_FS = 2000.0
SAMPLE_INTERVAL_FS = 2.0
ENERGY_INTERVAL_FS = 10.0
KRYLOV_DIMENSION = 6
PRE_EVENT_BALANCE_LIMIT_EV = 2.0e-6
PRIMARY_DISTANCES = (1, 2, 3, 4)


def _event_record(event) -> dict:
    return {
        "source_site": int(event.source_site),
        "target_site": int(event.target_site),
        "transition_start_time_fs": float(event.transition_start_time_fs),
        "accepted_time_fs": float(event.accepted_time_fs),
        "dx_sites": int(event.dx_sites),
        "dy_sites": int(event.dy_sites),
        "is_nearest_neighbor": bool(event.is_nearest_neighbor),
        "direction": str(event.direction),
    }


def _copy_with_vx_velocity(
    state: CoupledEhrenfestState, vx_velocity: np.ndarray
) -> CoupledEhrenfestState:
    return CoupledEhrenfestState(
        state.lattice.copy(),
        LatticeVelocity(
            state.velocity.u.copy(),
            np.asarray(vx_velocity, dtype=np.float64).copy(),
            state.velocity.vy.copy(),
        ),
        np.asarray(state.electronic_state, dtype=np.complex128).copy(),
    )


def _population(state: CoupledEhrenfestState) -> np.ndarray:
    p = np.abs(np.asarray(state.electronic_state, dtype=np.complex128)) ** 2
    p /= float(np.sum(p))
    return np.asarray(p, dtype=np.float64)


def _ipr(population: np.ndarray) -> float:
    p = np.asarray(population, dtype=np.float64)
    return float(np.sum(p * p))


def _driven_first_event_with_state(
    prepared: CoupledEhrenfestState,
    parameters,
    field,
) -> dict:
    steps = base._integer_stride(SEARCH_TIME_FS, DT_FS, "search_time_fs")
    sample_stride = base._integer_stride(SAMPLE_INTERVAL_FS, DT_FS, "sample_interval_fs")
    energy_stride = base._integer_stride(ENERGY_INTERVAL_FS, DT_FS, "energy_interval_fs")
    current = ip1n._copy_state(prepared)
    initial_energy = field_dynamic_total_energy(
        current.lattice,
        current.velocity,
        parameters,
        current.electronic_state,
        field,
        0.0,
    ).total
    initial_norm = float(np.linalg.norm(current.electronic_state))
    tracker = ip1n._branch_tracker(parameters.nx, parameters.ny, SAMPLE_INTERVAL_FS)
    tracker.update(_population(current), 0.0)

    work = 0.0
    max_balance = 0.0
    max_norm_error = 0.0
    max_zero_mode = 0.0
    first_event = None
    selected_state = None
    selected_time = None

    started = perf_counter()
    for step in range(steps):
        time_fs = step * DT_FS
        current, step_work, _, _ = coupled_field_verlet_step(
            current,
            parameters,
            field,
            time_fs,
            DT_FS,
            electronic_method="cfm4_lanczos",
            krylov_dimension=KRYLOV_DIMENSION,
        )
        new_time = time_fs + DT_FS
        work += float(step_work)

        if (step + 1) % energy_stride == 0:
            energy = field_dynamic_total_energy(
                current.lattice,
                current.velocity,
                parameters,
                current.electronic_state,
                field,
                new_time,
            ).total
            max_balance = max(max_balance, abs((energy - initial_energy) - work))
            max_norm_error = max(
                max_norm_error,
                abs(float(np.linalg.norm(current.electronic_state)) - initial_norm),
            )
            max_zero_mode = max(
                max_zero_mode,
                max(abs(value) for value in zero_mode_means(current).values()),
            )

        if (step + 1) % sample_stride == 0:
            event = tracker.update(_population(current), new_time)
            if event is not None:
                first_event = _event_record(event)
                selected_state = ip1n._copy_state(current)
                selected_time = float(new_time)
                break

    if selected_state is not None and selected_time is not None:
        energy = field_dynamic_total_energy(
            selected_state.lattice,
            selected_state.velocity,
            parameters,
            selected_state.electronic_state,
            field,
            selected_time,
        ).total
        max_balance = max(max_balance, abs((energy - initial_energy) - work))
        max_norm_error = max(
            max_norm_error,
            abs(float(np.linalg.norm(selected_state.electronic_state)) - initial_norm),
        )
        max_zero_mode = max(
            max_zero_mode,
            max(abs(value) for value in zero_mode_means(selected_state).values()),
        )

    return {
        "first_persistent_event": first_event,
        "selected_state": selected_state,
        "accumulated_external_work_eV": float(work),
        "maximum_energy_work_residual_eV": float(max_balance),
        "maximum_norm_error": float(max_norm_error),
        "maximum_zero_mode_mean": float(max_zero_mode),
        "numerically_stable": bool(
            max_balance <= PRE_EVENT_BALANCE_LIMIT_EV
            and extensive_energy_balance_passes(max_balance, parameters.n_sites)
            and max_norm_error < 1.0e-10
            and max_zero_mode < 1.0e-10
        ),
        "elapsed_seconds": float(perf_counter() - started),
    }


def _branch_point_covariates(selected_state, selected_event, held, parameters) -> dict:
    equilibrium = frozen_surface_equilibrium(
        parameters, selected_state.electronic_state, held
    )
    excitation = real_space_excitation_energies(
        selected_state.lattice, selected_state.velocity, equilibrium, parameters
    )
    modal = modal_energy_arrays(
        selected_state.lattice, selected_state.velocity, equilibrium, parameters
    )
    traveling = traveling_vx_energy_split(
        selected_state.lattice,
        selected_state.velocity,
        equilibrium,
        parameters,
        carrier_dx_sites=int(selected_event["dx_sites"]),
    )

    qx = harmonic_wave_numbers(parameters.nx)
    qy = harmonic_wave_numbers(parameters.ny)
    vx_by_qx = np.sum(modal.vx, axis=0)
    vy_by_qy = np.sum(modal.vy, axis=1)
    low_x = (np.abs(qx) > 1.0e-14) & (np.abs(qx) <= np.pi / 4.0 + 1.0e-14)
    low_y = (np.abs(qy) > 1.0e-14) & (np.abs(qy) <= np.pi / 4.0 + 1.0e-14)
    vx_total = float(np.sum(vx_by_qx))
    vy_total = float(np.sum(vy_by_qy))
    low_q = {
        "vx_lowest_nonzero_quarter_fraction": (
            0.0 if vx_total <= 1.0e-30 else float(np.sum(vx_by_qx[low_x]) / vx_total)
        ),
        "vy_lowest_nonzero_quarter_fraction": (
            0.0 if vy_total <= 1.0e-30 else float(np.sum(vy_by_qy[low_y]) / vy_total)
        ),
    }

    current = decompose_vx_current(
        selected_state.lattice.vx,
        selected_state.velocity.vx,
        equilibrium.vx,
        parameters,
        carrier_dx_sites=int(selected_event["dx_sites"]),
    )
    post_hop_site = int(selected_event["target_site"])
    forward_site = continuation_neighbor(
        post_hop_site,
        dx_sites=int(selected_event["dx_sites"]),
        dy_sites=int(selected_event["dy_sites"]),
        nx=parameters.nx,
        ny=parameters.ny,
    )
    trailing_vectors = {}
    s_axis = None
    for key, field in current.items():
        axis, profile = base._project_profiles(
            np.asarray(field, dtype=np.float64)[None, :, :],
            source_site=post_hop_site,
            target_site=forward_site,
            nx=parameters.nx,
            ny=parameters.ny,
            direction_sign=int(selected_event["dx_sites"]),
        )
        if s_axis is None:
            s_axis = axis
        trailing_vectors[key] = trailing_outward_matrix(
            profile, axis, distances_sites=PRIMARY_DISTANCES
        )[0]
    reconstruction = (
        current["retrograde"] + current["comoving"] + current["special"] + current["cross"]
    )
    return {
        "fixed_surface_excitation_eV": excitation,
        "modal_energy_sums_eV": modal.sums,
        "traveling_vx_energy_split": traveling,
        "low_q_fractions": low_q,
        "branch_point_trailing_current": {
            "distances_sites": list(PRIMARY_DISTANCES),
            "full_eV_per_fs": trailing_vectors["full"].tolist(),
            "retrograde_eV_per_fs": trailing_vectors["retrograde"].tolist(),
            "comoving_eV_per_fs": trailing_vectors["comoving"].tolist(),
            "special_eV_per_fs": trailing_vectors["special"].tolist(),
            "cross_eV_per_fs": trailing_vectors["cross"].tolist(),
            "retrograde_projection_attribution": float(
                projection_attribution(
                    trailing_vectors["retrograde"], trailing_vectors["full"]
                )
            ),
            "maximum_current_reconstruction_error_eV_per_fs": float(
                np.max(np.abs(current["full"] - reconstruction))
            ),
        },
    }


def _run_three_branches(native, reversed_state, replica, parameters, held, ts, event):
    states = {
        "native": ip1n._copy_state(native),
        "reversed": ip1n._copy_state(reversed_state),
        "native_replicate": ip1n._copy_state(replica),
    }
    initial_energies = {
        key: field_dynamic_total_energy(
            state.lattice, state.velocity, parameters, state.electronic_state, held, ts
        ).total
        for key, state in states.items()
    }
    initial_norms = {
        key: float(np.linalg.norm(state.electronic_state)) for key, state in states.items()
    }
    works = {key: 0.0 for key in states}
    max_energy_drift = {key: 0.0 for key in states}
    max_norm_error = {key: 0.0 for key in states}
    max_zero_mode = {key: 0.0 for key in states}
    trackers = {
        key: ip1n._branch_tracker(parameters.nx, parameters.ny, SAMPLE_INTERVAL_FS)
        for key in states
    }
    events = {key: [] for key in states}
    populations = {key: _population(state) for key, state in states.items()}
    for key in states:
        trackers[key].update(populations[key], ts)

    sample_offsets = [0.0]
    l1 = [0.0]
    l1_ctrl = [0.0]
    max_sites = {key: [int(np.argmax(populations[key]))] for key in states}
    ipr = {key: [_ipr(populations[key])] for key in states}
    max_l1 = 0.0
    max_l1_index = 0
    max_pop_native = populations["native"].copy()
    max_pop_reversed = populations["reversed"].copy()
    branch_population = populations["native"].copy()

    steps = base._integer_stride(CONTINUATION_FS, DT_FS, "continuation_fs")
    sample_stride = base._integer_stride(SAMPLE_INTERVAL_FS, DT_FS, "sample_interval_fs")
    energy_stride = base._integer_stride(ENERGY_INTERVAL_FS, DT_FS, "energy_interval_fs")
    started = perf_counter()

    for step in range(steps):
        time_fs = ts + step * DT_FS
        new_time = time_fs + DT_FS
        for key in ("native", "reversed", "native_replicate"):
            state, work, _, _ = coupled_field_verlet_step(
                states[key],
                parameters,
                held,
                time_fs,
                DT_FS,
                electronic_method="cfm4_lanczos",
                krylov_dimension=KRYLOV_DIMENSION,
            )
            states[key] = state
            works[key] += float(work)

        if (step + 1) % energy_stride == 0 or step + 1 == steps:
            for key, state in states.items():
                energy = field_dynamic_total_energy(
                    state.lattice,
                    state.velocity,
                    parameters,
                    state.electronic_state,
                    held,
                    new_time,
                ).total
                max_energy_drift[key] = max(
                    max_energy_drift[key], abs(energy - initial_energies[key])
                )
                max_norm_error[key] = max(
                    max_norm_error[key],
                    abs(float(np.linalg.norm(state.electronic_state)) - initial_norms[key]),
                )
                max_zero_mode[key] = max(
                    max_zero_mode[key],
                    max(abs(value) for value in zero_mode_means(state).values()),
                )

        if (step + 1) % sample_stride == 0:
            sample_offsets.append(float(new_time - ts))
            for key, state in states.items():
                populations[key] = _population(state)
                max_sites[key].append(int(np.argmax(populations[key])))
                ipr[key].append(_ipr(populations[key]))
                found = trackers[key].update(populations[key], new_time)
                if found is not None:
                    events[key].append(_event_record(found))
            value = float(np.sum(np.abs(populations["native"] - populations["reversed"])))
            ctrl = float(
                np.sum(
                    np.abs(populations["native"] - populations["native_replicate"])
                )
            )
            l1.append(value)
            l1_ctrl.append(ctrl)
            if value > max_l1:
                max_l1 = value
                max_l1_index = len(l1) - 1
                max_pop_native = populations["native"].copy()
                max_pop_reversed = populations["reversed"].copy()

    offsets = np.asarray(sample_offsets, dtype=np.float64)
    l1_arr = np.asarray(l1, dtype=np.float64)
    ctrl_arr = np.asarray(l1_ctrl, dtype=np.float64)

    def first_threshold(threshold: float):
        indices = np.flatnonzero(l1_arr >= threshold)
        return None if indices.size == 0 else float(offsets[int(indices[0])])

    native_first = first_x_event_within_window(events["native"], branch_time_fs=ts)
    reversed_first = first_x_event_within_window(events["reversed"], branch_time_fs=ts)
    start_delta = None
    if native_first is not None and reversed_first is not None:
        start_delta = abs(
            float(native_first["transition_start_time_fs"])
            - float(reversed_first["transition_start_time_fs"])
        )
    post_hop_site = int(event["target_site"])
    pre_hop_site = int(event["source_site"])

    summary = {
        "primary": {
            "D_i": float(np.max(l1_arr)),
            "D_i_ctrl": float(np.max(ctrl_arr)),
            "max_l1_sample_offset_fs": float(offsets[int(np.argmax(l1_arr))]),
            "first_l1_ge_0p10_offset_fs": first_threshold(0.10),
            "first_l1_ge_0p25_offset_fs": first_threshold(0.25),
        },
        "secondary_events": {
            "native_events": events["native"],
            "reversed_events": events["reversed"],
            "native_replicate_events": events["native_replicate"],
            "native_first_x_event": native_first,
            "reversed_first_x_event": reversed_first,
            "first_transition_start_abs_difference_fs": start_delta,
            "native_first_is_direct_recross": is_direct_recross(
                native_first, post_hop_site=post_hop_site, pre_hop_site=pre_hop_site
            ),
            "reversed_first_is_direct_recross": is_direct_recross(
                reversed_first, post_hop_site=post_hop_site, pre_hop_site=pre_hop_site
            ),
            "native_x_event_topology_history": x_event_topology_history(
                events["native"], branch_time_fs=ts
            ),
            "reversed_x_event_topology_history": x_event_topology_history(
                events["reversed"], branch_time_fs=ts
            ),
        },
        "branch_numerics": {
            key: {
                "accumulated_external_work_eV": float(works[key]),
                "maximum_energy_drift_eV": float(max_energy_drift[key]),
                "maximum_norm_error": float(max_norm_error[key]),
                "maximum_zero_mode_mean": float(max_zero_mode[key]),
            }
            for key in states
        },
        "elapsed_seconds": float(perf_counter() - started),
    }
    arrays = {
        "sample_offsets_fs": offsets,
        "population_l1_native_reversed": l1_arr,
        "population_l1_native_replicate": ctrl_arr,
        "branch_population": branch_population,
        "population_native_at_max_l1": max_pop_native,
        "population_reversed_at_max_l1": max_pop_reversed,
        "max_site_native": np.asarray(max_sites["native"], dtype=np.int64),
        "max_site_reversed": np.asarray(max_sites["reversed"], dtype=np.int64),
        "max_site_native_replicate": np.asarray(
            max_sites["native_replicate"], dtype=np.int64
        ),
        "ipr_native": np.asarray(ipr["native"], dtype=np.float64),
        "ipr_reversed": np.asarray(ipr["reversed"], dtype=np.float64),
        "ipr_native_replicate": np.asarray(
            ipr["native_replicate"], dtype=np.float64
        ),
        "max_l1_index": np.asarray([max_l1_index], dtype=np.int64),
    }
    return summary, arrays


def _false_reasons(checks: dict) -> list[str]:
    return [name for name, passed in checks.items() if not bool(passed)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--member-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    args = parser.parse_args()

    member_id = int(args.member_id)
    if member_id < 0 or member_id >= 32:
        raise ValueError("member-id must be in 0..31")

    initial, parameters, static = base._prepare_initial_state(SIZE, 1.0)
    member = peierls_velocity_member(
        parameters,
        member_id,
        ensemble_size=32,
        target_energy_eV=PREPARATION_ENERGY_EV,
        vx_energy_fraction=0.5,
        max_mode_index=4,
    )
    prepared = calibration._prepared_state(initial, member.velocity)
    field = UniformElectricField2D.from_millivolt_per_angstrom(
        FIELD_MV_PER_A,
        angle_radians=0.0,
        ax_angstrom=3.0,
        ay_angstrom=3.0,
    )

    free = calibration._field_free_control(
        prepared,
        parameters,
        dt_fs=DT_FS,
        sample_interval_fs=SAMPLE_INTERVAL_FS,
        energy_interval_fs=ENERGY_INTERVAL_FS,
        krylov_dimension=KRYLOV_DIMENSION,
    )
    driven = _driven_first_event_with_state(prepared, parameters, field)
    first_event = driven["first_persistent_event"]
    early = bool(
        first_event is not None
        and float(first_event["transition_start_time_fs"]) < EARLY_EVENT_CUTOFF_FS
    )
    first_is_x = bool(
        first_event is not None
        and first_event["is_nearest_neighbor"]
        and abs(int(first_event["dx_sites"])) == 1
        and int(first_event["dy_sites"]) == 0
    )
    pre_checks = {
        "static_relaxation_converged": bool(static["converged"]),
        "field_free_control_clean": bool(free["clean_no_persistent_relocation"]),
        "field_free_numerically_stable": bool(free["numerically_stable"]),
        "first_persistent_event_found": first_event is not None,
        "no_early_driven_event": not early,
        "first_persistent_event_is_nearest_neighbor_x": first_is_x,
        "driven_numerically_stable": bool(driven["numerically_stable"]),
        "driven_balance_le_2e-6_eV": bool(
            driven["maximum_energy_work_residual_eV"] <= PRE_EVENT_BALANCE_LIMIT_EV
        ),
        "pre_intervention_diagnostics_finite": bool(
            np.all(
                np.isfinite(
                    [
                        free["maximum_energy_drift_eV"],
                        free["maximum_norm_error"],
                        free["maximum_zero_mode_mean"],
                        driven["maximum_energy_work_residual_eV"],
                        driven["maximum_norm_error"],
                        driven["maximum_zero_mode_mean"],
                    ]
                )
            )
        ),
    }

    payload = {
        "scope": "IP2b one-member paired counterfactual production calculation",
        "member_id": member_id,
        "pair_id": int(member.pair_id),
        "pair_sign": int(member.pair_sign),
        "fixed_protocol": {
            "size": SIZE,
            "preparation_energy_eV": PREPARATION_ENERGY_EV,
            "field_mV_per_A": FIELD_MV_PER_A,
            "dt_fs": DT_FS,
            "search_time_fs": SEARCH_TIME_FS,
            "field_free_control_fs": FIELD_FREE_CONTROL_FS,
            "continuation_fs": CONTINUATION_FS,
            "sample_interval_fs": SAMPLE_INTERVAL_FS,
            "energy_interval_fs": ENERGY_INTERVAL_FS,
            "krylov_dimension": KRYLOV_DIMENSION,
        },
        "field_free": free,
        "driven_pre_intervention": {
            key: value
            for key, value in driven.items()
            if key != "selected_state"
        },
        "pre_intervention_checks": pre_checks,
        "intervention": None,
        "modal_covariates": None,
        "primary": None,
        "secondary_events": None,
        "branch_numerics": None,
        "valid_member": False,
        "rejection_reasons": [],
        "interpretation_guard": {
            "deterministic_design_not_random_sample": True,
            "native_replicate_is_numerical_control": True,
            "no_hopping_probability_rate_mobility_or_activation_energy": True,
        },
    }
    arrays = {
        "member_id": np.asarray([member_id], dtype=np.int64),
        "pair_id": np.asarray([member.pair_id], dtype=np.int64),
        "pair_sign": np.asarray([member.pair_sign], dtype=np.int64),
    }

    if not all(pre_checks.values()):
        payload["rejection_reasons"] = [
            f"pre_intervention:{name}" for name in _false_reasons(pre_checks)
        ]
    else:
        selected_state = driven["selected_state"]
        assert selected_state is not None and first_event is not None
        ts = float(first_event["accepted_time_fs"])
        held = HeldPeierlsPhase2D.from_driving_field(field, ts)
        phi_drive = np.asarray(field.phases(ts), dtype=np.float64)
        phi_held = np.asarray(held.phases(ts), dtype=np.float64)
        phase_error = float(np.max(np.abs(phi_drive - phi_held)))
        h_drive = build_sparse_field_hamiltonian(
            selected_state.lattice, parameters, field, ts
        )
        h_held = build_sparse_field_hamiltonian(
            selected_state.lattice, parameters, held, ts
        )
        hdiff = h_drive - h_held
        h_error = float(np.max(np.abs(hdiff.data))) if hdiff.nnz else 0.0

        native = ip1n._copy_state(selected_state)
        replica = ip1n._copy_state(selected_state)
        reversed_vx = reverse_non_special_vx_velocity(
            selected_state.velocity.vx, parameters
        )
        reversed_state = _copy_with_vx_velocity(selected_state, reversed_vx)

        fourier_errors = direction_reversal_fourier_errors(
            native.velocity.vx, reversed_state.velocity.vx, parameters
        )
        kinetic_error = abs(
            vx_velocity_kinetic_energy_eV(native.velocity.vx, parameters)
            - vx_velocity_kinetic_energy_eV(reversed_state.velocity.vx, parameters)
        )
        energy_native = field_dynamic_total_energy(
            native.lattice, native.velocity, parameters, native.electronic_state, held, ts
        ).total
        energy_reversed = field_dynamic_total_energy(
            reversed_state.lattice,
            reversed_state.velocity,
            parameters,
            reversed_state.electronic_state,
            held,
            ts,
        ).total
        matter_error = abs(energy_native - energy_reversed)

        equilibrium = frozen_surface_equilibrium(
            parameters, native.electronic_state, held
        )
        split_before = traveling_vx_energy_split(
            native.lattice,
            native.velocity,
            equilibrium,
            parameters,
            carrier_dx_sites=int(first_event["dx_sites"]),
        )
        split_after = traveling_vx_energy_split(
            reversed_state.lattice,
            reversed_state.velocity,
            equilibrium,
            parameters,
            carrier_dx_sites=int(first_event["dx_sites"]),
        )
        swap_error = max(
            abs(split_after["retrograde_eV"] - split_before["comoving_eV"]),
            abs(split_after["comoving_eV"] - split_before["retrograde_eV"]),
            abs(
                split_after["nondirectional_special_eV"]
                - split_before["nondirectional_special_eV"]
            ),
        )
        intervention_checks = {
            "phase_continuity": phase_error < 1.0e-14,
            "hamiltonian_continuity": h_error < 1.0e-13,
            "held_phase_rates_zero": max(
                abs(value) for value in held.phase_rates_per_fs()
            )
            < 1.0e-15,
            "coordinates_unchanged": bool(
                np.array_equal(native.lattice.u, reversed_state.lattice.u)
                and np.array_equal(native.lattice.vx, reversed_state.lattice.vx)
                and np.array_equal(native.lattice.vy, reversed_state.lattice.vy)
            ),
            "electronic_state_unchanged": bool(
                np.array_equal(
                    native.electronic_state, reversed_state.electronic_state
                )
            ),
            "u_vy_velocities_unchanged": bool(
                np.array_equal(native.velocity.u, reversed_state.velocity.u)
                and np.array_equal(native.velocity.vy, reversed_state.velocity.vy)
            ),
            "special_vx_sectors_unchanged": bool(
                fourier_errors[
                    "special_sector_unchanged_max_abs_A_per_fs"
                ]
                < 1.0e-15
            ),
            "non_special_vx_sectors_reversed": bool(
                fourier_errors[
                    "non_special_sector_sign_reversal_max_abs_A_per_fs"
                ]
                < 1.0e-15
            ),
            "vx_kinetic_energy_preserved": kinetic_error <= 1.0e-12,
            "matter_energy_preserved": matter_error <= 1.0e-12,
            "traveling_vx_energies_exchanged": swap_error <= 1.0e-12,
            "native_replicate_initial_state_exact": bool(
                np.array_equal(native.lattice.u, replica.lattice.u)
                and np.array_equal(native.lattice.vx, replica.lattice.vx)
                and np.array_equal(native.lattice.vy, replica.lattice.vy)
                and np.array_equal(native.velocity.u, replica.velocity.u)
                and np.array_equal(native.velocity.vx, replica.velocity.vx)
                and np.array_equal(native.velocity.vy, replica.velocity.vy)
                and np.array_equal(
                    native.electronic_state, replica.electronic_state
                )
            ),
        }
        payload["intervention"] = {
            "switch_time_fs": ts,
            "phase_continuity_error": phase_error,
            "hamiltonian_continuity_max_abs_eV": h_error,
            "fourier_errors": fourier_errors,
            "vx_kinetic_energy_error_eV": float(kinetic_error),
            "matter_energy_error_eV": float(matter_error),
            "traveling_energy_swap_error_eV": float(swap_error),
            "traveling_split_before": split_before,
            "traveling_split_after": split_after,
            "checks": intervention_checks,
        }

        if not all(intervention_checks.values()):
            payload["rejection_reasons"] = [
                f"intervention:{name}"
                for name in _false_reasons(intervention_checks)
            ]
        else:
            payload["modal_covariates"] = _branch_point_covariates(
                selected_state, first_event, held, parameters
            )
            branch, branch_arrays = _run_three_branches(
                native,
                reversed_state,
                replica,
                parameters,
                held,
                ts,
                first_event,
            )
            payload["primary"] = branch["primary"]
            payload["secondary_events"] = branch["secondary_events"]
            payload["branch_numerics"] = branch["branch_numerics"]
            payload["branch_elapsed_seconds"] = branch["elapsed_seconds"]
            arrays.update(branch_arrays)

            tolerance = size_scaled_energy_balance_tolerance_eV(parameters.n_sites)
            branch_checks = {}
            for key, values in branch["branch_numerics"].items():
                branch_checks[f"{key}_external_work_zero"] = (
                    abs(values["accumulated_external_work_eV"]) < 1.0e-12
                )
                branch_checks[f"{key}_energy_conservation"] = (
                    values["maximum_energy_drift_eV"] < tolerance
                    and extensive_energy_balance_passes(
                        values["maximum_energy_drift_eV"], parameters.n_sites
                    )
                )
                branch_checks[f"{key}_electronic_norm"] = (
                    values["maximum_norm_error"] < 1.0e-10
                )
                branch_checks[f"{key}_projected_zero_modes"] = (
                    values["maximum_zero_mode_mean"] < 1.0e-10
                )
            branch_checks["finite_primary_series"] = bool(
                np.all(
                    np.isfinite(
                        [
                            payload["primary"]["D_i"],
                            payload["primary"]["D_i_ctrl"],
                        ]
                    )
                )
            )
            payload["branch_numerical_checks"] = branch_checks
            if all(branch_checks.values()):
                payload["valid_member"] = True
            else:
                payload["rejection_reasons"] = [
                    f"branch:{name}" for name in _false_reasons(branch_checks)
                ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(args.arrays, **arrays)
    print(
        json.dumps(
            {
                "member_id": member_id,
                "pair_id": int(member.pair_id),
                "valid_member": payload["valid_member"],
                "rejection_reasons": payload["rejection_reasons"],
                "primary": payload["primary"],
            },
            indent=2,
        )
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.arrays}")


if __name__ == "__main__":
    main()
