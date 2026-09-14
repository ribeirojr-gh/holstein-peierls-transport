#!/usr/bin/env python3
"""IP1q mode-resolved decomposition of autonomous isotropic post-hop memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

import ip1l_interhop_wake_memory as base
import ip1n_gauge_continuous_field_release as ip1n
from holstein_peierls.dynamics.coupled import CoupledEhrenfestState
from holstein_peierls.dynamics.driven import (
    coupled_field_verlet_step,
    field_dynamic_total_energy,
    field_ehrenfest_force,
)
from holstein_peierls.dynamics.ehrenfest import LatticeVelocity, lattice_masses_fs
from holstein_peierls.dynamics.field import UniformElectricField2D
from holstein_peierls.dynamics.field_release import HeldPeierlsPhase2D
from holstein_peierls.dynamics.frozen_electronic_surface import (
    frozen_surface_total_energy,
    frozen_surface_verlet_step,
)
from holstein_peierls.dynamics.mode_memory import (
    axis_resolved_velocity_spectrum,
    frozen_surface_equilibrium,
    harmonic_wave_numbers,
    integrated_velocity_spectrum,
    modal_energy_arrays,
    q_participation_ratio,
    real_space_excitation_energies,
    ridge_diagnostics,
    traveling_vx_energy_split,
)
from holstein_peierls.dynamics.numerical_validation import (
    extensive_energy_balance_passes,
    size_scaled_energy_balance_tolerance_eV,
)
from holstein_peierls.dynamics.phonon_recurrence import intermolecular_recurrence_scales
from holstein_peierls.dynamics.thermal import zero_mode_means


LATTICE_SPACING_A = 3.0
RIDGE_MINIMUM_Q_ENERGY_FRACTION = 0.005


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


def _max_force(force) -> float:
    return float(
        max(
            np.max(np.abs(force.u)),
            np.max(np.abs(force.vx)),
            np.max(np.abs(force.vy)),
        )
    )


def _paired_abs_q_summary(
    energy_by_q: np.ndarray,
    q_axis: np.ndarray,
    *,
    omega_by_q: np.ndarray,
    group_speed_by_q_sites_per_ps: np.ndarray,
) -> dict:
    energy = np.asarray(energy_by_q, dtype=np.float64)
    q = np.asarray(q_axis, dtype=np.float64)
    omega = np.asarray(omega_by_q, dtype=np.float64)
    vg = np.asarray(group_speed_by_q_sites_per_ps, dtype=np.float64)
    n = energy.size
    sectors = []
    for k in range(n // 2 + 1):
        if k == 0 or (n % 2 == 0 and k == n // 2):
            indices = [k]
        else:
            indices = [k, (-k) % n]
        e = float(np.sum(energy[indices]))
        qabs = abs(float(q[k]))
        sectors.append(
            {
                "absolute_q_rad_per_site": qabs,
                "energy_eV": e,
                "energy_fraction": 0.0,
                "wavelength_sites": None if qabs <= 1.0e-14 else float(2.0 * np.pi / qabs),
                "omega_per_fs": float(omega[k]),
                "group_speed_sites_per_ps": abs(float(vg[k])),
            }
        )
    total = float(sum(item["energy_eV"] for item in sectors))
    if total > 0.0:
        for item in sectors:
            item["energy_fraction"] = float(item["energy_eV"] / total)
    top = sorted(sectors, key=lambda item: item["energy_eV"], reverse=True)[:5]
    low = sum(
        item["energy_eV"]
        for item in sectors
        if item["absolute_q_rad_per_site"] > 1.0e-14
        and item["absolute_q_rad_per_site"] <= 0.25 * np.pi + 1.0e-14
    )
    return {
        "total_energy_eV": total,
        "participation_ratio_raw_fft_sectors": q_participation_ratio(energy),
        "lowest_nonzero_quarter_bz_energy_fraction": 0.0 if total <= 0.0 else float(low / total),
        "top_five_absolute_q_sectors": top,
        "all_absolute_q_sectors": sectors,
    }


def _traveling_vx_q_summary(
    lattice,
    velocity,
    equilibrium,
    parameters,
    *,
    carrier_dx_sites: int,
) -> dict:
    mass_v = lattice_masses_fs(parameters)[1]
    qfield = np.fft.fft2(lattice.vx - equilibrium.vx, norm="ortho")
    vfield = np.fft.fft2(velocity.vx, norm="ortho")
    qx = harmonic_wave_numbers(parameters.nx)
    c = float(np.sqrt(parameters.k2 / mass_v))
    omega = 2.0 * c * np.abs(np.sin(0.5 * qx))
    positive = (qx > 1.0e-14) & (qx < np.pi - 1.0e-14)
    records = []
    retro_total = 0.0
    comoving_total = 0.0
    weighted_retro_speed = 0.0
    for ix in np.flatnonzero(positive):
        w = float(omega[ix])
        q = qfield[:, ix]
        v = vfield[:, ix]
        a_plus = 0.5 * (q + 1.0j * v / w)
        a_minus = 0.5 * (q - 1.0j * v / w)
        plus = float(np.sum(2.0 * mass_v * w * w * np.abs(a_plus) ** 2))
        minus = float(np.sum(2.0 * mass_v * w * w * np.abs(a_minus) ** 2))
        if int(carrier_dx_sites) < 0:
            retro = plus
            comoving = minus
        else:
            retro = minus
            comoving = plus
        speed = float(c * np.cos(0.5 * qx[ix]) * 1000.0)
        retro_total += retro
        comoving_total += comoving
        weighted_retro_speed += retro * abs(speed)
        records.append(
            {
                "positive_qx_rad_per_site": float(qx[ix]),
                "omega_per_fs": w,
                "absolute_group_speed_sites_per_ps": abs(speed),
                "plus_x_energy_eV": plus,
                "minus_x_energy_eV": minus,
                "retrograde_energy_eV": retro,
                "comoving_energy_eV": comoving,
            }
        )
    return {
        "retrograde_energy_eV": float(retro_total),
        "comoving_energy_eV": float(comoving_total),
        "retrograde_fraction_of_direction_resolved": (
            0.0 if retro_total + comoving_total <= 1.0e-30
            else float(retro_total / (retro_total + comoving_total))
        ),
        "retrograde_energy_weighted_group_speed_sites_per_ps": (
            0.0 if retro_total <= 1.0e-30 else float(weighted_retro_speed / retro_total)
        ),
        "q_records": records,
    }


def _polarization_classification(u_fraction: float, peierls_fraction: float) -> str:
    if peierls_fraction >= 2.0 / 3.0:
        return "Peierls dominated"
    if u_fraction >= 2.0 / 3.0:
        return "Holstein dominated"
    return "mixed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=40)
    parser.add_argument("--field-mv-per-A", dest="field_mV_per_A", type=float, default=10.0)
    parser.add_argument("--dt-fs", type=float, default=0.2)
    parser.add_argument("--search-time-fs", type=float, default=4000.0)
    parser.add_argument("--continuation-fs", type=float, default=10000.0)
    parser.add_argument("--event-sample-interval-fs", type=float, default=2.0)
    parser.add_argument("--state-sample-interval-fs", type=float, default=10.0)
    parser.add_argument("--energy-sample-interval-fs", type=float, default=10.0)
    parser.add_argument("--krylov-dimension", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    args = parser.parse_args()

    initial, parameters, static = base._prepare_initial_state(args.size, 1.0)
    field = UniformElectricField2D.from_millivolt_per_angstrom(
        args.field_mV_per_A,
        angle_radians=0.0,
        ax_angstrom=LATTICE_SPACING_A,
        ay_angstrom=LATTICE_SPACING_A,
    )
    search_steps = base._integer_stride(args.search_time_fs, args.dt_fs, "search_time_fs")
    event_stride = base._integer_stride(
        args.event_sample_interval_fs, args.dt_fs, "event_sample_interval_fs"
    )
    energy_stride = base._integer_stride(
        args.energy_sample_interval_fs, args.dt_fs, "energy_sample_interval_fs"
    )
    tracker = ip1n._branch_tracker(
        parameters.nx, parameters.ny, args.event_sample_interval_fs
    )
    current = ip1n._copy_state(initial)
    populations = np.abs(current.electronic_state) ** 2
    tracker.update(populations / float(np.sum(populations)), 0.0)
    drive_initial_energy = field_dynamic_total_energy(
        current.lattice,
        current.velocity,
        parameters,
        current.electronic_state,
        field,
        0.0,
    ).total
    drive_work = 0.0
    max_drive_balance = 0.0
    pre_events = []
    selected_event = None
    selected_state = None
    start = perf_counter()
    for step in range(search_steps):
        time_fs = step * args.dt_fs
        current, work, _, _ = coupled_field_verlet_step(
            current,
            parameters,
            field,
            time_fs,
            args.dt_fs,
            electronic_method="cfm4_lanczos",
            krylov_dimension=args.krylov_dimension,
        )
        new_time = time_fs + args.dt_fs
        drive_work += float(work)
        if (step + 1) % energy_stride == 0:
            energy = field_dynamic_total_energy(
                current.lattice,
                current.velocity,
                parameters,
                current.electronic_state,
                field,
                new_time,
            ).total
            max_drive_balance = max(
                max_drive_balance, abs((energy - drive_initial_energy) - drive_work)
            )
        if (step + 1) % event_stride == 0:
            populations = np.abs(current.electronic_state) ** 2
            populations /= float(np.sum(populations))
            event = tracker.update(populations, new_time)
            if event is not None:
                record = _event_record(event)
                pre_events.append(record)
                if (
                    selected_event is None
                    and record["is_nearest_neighbor"]
                    and abs(record["dx_sites"]) == 1
                    and record["dy_sites"] == 0
                ):
                    selected_event = record
                    selected_state = ip1n._copy_state(current)
                    break
    search_elapsed = perf_counter() - start

    analysis = None
    serialized: dict[str, np.ndarray] = {}
    tolerance = size_scaled_energy_balance_tolerance_eV(parameters.n_sites)
    numerical_checks = {
        "static_relaxation_converged": bool(static["converged"]),
        "natural_x_event_found": bool(selected_event is not None and selected_state is not None),
        "pre_switch_energy_work_balance": bool(
            extensive_energy_balance_passes(max_drive_balance, parameters.n_sites)
        ),
    }

    if selected_event is not None and selected_state is not None:
        ts = float(selected_event["accepted_time_fs"])
        held = HeldPeierlsPhase2D.from_driving_field(field, ts)
        fixed_psi = np.asarray(selected_state.electronic_state, dtype=np.complex128).copy()
        equilibrium = frozen_surface_equilibrium(parameters, fixed_psi, held)
        equilibrium_force = field_ehrenfest_force(
            equilibrium, parameters, fixed_psi, held, ts
        )
        equilibrium_force_residual = _max_force(equilibrium_force)
        equilibrium_state = CoupledEhrenfestState(
            equilibrium.copy(),
            LatticeVelocity.zeros(parameters.ny, parameters.nx),
            fixed_psi.copy(),
        )
        equilibrium_total_energy = frozen_surface_total_energy(
            equilibrium_state, parameters, fixed_psi, held, ts
        )

        branch_steps = base._integer_stride(
            args.continuation_fs, args.dt_fs, "continuation_fs"
        )
        state_stride = base._integer_stride(
            args.state_sample_interval_fs, args.dt_fs, "state_sample_interval_fs"
        )
        current = ip1n._copy_state(selected_state)
        branch_initial_total = frozen_surface_total_energy(
            current, parameters, fixed_psi, held, ts
        )
        max_energy_drift = 0.0
        max_zero_mode = 0.0
        max_parseval_error = 0.0
        max_surface_excitation_error = 0.0

        times = []
        u_frames = []
        vx_frames = []
        vy_frames = []
        vu_frames = []
        vvx_frames = []
        vvy_frames = []
        eu_series = []
        evx_series = []
        evy_series = []
        split_retro_series = []
        split_comoving_series = []
        zero_vx_series = []
        zero_vy_series = []

        modal_initial = None
        split_initial = None

        def record_state(time_fs: float) -> None:
            nonlocal modal_initial, split_initial, max_parseval_error, max_surface_excitation_error
            times.append(float(time_fs))
            u_frames.append(np.asarray(current.lattice.u, dtype=np.float32).copy())
            vx_frames.append(np.asarray(current.lattice.vx, dtype=np.float32).copy())
            vy_frames.append(np.asarray(current.lattice.vy, dtype=np.float32).copy())
            vu_frames.append(np.asarray(current.velocity.u, dtype=np.float32).copy())
            vvx_frames.append(np.asarray(current.velocity.vx, dtype=np.float32).copy())
            vvy_frames.append(np.asarray(current.velocity.vy, dtype=np.float32).copy())
            real = real_space_excitation_energies(
                current.lattice, current.velocity, equilibrium, parameters
            )
            modal = modal_energy_arrays(
                current.lattice, current.velocity, equilibrium, parameters
            )
            sums = modal.sums
            max_parseval_error = max(
                max_parseval_error,
                max(abs(real[key] - sums[key]) for key in ("u_eV", "vx_eV", "vy_eV", "total_eV")),
            )
            surface_excitation = frozen_surface_total_energy(
                current, parameters, fixed_psi, held, time_fs
            ) - equilibrium_total_energy
            max_surface_excitation_error = max(
                max_surface_excitation_error, abs(surface_excitation - real["total_eV"])
            )
            eu_series.append(sums["u_eV"])
            evx_series.append(sums["vx_eV"])
            evy_series.append(sums["vy_eV"])
            split = traveling_vx_energy_split(
                current.lattice,
                current.velocity,
                equilibrium,
                parameters,
                carrier_dx_sites=int(selected_event["dx_sites"]),
            )
            split_retro_series.append(split["retrograde_eV"])
            split_comoving_series.append(split["comoving_eV"])
            zero_vx_series.append(float(np.sum(modal.vx[:, 0])))
            zero_vy_series.append(float(np.sum(modal.vy[0, :])))
            if modal_initial is None:
                modal_initial = modal
                split_initial = split

        record_state(ts)
        branch_start = perf_counter()
        for step in range(branch_steps):
            time_fs = ts + step * args.dt_fs
            current = frozen_surface_verlet_step(
                current,
                parameters,
                fixed_psi,
                held,
                time_fs,
                args.dt_fs,
            )
            new_time = time_fs + args.dt_fs
            max_zero_mode = max(
                max_zero_mode,
                max(abs(value) for value in zero_mode_means(current).values()),
            )
            if (step + 1) % energy_stride == 0 or step + 1 == branch_steps:
                energy = frozen_surface_total_energy(
                    current, parameters, fixed_psi, held, new_time
                )
                max_energy_drift = max(max_energy_drift, abs(energy - branch_initial_total))
            if (step + 1) % state_stride == 0:
                record_state(new_time)
        branch_elapsed = perf_counter() - branch_start

        times_arr = np.asarray(times, dtype=np.float64)
        u_frames_arr = np.stack(u_frames, axis=0)
        vx_frames_arr = np.stack(vx_frames, axis=0)
        vy_frames_arr = np.stack(vy_frames, axis=0)
        vu_frames_arr = np.stack(vu_frames, axis=0)
        vvx_frames_arr = np.stack(vvx_frames, axis=0)
        vvy_frames_arr = np.stack(vvy_frames, axis=0)
        eu = np.asarray(eu_series, dtype=np.float64)
        evx = np.asarray(evx_series, dtype=np.float64)
        evy = np.asarray(evy_series, dtype=np.float64)
        total_modal = eu + evx + evy

        component_drift = {
            "u_eV": float(np.max(np.abs(eu - eu[0]))),
            "vx_eV": float(np.max(np.abs(evx - evx[0]))),
            "vy_eV": float(np.max(np.abs(evy - evy[0]))),
            "total_eV": float(np.max(np.abs(total_modal - total_modal[0]))),
        }
        split_retro = np.asarray(split_retro_series, dtype=np.float64)
        split_comoving = np.asarray(split_comoving_series, dtype=np.float64)
        split_drift = {
            "retrograde_eV": float(np.max(np.abs(split_retro - split_retro[0]))),
            "comoving_eV": float(np.max(np.abs(split_comoving - split_comoving[0]))),
        }

        assert modal_initial is not None and split_initial is not None
        initial_sums = modal_initial.sums
        total0 = initial_sums["total_eV"]
        u_fraction = float(initial_sums["u_eV"] / total0)
        vx_fraction = float(initial_sums["vx_eV"] / total0)
        vy_fraction = float(initial_sums["vy_eV"] / total0)
        peierls_fraction = vx_fraction + vy_fraction

        qx = harmonic_wave_numbers(parameters.nx)
        qy = harmonic_wave_numbers(parameters.ny)
        mass_u, mass_v = lattice_masses_fs(parameters)
        c_v = float(np.sqrt(parameters.k2 / mass_v))
        omega_u = float(np.sqrt(parameters.k1 / mass_u))
        omega_vx = 2.0 * c_v * np.abs(np.sin(0.5 * qx))
        omega_vy = 2.0 * c_v * np.abs(np.sin(0.5 * qy))
        vg_x = c_v * np.sign(qx) * np.cos(0.5 * qx) * 1000.0
        vg_y = c_v * np.sign(qy) * np.cos(0.5 * qy) * 1000.0
        vx_q_energy = np.sum(modal_initial.vx, axis=0)
        vy_q_energy = np.sum(modal_initial.vy, axis=1)
        vx_q_summary = _paired_abs_q_summary(
            vx_q_energy,
            qx,
            omega_by_q=omega_vx,
            group_speed_by_q_sites_per_ps=vg_x,
        )
        vy_q_summary = _paired_abs_q_summary(
            vy_q_energy,
            qy,
            omega_by_q=omega_vy,
            group_speed_by_q_sites_per_ps=vg_y,
        )
        traveling_q = _traveling_vx_q_summary(
            selected_state.lattice,
            selected_state.velocity,
            equilibrium,
            parameters,
            carrier_dx_sites=int(selected_event["dx_sites"]),
        )

        qx_spec, omega_axis, vx_qomega = axis_resolved_velocity_spectrum(
            vvx_frames_arr,
            args.state_sample_interval_fs,
            dispersive_axis="x",
        )
        qy_spec, omega_axis_y, vy_qomega = axis_resolved_velocity_spectrum(
            vvy_frames_arr,
            args.state_sample_interval_fs,
            dispersive_axis="y",
        )
        omega_u_axis, u_power = integrated_velocity_spectrum(
            vu_frames_arr, args.state_sample_interval_fs
        )
        vx_ridge = ridge_diagnostics(
            vx_qomega,
            omega_axis,
            omega_vx,
            vx_q_energy,
            minimum_energy_fraction=RIDGE_MINIMUM_Q_ENERGY_FRACTION,
        )
        vy_ridge = ridge_diagnostics(
            vy_qomega,
            omega_axis_y,
            omega_vy,
            vy_q_energy,
            minimum_energy_fraction=RIDGE_MINIMUM_Q_ENERGY_FRACTION,
        )
        u_peak_index = 1 + int(np.argmax(u_power[1:]))
        u_peak_omega = float(omega_u_axis[u_peak_index])
        u_bin_width = float(omega_u_axis[1] - omega_u_axis[0])
        u_spectral = {
            "analytic_omega_per_fs": omega_u,
            "peak_omega_per_fs": u_peak_omega,
            "absolute_error_per_fs": abs(u_peak_omega - omega_u),
            "frequency_bin_width_per_fs": u_bin_width,
            "within_one_frequency_bin": bool(abs(u_peak_omega - omega_u) <= u_bin_width),
        }

        electronic_unchanged = bool(
            np.array_equal(np.asarray(current.electronic_state), fixed_psi)
        )
        max_zero_modal_energy = float(
            max(np.max(zero_vx_series), np.max(zero_vy_series))
        )
        analysis = {
            "event": selected_event,
            "switch_time_fs": ts,
            "held_phase_rates_per_fs": list(held.phase_rates_per_fs()),
            "equilibrium_force_residual": equilibrium_force_residual,
            "equilibrium_total_energy_eV": float(equilibrium_total_energy),
            "branch_initial_total_energy_eV": float(branch_initial_total),
            "branch_elapsed_seconds": float(branch_elapsed),
            "maximum_frozen_surface_energy_drift_eV": float(max_energy_drift),
            "maximum_parseval_modal_energy_error_eV": float(max_parseval_error),
            "maximum_surface_excitation_closure_error_eV": float(max_surface_excitation_error),
            "maximum_zero_mode_modal_energy_eV": max_zero_modal_energy,
            "electronic_state_bitwise_unchanged": electronic_unchanged,
            "polarization_energy_at_switch_eV": initial_sums,
            "polarization_fractions_at_switch": {
                "Holstein_u": u_fraction,
                "Peierls_vx": vx_fraction,
                "Peierls_vy": vy_fraction,
                "Peierls_total": peierls_fraction,
            },
            "polarization_classification": _polarization_classification(
                u_fraction, peierls_fraction
            ),
            "polarization_energy_max_drift_eV": component_drift,
            "traveling_vx_split_at_switch": split_initial,
            "traveling_vx_q_summary_at_switch": traveling_q,
            "traveling_vx_energy_max_drift_eV": split_drift,
            "vx_q_summary": vx_q_summary,
            "vy_q_summary": vy_q_summary,
            "u_spectral_summary": u_spectral,
            "vx_qomega_ridge": vx_ridge,
            "vy_qomega_ridge": vy_ridge,
            "state_sample_count": int(times_arr.size),
            "state_sample_interval_fs": float(args.state_sample_interval_fs),
            "spectral_duration_fs": float(times_arr[-1] - times_arr[0]),
        }

        numerical_checks.update(
            {
                "selected_event_is_first_persistent_event": bool(
                    pre_events and pre_events[0] == selected_event
                ),
                "held_phase_rates_zero": bool(
                    max(abs(value) for value in held.phase_rates_per_fs()) < 1.0e-15
                ),
                "electronic_state_bitwise_unchanged": electronic_unchanged,
                "frozen_surface_energy_conservation": bool(
                    extensive_energy_balance_passes(max_energy_drift, parameters.n_sites)
                ),
                "frozen_surface_equilibrium_force": bool(
                    equilibrium_force_residual < 1.0e-12
                ),
                "parseval_modal_energy_closure": bool(max_parseval_error < 1.0e-9),
                "surface_excitation_energy_closure": bool(
                    max_surface_excitation_error < 1.0e-9
                ),
                "polarization_modal_energies_conserved": bool(
                    max(component_drift.values()) < 1.0e-8
                ),
                "traveling_vx_energies_conserved": bool(
                    max(split_drift.values()) < 1.0e-8
                ),
                "zero_mode_modal_energy_negligible": bool(
                    max_zero_modal_energy < 1.0e-10
                ),
                "finite_modal_and_spectral_arrays": bool(
                    np.all(np.isfinite(vx_qomega))
                    and np.all(np.isfinite(vy_qomega))
                    and np.all(np.isfinite(u_power))
                    and np.all(np.isfinite(total_modal))
                ),
                "complete_requested_frozen_steps": bool(
                    np.isclose(times_arr[-1] - ts, args.continuation_fs, atol=1.0e-9)
                ),
            }
        )

        serialized = {
            "times_fs": times_arr,
            "u_A": u_frames_arr,
            "vx_A": vx_frames_arr,
            "vy_A": vy_frames_arr,
            "velocity_u_A_per_fs": vu_frames_arr,
            "velocity_vx_A_per_fs": vvx_frames_arr,
            "velocity_vy_A_per_fs": vvy_frames_arr,
            "equilibrium_u_A": np.asarray(equilibrium.u, dtype=np.float64),
            "equilibrium_vx_A": np.asarray(equilibrium.vx, dtype=np.float64),
            "equilibrium_vy_A": np.asarray(equilibrium.vy, dtype=np.float64),
            "modal_energy_u_eV": eu,
            "modal_energy_vx_eV": evx,
            "modal_energy_vy_eV": evy,
            "traveling_retrograde_vx_energy_eV": split_retro,
            "traveling_comoving_vx_energy_eV": split_comoving,
            "qx_rad_per_site": qx_spec,
            "qy_rad_per_site": qy_spec,
            "omega_per_fs": omega_axis,
            "vx_qomega_power": vx_qomega,
            "vy_qomega_power": vy_qomega,
            "u_omega_per_fs": omega_u_axis,
            "u_velocity_power": u_power,
            "initial_modal_energy_u_qy_qx_eV": modal_initial.u,
            "initial_modal_energy_vx_qy_qx_eV": modal_initial.vx,
            "initial_modal_energy_vy_qy_qx_eV": modal_initial.vy,
        }

    numerical_pass = bool(all(numerical_checks.values()))
    payload = {
        "scope": "IP1q mode-resolved decomposition of autonomous isotropic post-hop memory on the frozen-electronic surface",
        "size": int(args.size),
        "anisotropy_ratio": 1.0,
        "initial_field_mV_per_A": float(args.field_mV_per_A),
        "temperature_K": 0.0,
        "thermostat": None,
        "IDC": None,
        "dt_fs": float(args.dt_fs),
        "continuation_fs": float(args.continuation_fs),
        "event_sample_interval_fs": float(args.event_sample_interval_fs),
        "state_sample_interval_fs": float(args.state_sample_interval_fs),
        "search_elapsed_seconds": float(search_elapsed),
        "static": static,
        "pre_switch_persistent_events": pre_events,
        "analysis": analysis,
        "energy_balance_tolerance_eV": float(tolerance),
        "numerical_checks": numerical_checks,
        "numerical_pass": numerical_pass,
        "interpretation_guard": {
            "fixed_electronic_force_shifts_equilibrium_but_not_harmonic_hessian": True,
            "mode_group_velocity_is_harmonic_model_diagnostic_not_mobility": True,
            "held_boundary_twist_is_static_and_enters_the_fixed_equilibrium": True,
            "undamped_continuation_does_not_define_material_phonon_lifetime": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if serialized:
        np.savez_compressed(args.trajectory, **serialized)

    lines = [
        "# IP1q mode-resolved autonomous memory",
        "",
        f"Cell {args.size}x{args.size}; isotropic; initial E=+{args.field_mV_per_A:g} mV/A; frozen continuation={args.continuation_fs/1000:g} ps.",
        "",
    ]
    if analysis is None:
        lines.append("No qualifying natural x event was found before the search cutoff.")
    else:
        fractions = analysis["polarization_fractions_at_switch"]
        split = analysis["traveling_vx_split_at_switch"]
        tq = analysis["traveling_vx_q_summary_at_switch"]
        lines.extend(
            [
                f"Event {analysis['event']['source_site']}->{analysis['event']['target_site']} ({analysis['event']['direction']}): start={analysis['event']['transition_start_time_fs']:.1f} fs; switch={analysis['switch_time_fs']:.1f} fs.",
                f"Frozen equilibrium force residual: {analysis['equilibrium_force_residual']:.3e}.",
                f"Maximum total frozen energy drift: {analysis['maximum_frozen_surface_energy_drift_eV']:.6e} eV.",
                f"Maximum real/modal Parseval mismatch: {analysis['maximum_parseval_modal_energy_error_eV']:.6e} eV.",
                "",
                "## Polarization content at switch",
                "",
                f"- Holstein u fraction: {fractions['Holstein_u']:.6f};",
                f"- Peierls vx fraction: {fractions['Peierls_vx']:.6f};",
                f"- Peierls vy fraction: {fractions['Peierls_vy']:.6f};",
                f"- total Peierls fraction: {fractions['Peierls_total']:.6f};",
                f"- classification: **{analysis['polarization_classification']}**.",
                "",
                "## Direction-resolved Peierls-x content",
                "",
                f"- retrograde fraction of direction-resolved vx energy: {split['retrograde_fraction_of_direction_resolved']:.6f};",
                f"- direction-resolved fraction of total vx energy: {split['direction_resolved_fraction_of_vx']:.6f};",
                f"- retrograde energy-weighted |group velocity|: {tq['retrograde_energy_weighted_group_speed_sites_per_ps']:.6f} sites/ps;",
                "",
                "## q-space summaries",
                "",
                f"- vx q participation ratio: {analysis['vx_q_summary']['participation_ratio_raw_fft_sectors']:.3f};",
                f"- vx low-|q| quarter-BZ fraction: {analysis['vx_q_summary']['lowest_nonzero_quarter_bz_energy_fraction']:.6f};",
                f"- vy q participation ratio: {analysis['vy_q_summary']['participation_ratio_raw_fft_sectors']:.3f};",
                f"- vy low-|q| quarter-BZ fraction: {analysis['vy_q_summary']['lowest_nonzero_quarter_bz_energy_fraction']:.6f};",
                "",
                "## q-omega validation",
                "",
                f"- u analytic omega={analysis['u_spectral_summary']['analytic_omega_per_fs']:.6e} fs^-1; peak={analysis['u_spectral_summary']['peak_omega_per_fs']:.6e}; within one bin={'YES' if analysis['u_spectral_summary']['within_one_frequency_bin'] else 'NO'};",
                f"- vx occupied-sector weighted ridge error={analysis['vx_qomega_ridge']['energy_weighted_mean_absolute_error_per_fs']}; bin={analysis['vx_qomega_ridge']['frequency_bin_width_per_fs']:.6e}; within one bin={'YES' if analysis['vx_qomega_ridge']['within_one_frequency_bin'] else 'NO'};",
                f"- vy occupied-sector weighted ridge error={analysis['vy_qomega_ridge']['energy_weighted_mean_absolute_error_per_fs']}; bin={analysis['vy_qomega_ridge']['frequency_bin_width_per_fs']:.6e}; within one bin={'YES' if analysis['vy_qomega_ridge']['within_one_frequency_bin'] else 'NO'};",
            ]
        )
    lines.extend(["", "## Numerical gates", ""])
    for name, value in numerical_checks.items():
        lines.append(f"- {name}: {'PASS' if value else 'FAIL'}")
    lines.extend(["", f"Numerical status: {'PASS' if numerical_pass else 'FAIL'}"])
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"\nWrote {args.output}")
    print(f"Wrote {args.markdown}")
    if serialized:
        print(f"Wrote {args.trajectory}")
    if not numerical_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
