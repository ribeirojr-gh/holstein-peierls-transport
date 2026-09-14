#!/usr/bin/env python3
"""IP1g controlled one-site charge-relocation lattice-radiation diagnostic.

A relaxed zero-temperature polaron is prepared at the lattice center. At t=0
only the electronic wavefunction is translated by one site toward +x; the
relaxed lattice and zero lattice velocity are unchanged. Deterministic
zero-field Ehrenfest dynamics then follow the re-dressing radiation.

This is an imposed charge-relocation quench, not a natural hopping event. It
cannot define a hopping rate, activation energy, mobility or material phonon
lifetime.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from holstein_peierls.dynamics.coupled import CoupledEhrenfestState, coupled_verlet_step
from holstein_peierls.dynamics.ehrenfest import dynamic_total_energy
from holstein_peierls.dynamics.hopping_observables import site_xy
from holstein_peierls.dynamics.numerical_validation import (
    extensive_energy_balance_passes,
    size_scaled_energy_balance_tolerance_eV,
)
from holstein_peierls.dynamics.phonon_wake import (
    IntermolecularEnergyFlux,
    backward_excess_centroid_sites,
    event_aligned_coordinates,
    fit_group_velocity_sites_per_fs,
    intermolecular_energy_flux,
    local_lattice_energy_density,
    longitudinal_flux_field,
    longitudinal_profile,
    wake_window_metrics,
)
from holstein_peierls.dynamics.single_relocation import translate_electronic_state
from holstein_peierls.dynamics.thermal import zero_mode_means

from ip1c_event_conditioned_dressed_hopping import _integer_stride, _prepare_initial_state  # noqa: E402


TRANSVERSE_HALF_WIDTH_SITES = 5
CORE_HALF_WIDTH_SITES = 2
GROUP_FIT_START_FS = 200.0
GROUP_FIT_END_FS = 4000.0
WINDOWS_FS = ((0.0, 500.0), (500.0, 1000.0), (1000.0, 2000.0), (2000.0, 5000.0))


def _profile(values: np.ndarray, coordinates) -> tuple[np.ndarray, np.ndarray]:
    return longitudinal_profile(
        values,
        coordinates,
        transverse_half_width_sites=TRANSVERSE_HALF_WIDTH_SITES,
    )


def _condition(
    size: int,
    ratio: float,
    dt_fs: float,
    final_time_fs: float,
    sample_interval_fs: float,
    krylov_dimension: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    relaxed, parameters, static = _prepare_initial_state(size, ratio)
    source = int(np.argmax(np.abs(relaxed.electronic_state) ** 2))
    sx, sy = site_xy(source, size, size)
    target = sy * size + ((sx + 1) % size)
    psi_shifted = translate_electronic_state(
        relaxed.electronic_state, size, size, dx_sites=1
    )
    current = CoupledEhrenfestState(
        relaxed.lattice.copy(), relaxed.velocity.copy(), psi_shifted
    )
    coordinates = event_aligned_coordinates(source, target, size, size)

    steps = _integer_stride(final_time_fs, dt_fs, "final_time_fs")
    stride = _integer_stride(sample_interval_fs, dt_fs, "sample_interval_fs")
    n_samples = steps // stride + 1
    times = np.empty(n_samples)
    source_pop = np.empty(n_samples)
    target_pop = np.empty(n_samples)
    total_energy = np.empty(n_samples)
    norm_error = np.empty(n_samples)
    zero_mode_mag = np.empty(n_samples)
    u_profiles: list[np.ndarray] = []
    long_profiles: list[np.ndarray] = []
    trans_profiles: list[np.ndarray] = []
    flux_profiles: list[np.ndarray] = []
    s_axis: np.ndarray | None = None
    initial_norm = float(np.vdot(current.electronic_state, current.electronic_state).real)

    def record(index: int, time_fs: float, state: CoupledEhrenfestState) -> None:
        nonlocal s_axis
        density = local_lattice_energy_density(state.lattice, state.velocity, parameters)
        flux = intermolecular_energy_flux(state.lattice, state.velocity, parameters)
        s, u = _profile(density.u, coordinates)
        _, ex = _profile(density.vx, coordinates)
        _, ey = _profile(density.vy, coordinates)
        projected = longitudinal_flux_field(
            IntermolecularEnergyFlux(flux.jx, flux.jy), coordinates
        )
        _, j = _profile(projected, coordinates)
        if s_axis is None:
            s_axis = np.asarray(s, dtype=np.int64)
        u_profiles.append(np.asarray(u, dtype=np.float64))
        long_profiles.append(np.asarray(ex, dtype=np.float64))
        trans_profiles.append(np.asarray(ey, dtype=np.float64))
        flux_profiles.append(np.asarray(j, dtype=np.float64))
        psi = np.asarray(state.electronic_state, dtype=np.complex128)
        norm = float(np.vdot(psi, psi).real)
        pop = np.abs(psi) ** 2 / norm
        means = zero_mode_means(state)
        times[index] = time_fs
        source_pop[index] = float(pop[source])
        target_pop[index] = float(pop[target])
        total_energy[index] = float(
            dynamic_total_energy(state.lattice, state.velocity, parameters, psi).total
        )
        norm_error[index] = abs(norm - initial_norm)
        zero_mode_mag[index] = max(abs(v) for v in means.values())

    start = perf_counter()
    record(0, 0.0, current)
    sample_index = 1
    h_eval = 0
    h_apply = 0
    for step in range(1, steps + 1):
        current, evaluations, applications = coupled_verlet_step(
            current,
            parameters,
            dt_fs,
            electronic_method="cfm4_lanczos",
            krylov_dimension=krylov_dimension,
        )
        h_eval += evaluations
        h_apply += applications
        if step % stride == 0:
            record(sample_index, step * dt_fs, current)
            sample_index += 1
    elapsed = perf_counter() - start
    if sample_index != n_samples or s_axis is None:
        raise RuntimeError("unexpected IP1g sample count")

    u = np.asarray(u_profiles)
    long_e = np.asarray(long_profiles)
    trans_e = np.asarray(trans_profiles)
    flux = np.asarray(flux_profiles)
    inter = long_e + trans_e
    u_ex = u - u[0]
    long_ex = long_e - long_e[0]
    trans_ex = trans_e - trans_e[0]
    inter_ex = inter - inter[0]
    flux_ex = flux - flux[0]

    windows: dict[str, dict] = {}
    for start_fs, end_fs in WINDOWS_FS:
        mask = (times >= start_fs) & (times < end_fs + 1.0e-9)
        inter_mean = np.mean(inter_ex[mask], axis=0)
        flux_mean = np.mean(flux_ex[mask], axis=0)
        metrics = wake_window_metrics(
            inter_mean, flux_mean, s_axis, core_half_width_sites=CORE_HALF_WIDTH_SITES
        )
        outer = np.abs(s_axis) > CORE_HALF_WIDTH_SITES
        long_pos = float(np.sum(np.clip(np.mean(long_ex[mask], axis=0)[outer], 0.0, None)))
        trans_pos = float(np.sum(np.clip(np.mean(trans_ex[mask], axis=0)[outer], 0.0, None)))
        u_pos = float(np.sum(np.clip(np.mean(u_ex[mask], axis=0)[outer], 0.0, None)))
        denom = long_pos + trans_pos
        windows[f"{int(start_fs)}:{int(end_fs)}fs"] = {
            **asdict(metrics),
            "longitudinal_positive_excess_eV": long_pos,
            "transverse_positive_excess_eV": trans_pos,
            "intramolecular_positive_excess_eV": u_pos,
            "transverse_inter_fraction": None if denom <= 1.0e-20 else trans_pos / denom,
        }

    fit_mask = (times >= GROUP_FIT_START_FS) & (times <= GROUP_FIT_END_FS)
    fit_t: list[float] = []
    fit_c: list[float] = []
    for t, profile in zip(times[fit_mask], inter_ex[fit_mask], strict=True):
        center = backward_excess_centroid_sites(
            profile, s_axis, core_half_width_sites=CORE_HALF_WIDTH_SITES
        )
        if center is not None:
            fit_t.append(float(t))
            fit_c.append(float(center))
    group_fit = fit_group_velocity_sites_per_fs(
        np.asarray(fit_t), np.asarray(fit_c), minimum_points=20
    )

    residual = total_energy - total_energy[0]
    max_residual = float(np.max(np.abs(residual)))
    tolerance = size_scaled_energy_balance_tolerance_eV(size * size)
    condition = {
        "anisotropy_ratio": float(ratio),
        "static": static,
        "source_site": source,
        "target_site": target,
        "launch_direction": "+x",
        "quench_description": "translate electronic wavefunction +1 x site; keep relaxed lattice and zero velocity",
        "elapsed_seconds": float(elapsed),
        "steps": steps,
        "sample_count": n_samples,
        "hamiltonian_evaluations": h_eval,
        "hamiltonian_applications": h_apply,
        "initial_source_population": float(source_pop[0]),
        "initial_target_population": float(target_pop[0]),
        "final_source_population": float(source_pop[-1]),
        "final_target_population": float(target_pop[-1]),
        "maximum_total_energy_residual_eV": max_residual,
        "energy_balance_tolerance_eV": tolerance,
        "energy_balance_pass": extensive_energy_balance_passes(max_residual, size * size),
        "maximum_electronic_norm_error": float(np.max(norm_error)),
        "maximum_zero_mode_magnitude": float(np.max(zero_mode_mag)),
        "windows": windows,
        "backward_centroid_group_fit": group_fit,
    }
    profiles = {
        "time_fs": times,
        "s_sites": np.asarray(s_axis, dtype=np.float64),
        "u_excess": u_ex,
        "longitudinal_excess": long_ex,
        "transverse_excess": trans_ex,
        "intermolecular_excess": inter_ex,
        "longitudinal_flux": flux_ex,
        "source_population": source_pop,
        "target_population": target_pop,
        "total_energy_residual_eV": residual,
    }
    return condition, profiles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=40)
    parser.add_argument("--ratios", type=float, nargs="+", default=[1.0, 0.15])
    parser.add_argument("--dt-fs", type=float, default=0.2)
    parser.add_argument("--final-time-fs", type=float, default=5000.0)
    parser.add_argument("--sample-interval-fs", type=float, default=2.0)
    parser.add_argument("--krylov-dimension", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    args = parser.parse_args()

    conditions: list[dict] = []
    arrays: dict[str, np.ndarray] = {}
    for ratio in args.ratios:
        condition, profiles = _condition(
            args.size,
            ratio,
            args.dt_fs,
            args.final_time_fs,
            args.sample_interval_fs,
            args.krylov_dimension,
        )
        conditions.append(condition)
        prefix = f"r{str(float(ratio)).replace('.', 'p')}"
        for name, values in profiles.items():
            arrays[f"{prefix}_{name}"] = values

    checks = {
        "all_static_relaxations_converged": all(c["static"]["converged"] for c in conditions),
        "deterministic_energy_balance": all(c["energy_balance_pass"] for c in conditions),
        "electronic_norm": max(c["maximum_electronic_norm_error"] for c in conditions) < 1.0e-10,
        "intermolecular_zero_modes": max(c["maximum_zero_mode_magnitude"] for c in conditions) < 1.0e-10,
        "finite_wake_diagnostics": all(np.isfinite(c["windows"]["0:500fs"]["signed_asymmetry"]) for c in conditions),
    }
    payload = {
        "scope": "IP1g deterministic controlled one-site electronic-relocation re-dressing radiation diagnostic; no hopping-rate/mobility/material-lifetime claim",
        "size": args.size,
        "dt_fs": args.dt_fs,
        "final_time_fs": args.final_time_fs,
        "sample_interval_fs": args.sample_interval_fs,
        "temperature_K": 0.0,
        "thermostat": None,
        "IDC": None,
        "external_field_after_launch": None,
        "conditions": conditions,
        "numerical_checks": checks,
        "numerical_pass": bool(all(checks.values())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(args.profiles, **arrays)

    lines = [
        "# IP1g controlled single-relocation lattice-radiation experiment",
        "",
        f"Cell {args.size}x{args.size}; deterministic T=0; final={args.final_time_fs/1000.0:.1f} ps; +x one-site electronic quench; no thermostat, IDC or field after launch.",
        "",
        "| J0y/J0x | max E residual [eV] | wake asym 0:500 | back flux [eV/fs] | front flux [eV/fs] | flux bias | backward-centroid slope [sites/ps] | R2 | transverse fraction | final target pop |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in conditions:
        early = c["windows"]["0:500fs"]
        fit = c["backward_centroid_group_fit"]
        vel = "NA" if fit["velocity_sites_per_ps"] is None else f"{fit['velocity_sites_per_ps']:.3f}"
        r2 = "NA" if fit["r_squared"] is None else f"{fit['r_squared']:.3f}"
        tf = early["transverse_inter_fraction"]
        tf_text = "NA" if tf is None else f"{tf:.3f}"
        lines.append(
            f"| {c['anisotropy_ratio']:.2f} | {c['maximum_total_energy_residual_eV']:.6e} | "
            f"{early['signed_asymmetry']:.3f} | {early['backward_outward_flux_eV_per_fs']:.6e} | "
            f"{early['forward_outward_flux_eV_per_fs']:.6e} | {early['flux_bias']:.3f} | "
            f"{vel} | {r2} | {tf_text} | {c['final_target_population']:.3f} |"
        )
    lines.extend(["", "## Numerical gates", ""])
    lines.extend(f"- {name}: {'PASS' if value else 'FAIL'}" for name, value in checks.items())
    lines.extend(
        [
            "",
            f"Numerical status: {'PASS' if payload['numerical_pass'] else 'FAIL'}",
            "",
            "This imposed re-dressing impulse isolates lattice radiation from thermal background, IDC and overlapping natural hops. It is not a natural hopping or mobility calculation.",
        ]
    )
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if not payload["numerical_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
