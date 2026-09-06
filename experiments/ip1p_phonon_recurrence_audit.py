#!/usr/bin/env python3
"""IP1p periodic-boundary phonon recurrence audit.

This lightweight stage evaluates the harmonic intermolecular propagation scale
of the current Holstein-Peierls lattice and the damping regime of the Langevin
bath.  It is a finite-size/protocol diagnostic only; it does not infer a
material phonon lifetime or transport coefficient.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from holstein_peierls.dynamics.phonon_recurrence import (
    conservative_event_window_cutoff_fs,
    intermolecular_recurrence_scales,
)
from holstein_peierls.parameters import StaticPolaronParameters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=20)
    parser.add_argument("--ny", type=int, default=20)
    parser.add_argument("--lattice-spacing-A", type=float, default=3.0)
    parser.add_argument("--gamma-v-per-fs", type=float, default=0.01)
    parser.add_argument("--event-half-window-fs", type=float, default=500.0)
    parser.add_argument("--planned-final-time-fs", type=float, default=10000.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    parameters = StaticPolaronParameters(nx=args.nx, ny=args.ny)
    scales = intermolecular_recurrence_scales(
        parameters,
        lattice_spacing_A=args.lattice_spacing_A,
        gamma_v_per_fs=args.gamma_v_per_fs,
    )
    cutoff = conservative_event_window_cutoff_fs(
        scales,
        event_half_window_fs=args.event_half_window_fs,
    )
    payload = {
        "parameters": {
            "nx": args.nx,
            "ny": args.ny,
            "k2_eV_per_A2": parameters.k2,
            "m2_legacy": parameters.m2,
            "lattice_spacing_A": args.lattice_spacing_A,
            "gamma_v_per_fs": args.gamma_v_per_fs,
            "event_half_window_fs": args.event_half_window_fs,
            "planned_final_time_fs": args.planned_final_time_fs,
        },
        "scales": asdict(scales),
        "stationary_carrier_event_cutoff_fs": cutoff,
        "planned_run_finishes_before_stationary_wrap": bool(
            args.planned_final_time_fs <= scales.earliest_stationary_carrier_wrap_fs
        ),
        "planned_complete_event_windows_finish_before_stationary_wrap": bool(
            args.planned_final_time_fs - args.event_half_window_fs <= cutoff
        ),
        "interpretation": {
            "recurrence_time_is_stationary_carrier_ballistic_diagnostic": True,
            "moving_carrier_can_change_actual_collision_time": True,
            "langevin_gamma_is_numerical_not_material_calibrated": True,
            "strong_damping_can_suppress_physical_phonon_memory": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = [
        "# IP1p phonon PBC recurrence audit",
        "",
        f"Cell: {args.nx}x{args.ny}; a={args.lattice_spacing_A:.3f} A; gamma_v={args.gamma_v_per_fs:.5f} fs^-1.",
        "",
        f"- harmonic maximum group velocity: {scales.max_group_velocity_sites_per_fs*1000.0:.6f} sites/ps ({scales.max_group_velocity_A_per_fs*1000.0:.6f} A/ps)",
        f"- stationary-carrier full-wrap time x: {scales.wrap_time_x_fs/1000.0:.6f} ps",
        f"- stationary-carrier full-wrap time y: {scales.wrap_time_y_fs/1000.0:.6f} ps",
        f"- latest event start whose +{args.event_half_window_fs:.0f} fs analysis edge precedes that wrap: {cutoff/1000.0:.6f} ps",
        f"- omega_max: {scales.omega_max_per_fs:.8f} fs^-1",
        f"- damping ratio at omega_max: {scales.damping_ratio_at_omega_max:.6f}",
        f"- all non-zero harmonic intermolecular modes overdamped at this gamma: {scales.all_nonzero_harmonic_modes_overdamped}",
        "",
        "The wrap time assumes a stationary carrier and undamped maximum-group-velocity packet. A moving carrier can alter the actual collision time. The current Langevin damping is a numerical bath parameter, so suppression of recurrence by damping must not be confused with a material phonon lifetime.",
    ]
    args.markdown.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
