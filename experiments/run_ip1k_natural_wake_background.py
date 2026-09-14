#!/usr/bin/env python3
"""Stable command-line entry point for the IP1k natural-wake background audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import ip1k_natural_wake_background as core
from holstein_peierls.dynamics.phonon_recurrence import intermolecular_recurrence_scales
from holstein_peierls.parameters import StaticPolaronParameters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=40)
    parser.add_argument("--ratios", nargs="+", type=float, default=[1.0, 0.15])
    parser.add_argument("--field-mv-per-A", dest="field_mV_per_A", type=float, default=10.0)
    parser.add_argument("--dt-fs", type=float, default=0.2)
    parser.add_argument("--final-time-fs", type=float, default=5000.0)
    parser.add_argument("--sample-interval-fs", type=float, default=2.0)
    parser.add_argument("--energy-sample-interval-fs", type=float, default=10.0)
    parser.add_argument("--krylov-dimension", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    args = parser.parse_args()

    recurrence = intermolecular_recurrence_scales(
        StaticPolaronParameters(nx=args.size, ny=args.size),
        lattice_spacing_A=core.LATTICE_SPACING_A,
        gamma_v_per_fs=0.0,
    )
    vmax = float(recurrence.max_group_velocity_sites_per_fs * 1000.0)

    conditions: list[dict] = []
    profile_output: dict[str, np.ndarray] = {}
    for ratio in args.ratios:
        result, profiles = core._run_condition(
            size=args.size,
            anisotropy_ratio=float(ratio),
            field_mV_per_A=args.field_mV_per_A,
            dt_fs=args.dt_fs,
            final_time_fs=args.final_time_fs,
            sample_interval_fs=args.sample_interval_fs,
            energy_sample_interval_fs=args.energy_sample_interval_fs,
            krylov_dimension=args.krylov_dimension,
            vmax_sites_per_ps=vmax,
        )
        conditions.append(result)
        key = f"r{str(float(ratio)).replace('.', 'p')}"
        for name, values in profiles.items():
            profile_output[f"{key}_{name}"] = values

    numerical_checks = {
        "all_conditions_numerically_pass": bool(all(item["numerical_pass"] for item in conditions)),
        "profiles_written_for_every_complete_real_event": bool(
            len(profile_output)
            == 3 * sum(item["complete_real_event_count"] for item in conditions)
        ),
    }
    replication = core._replication_summary(conditions)
    payload = {
        "scope": "IP1k natural-hop wake replication against matched event-free same-trajectory background; no transport coefficient or material phonon parameter claim",
        "size": int(args.size),
        "ratios": [float(value) for value in args.ratios],
        "field_mV_per_A": float(args.field_mV_per_A),
        "temperature_K": 0.0,
        "thermostat": None,
        "IDC": None,
        "dt_fs": float(args.dt_fs),
        "final_time_fs": float(args.final_time_fs),
        "sample_interval_fs": float(args.sample_interval_fs),
        "harmonic_max_group_velocity_sites_per_ps": vmax,
        "baseline_offsets_fs": [core.BASELINE_START_OFFSET_FS, core.BASELINE_END_OFFSET_FS],
        "post_end_offset_fs": core.POST_END_OFFSET_FS,
        "pseudo_event_cadence_fs": core.PSEUDO_CADENCE_FS,
        "delay_search_window_fs": [core.LAG_MIN_FS, core.LAG_MAX_FS],
        "minimum_delay_correlation": core.MIN_CORRELATION,
        "background_percentile_threshold": core.BACKGROUND_PERCENTILE_THRESHOLD,
        "conditions": conditions,
        "replication_decision": replication,
        "numerical_checks": numerical_checks,
        "numerical_pass": bool(all(numerical_checks.values())),
        "interpretation_guard": {
            "pseudo_windows_are_correlated_not_independent_samples": True,
            "percentiles_are_robustness_discriminators_not_formal_p_values": True,
            "field_remains_on_in_real_and_pseudo_windows": True,
            "d1_to_d2_delay_is_not_unique_normal_mode_group_velocity": True,
            "weak_retrograde_and_forward_dominated_total_radiation_can_coexist": True,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(args.profiles, **profile_output)

    lines = [
        "# IP1k natural wake replication and matched background",
        "",
        f"Cell {args.size}x{args.size}; T=0; E=+{args.field_mV_per_A:g} mV/A; dt={args.dt_fs:g} fs; final={args.final_time_fs/1000:g} ps.",
        f"Harmonic vmax: {vmax:.4f} sites/ps. Delay gate: {core.LAG_MIN_FS:g}-{core.LAG_MAX_FS:g} fs, correlation >= {core.MIN_CORRELATION:.2f}.",
        "",
        "| J0y/J0x | complete real events | pseudo windows | background-separated real events | replicated? |",
        "|---:|---:|---:|---:|---:|",
    ]
    for condition in conditions:
        replicated = bool(
            np.isclose(condition["anisotropy_ratio"], 0.15)
            and condition["complete_real_event_count"] >= 2
            and condition["background_separated_event_count"] >= 2
        )
        lines.append(
            f"| {condition['anisotropy_ratio']:.2f} | {condition['complete_real_event_count']} | "
            f"{condition['pseudo_event_count']} | {condition['background_separated_event_count']} | "
            f"{'YES' if replicated else 'NO'} |"
        )
        for index, analysis in enumerate(condition["complete_event_analyses"], start=1):
            event = analysis["event"]
            metrics = analysis["real_metrics"]
            back = metrics["backward_d1_to_d2_delay"]
            lines.extend(
                [
                    "",
                    f"### J0y/J0x={condition['anisotropy_ratio']:.2f}, event {index}",
                    "",
                    f"- event: {event['source_site']} -> {event['target_site']} ({event['direction']}), t0={event['transition_start_time_fs']:.1f} fs;",
                    f"- backward positive energy: {metrics['backward_positive_outward_energy_eV']:.6e} eV; percentile={analysis['backward_positive_energy_percentile']:.1f};",
                    f"- backward peak flux: {metrics['backward_peak_outward_flux_eV_per_fs']:.6e} eV/fs; percentile={analysis['backward_peak_flux_percentile']:.1f};",
                    f"- backward delay: {back['lag_fs']:.1f} fs, speed={back['speed_sites_per_ps']:.4f} sites/ps, corr={back['correlation']:.6f};",
                    f"- packet qualified: {'YES' if analysis['backward_packet_qualified'] else 'NO'};",
                    f"- background separated: {'YES' if analysis['background_separated'] else 'NO'}.",
                ]
            )
    lines.extend(
        [
            "",
            "## Replication decision",
            "",
            f"- anisotropic two complete events: {'YES' if replication['anisotropic_two_complete_events'] else 'NO'};",
            f"- anisotropic two background-separated events: {'YES' if replication['anisotropic_two_background_separated_events'] else 'NO'};",
            f"- replicated event-associated retrograde evidence: {'YES' if replication['replicated_event_associated_retrograde_evidence'] else 'NO'};",
            "",
            "Physical replication is not a numerical acceptance gate.",
            "",
            "## Numerical gates",
            "",
        ]
    )
    for name, value in numerical_checks.items():
        lines.append(f"- {name}: {'PASS' if value else 'FAIL'}")
    lines.append("")
    lines.append(f"Numerical status: {'PASS' if payload['numerical_pass'] else 'FAIL'}")
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"\nWrote {args.output}")
    print(f"Wrote {args.markdown}")
    print(f"Wrote {args.profiles}")
    if not payload["numerical_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
