#!/usr/bin/env python3
"""IP1h posthoc boundary-flux reanalysis of completed IP1g trajectories.

No dynamics are rerun.  IP1g stored carrier-aligned longitudinal lattice-energy
current profiles.  This stage replaces the earlier half-space summed-current
quantity by the physically cleaner current through fixed boundaries and tracks
packet propagation by one-site cross-correlation delays.

This establishes whether a lattice-radiation packet actually propagates toward
-s and/or +s after the controlled electronic relocation.  It does not establish
a natural hopping rate, mobility, calibrated phonon lifetime, or a unique
normal-mode group velocity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from holstein_peierls.dynamics.phonon_recurrence import intermolecular_recurrence_scales
from holstein_peierls.dynamics.wavepacket_flux import boundary_chain_summary
from holstein_peierls.parameters import StaticPolaronParameters


def _ratio_key(value: float) -> str:
    return f"r{str(float(value)).replace('.', 'p')}"


def _boundary_record(summary: dict, distance: int) -> dict:
    for record in summary["boundaries"]:
        if int(record["distance_sites"]) == int(distance):
            return record
    raise KeyError(distance)


def _directionality(back: float, front: float) -> float:
    denominator = abs(float(back)) + abs(float(front))
    return 0.0 if denominator <= 1.0e-30 else float((back - front) / denominator)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--distances", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    parser.add_argument("--lag-min-fs", type=float, default=300.0)
    parser.add_argument("--lag-max-fs", type=float, default=1000.0)
    args = parser.parse_args()

    artifact_dir = args.artifact_dir.resolve()
    source_json = artifact_dir / "ip1g-single-relocation-wake.json"
    source_npz = artifact_dir / "ip1g-single-relocation-wake-profiles.npz"
    if not source_json.exists() or not source_npz.exists():
        raise FileNotFoundError("IP1g JSON/NPZ artifacts not found in artifact_dir")

    source = json.loads(source_json.read_text(encoding="utf-8"))
    profiles = np.load(source_npz)
    size = int(source["size"])
    parameters = StaticPolaronParameters(nx=size, ny=size)
    recurrence = intermolecular_recurrence_scales(
        parameters,
        lattice_spacing_A=3.0,
        gamma_v_per_fs=0.0,
    )
    vmax = float(recurrence.max_group_velocity_sites_per_fs * 1000.0)
    distances = tuple(int(value) for value in args.distances)

    records: list[dict] = []
    for condition in source["conditions"]:
        ratio = float(condition["anisotropy_ratio"])
        key = _ratio_key(ratio)
        times = np.asarray(profiles[f"{key}_time_fs"], dtype=np.float64)
        s_axis = np.asarray(profiles[f"{key}_s_sites"], dtype=np.int64)
        flux = np.asarray(profiles[f"{key}_longitudinal_flux"], dtype=np.float64)

        backward = boundary_chain_summary(
            times,
            flux,
            s_axis,
            side="backward",
            distances_sites=distances,
            lag_min_fs=args.lag_min_fs,
            lag_max_fs=args.lag_max_fs,
        )
        forward = boundary_chain_summary(
            times,
            flux,
            s_axis,
            side="forward",
            distances_sites=distances,
            lag_min_fs=args.lag_min_fs,
            lag_max_fs=args.lag_max_fs,
        )

        distance_records = []
        for distance in distances:
            b = _boundary_record(backward, distance)
            f = _boundary_record(forward, distance)
            eb = float(b["positive_outward_energy_eV"])
            ef = float(f["positive_outward_energy_eV"])
            nb = float(b["net_outward_energy_eV"])
            nf = float(f["net_outward_energy_eV"])
            distance_records.append(
                {
                    "distance_sites": int(distance),
                    "backward_positive_outward_energy_eV": eb,
                    "forward_positive_outward_energy_eV": ef,
                    "positive_energy_directionality": _directionality(eb, ef),
                    "backward_net_outward_energy_eV": nb,
                    "forward_net_outward_energy_eV": nf,
                    "net_energy_directionality": _directionality(nb, nf),
                }
            )

        def annotate(chain: dict) -> dict:
            delays = []
            for item in chain["delays"]:
                speed = float(item["speed_sites_per_ps"])
                correlation = float(item["correlation"])
                delays.append(
                    {
                        **item,
                        "within_harmonic_vmax": bool(speed <= 1.05 * vmax),
                        "high_correlation": bool(correlation >= 0.80),
                    }
                )
            valid = [
                float(item["speed_sites_per_ps"])
                for item in delays
                if item["within_harmonic_vmax"] and item["high_correlation"]
            ]
            return {
                **chain,
                "delays": delays,
                "validated_median_speed_sites_per_ps": float(np.median(valid)) if valid else None,
                "validated_pair_count": int(len(valid)),
            }

        records.append(
            {
                "anisotropy_ratio": ratio,
                "backward": annotate(backward),
                "forward": annotate(forward),
                "boundary_directionality": distance_records,
            }
        )

    numerical_checks = {
        "source_ip1g_numerical_pass": bool(source.get("numerical_pass", False)),
        "finite_boundary_metrics": bool(
            all(
                np.isfinite(item["positive_energy_directionality"])
                for record in records
                for item in record["boundary_directionality"]
            )
        ),
        "finite_delay_metrics": bool(
            all(
                np.isfinite(item["lag_fs"])
                and np.isfinite(item["speed_sites_per_ps"])
                and np.isfinite(item["correlation"])
                for record in records
                for side in ("backward", "forward")
                for item in record[side]["delays"]
            )
        ),
    }
    physical_screen = {
        "backward_packet_validated_pair_counts": {
            str(record["anisotropy_ratio"]): int(record["backward"]["validated_pair_count"])
            for record in records
        },
        "forward_packet_validated_pair_counts": {
            str(record["anisotropy_ratio"]): int(record["forward"]["validated_pair_count"])
            for record in records
        },
    }
    payload = {
        "scope": "IP1h posthoc fixed-boundary harmonic energy-flux and propagation-delay audit of IP1g; no dynamics rerun and no mobility/rate/material-lifetime claim",
        "source_artifact_dir": str(artifact_dir),
        "size": size,
        "distances_sites": list(distances),
        "lag_search_fs": [float(args.lag_min_fs), float(args.lag_max_fs)],
        "harmonic_max_group_velocity_sites_per_ps": vmax,
        "records": records,
        "numerical_checks": numerical_checks,
        "numerical_pass": bool(all(numerical_checks.values())),
        "physical_screen": physical_screen,
        "interpretation_guard": {
            "fixed_boundary_flux_supersedes_half_space_summed_current_for_directionality": True,
            "cross_correlation_speed_is_flux_packet_speed_not_unique_normal_mode_group_velocity": True,
            "controlled_quench_is_not_natural_transport": True,
            "absence_of_a_validated_packet_is_a_physical_result_not_a_numerical_failure": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# IP1h fixed-boundary lattice-radiation reanalysis",
        "",
        "No dynamics were rerun. Fixed-boundary flux replaces the earlier half-space summed-current quantity.",
        "",
        f"Harmonic maximum group velocity: {vmax:.4f} sites/ps.",
        "",
        "| J0y/J0x | back packet speed [sites/ps] | back pairs | front packet speed [sites/ps] | front pairs | directionality d=2 | d=4 | d=6 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        by_distance = {item["distance_sites"]: item for item in record["boundary_directionality"]}
        back_speed = record["backward"]["validated_median_speed_sites_per_ps"]
        front_speed = record["forward"]["validated_median_speed_sites_per_ps"]
        lines.append(
            f"| {record['anisotropy_ratio']:.2f} | "
            f"{'NA' if back_speed is None else f'{back_speed:.3f}'} | {record['backward']['validated_pair_count']} | "
            f"{'NA' if front_speed is None else f'{front_speed:.3f}'} | {record['forward']['validated_pair_count']} | "
            f"{by_distance[2]['positive_energy_directionality']:.3f} | "
            f"{by_distance[4]['positive_energy_directionality']:.3f} | "
            f"{by_distance[6]['positive_energy_directionality']:.3f} |"
        )
    lines.extend(["", "## Numerical gates", ""])
    for name, value in numerical_checks.items():
        lines.append(f"- {name}: {'PASS' if value else 'FAIL'}")
    lines.extend(
        [
            "",
            f"Numerical status: {'PASS' if payload['numerical_pass'] else 'FAIL'}",
            "",
            "A positive boundary directionality means more positive outward lattice energy crossed the backward (-s) boundary than the forward (+s) boundary at the same distance. Packet speeds are inferred from correlations between successive fixed boundaries and are checked against the harmonic maximum group velocity. The imposed IP1g quench remains a diagnostic impulse experiment, not natural carrier transport.",
        ]
    )
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if not payload["numerical_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
