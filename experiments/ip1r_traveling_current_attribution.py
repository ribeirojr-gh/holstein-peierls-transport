#!/usr/bin/env python3
"""IP1r exact traveling-wave attribution of the local trailing x-current."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from holstein_peierls.dynamics.mode_memory import traveling_vx_energy_split
from holstein_peierls.dynamics.phonon_wake import event_aligned_coordinates, longitudinal_profile
from holstein_peierls.dynamics.single_hop_memory import continuation_neighbor
from holstein_peierls.dynamics.traveling_current_attribution import (
    decompose_vx_current,
    projection_attribution,
    row_cosine_similarity,
    squared_norm_ratio,
    traveling_vx_fields,
)
from holstein_peierls.dynamics.wavepacket_flux import outward_boundary_flux_series
from holstein_peierls.lattice import LatticeState
from holstein_peierls.parameters import StaticPolaronParameters
from holstein_peierls.dynamics.ehrenfest import LatticeVelocity


PRIMARY_DISTANCES = (1, 2, 3, 4)
TRANSVERSE_HALF_WIDTH_SITES = 3
EARLY_END_OFFSET_FS = 1000.0
LATE_START_OFFSET_FS = 1000.0
DOMINANCE_THRESHOLD = 2.0 / 3.0


def _trailing_vector(
    current_jx: np.ndarray,
    *,
    coordinates,
    carrier_dx_sites: int,
) -> np.ndarray:
    s_axis, profile = longitudinal_profile(
        int(carrier_dx_sites) * np.asarray(current_jx, dtype=np.float64),
        coordinates,
        transverse_half_width_sites=TRANSVERSE_HALF_WIDTH_SITES,
    )
    values = []
    for distance in PRIMARY_DISTANCES:
        series = outward_boundary_flux_series(
            profile[None, :], s_axis, distance, side="backward"
        )
        values.append(float(series[0]))
    return np.asarray(values, dtype=np.float64)


def _window_summary(
    times_fs: np.ndarray,
    component_vectors: dict[str, np.ndarray],
    mask: np.ndarray,
) -> dict:
    full = component_vectors["full"][mask]
    if full.size == 0:
        raise ValueError("attribution window contains no samples")
    out = {}
    for name in ("retrograde", "comoving", "special", "cross"):
        comp = component_vectors[name][mask]
        cosine = row_cosine_similarity(comp, full)
        rms = np.sqrt(np.mean(comp * comp, axis=1))
        boundary_net = []
        local_t = times_fs[mask]
        for column, distance in enumerate(PRIMARY_DISTANCES):
            boundary_net.append(
                {
                    "distance_sites": int(distance),
                    "net_outward_energy_eV": float(
                        np.trapezoid(comp[:, column], local_t)
                    ),
                }
            )
        out[name] = {
            "projection_fraction": projection_attribution(comp, full),
            "squared_norm_ratio": squared_norm_ratio(comp, full),
            "mean_cosine_similarity": float(np.mean(cosine)),
            "median_cosine_similarity": float(np.median(cosine)),
            "mean_rms_trailing_amplitude_eV_per_fs": float(np.mean(rms)),
            "boundary_net_outward_energy": boundary_net,
        }
    projection_sum = float(sum(out[name]["projection_fraction"] for name in out))
    full_rms = np.sqrt(np.mean(full * full, axis=1))
    return {
        "sample_count": int(np.count_nonzero(mask)),
        "start_fs": float(times_fs[mask][0]),
        "end_fs": float(times_fs[mask][-1]),
        "full_mean_rms_trailing_amplitude_eV_per_fs": float(np.mean(full_rms)),
        "components": out,
        "projection_sum": projection_sum,
        "projection_closure_error": abs(projection_sum - 1.0),
    }


def _classification(late_summary: dict) -> str:
    retro = float(late_summary["components"]["retrograde"]["projection_fraction"])
    comoving = float(late_summary["components"]["comoving"]["projection_fraction"])
    if retro >= DOMINANCE_THRESHOLD:
        return "local retrograde-dominated"
    if comoving >= DOMINANCE_THRESHOLD:
        return "local co-moving-dominated"
    return "mixed/interference"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip1q-json", type=Path, required=True)
    parser.add_argument("--ip1q-trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.ip1q_json.read_text(encoding="utf-8"))
    arrays = np.load(args.ip1q_trajectory)
    if not bool(source.get("numerical_pass", False)):
        raise RuntimeError("source IP1q trajectory is not numerically validated")
    analysis = source["analysis"]
    event = analysis["event"]
    size = int(source["size"])
    parameters = StaticPolaronParameters(nx=size, ny=size)
    parameters = StaticPolaronParameters(
        **{
            **parameters.to_dict(),
            "j0y": parameters.j0x,
            "polaron_position": parameters.polaron_position,
        }
    )

    times = np.asarray(arrays["times_fs"], dtype=np.float64)
    vx_frames = np.asarray(arrays["vx_A"], dtype=np.float64)
    velocity_frames = np.asarray(arrays["velocity_vx_A_per_fs"], dtype=np.float64)
    equilibrium = np.asarray(arrays["equilibrium_vx_A"], dtype=np.float64)
    if vx_frames.shape != velocity_frames.shape or vx_frames.shape[0] != times.size:
        raise ValueError("IP1q trajectory arrays have incompatible shapes")

    carrier_dx = int(event["dx_sites"])
    residence_source = int(event["target_site"])
    residence_target = continuation_neighbor(
        residence_source,
        dx_sites=carrier_dx,
        dy_sites=int(event["dy_sites"]),
        nx=size,
        ny=size,
    )
    coordinates = event_aligned_coordinates(
        residence_source, residence_target, size, size
    )

    component_vectors = {
        key: np.zeros((times.size, len(PRIMARY_DISTANCES)), dtype=np.float64)
        for key in ("full", "retrograde", "comoving", "special", "cross")
    }
    max_coordinate_reconstruction_error = 0.0
    max_velocity_reconstruction_error = 0.0
    max_current_reconstruction_error = 0.0

    for index in range(times.size):
        fields = traveling_vx_fields(
            vx_frames[index], velocity_frames[index], equilibrium, parameters
        )
        reconstructed_coordinate = (
            equilibrium
            + fields.plus_displacement
            + fields.minus_displacement
            + fields.special_displacement
        )
        reconstructed_velocity = (
            fields.plus_velocity + fields.minus_velocity + fields.special_velocity
        )
        max_coordinate_reconstruction_error = max(
            max_coordinate_reconstruction_error,
            float(np.max(np.abs(reconstructed_coordinate - vx_frames[index]))),
        )
        max_velocity_reconstruction_error = max(
            max_velocity_reconstruction_error,
            float(np.max(np.abs(reconstructed_velocity - velocity_frames[index]))),
        )
        currents = decompose_vx_current(
            vx_frames[index],
            velocity_frames[index],
            equilibrium,
            parameters,
            carrier_dx_sites=carrier_dx,
        )
        reconstructed_current = (
            currents["retrograde"]
            + currents["comoving"]
            + currents["special"]
            + currents["cross"]
        )
        max_current_reconstruction_error = max(
            max_current_reconstruction_error,
            float(np.max(np.abs(reconstructed_current - currents["full"]))),
        )
        for name in component_vectors:
            component_vectors[name][index] = _trailing_vector(
                currents[name],
                coordinates=coordinates,
                carrier_dx_sites=carrier_dx,
            )

    switch = float(analysis["switch_time_fs"])
    relative = times - switch
    early_mask = relative <= EARLY_END_OFFSET_FS + 1.0e-10
    late_mask = relative >= LATE_START_OFFSET_FS - 1.0e-10
    early = _window_summary(times, component_vectors, early_mask)
    late = _window_summary(times, component_vectors, late_mask)

    initial_lattice = LatticeState(
        np.asarray(arrays["u_A"][0], dtype=np.float64),
        np.asarray(vx_frames[0], dtype=np.float64),
        np.asarray(arrays["vy_A"][0], dtype=np.float64),
    )
    initial_velocity = LatticeVelocity(
        np.asarray(arrays["velocity_u_A_per_fs"][0], dtype=np.float64),
        np.asarray(velocity_frames[0], dtype=np.float64),
        np.asarray(arrays["velocity_vy_A_per_fs"][0], dtype=np.float64),
    )
    equilibrium_state = LatticeState(
        np.asarray(arrays["equilibrium_u_A"], dtype=np.float64),
        equilibrium,
        np.asarray(arrays["equilibrium_vy_A"], dtype=np.float64),
    )
    global_split = traveling_vx_energy_split(
        initial_lattice,
        initial_velocity,
        equilibrium_state,
        parameters,
        carrier_dx_sites=carrier_dx,
    )

    classification = _classification(late)
    numerical_checks = {
        "source_ip1q_numerical_pass": bool(source.get("numerical_pass", False)),
        "traveling_coordinate_reconstruction": bool(
            max_coordinate_reconstruction_error < 1.0e-12
        ),
        "traveling_velocity_reconstruction": bool(
            max_velocity_reconstruction_error < 1.0e-12
        ),
        "current_reconstruction": bool(max_current_reconstruction_error < 1.0e-12),
        "early_projection_closure": bool(early["projection_closure_error"] < 1.0e-10),
        "late_projection_closure": bool(late["projection_closure_error"] < 1.0e-10),
        "finite_attribution_metrics": bool(
            np.all(
                np.isfinite(
                    np.concatenate([component_vectors[name].ravel() for name in component_vectors])
                )
            )
        ),
    }
    numerical_pass = bool(all(numerical_checks.values()))
    payload = {
        "scope": "IP1r exact traveling-wave attribution of the autonomous isotropic local trailing x-current pattern",
        "source_ip1q_json": str(args.ip1q_json),
        "source_ip1q_trajectory": str(args.ip1q_trajectory),
        "event": event,
        "switch_time_fs": switch,
        "post_hop_frame_source_site": residence_source,
        "post_hop_frame_target_site": residence_target,
        "primary_distances_sites": list(PRIMARY_DISTANCES),
        "maximum_coordinate_reconstruction_error_A": max_coordinate_reconstruction_error,
        "maximum_velocity_reconstruction_error_A_per_fs": max_velocity_reconstruction_error,
        "maximum_current_reconstruction_error_eV_per_fs": max_current_reconstruction_error,
        "global_vx_traveling_energy_split": global_split,
        "early_attribution": early,
        "late_attribution": late,
        "late_local_classification": classification,
        "numerical_checks": numerical_checks,
        "numerical_pass": numerical_pass,
        "interpretation_guard": {
            "global_vx_energy_and_local_trailing_current_are_distinct_observables": True,
            "vy_total_energy_dominance_is_not_direct_jx_attribution": True,
            "cross_term_is_bilinear_remainder_not_normal_mode": True,
            "local_retrograde_dominance_does_not_imply_global_retrograde_energy_dominance": True,
            "no_transport_coefficient_or_material_lifetime_claim": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(
        args.trajectory,
        times_fs=times,
        **{f"{name}_trailing_d1_d4_eV_per_fs": values for name, values in component_vectors.items()},
    )

    lines = [
        "# IP1r traveling-wave attribution of local trailing x-current",
        "",
        f"Event {event['source_site']}->{event['target_site']} ({event['direction']}), switch={switch:.1f} fs.",
        f"Global direction-resolved vx retrograde fraction: {global_split['retrograde_fraction_of_direction_resolved']:.6f}.",
        "",
        "## Exact reconstruction",
        "",
        f"- coordinate max error: {max_coordinate_reconstruction_error:.3e} A;",
        f"- velocity max error: {max_velocity_reconstruction_error:.3e} A/fs;",
        f"- jx max error: {max_current_reconstruction_error:.3e} eV/fs.",
        "",
        "## Late attribution",
        "",
    ]
    for name, record in late["components"].items():
        lines.append(
            f"- {name}: projection={record['projection_fraction']:.6f}; norm2/full={record['squared_norm_ratio']:.6f}; mean cosine={record['mean_cosine_similarity']:.6f};"
        )
    lines.extend(
        [
            f"- projection closure: {late['projection_sum']:.12f};",
            f"- classification: **{classification}**.",
            "",
            "## Numerical gates",
            "",
        ]
    )
    for name, value in numerical_checks.items():
        lines.append(f"- {name}: {'PASS' if value else 'FAIL'}")
    lines.extend(["", f"Numerical status: {'PASS' if numerical_pass else 'FAIL'}"])
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"\nWrote {args.output}")
    print(f"Wrote {args.markdown}")
    print(f"Wrote {args.trajectory}")
    if not numerical_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
