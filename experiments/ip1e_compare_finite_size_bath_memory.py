#!/usr/bin/env python3
"""Compare IP1d mechanism diagnostics across IP1e finite-size/bath controls.

This script consumes already completed IP1d JSON artifacts. It performs no
additional dynamics and applies no physical pass threshold. The purpose is to
make the 20x20 -> 40x40 and gamma_v=0.01 -> 0.002 sensitivity transparent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not bool(payload.get("numerical_pass", False)):
        raise ValueError(f"underlying IP1d artifact is not numerical PASS: {path}")
    return payload


def _find_condition(payload: dict, ratio: float, temperature_K: float) -> dict:
    for condition in payload["conditions"]:
        if abs(float(condition["anisotropy_ratio"]) - float(ratio)) < 1.0e-12 and abs(
            float(condition["temperature_K"]) - float(temperature_K)
        ) < 1.0e-9:
            return condition
    raise KeyError(f"condition ratio={ratio}, T={temperature_K} K not found")


def _extract(condition: dict) -> dict:
    a = condition["aggregate"]
    early = a["prehop_matched_bins"]["-100:-80fs"]
    far = a["prehop_matched_bins"]["-500:-400fs"]
    current = a["current_windows"]["100fs"]
    return {
        "complete_events": int(a["complete_window_event_count"]),
        "temperature_mean_K": float(a["temperature_mean_K"]),
        "q_matched_advantage_mean": float(a["q_matched_advantage_mean"]),
        "q_matched_advantage_positive_fraction": float(
            a["q_matched_advantage_positive_fraction"]
        ),
        "q_true_maximum_fraction": float(a["q_true_maximum_fraction"]),
        "current100_parallel_positive_fraction": float(
            current["true_parallel_positive_fraction"]
        ),
        "current100_true_maximum_fraction": float(current["true_maximum_fraction"]),
        "bond_advantage_100_80_eV": float(early["matched_advantage_mean_eV"]),
        "bond_top1_100_80_fraction": float(early["true_top1_fraction"]),
        "bond_advantage_500_400_eV": float(far["matched_advantage_mean_eV"]),
        "bond_top1_500_400_fraction": float(far["true_top1_fraction"]),
        "template_signal_at_event_mean": float(a["template_signal_at_event_mean"]),
        "idc_associated_fraction": float(a["idc_associated_fraction"]),
        "maximum_generalized_energy_residual_eV": float(
            a["maximum_generalized_energy_residual_eV"]
        ),
        "maximum_electronic_norm_error": float(a["maximum_electronic_norm_error"]),
        "maximum_zero_mode_magnitude": float(a["maximum_zero_mode_magnitude"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline20", type=Path, required=True)
    parser.add_argument("--large40", type=Path, required=True)
    parser.add_argument("--weak40", type=Path, required=True)
    parser.add_argument("--temperature-K", type=float, default=300.0)
    parser.add_argument("--ratios", type=float, nargs="+", default=[1.0, 0.15])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    protocols = {
        "20x20_gamma0.01": _load(args.baseline20),
        "40x40_gamma0.01": _load(args.large40),
        "40x40_gamma0.002": _load(args.weak40),
    }
    records: list[dict] = []
    for name, payload in protocols.items():
        for ratio in args.ratios:
            condition = _find_condition(payload, ratio, args.temperature_K)
            records.append(
                {
                    "protocol": name,
                    "size": int(payload["size"]),
                    "gamma_u_per_fs": float(payload["gamma_u_per_fs"]),
                    "gamma_v_per_fs": float(payload["gamma_v_per_fs"]),
                    "anisotropy_ratio": float(ratio),
                    "temperature_K": float(args.temperature_K),
                    **_extract(condition),
                }
            )

    result = {
        "scope": "IP1e physical sensitivity comparison only; no mobility/diffusion/activation/rate claim",
        "underlying_numerical_pass": True,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "# IP1e finite-size and bath-memory comparison",
        "",
        "All rows come from numerically passing IP1d-style dynamics. Agreement/disagreement is a physical sensitivity result, not a numerical gate.",
        "",
        "| protocol | J0y/J0x | complete events | <Delta q true-cf> | q advantage + | current + +/-100 fs | bond adv -100:-80 [meV] | bond top1 -100:-80 | bond adv -500:-400 [meV] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            f"| {record['protocol']} | {record['anisotropy_ratio']:.2f} | "
            f"{record['complete_events']} | {record['q_matched_advantage_mean']:.4f} | "
            f"{record['q_matched_advantage_positive_fraction']:.3f} | "
            f"{record['current100_parallel_positive_fraction']:.3f} | "
            f"{1000.0*record['bond_advantage_100_80_eV']:.2f} | "
            f"{record['bond_top1_100_80_fraction']:.3f} | "
            f"{1000.0*record['bond_advantage_500_400_eV']:.2f} |"
        )
    lines.extend(
        [
            "",
            "No event count in this table is interpreted as a production hopping rate. The two-seed protocols are sensitivity screens only.",
        ]
    )
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
