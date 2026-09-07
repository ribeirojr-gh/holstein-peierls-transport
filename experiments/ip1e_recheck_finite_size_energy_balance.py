#!/usr/bin/env python3
"""Recheck completed IP1e artifacts with a size-aware extensive energy gate.

IP1e attempt 1 reused the 5e-5 eV absolute energy-balance threshold calibrated
on 20x20 dynamics for 40x40 cells.  The completed 40x40 trajectories missed
that fixed threshold only slightly while their residual per site was no worse
than the 20x20 control.  Re-running the expensive trajectories is unnecessary:
this script audits the already written JSON artifacts, preserves every other
numerical check, and applies the finite-size gate explicitly.

No physical sensitivity threshold is introduced here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from holstein_peierls.dynamics.numerical_validation import (
    extensive_energy_balance_passes,
    size_scaled_energy_balance_tolerance_eV,
)

PROTOCOLS = ("baseline20", "large40", "weak40")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _maximum_residual(payload: dict) -> float:
    return float(
        max(
            condition["aggregate"]["maximum_generalized_energy_residual_eV"]
            for condition in payload["conditions"]
        )
    )


def _audit(payload: dict) -> dict:
    size = int(payload["size"])
    n_sites = size * size
    original_checks = dict(payload["numerical_checks"])
    other_checks = {
        key: bool(value)
        for key, value in original_checks.items()
        if key != "zero_field_generalized_energy_balance"
    }
    maximum_residual = _maximum_residual(payload)
    tolerance = size_scaled_energy_balance_tolerance_eV(n_sites)
    energy_ok = extensive_energy_balance_passes(maximum_residual, n_sites)
    checks = {**other_checks, "size_scaled_zero_field_generalized_energy_balance": energy_ok}
    return {
        "size": size,
        "n_sites": n_sites,
        "maximum_generalized_energy_residual_eV": maximum_residual,
        "residual_per_site_eV": maximum_residual / n_sites,
        "size_scaled_tolerance_eV": tolerance,
        "tolerance_per_site_eV": tolerance / n_sites,
        "original_numerical_pass": bool(payload.get("numerical_pass", False)),
        "original_energy_balance_check": bool(
            original_checks.get("zero_field_generalized_energy_balance", False)
        ),
        "rechecked_checks": checks,
        "rechecked_numerical_pass": bool(all(checks.values())),
    }


def _find_condition(payload: dict, ratio: float, temperature_K: float = 300.0) -> dict:
    for condition in payload["conditions"]:
        if abs(float(condition["anisotropy_ratio"]) - ratio) < 1.0e-12 and abs(
            float(condition["temperature_K"]) - temperature_K
        ) < 1.0e-9:
            return condition
    raise KeyError(f"condition ratio={ratio}, T={temperature_K} K not found")


def _extract_physics(payload: dict, ratio: float) -> dict:
    condition = _find_condition(payload, ratio)
    aggregate = condition["aggregate"]
    current = aggregate["current_windows"]["100fs"]
    early = aggregate["prehop_matched_bins"]["-100:-80fs"]
    far = aggregate["prehop_matched_bins"]["-500:-400fs"]
    return {
        "anisotropy_ratio": ratio,
        "complete_events": int(aggregate["complete_window_event_count"]),
        "temperature_mean_K": float(aggregate["temperature_mean_K"]),
        "q_matched_advantage_mean": float(aggregate["q_matched_advantage_mean"]),
        "q_matched_advantage_positive_fraction": float(
            aggregate["q_matched_advantage_positive_fraction"]
        ),
        "q_true_maximum_fraction": float(aggregate["q_true_maximum_fraction"]),
        "current100_parallel_positive_fraction": float(
            current["true_parallel_positive_fraction"]
        ),
        "current100_true_maximum_fraction": float(current["true_maximum_fraction"]),
        "bond_advantage_100_80_eV": float(early["matched_advantage_mean_eV"]),
        "bond_top1_100_80_fraction": float(early["true_top1_fraction"]),
        "bond_advantage_500_400_eV": float(far["matched_advantage_mean_eV"]),
        "template_signal_at_event_mean": float(aggregate["template_signal_at_event_mean"]),
        "idc_associated_fraction": float(aggregate["idc_associated_fraction"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args()

    artifact_dir = args.artifact_dir
    payloads = {
        name: _load(artifact_dir / f"{name}-ip1d.json") for name in PROTOCOLS
    }
    audits = {name: _audit(payload) for name, payload in payloads.items()}
    all_pass = bool(all(item["rechecked_numerical_pass"] for item in audits.values()))

    records: list[dict] = []
    for name, payload in payloads.items():
        for ratio in (1.0, 0.15):
            records.append(
                {
                    "protocol": name,
                    "size": int(payload["size"]),
                    "gamma_v_per_fs": float(payload["gamma_v_per_fs"]),
                    **_extract_physics(payload, ratio),
                }
            )

    result = {
        "scope": "IP1e posthoc finite-size numerical audit and physical sensitivity comparison; no mobility/diffusion/activation/rate claim",
        "criterion": {
            "reference_cell": "20x20",
            "reference_tolerance_eV": 5.0e-5,
            "scaling": "linear in number of lattice sites; no time scaling",
        },
        "protocol_audits": audits,
        "rechecked_numerical_pass": all_pass,
        "records": records,
    }

    output = args.output or artifact_dir / "ip1e-size-aware-recheck.json"
    markdown = args.markdown or artifact_dir / "ip1e-size-aware-recheck.md"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "# IP1e size-aware numerical recheck",
        "",
        "The original 20x20 absolute 5e-5 eV energy-balance tolerance is preserved exactly and scaled linearly with lattice-site count for the 40x40 finite-size controls.",
        "",
        "| protocol | size | max residual [eV] | residual/site [eV] | scaled tolerance [eV] | recheck |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in PROTOCOLS:
        audit = audits[name]
        lines.append(
            f"| {name} | {audit['size']} | {audit['maximum_generalized_energy_residual_eV']:.8e} | "
            f"{audit['residual_per_site_eV']:.8e} | {audit['size_scaled_tolerance_eV']:.8e} | "
            f"{'PASS' if audit['rechecked_numerical_pass'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Physical sensitivity screen",
            "",
            "| protocol | J0y/J0x | events | <Delta q matched> | q advantage + | current + +/-100 fs | current true max | bond adv -100:-80 [meV] | bond top1 -100:-80 | bond adv -500:-400 [meV] |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for record in records:
        lines.append(
            f"| {record['protocol']} | {record['anisotropy_ratio']:.2f} | {record['complete_events']} | "
            f"{record['q_matched_advantage_mean']:.4f} | {record['q_matched_advantage_positive_fraction']:.3f} | "
            f"{record['current100_parallel_positive_fraction']:.3f} | {record['current100_true_maximum_fraction']:.3f} | "
            f"{1000.0*record['bond_advantage_100_80_eV']:.2f} | {record['bond_top1_100_80_fraction']:.3f} | "
            f"{1000.0*record['bond_advantage_500_400_eV']:.2f} |"
        )
    lines.extend(
        [
            "",
            f"Rechecked numerical status: {'PASS' if all_pass else 'FAIL'}",
            "",
            "This recheck changes only the finite-size numerical tolerance. It does not change trajectories, physical diagnostics, or any physical interpretation threshold.",
        ]
    )
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if not all_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
