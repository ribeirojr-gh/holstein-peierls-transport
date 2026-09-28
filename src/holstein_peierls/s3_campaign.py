"""Manifest validation and observable-based aggregation for Paper-1 S3."""

from __future__ import annotations

from collections import defaultdict
from math import isfinite
from typing import Any, Iterable


BRANCHES = ("onsite", "intersite_x", "intersite_y", "diagonal", "separated")


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Validate the compact, prospective S3 bipolaron campaign manifest."""
    if manifest.get("schema_version") != 1:
        raise ValueError("S3 manifest schema_version must be 1")
    if manifest.get("stage") not in {
        "pilot",
        "screen",
        "production",
        "finite_size",
    }:
        raise ValueError("S3 manifest stage is unsupported")

    sizes = manifest.get("lattice_sizes")
    if not isinstance(sizes, list) or not sizes:
        raise ValueError("lattice_sizes must be a non-empty list")
    if any(not isinstance(size, int) or size < 4 or size % 2 for size in sizes):
        raise ValueError("lattice_sizes must contain even integers >= 4")

    grid = manifest.get("grid", {})
    if "slices" in grid:
        slices = grid.get("slices")
        if not isinstance(slices, list) or not slices:
            raise ValueError("grid.slices must be a non-empty list")
        for index, row in enumerate(slices):
            for name in ("U_eV", "V1_eV"):
                values = row.get(name)
                if not isinstance(values, list) or not values:
                    raise ValueError(f"grid.slices[{index}].{name} must be non-empty")
                if any(
                    not isinstance(value, (int, float)) or not isfinite(value)
                    for value in values
                ):
                    raise ValueError(f"grid.slices[{index}].{name} must be finite")
                if any(value < 0.0 for value in values):
                    raise ValueError(f"grid.slices[{index}].{name} must be non-negative")
            coupling = row.get("coupling_scale")
            if (
                not isinstance(coupling, (int, float))
                or not isfinite(coupling)
                or coupling <= 0.0
            ):
                raise ValueError(
                    f"grid.slices[{index}].coupling_scale must be positive and finite"
                )
    else:
        for name in ("U_eV", "V1_eV", "coupling_scale"):
            values = grid.get(name)
            if not isinstance(values, list) or not values:
                raise ValueError(f"grid.{name} must be a non-empty list")
            if any(
                not isinstance(value, (int, float)) or not isfinite(value)
                for value in values
            ):
                raise ValueError(f"grid.{name} must contain finite numbers")
        if any(value < 0.0 for value in grid["U_eV"] + grid["V1_eV"]):
            raise ValueError("U_eV and V1_eV must be non-negative")
        if any(value <= 0.0 for value in grid["coupling_scale"]):
            raise ValueError("coupling_scale must be positive")

    branches = manifest.get("branches")
    if branches != list(BRANCHES):
        raise ValueError("branches must contain the complete locked S3 seed order")

    model = manifest.get("model", {})
    required_positive = (
        "Jx_eV",
        "Jy_eV",
        "K1_eV_per_A2",
        "K2_eV_per_A2",
        "base_alpha_intra_eV_per_A",
        "base_alpha_x_eV_per_A",
        "base_alpha_y_eV_per_A",
    )
    for name in required_positive:
        value = model.get(name)
        if not isinstance(value, (int, float)) or not isfinite(value) or value <= 0.0:
            raise ValueError(f"model.{name} must be a positive finite number")
    if model.get("boundary_conditions") != "periodic":
        raise ValueError("S3 currently requires periodic boundary conditions")

    numerical = manifest.get("numerical", {})
    if numerical.get("max_iterations", 0) <= 0:
        raise ValueError("numerical.max_iterations must be positive")
    criteria = manifest.get("classification", {})
    for name in (
        "energy_tie_tolerance_eV",
        "robust_binding_threshold_eV",
        "linear_peierls_ratio_max",
    ):
        value = criteria.get(name)
        if not isinstance(value, (int, float)) or not isfinite(value) or value <= 0.0:
            raise ValueError(f"classification.{name} must be positive and finite")


def expand_tasks(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a validated manifest into deterministic branch tasks."""
    validate_manifest(manifest)
    model = manifest["model"]
    tasks: list[dict[str, Any]] = []
    grid = manifest["grid"]
    if "slices" in grid:
        parameter_points = [
            (float(row["coupling_scale"]), float(u), float(v1))
            for row in grid["slices"]
            for u in row["U_eV"]
            for v1 in row["V1_eV"]
        ]
    else:
        parameter_points = [
            (float(coupling_scale), float(u), float(v1))
            for coupling_scale in grid["coupling_scale"]
            for u in grid["U_eV"]
            for v1 in grid["V1_eV"]
        ]
    for size in manifest["lattice_sizes"]:
        for coupling_scale, hubbard_u, v1 in parameter_points:
            for branch in manifest["branches"]:
                tasks.append(
                    {
                        "size": int(size),
                        "coupling_scale": coupling_scale,
                        "U_eV": hubbard_u,
                        "V1_eV": v1,
                        "branch": str(branch),
                        "alpha_intra_eV_per_A": coupling_scale
                        * float(model["base_alpha_intra_eV_per_A"]),
                        "alpha_x_eV_per_A": coupling_scale
                        * float(model["base_alpha_x_eV_per_A"]),
                        "alpha_y_eV_per_A": coupling_scale
                        * float(model["base_alpha_y_eV_per_A"]),
                    }
                )
    return tasks


def task_id(task: dict[str, Any]) -> str:
    """Return a filesystem-safe, deterministic ID for one branch task."""
    def token(value: float) -> str:
        return f"{value:.6f}".replace("-", "m").replace(".", "p")

    return (
        f"n{int(task['size']):03d}-g{token(float(task['coupling_scale']))}"
        f"-u{token(float(task['U_eV']))}-v{token(float(task['V1_eV']))}"
        f"-{task['branch']}"
    )


def classify_topology(record: dict[str, Any]) -> str:
    """Classify a converged pair state by observables, never by seed label."""
    local = (
        float(record["P_onsite"])
        + float(record["P_nn"])
        + float(record["P_diagonal"])
    )
    if local < 0.10 and float(record["mean_r"]) > 2.0:
        return "separated"
    channels = {
        "onsite": float(record["P_onsite"]),
        "intersite_x": float(record["P_nn_x"]),
        "intersite_y": float(record["P_nn_y"]),
        "diagonal": float(record["P_diagonal"]),
    }
    label = max(channels, key=channels.get)
    return label if channels[label] >= 0.25 else "mixed"


def summarize_campaign(
    manifest: dict[str, Any], records: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    """Aggregate all branch records into observable-classified parameter points."""
    validate_manifest(manifest)
    expected = {
        (task["size"], task["coupling_scale"], task["U_eV"], task["V1_eV"], task["branch"])
        for task in expand_tasks(manifest)
    }
    indexed: dict[tuple[int, float, float, float, str], dict[str, Any]] = {}
    for record in records:
        key = (
            int(record["size"]),
            float(record["coupling_scale"]),
            float(record["U_eV"]),
            float(record["V1_eV"]),
            str(record["branch"]),
        )
        if key in indexed:
            raise ValueError(f"duplicate S3 branch record: {key}")
        indexed[key] = dict(record)

    unexpected = set(indexed) - expected
    if unexpected:
        raise ValueError(f"unexpected S3 branch records: {sorted(unexpected)!r}")

    grouped: dict[tuple[int, float, float, float], list[dict[str, Any]]] = defaultdict(list)
    for key, record in indexed.items():
        grouped[key[:4]].append(record)

    criteria = manifest["classification"]
    tie = float(criteria["energy_tie_tolerance_eV"])
    robust = float(criteria["robust_binding_threshold_eV"])
    peierls_max = float(criteria["linear_peierls_ratio_max"])
    model = manifest["model"]
    isotropic_xy = bool(
        abs(float(model["Jx_eV"]) - float(model["Jy_eV"])) <= 1.0e-14
        and abs(
            float(model["base_alpha_x_eV_per_A"])
            - float(model["base_alpha_y_eV_per_A"])
        )
        <= 1.0e-14
    )
    points: list[dict[str, Any]] = []
    for point_key in sorted(grouped):
        branches = sorted(grouped[point_key], key=lambda item: str(item["branch"]))
        labels = {str(item["branch"]) for item in branches}
        complete = labels == set(BRANCHES)
        all_converged = complete and all(
            bool(item["converged"])
            and float(item["final_max_update_A"]) < 1.0e-8
            and float(item["final_max_gradient_eV_per_A"]) < 1.0e-6
            for item in branches
        )
        if not complete:
            points.append(
                {
                    "size": point_key[0],
                    "coupling_scale": point_key[1],
                    "U_eV": point_key[2],
                    "V1_eV": point_key[3],
                    "complete": False,
                    "all_branches_converged": False,
                    "classification": "incomplete",
                }
            )
            continue

        separated = next(item for item in branches if item["branch"] == "separated")
        best = min(branches, key=lambda item: float(item["total_energy_eV"]))
        binding = float(separated["total_energy_eV"]) - float(best["total_energy_eV"])
        raw_observable_topology = classify_topology(best)
        observable_topology = (
            "axial"
            if isotropic_xy
            and raw_observable_topology in {"intersite_x", "intersite_y"}
            else raw_observable_topology
        )
        if observable_topology == "separated":
            classification = "separated"
            binding_status = "not_applicable_separated_topology"
        elif binding <= tie:
            classification = "separated"
            binding_status = "unresolved_or_unbound"
        elif binding < robust:
            classification = f"marginal_{observable_topology}"
            binding_status = "marginal"
        else:
            classification = observable_topology
            binding_status = "robust"
        linear_peierls = bool(
            float(best["max_delta_tx_over_Jx"]) <= peierls_max
            and float(best["max_delta_ty_over_Jy"]) <= peierls_max
        )
        points.append(
            {
                "size": point_key[0],
                "coupling_scale": point_key[1],
                "U_eV": point_key[2],
                "V1_eV": point_key[3],
                "complete": True,
                "all_branches_converged": all_converged,
                "selected_seed": str(best["branch"]),
                "classification": classification,
                "observable_topology": observable_topology,
                "raw_observable_topology": raw_observable_topology,
                "binding_status": binding_status,
                "best_energy_eV": float(best["total_energy_eV"]),
                "separated_energy_eV": float(separated["total_energy_eV"]),
                "binding_vs_separated_eV": binding,
                "P_onsite": float(best["P_onsite"]),
                "P_nn_x": float(best["P_nn_x"]),
                "P_nn_y": float(best["P_nn_y"]),
                "P_diagonal": float(best["P_diagonal"]),
                "mean_r": float(best["mean_r"]),
                "max_delta_tx_over_Jx": float(best["max_delta_tx_over_Jx"]),
                "max_delta_ty_over_Jy": float(best["max_delta_ty_over_Jy"]),
                "linear_peierls_gate": linear_peierls,
                "quantitative_at_this_size": bool(all_converged and linear_peierls),
                "finite_size_status": "pending",
            }
        )

    return {
        "campaign_id": manifest["campaign_id"],
        "stage": manifest["stage"],
        "expected_branch_records": len(expected),
        "completed_branch_records": len(indexed),
        "complete": set(indexed) == expected,
        "point_count": len(points),
        "points": points,
        "interpretation_guard": {
            "seed_labels_are_not_phases": True,
            "finite_size_status_must_pass_before_boundary_promotion": True,
            "linear_peierls_gate_is_required_for_quantitative_use": True,
            "nonproduction_results_are_not_paper_data": manifest["stage"]
            in {"pilot", "screen"},
        },
    }
