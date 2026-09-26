"""Locked S1R deterministic root-manifold aggregation and diagnostics."""

from __future__ import annotations

from math import isfinite

import numpy as np


OPTIMIZERS = ("preconditioned", "rprop")
STRUCTURAL_SEEDS = ("onsite", "bond_x", "bond_y")
SINGLET_ROOT_IDS = tuple(range(16))
ENERGY_EQUIVALENCE_EV = 1.0e-5
CLUSTER_SCALE_EV = 1.0e-5
MINIMUM_AUXILIARY_GAP_EV = 1.0e-12
MINIMUM_SINGLET_CONVERGED = 45
PARTICLE_NUMBER_TOLERANCE = 1.0e-10
UPDATE_TOLERANCE_A = 1.0e-8
GRADIENT_TOLERANCE_EV_PER_A = 1.0e-6


def _expected_keys() -> set[tuple[str, str, str, int]]:
    singlets = {
        (optimizer, "singlet", seed, root_id)
        for optimizer in OPTIMIZERS
        for seed in STRUCTURAL_SEEDS
        for root_id in SINGLET_ROOT_IDS
    }
    triplets = {
        (optimizer, "triplet", seed, 0)
        for optimizer in OPTIMIZERS
        for seed in STRUCTURAL_SEEDS
    }
    return singlets | triplets


def _record_key(record: dict) -> tuple[str, str, str, int]:
    return (
        str(record["structural_optimizer"]),
        str(record["multiplicity"]),
        str(record["structural_seed"]),
        int(record["root_seed_id"]),
    )


def _strict_valid(record: dict) -> bool:
    required = (
        "initial_auxiliary_gap_eV",
        "final_max_update_A",
        "final_max_gradient_eV_per_A",
        "total_referenced_energy_eV",
        "particle_number_change",
    )
    if not bool(record.get("completed", False)):
        return False
    if not bool(record.get("converged", False)):
        return False
    if not bool(record.get("neutral_orbitals_converged", False)):
        return False
    if not bool(record.get("excited_orbitals_converged", False)):
        return False
    try:
        values = [float(record[name]) for name in required]
    except (KeyError, TypeError, ValueError):
        return False
    if not all(isfinite(value) for value in values):
        return False
    return bool(
        float(record["initial_auxiliary_gap_eV"]) > MINIMUM_AUXILIARY_GAP_EV
        and float(record["final_max_update_A"]) < UPDATE_TOLERANCE_A
        and float(record["final_max_gradient_eV_per_A"])
        < GRADIENT_TOLERANCE_EV_PER_A
    )


def _record_sort_key(record: dict) -> tuple[float, int, int]:
    seed_order = STRUCTURAL_SEEDS.index(str(record["structural_seed"]))
    return (
        float(record["total_referenced_energy_eV"]),
        seed_order,
        int(record["root_seed_id"]),
    )


def _clusters(records: list[dict]) -> list[dict]:
    ordered = sorted(records, key=_record_sort_key)
    clusters: list[list[dict]] = []
    for record in ordered:
        energy = float(record["total_referenced_energy_eV"])
        if (
            not clusters
            or energy
            - float(clusters[-1][0]["total_referenced_energy_eV"])
            > CLUSTER_SCALE_EV
        ):
            clusters.append([record])
        else:
            clusters[-1].append(record)
    return [
        {
            "cluster_index": index,
            "minimum_energy_eV": float(group[0]["total_referenced_energy_eV"]),
            "maximum_energy_eV": float(group[-1]["total_referenced_energy_eV"]),
            "span_eV": float(
                group[-1]["total_referenced_energy_eV"]
                - group[0]["total_referenced_energy_eV"]
            ),
            "count": len(group),
            "members": [
                {
                    "structural_seed": str(record["structural_seed"]),
                    "root_seed_id": int(record["root_seed_id"]),
                    "energy_eV": float(record["total_referenced_energy_eV"]),
                }
                for record in group
            ],
        }
        for index, group in enumerate(clusters)
    ]


def _d4_matrices() -> tuple[np.ndarray, ...]:
    return tuple(
        np.asarray(matrix, dtype=np.int64)
        for matrix in (
            ((1, 0), (0, 1)),
            ((0, -1), (1, 0)),
            ((-1, 0), (0, -1)),
            ((0, 1), (-1, 0)),
            ((-1, 0), (0, 1)),
            ((1, 0), (0, -1)),
            ((0, 1), (1, 0)),
            ((0, -1), (-1, 0)),
        )
    )


def _transform_scalar(
    field: np.ndarray,
    matrix: np.ndarray,
    tx: int,
    ty: int,
) -> np.ndarray:
    values = np.asarray(field, dtype=np.float64)
    n_y, n_x = values.shape
    if n_x != n_y:
        raise ValueError("S1R D4 diagnostics require a square field")
    n = n_x
    out = np.empty_like(values)
    for y in range(n):
        for x in range(n):
            xp, yp = matrix @ np.asarray([x, y], dtype=np.int64)
            xp = int((xp + tx) % n)
            yp = int((yp + ty) % n)
            out[yp, xp] = values[y, x]
    return out


def _transform_vector_field(
    vx: np.ndarray,
    vy: np.ndarray,
    matrix: np.ndarray,
    tx: int,
    ty: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_values = np.asarray(vx, dtype=np.float64)
    y_values = np.asarray(vy, dtype=np.float64)
    if x_values.shape != y_values.shape or x_values.shape[0] != x_values.shape[1]:
        raise ValueError("S1R vector diagnostic requires matching square fields")
    n = x_values.shape[0]
    out_x = np.empty_like(x_values)
    out_y = np.empty_like(y_values)
    for y in range(n):
        for x in range(n):
            xp, yp = matrix @ np.asarray([x, y], dtype=np.int64)
            xp = int((xp + tx) % n)
            yp = int((yp + ty) % n)
            vector = matrix @ np.asarray(
                [x_values[y, x], y_values[y, x]], dtype=np.float64
            )
            out_x[yp, xp] = float(vector[0])
            out_y[yp, xp] = float(vector[1])
    return out_x, out_y


def _promoted_signature_diagnostics(
    preconditioned: dict,
    rprop: dict,
) -> dict:
    density_p = np.asarray(preconditioned["excitation_density"], dtype=np.float64)
    density_r = np.asarray(rprop["excitation_density"], dtype=np.float64)
    rdm_p = np.asarray(preconditioned["excitation_rdm"], dtype=np.float64)
    rdm_r = np.asarray(rprop["excitation_rdm"], dtype=np.float64)
    u_p = np.asarray(preconditioned["lattice_u_A"], dtype=np.float64)
    vx_p = np.asarray(preconditioned["lattice_vx_A"], dtype=np.float64)
    vy_p = np.asarray(preconditioned["lattice_vy_A"], dtype=np.float64)
    u_r = np.asarray(rprop["lattice_u_A"], dtype=np.float64)
    vx_r = np.asarray(rprop["lattice_vx_A"], dtype=np.float64)
    vy_r = np.asarray(rprop["lattice_vy_A"], dtype=np.float64)

    if density_p.shape != (4, 4) or density_r.shape != (4, 4):
        raise ValueError("S1R promoted density must be 4x4")
    if rdm_p.shape != (16, 16) or rdm_r.shape != (16, 16):
        raise ValueError("S1R promoted RDM must be 16x16")

    raw_density = density_r - density_p
    raw_rdm = rdm_r - rdm_p

    best_density: dict | None = None
    best_lattice: dict | None = None
    for matrix_index, matrix in enumerate(_d4_matrices()):
        for ty in range(4):
            for tx in range(4):
                density_t = _transform_scalar(density_r, matrix, tx, ty)
                density_delta = density_t - density_p
                density_rms = float(np.sqrt(np.mean(density_delta**2)))
                density_max = float(np.max(np.abs(density_delta)))
                if (
                    best_density is None
                    or (density_rms, density_max)
                    < (best_density["rms_difference"], best_density["max_abs_difference"])
                ):
                    best_density = {
                        "matrix_index": matrix_index,
                        "matrix": matrix.tolist(),
                        "translation_x": tx,
                        "translation_y": ty,
                        "rms_difference": density_rms,
                        "max_abs_difference": density_max,
                    }

                u_t = _transform_scalar(u_r, matrix, tx, ty)
                vx_t, vy_t = _transform_vector_field(vx_r, vy_r, matrix, tx, ty)
                lattice_delta = np.concatenate(
                    [
                        (u_t - u_p).reshape(-1),
                        (vx_t - vx_p).reshape(-1),
                        (vy_t - vy_p).reshape(-1),
                    ]
                )
                lattice_rms = float(np.sqrt(np.mean(lattice_delta**2)))
                lattice_max = float(np.max(np.abs(lattice_delta)))
                if (
                    best_lattice is None
                    or (lattice_rms, lattice_max)
                    < (best_lattice["rms_difference_A"], best_lattice["max_abs_difference_A"])
                ):
                    best_lattice = {
                        "matrix_index": matrix_index,
                        "matrix": matrix.tolist(),
                        "translation_x": tx,
                        "translation_y": ty,
                        "rms_difference_A": lattice_rms,
                        "max_abs_difference_A": lattice_max,
                    }

    assert best_density is not None and best_lattice is not None
    return {
        "raw_excitation_density_l2_difference": float(np.linalg.norm(raw_density)),
        "raw_excitation_density_max_abs_difference": float(
            np.max(np.abs(raw_density))
        ),
        "raw_excitation_rdm_frobenius_difference": float(np.linalg.norm(raw_rdm)),
        "best_D4_translation_density_alignment": best_density,
        "best_D4_translation_lattice_alignment": best_lattice,
    }


def _environment_pinned(record: dict) -> bool:
    environment = record.get("environment", {})
    threads = environment.get("thread_environment", {})
    return bool(
        environment.get("python") == "3.12.14"
        and environment.get("numpy") == "2.5.3"
        and environment.get("scipy") == "1.18.1"
        and threads.get("OPENBLAS_NUM_THREADS") == "1"
        and threads.get("OMP_NUM_THREADS") == "1"
        and threads.get("MKL_NUM_THREADS") == "1"
    )


def summarize_s1r(
    records: list[dict],
    *,
    full_pytest_passed: bool,
) -> dict:
    """Apply the complete preregistered S1R production gates."""
    if len(records) != 102:
        raise ValueError("S1R requires exactly 102 branch records")
    keys = [_record_key(record) for record in records]
    if set(keys) != _expected_keys() or len(set(keys)) != 102:
        raise ValueError("S1R records do not cover the exact preregistered design")

    index = {key: record for key, record in zip(keys, records, strict=True)}
    singlet_by_optimizer: dict[str, list[dict]] = {}
    triplet_by_optimizer: dict[str, list[dict]] = {}
    singlet_valid: dict[str, list[dict]] = {}
    triplet_valid: dict[str, list[dict]] = {}

    for optimizer in OPTIMIZERS:
        singlet_by_optimizer[optimizer] = [
            index[(optimizer, "singlet", seed, root_id)]
            for seed in STRUCTURAL_SEEDS
            for root_id in SINGLET_ROOT_IDS
        ]
        triplet_by_optimizer[optimizer] = [
            index[(optimizer, "triplet", seed, 0)]
            for seed in STRUCTURAL_SEEDS
        ]
        singlet_valid[optimizer] = [
            record
            for record in singlet_by_optimizer[optimizer]
            if _strict_valid(record)
        ]
        triplet_valid[optimizer] = [
            record
            for record in triplet_by_optimizer[optimizer]
            if _strict_valid(record)
        ]

    all_auxiliary_unique = all(
        bool(record.get("completed", False))
        and isfinite(float(record.get("initial_auxiliary_gap_eV", float("nan"))))
        and float(record["initial_auxiliary_gap_eV"]) > MINIMUM_AUXILIARY_GAP_EV
        for optimizer in OPTIMIZERS
        for record in singlet_by_optimizer[optimizer]
    )

    promoted_singlet = {
        optimizer: (
            None
            if not singlet_valid[optimizer]
            else min(singlet_valid[optimizer], key=_record_sort_key)
        )
        for optimizer in OPTIMIZERS
    }
    promoted_triplet = {
        optimizer: (
            None
            if not triplet_valid[optimizer]
            else min(triplet_valid[optimizer], key=_record_sort_key)
        )
        for optimizer in OPTIMIZERS
    }

    complete_promoted = all(
        promoted_singlet[optimizer] is not None
        and promoted_triplet[optimizer] is not None
        for optimizer in OPTIMIZERS
    )

    singlet_difference = None
    triplet_difference = None
    signature_diagnostics = None
    if complete_promoted:
        singlet_difference = abs(
            float(promoted_singlet["preconditioned"]["total_referenced_energy_eV"])
            - float(promoted_singlet["rprop"]["total_referenced_energy_eV"])
        )
        triplet_difference = abs(
            float(promoted_triplet["preconditioned"]["total_referenced_energy_eV"])
            - float(promoted_triplet["rprop"]["total_referenced_energy_eV"])
        )
        signature_diagnostics = _promoted_signature_diagnostics(
            promoted_singlet["preconditioned"],
            promoted_singlet["rprop"],
        )

    promoted_particle_ok = bool(
        complete_promoted
        and all(
            abs(float(promoted_singlet[optimizer]["particle_number_change"]))
            < PARTICLE_NUMBER_TOLERANCE
            for optimizer in OPTIMIZERS
        )
    )
    promoted_structural_ok = bool(
        complete_promoted
        and all(
            float(record["final_max_update_A"]) < UPDATE_TOLERANCE_A
            and float(record["final_max_gradient_eV_per_A"])
            < GRADIENT_TOLERANCE_EV_PER_A
            for optimizer in OPTIMIZERS
            for record in (
                promoted_singlet[optimizer],
                promoted_triplet[optimizer],
            )
        )
    )
    environment_pinned = all(
        _environment_pinned(record)
        for record in records
        if bool(record.get("completed", False))
    )

    gates = {
        "all_initial_auxiliary_seeds_unique_above_1e-12_eV": bool(
            all_auxiliary_unique
        ),
        "at_least_45_of_48_preconditioned_singlets_converged": bool(
            len(singlet_valid["preconditioned"]) >= MINIMUM_SINGLET_CONVERGED
        ),
        "at_least_45_of_48_rprop_singlets_converged": bool(
            len(singlet_valid["rprop"]) >= MINIMUM_SINGLET_CONVERGED
        ),
        "all_six_triplet_controls_converged": bool(
            len(triplet_valid["preconditioned"]) == 3
            and len(triplet_valid["rprop"]) == 3
        ),
        "promoted_singlet_minima_agree_within_1e-5_eV": bool(
            singlet_difference is not None
            and singlet_difference <= ENERGY_EQUIVALENCE_EV
        ),
        "promoted_singlet_particle_number_changes_below_1e-10": promoted_particle_ok,
        "promoted_triplet_minima_agree_within_1e-5_eV": bool(
            triplet_difference is not None
            and triplet_difference <= ENERGY_EQUIVALENCE_EV
        ),
        "promoted_states_pass_strict_structural_gates": promoted_structural_ok,
        "full_repository_pytest_passed": bool(full_pytest_passed),
        "production_environment_exactly_pinned": bool(environment_pinned),
    }

    def _small_record(record: dict | None) -> dict | None:
        if record is None:
            return None
        return {
            "structural_optimizer": str(record["structural_optimizer"]),
            "multiplicity": str(record["multiplicity"]),
            "structural_seed": str(record["structural_seed"]),
            "root_seed_id": int(record["root_seed_id"]),
            "energy_eV": float(record["total_referenced_energy_eV"]),
            "particle_number_change": float(record["particle_number_change"]),
            "final_max_update_A": float(record["final_max_update_A"]),
            "final_max_gradient_eV_per_A": float(
                record["final_max_gradient_eV_per_A"]
            ),
        }

    return {
        "record_count": len(records),
        "strict_valid_singlet_counts": {
            optimizer: len(singlet_valid[optimizer]) for optimizer in OPTIMIZERS
        },
        "strict_valid_triplet_counts": {
            optimizer: len(triplet_valid[optimizer]) for optimizer in OPTIMIZERS
        },
        "failed_or_invalid_records": [
            {
                "key": list(_record_key(record)),
                "completed": bool(record.get("completed", False)),
                "converged": bool(record.get("converged", False)),
                "execution_error": record.get("execution_error"),
            }
            for record in records
            if not _strict_valid(record)
        ],
        "promoted_singlet": {
            optimizer: _small_record(promoted_singlet[optimizer])
            for optimizer in OPTIMIZERS
        },
        "promoted_triplet": {
            optimizer: _small_record(promoted_triplet[optimizer])
            for optimizer in OPTIMIZERS
        },
        "promoted_singlet_energy_abs_difference_eV": singlet_difference,
        "promoted_triplet_energy_abs_difference_eV": triplet_difference,
        "singlet_energy_clusters": {
            optimizer: _clusters(singlet_valid[optimizer])
            for optimizer in OPTIMIZERS
        },
        "promoted_signature_diagnostics": signature_diagnostics,
        "gates": gates,
        "s1r_pass": bool(all(gates.values())),
        "interpretation_guard": {
            "s1_remains_failed_as_original_experiment": True,
            "s1p_environment_dependence_is_numerical_not_physical": True,
            "auxiliary_root_seed_absent_from_physical_energy_and_force": True,
            "cluster_scale_is_numerical_not_physical_uncertainty": True,
        },
    }
