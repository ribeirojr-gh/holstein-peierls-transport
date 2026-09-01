import math

import numpy as np

from holstein_peierls.dimer_finite_difference import (
    FiniteDifferenceScanPlan,
    RigidDimerReference,
    RigidMoleculeGeometry,
)
from holstein_peierls.fhi_aims_fodft import FinalFodftSelection
from holstein_peierls.fhi_aims_fodft_scan import build_fodft_scan_manifest


def _reference() -> RigidDimerReference:
    molecule_a = RigidMoleculeGeometry(
        symbols=("C",),
        coordinates_angstrom=((0.0, 0.0, 0.0),),
        pivot_angstrom=(0.0, 0.0, 0.0),
    )
    molecule_b = RigidMoleculeGeometry(
        symbols=("C", "H"),
        coordinates_angstrom=((3.0, 0.0, 0.0), (4.0, 0.0, 0.0)),
        pivot_angstrom=(3.0, 0.0, 0.0),
    )
    return RigidDimerReference(
        molecule_a=molecule_a,
        molecule_b=molecule_b,
        local_axes_global=tuple(map(tuple, np.eye(3).tolist())),
    )


def _plan() -> FiniteDifferenceScanPlan:
    return FiniteDifferenceScanPlan(
        (0.0025, 0.005, 0.01),
        tuple(math.radians(value) for value in (0.25, 0.5, 1.0)),
    )


def test_one_oriented_bond_manifest_has_36_coupling_jobs_and_108_calculations() -> None:
    manifest = build_fodft_scan_manifest(
        "diag_plus_AB_forward",
        _reference(),
        _plan(),
        FinalFodftSelection(50, 50),
    )
    assert manifest.transfer_integral_job_count == 36
    assert manifest.conservative_fhi_aims_calculation_count == 108
    assert len({job.bundle.layout.root_name for job in manifest.jobs}) == 36


def test_manifest_geometry_tracks_each_perturbation_in_final_and_fragment_two() -> None:
    manifest = build_fodft_scan_manifest(
        "a_AA", _reference(), _plan(), FinalFodftSelection(20, 20)
    )
    job = next(
        item
        for item in manifest.jobs
        if item.perturbation.coordinate_label == "translation_long"
        and item.perturbation.amount == 0.0025
    )
    fragment_b_atoms = [
        line
        for line in job.bundle.fragment_2_geometry_in.splitlines()
        if line.startswith("atom ")
    ]
    dimer_atoms = [
        line for line in job.bundle.dimer_geometry_in.splitlines() if line.startswith("atom ")
    ]
    assert "3.002500000000" in fragment_b_atoms[0]
    assert dimer_atoms[-2:] == fragment_b_atoms
