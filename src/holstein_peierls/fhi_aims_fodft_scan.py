"""Deterministic manifests connecting G5e dimer scans to G5f FHI-aims inputs."""

from __future__ import annotations

from dataclasses import dataclass

from .dimer_finite_difference import (
    DimerPerturbation,
    FiniteDifferenceScanPlan,
    RigidDimerReference,
    prepare_scan_geometries,
)
from .fhi_aims_fodft import (
    FinalFodftSelection,
    FodftInputBundle,
    build_fodft_input_bundle,
    perturbation_job_tag,
)


@dataclass(frozen=True, slots=True)
class FodftScanJob:
    """One displaced geometry and its three-step FHI-aims FO-DFT input bundle."""

    family_label: str
    perturbation: DimerPerturbation
    bundle: FodftInputBundle


@dataclass(frozen=True, slots=True)
class FodftScanManifest:
    """All transfer-integral jobs for one explicitly oriented bond family."""

    family_label: str
    jobs: tuple[FodftScanJob, ...]

    def __post_init__(self) -> None:
        if not self.family_label:
            raise ValueError("family_label must be non-empty")
        if not self.jobs:
            raise ValueError("scan manifest must contain at least one job")
        if any(job.family_label != self.family_label for job in self.jobs):
            raise ValueError("all jobs must belong to the manifest family")
        roots = [job.bundle.layout.root_name for job in self.jobs]
        if len(roots) != len(set(roots)):
            raise ValueError("scan job root names must be unique")

    @property
    def transfer_integral_job_count(self) -> int:
        return len(self.jobs)

    @property
    def conservative_fhi_aims_calculation_count(self) -> int:
        """Count fragment1 + fragment2 + final calculations without reuse assumptions."""
        return sum(job.bundle.layout.calculation_count for job in self.jobs)


def build_fodft_scan_manifest(
    family_label: str,
    reference: RigidDimerReference,
    plan: FiniteDifferenceScanPlan,
    selection: FinalFodftSelection,
) -> FodftScanManifest:
    """Render one complete G5e scan for an explicitly oriented bond family."""
    prepared = prepare_scan_geometries(reference, plan)
    jobs = tuple(
        FodftScanJob(
            family_label=family_label,
            perturbation=perturbation,
            bundle=build_fodft_input_bundle(
                geometry,
                selection,
                root_name=perturbation_job_tag(family_label, perturbation),
            ),
        )
        for perturbation, geometry in prepared
    )
    return FodftScanManifest(family_label=family_label, jobs=jobs)
