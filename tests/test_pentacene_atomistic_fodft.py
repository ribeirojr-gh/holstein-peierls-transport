from holstein_peierls.fhi_aims_fodft import (
    FinalFodftSelection,
    build_fodft_input_bundle,
)
from holstein_peierls.fhi_aims_fodft_scan import build_fodft_scan_manifest
from holstein_peierls.materials.pentacene_atomistic import (
    pentacene_293k_atomistic_dimer,
)
from holstein_peierls.materials.pentacene_g5e import pentacene_g5e_scan_plan


def _atom_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("atom ")]


def test_atomistic_pentacene_dimer_renders_36_plus_36_atoms_in_fodft_order() -> None:
    dimer = pentacene_293k_atomistic_dimer("diag_plus_AB_forward")
    # State 1 is a renderer-only placeholder in this test.  G5g explicitly does
    # not infer or promote production HOMO indices.
    bundle = build_fodft_input_bundle(
        dimer.reference,
        FinalFodftSelection(1, 1),
        root_name="atomistic_smoke",
    )
    fragment_a = _atom_lines(bundle.fragment_1_geometry_in)
    fragment_b = _atom_lines(bundle.fragment_2_geometry_in)
    combined = _atom_lines(bundle.dimer_geometry_in)
    assert len(fragment_a) == 36
    assert len(fragment_b) == 36
    assert len(combined) == 72
    assert combined == fragment_a + fragment_b


def test_one_atomistic_pentacene_family_generates_complete_36_job_g5e_g5f_manifest() -> None:
    dimer = pentacene_293k_atomistic_dimer("diag_minus_AB_forward")
    manifest = build_fodft_scan_manifest(
        dimer.family_label,
        dimer.reference,
        pentacene_g5e_scan_plan(),
        FinalFodftSelection(1, 1),
    )
    assert manifest.transfer_integral_job_count == 36
    assert manifest.conservative_fhi_aims_calculation_count == 108
    assert all(len(_atom_lines(job.bundle.dimer_geometry_in)) == 72 for job in manifest.jobs)
