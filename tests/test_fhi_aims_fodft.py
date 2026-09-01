import numpy as np
import pytest

from holstein_peierls.dimer_finite_difference import (
    DimerPerturbation,
    RigidDimerReference,
    RigidMoleculeGeometry,
)
from holstein_peierls.fhi_aims_fodft import (
    FinalFodftSelection,
    FodftJobLayout,
    build_fodft_input_bundle,
    g5f_output_parser_status,
    perturbation_job_tag,
    render_dimer_geometry_in,
    render_final_fodft_control_block,
    render_fragment_fodft_control_block,
    render_molecule_geometry_in,
)


def _molecule_a() -> RigidMoleculeGeometry:
    return RigidMoleculeGeometry(
        symbols=("C", "H"),
        coordinates_angstrom=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        pivot_angstrom=(0.0, 0.0, 0.0),
    )


def _molecule_b() -> RigidMoleculeGeometry:
    return RigidMoleculeGeometry(
        symbols=("N", "H", "H"),
        coordinates_angstrom=((3.0, 0.0, 0.0), (3.0, 1.0, 0.0), (3.0, 0.0, 1.0)),
        pivot_angstrom=(3.0, 0.0, 0.0),
    )


def _reference() -> RigidDimerReference:
    return RigidDimerReference(
        molecule_a=_molecule_a(),
        molecule_b=_molecule_b(),
        local_axes_global=tuple(map(tuple, np.eye(3).tolist())),
    )


def test_fragment_geometry_renderer_preserves_atomic_order_and_precision() -> None:
    rendered = render_molecule_geometry_in(_molecule_a()).splitlines()
    assert rendered[1] == "atom 0.000000000000 0.000000000000 0.000000000000 C"
    assert rendered[2] == "atom 1.000000000000 0.000000000000 0.000000000000 H"


def test_dimer_geometry_is_exact_fragment_a_then_fragment_b_concatenation() -> None:
    rendered = render_dimer_geometry_in(_reference()).splitlines()
    atom_lines = [line for line in rendered if line.startswith("atom ")]
    expected_a = [
        line
        for line in render_molecule_geometry_in(_molecule_a()).splitlines()
        if line.startswith("atom ")
    ]
    expected_b = [
        line
        for line in render_molecule_geometry_in(_molecule_b()).splitlines()
        if line.startswith("atom ")
    ]
    assert atom_lines == expected_a + expected_b
    assert rendered.index("# fragment 1 / molecule A") < rendered.index(
        "# fragment 2 / molecule B"
    )


def test_fragment_control_block_uses_h2n_default_and_not_deltaplus() -> None:
    block = render_fragment_fodft_control_block()
    assert block == "fo_dft fragment\nfo_flavour default\n"
    assert "fo_deltaplus" not in block
    assert "reset" not in block


def test_final_control_block_contains_only_documented_hole_selection_tags() -> None:
    selection = FinalFodftSelection(51, 51)
    block = render_final_fodft_control_block(selection)
    assert block.splitlines() == [
        "fo_dft final",
        "fo_flavour default",
        "fo_folders frag1 frag2",
        "fo_orbitals 51 51 1 1 hole",
        "fo_verbosity 1",
    ]
    assert "fo_deltaplus" not in block


def test_final_selection_rejects_invalid_orbital_indices_ranges_and_verbosity() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        FinalFodftSelection(0, 10)
    with pytest.raises(ValueError, match="positive integer"):
        FinalFodftSelection(10, 10, range_fragment_1=0)
    with pytest.raises(ValueError, match="verbosity"):
        FinalFodftSelection(10, 10, verbosity=3)
    with pytest.raises(ValueError, match="transport_type"):
        FinalFodftSelection(10, 10, transport_type="proton")


def test_folder_layout_rejects_path_traversal_and_duplicate_calculation_names() -> None:
    with pytest.raises(ValueError, match="folder names"):
        FodftJobLayout("../unsafe")
    with pytest.raises(ValueError, match="distinct"):
        FodftJobLayout("root", fragment_1_folder="frag", fragment_2_folder="frag")


def test_perturbation_job_tag_is_deterministic_and_path_safe() -> None:
    perturbation = DimerPerturbation("translation_long", -0.0025, "angstrom")
    first = perturbation_job_tag("diag_plus_AB_forward", perturbation)
    second = perturbation_job_tag("diag_plus_AB_forward", perturbation)
    assert first == second == "diag_plus_AB_forward__translation_long__minus__0p0025A"
    assert "/" not in first
    assert " " not in first


def test_input_bundle_exposes_partial_control_blocks_not_fake_complete_control_files() -> None:
    bundle = build_fodft_input_bundle(
        _reference(), FinalFodftSelection(20, 21), root_name="bond_scan_001"
    )
    files = bundle.file_map()
    assert len(files) == 6
    assert "bond_scan_001/frag1/fodft.control" in files
    assert "bond_scan_001/frag2/fodft.control" in files
    assert "bond_scan_001/dimer/fodft.control" in files
    assert not any(path.endswith("/control.in") for path in files)
    assert bundle.layout.calculation_count == 3


def test_bundle_fragment_geometries_match_dimer_fragments_in_exact_order() -> None:
    bundle = build_fodft_input_bundle(
        _reference(), FinalFodftSelection(20, 20), root_name="ordered"
    )
    fragment_a_atoms = [
        line for line in bundle.fragment_1_geometry_in.splitlines() if line.startswith("atom ")
    ]
    fragment_b_atoms = [
        line for line in bundle.fragment_2_geometry_in.splitlines() if line.startswith("atom ")
    ]
    dimer_atoms = [
        line for line in bundle.dimer_geometry_in.splitlines() if line.startswith("atom ")
    ]
    assert dimer_atoms == fragment_a_atoms + fragment_b_atoms


def test_output_parser_is_explicitly_deferred_until_exact_format_is_known() -> None:
    status, reason = g5f_output_parser_status()
    assert status == "deferred"
    assert "full_hab_submatrix" in reason
    assert "exact text format" in reason
