"""Guard IP2a-4 against accidental changes to its locked production pilot."""

from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "ip2a4_refined_complete_calibration.py"


@pytest.mark.parametrize(
    ("option", "value", "diagnostic"),
    [
        ("--dt-fs", "0.2", "dt_fs=0.10 fs"),
        ("--size", "20", "lattice size and field"),
        ("--field-mv-per-A", "5.0", "lattice size and field"),
        ("--sample-interval-fs", "4.0", "event sampling is fixed"),
        ("--energy-sample-interval-fs", "20.0", "energy sampling is fixed"),
        ("--krylov-dimension", "8", "Krylov dimension 6"),
        ("--search-time-fs", "5000.0", "search_time_fs must remain 4000"),
    ],
)
def test_locked_ip2a4_protocol_rejects_changed_settings(
    tmp_path, option, value, diagnostic
):
    output = tmp_path / "should_not_exist.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            option,
            value,
            "--output",
            str(output),
            "--markdown",
            str(tmp_path / "unused.md"),
            "--arrays",
            str(tmp_path / "unused.npz"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert diagnostic in proc.stderr
    assert not output.exists()


def test_locked_ip2a4_rejects_changed_candidate_energies(tmp_path):
    output = tmp_path / "should_not_exist.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--candidate-energies-eV",
            "1e-5",
            "1e-4",
            "--output",
            str(output),
            "--markdown",
            str(tmp_path / "unused.md"),
            "--arrays",
            str(tmp_path / "unused.npz"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "candidate energies must match" in proc.stderr
    assert not output.exists()
