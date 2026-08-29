# holstein-peierls-transport

Modern, reproducible implementation of the semiclassical two-dimensional
Holstein-Peierls model used to study polaron formation and charge transport in
molecular organic semiconductors.

## Status

The current stable release is `0.1.0a1` and covers **static polaron formation
only**. Development toward the optimized CPU solver is performed separately
before changes are consolidated into the next version. The dynamical `hp2D.f90`
workflow will be migrated only after the static solver remains numerically
consistent with the validated reference implementation.

## Scientific model

The static calculation couples one electronic state per molecular site to three
classical lattice coordinates per site: the intramolecular Holstein coordinate
`u` and two intermolecular Peierls coordinates `vx` and `vy`. Periodic boundary
conditions are used in both lattice directions. The lattice is relaxed with the
RPROP algorithm while the electronic state is kept in the instantaneous ground
state.

Key references underlying the archived implementation include:

- E. Mozafari and S. Stafstrom, *Physics Letters A* **376**, 1807-1811 (2012).
- E. Mozafari and S. Stafstrom, *Journal of Chemical Physics* **138**, 184104 (2013).
- E. Mozafari, *A Theoretical Study of Charge Transport in Molecular Crystals*, Linkoping University (2012).

## Reproducibility and backups

The original Fortran source and generated include files are archived separately
in Google Drive under `CODIGOS/holstein-peierls-transport/legacy`. They are not
silently edited. Consolidated Python versions are additionally archived under
`CODIGOS/holstein-peierls-transport/releases`, so GitHub and Google Drive provide
independent copies of publication-relevant source snapshots.

Historical behaviours that may look unusual are represented in an explicit
compatibility path and documented before any modernized alternative is
introduced.

## Getting the source

### Option A: GitHub ZIP download

For a private repository this is often the simplest route on a new computer.
Use **Code -> Download ZIP** in GitHub, extract the archive, and open a terminal
in the extracted repository root (the directory containing `pyproject.toml`).

### Option B: authenticated Git clone

A private repository requires GitHub authentication. If GitHub CLI is
installed, the recommended setup is:

```bash
gh auth login
gh repo clone ribeirojr-gh/holstein-peierls-transport
cd holstein-peierls-transport
```

A normal HTTPS `git clone` also works when Git Credential Manager or another
GitHub credential method has already been configured.

## Installation

Python 3.11 or newer is required. Creating an isolated virtual environment is
recommended.

Linux, macOS, or WSL:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pytest
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
```

## First static-polaron run

A legacy-format example input is included in the repository:

```bash
hp-polaron \
  --parameters examples/static_polaron/parameters1.inc \
  --solver dense_lowest \
  --output run-static
```

Use `--solver dense_full` for the closest numerical analogue of the full LAPACK
diagonalization used by `rprop.f90`. The modern default requires `u`, `vx`, and
`vy` to converge. Use `--legacy-convergence` only for regression against the
historical `u`-only stopping rule.

## Development stages

1. Static CPU reference solver and legacy regression tests.
2. CPU optimization, sparse/operator formulation, and parameter sweeps.
3. Optional GPU backend.
4. Time-dependent polaron dynamics.
5. Disorder, finite temperature, transport observables, and publication-grade workflows.
