# holstein-peierls-transport

Modern, reproducible implementation of the semiclassical two-dimensional
Holstein-Peierls model used to study polaron formation and charge transport in
molecular organic semiconductors.

## Status

Version `0.2.0a1` covers **static polaron formation** and introduces an optimized
CPU path while preserving the validated `0.1.0a1` reference behavior. The
dynamical `hp2D.f90` workflow remains intentionally deferred until the static
solver and its performance-oriented reformulations are fully consolidated.

The optimized implementation now provides a direct sparse Hamiltonian, an O(N)
analytical gradient, iterative ground-state solution, and reuse of electronic
states between RPROP iterations. A strict reference path remains available for
regression against the archived Fortran implementation.

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

## Recommended static-polaron run

The optimized sparse CPU path is requested explicitly:

```bash
hp-polaron \
  --parameters examples/static_polaron/parameters1.inc \
  --solver sparse \
  --gradient optimized \
  --output run-static
```

For the closest numerical analogue of the archived static Fortran program, use:

```bash
hp-polaron \
  --parameters examples/static_polaron/parameters1.inc \
  --solver dense_full \
  --gradient reference \
  --legacy-convergence \
  --output run-reference
```

The CLI keeps `dense_lowest` as its default eigensolver in `0.2.0a1`; therefore,
choosing the sparse iterative solver is explicit. The modern default stopping
criterion requires `u`, `vx`, and `vy` to converge. Use
`--legacy-convergence` only when reproducing the historical u-only stopping
rule.

## Validation and benchmarks

The optimized gradient is algebraically equivalent to the reference expression
but avoids the complete N x N density matrix. Validation accounts explicitly
for periodic translations of a localized polaron and constant zero modes of the
Peierls coordinates. See `docs/cpu-optimization-v0.2.md` for the numerical
comparison and reproducible benchmark commands.

On the current GitHub CPU benchmark, the optimized gradient is approximately
13x, 60x, and 301x faster than the reference gradient for 20x20, 40x40, and
80x80 lattices, respectively. The sparse ground-state solve is approximately
61x faster than `dense_lowest` for the tested 40x40 case. These values are
hardware- and library-dependent and are reported as implementation benchmarks,
not universal performance claims.

## Development stages

1. Static CPU reference solver and legacy regression tests.
2. CPU optimization, sparse/operator formulation, and parameter sweeps.
3. Optional GPU backend.
4. Time-dependent polaron dynamics.
5. Disorder, finite temperature, transport observables, and publication-grade workflows.
