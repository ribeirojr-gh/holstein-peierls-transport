# holstein-peierls-transport

Modern, reproducible implementation of the semiclassical two-dimensional
Holstein-Peierls model used to study polaron formation and charge transport in
molecular organic semiconductors.

## Status

The current `0.1.0a1` implementation covers **static polaron formation only**. It
reproduces the structure and historical update rules of the archived
`rprop.f90` program before performance optimizations or GPU support are added.
The dynamical `hp2D.f90` workflow will be migrated only after the static solver
has passed numerical regression tests against legacy outputs.

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

## Reproducibility policy

The original Fortran source and generated include files are archived separately
in Google Drive under `CODIGOS/holstein-peierls-transport/legacy`. They are not
silently edited. Historical behaviours that may look unusual are represented in
an explicit compatibility path and documented before any modernized alternative
is introduced.

## Development stages

1. Static CPU reference solver and legacy regression tests.
2. CPU optimization, sparse/operator formulation, and parameter sweeps.
3. Optional GPU backend.
4. Time-dependent polaron dynamics.
5. Disorder, finite temperature, transport observables, and publication-grade workflows.

## Quick start

```bash
python -m pip install -e .
hp-polaron --parameters parameters1.inc --solver dense_lowest --output run-static
```

Use `--solver dense_full` for the closest numerical analogue of the full LAPACK
diagonalization used by `rprop.f90`. The modern default requires `u`, `vx`, and
`vy` to converge. Use `--legacy-convergence` only for regression against the
historical `u`-only stopping rule.
