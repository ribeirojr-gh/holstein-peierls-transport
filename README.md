# holstein-peierls-transport

Modern, reproducible implementation of the semiclassical two-dimensional
Holstein-Peierls model used to study polaron formation and charge transport in
molecular organic semiconductors.

## Status

Version `0.3.0a1` retains the validated **static one-polaron** implementation
from `0.2.0a1` and adds the first experimental **correlated static two-particle**
solver for singlet bipolarons. The new two-particle path supports adiabatic
Holstein-Hubbard and Holstein-Peierls-Hubbard calculations, matrix-free
electronic operators, full `u`/`vx`/`vy` lattice relaxation, pair observables,
and strict residual-gradient convergence.

The existing one-polaron path remains unchanged and is still the stable
production interface. The two-particle implementation is intentionally isolated
under `holstein_peierls.two_particle` and should be treated as research code
while long-range screened Coulomb interactions, triplet states, and
material-specific parameterization are still under development.

Time-dependent `hp2D.f90` dynamics remain intentionally deferred. The current
research priority is to establish and validate static correlated two-particle
physics before returning to dynamical propagation.

## Scientific model

The static one-polaron calculation couples one electronic state per molecular
site to three classical lattice coordinates per site: the intramolecular
Holstein coordinate `u` and two intermolecular Peierls coordinates `vx` and
`vy`. Periodic boundary conditions are used in both lattice directions. The
lattice is relaxed with RPROP while the electronic state is kept in the
instantaneous ground state.

The experimental bipolaron extension replaces the one-particle electronic state
with a correlated singlet wavefunction `Psi(i,j)` and adds onsite Hubbard
repulsion. The spin-summed reduced one-particle density matrix has trace two and
provides the Holstein and Peierls lattice forces. The two-particle Hamiltonian is
applied matrix-free, avoiding construction of an `N^2 x N^2` dense matrix.

Key references underlying the archived one-polaron implementation include:

- E. Mozafari and S. Stafstrom, *Physics Letters A* **376**, 1807-1811 (2012).
- E. Mozafari and S. Stafstrom, *Journal of Chemical Physics* **138**, 184104 (2013).
- E. Mozafari, *A Theoretical Study of Charge Transport in Molecular Crystals*, Linkoping University (2012).

The working two-particle literature map and validation rationale are documented
in `docs/two-particle-literature-gap.md`, `docs/two-particle-static-theory.md`,
`docs/bipolaron-validation-plan.md`, and
`docs/bipolaron-validation-results.md`.

## Reproducibility and backups

The original Fortran source and generated include files are archived separately
in Google Drive under `CODIGOS/holstein-peierls-transport/legacy`. They are not
silently edited. Consolidated Python versions are additionally archived under
`CODIGOS/holstein-peierls-transport/releases`, so GitHub and Google Drive provide
independent copies of publication-relevant source snapshots.

Historical one-polaron behaviours that may look unusual are represented in an
explicit compatibility path and documented before any modernized alternative is
introduced. The experimental two-particle solver uses a non-backtracking RPROP
variant and is validated by stationary energies, residual gradients, analytic
limits, pair observables, and finite-size behavior rather than by reproducing
the historical RPROP iteration trajectory.

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

The CLI keeps `dense_lowest` as its default eigensolver; therefore choosing the
sparse iterative solver is explicit. The modern default stopping criterion
requires `u`, `vx`, and `vy` to converge. Use `--legacy-convergence` only when
reproducing the historical u-only stopping rule.

## Experimental static bipolaron solver

The two-particle API is currently Python-only and intentionally not exposed as a
stable CLI. Core objects are available under:

```python
from holstein_peierls.two_particle import (
    BipolaronParameters,
    pair_observables,
    relax_static_holstein_peierls_bipolaron,
)
```

The validated large-cell benchmark uses a strongly anisotropic molecular-crystal
reference set. At `40x40`, `alpha_x = 0.10 eV/angstrom` and
`alpha_y = 0.12 eV/angstrom`, the intersite-x state remains bound by about
`43.91 meV` at `U = 0.525 eV` and `6.81 meV` at `U = 1.0 eV`, while the
separated reference expands to a mean pair distance of about 28 sites.

These results demonstrate a numerically robust intersite-x bipolaron in the
validated model domain, but they do **not** yet constitute a material-specific
pentacene phase diagram. See `docs/bipolaron-validation-results.md` for the
controlled Peierls range, finite-size checks, and limitations.

## Validation and benchmarks

For the one-polaron solver, the optimized gradient is algebraically equivalent
to the reference expression but avoids the complete `N x N` density matrix.
Validation accounts explicitly for periodic translations of a localized polaron
and constant zero modes of the Peierls coordinates. See
`docs/cpu-optimization-v0.2.md` for the numerical comparison and reproducible
benchmark commands.

On the recorded GitHub CPU benchmark, the optimized one-polaron gradient is
approximately 13x, 60x, and 301x faster than the reference gradient for 20x20,
40x40, and 80x80 lattices, respectively. The sparse ground-state solve is
approximately 61x faster than `dense_lowest` for the tested 40x40 case. These
values are hardware- and library-dependent implementation benchmarks, not
universal performance claims.

For the two-particle solver, regression tests cover the noninteracting limit,
singlet exchange symmetry, reduced-density-matrix trace/Hermiticity, Hubbard
energy, analytic Holstein threshold, finite-difference Holstein/Peierls
gradients, zero-Peierls reduction, stationary residual gradients, periodic pair
observables, and finite-size localization. Research scans are retained under
`experiments/`; the strict large-cell benchmark is documented in
`docs/bipolaron-validation-results.md`.

## Development stages

1. Validated static one-polaron reference solver and legacy regression tests.
2. Optimized sparse CPU one-polaron implementation.
3. Experimental correlated static singlet bipolaron solver and validation.
4. Long-range screened Coulomb interaction, triplet sector, and material-specific bipolaron phase diagrams.
5. Correlated electron-hole exciton solver with separate HOMO/LUMO parameters.
6. Time-dependent dynamics, transport observables, disorder, and finite temperature.
7. Optional GPU acceleration where two-particle or dynamical workloads justify it.
