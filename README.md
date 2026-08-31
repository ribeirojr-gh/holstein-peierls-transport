# holstein-peierls-transport

Modern, reproducible implementation of the semiclassical two-dimensional
Holstein-Peierls model used to study polaron formation and charge transport in
molecular organic semiconductors.

## Status

Version `0.5.0a1` retains the validated **static one-polaron** implementation
from `0.2.0a1`, the correlated static singlet-bipolaron solver introduced in
`0.3.0a1`, and the extended-Hubbard nearest-neighbour interaction and
anisotropy/isotropy validation from `0.4.0a1`. It adds a validated, opt-in
**screened long-range Coulomb interaction** for distinct sites, with explicit
physical lattice spacings and dielectric screening, minimum-image periodic
geometry, a controlled short-range `V1` replacement rule, and an
infinite-separation dissociation reference based on `2 E_polaron`.

The correlated two-particle path supports adiabatic Holstein-Hubbard and
Holstein-Peierls-Hubbard calculations, `U + V1` interactions, screened
long-range offsite repulsion, matrix-free electronic operators, full
`u`/`vx`/`vy` lattice relaxation, onsite/axial/diagonal/separated branch
searches, pair observables, and strict residual-gradient convergence.

The existing one-polaron path remains unchanged and is still the stable
production interface. The two-particle implementation is isolated under
`holstein_peierls.two_particle` and should be treated as validated research code.
The long-range model in `0.5.0a1` is deliberately a general frozen-distance
screening model rather than a material-specific dielectric parameterization.
Triplet states, material-specific bipolaron phase diagrams, and direct
Coulomb-lattice force terms remain future extensions.

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

The bipolaron extension replaces the one-particle electronic state with a
correlated singlet wavefunction `Psi(i,j)`. The electronic interaction contains
onsite Hubbard repulsion `U`, an optional positive isotropic nearest-neighbour
repulsion `V1`, and an optional screened offsite continuum tail

`V(r) = e^2 / (4 pi epsilon_0 epsilon_r r)`.

The long-range interaction requires explicit positive `a_x`, `a_y`, and
`epsilon_r`. Periodic finite cells use minimum-image distances rather than an
Ewald sum. If a nonzero `V1` is combined with the long-range tail, `V1` replaces
the continuum value on the four cardinal nearest-neighbour bonds instead of
being added to it, avoiding implicit short-range double counting. In this first
validated implementation Coulomb distances are frozen to the equilibrium
lattice geometry, so the long-range interaction modifies structural forces only
through the correlated electronic state and does not yet add an explicit
electrostatic contribution to the Peierls-coordinate gradients.

The spin-summed reduced one-particle density matrix has trace two and provides
the Holstein and Peierls lattice forces. The two-particle Hamiltonian is applied
matrix-free, avoiding construction of an `N^2 x N^2` dense matrix.

Key references underlying the archived one-polaron implementation include:

- E. Mozafari and S. Stafstrom, *Physics Letters A* **376**, 1807-1811 (2012).
- E. Mozafari and S. Stafstrom, *Journal of Chemical Physics* **138**, 184104 (2013).
- E. Mozafari, *A Theoretical Study of Charge Transport in Molecular Crystals*, Linkoping University (2012).

The working two-particle theory and validation record is documented in
`docs/two-particle-literature-gap.md`, `docs/two-particle-static-theory.md`,
`docs/bipolaron-validation-plan.md`, `docs/bipolaron-validation-results.md`,
`docs/bipolaron-v1-validation.md`, `docs/bipolaron-isotropy-validation.md`, and
`docs/screened-long-range-coulomb.md`.

## Reproducibility and backups

The original Fortran source and generated include files are archived separately
in Google Drive under `CODIGOS/holstein-peierls-transport/legacy`. They are not
silently edited. Consolidated Python versions are additionally archived under
`CODIGOS/holstein-peierls-transport/releases`, so GitHub and Google Drive provide
independent copies of publication-relevant source snapshots.

Historical one-polaron behaviours that may look unusual are represented in an
explicit compatibility path and documented before any modernized alternative is
introduced. The two-particle solver uses a non-backtracking RPROP variant and is
validated by stationary energies, residual gradients, analytic limits, pair
observables, symmetry checks, interaction identities, and finite-size behavior
rather than by reproducing the historical RPROP iteration trajectory.

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

## Static bipolaron solver

The two-particle API is currently Python-only and intentionally not exposed as a
stable CLI. Core objects are available under:

```python
from holstein_peierls.two_particle import (
    BipolaronParameters,
    pair_observables,
    relax_static_holstein_peierls_bipolaron,
)
```

For the strongly anisotropic reference set `Jx = 0.100 eV`, `Jy = 0.015 eV`,
`alpha_x = 0.10 eV/angstrom`, and `alpha_y = 0.12 eV/angstrom`, strict `40x40`
calculations give the extended-Hubbard boundaries

- `U = 1.000 eV`: intersite-x -> separated at `V1 ~= 7.9113 meV`;
- `U = 0.525 eV`: intersite-x -> onsite at `V1 ~= 30.1867 meV`;
- `U = 0.525 eV`: onsite -> separated at `V1 ~= 185.4393 meV`.

The first `U = 0.525 eV` structural boundary is numerically resolved but is
borderline relative to the project's conservative 25% linear-Peierls working
criterion.

A bandwidth-matched fully isotropic control uses
`Jx = Jy = 0.0575 eV` and `alpha_x = alpha_y = 0.10 eV/angstrom`. In this
regime the x- and y-oriented axial states are rotationally degenerate and a new
controlled phase topology appears for `U = 1.0 eV`:

`axial bipolaron -> diagonal bipolaron -> separated polarons`,

with strict `40x40` boundaries at approximately `3.65887 meV` and
`15.97119 meV`. For `U = 0.525 eV`, the isotropic onsite state dissociates at
approximately `308.57480 meV`. These final isotropic boundaries remain inside
the conservative linear-Peierls window.

These short-range calculations demonstrate that hopping anisotropy changes the
stationary energy landscape and phase topology and that isotropy must be
separated from total-bandwidth changes. They do **not** constitute a
material-specific pentacene phase diagram.

### Screened long-range Coulomb validation

Version `0.5.0a1` adds a generic isotropic long-range control with

`Jx = Jy = 0.0575 eV`, `alpha_x = alpha_y = 0.10 eV/angstrom`,
`a_x = a_y = 7.0 angstrom`, and `V1 = 0`.

The correct infinite-separation reference is

`E_diss = 2 E_polaron`,

because a finite periodic `separated` branch retains a positive minimum-image
Coulomb repulsion. The calculated excess
`E_separated(L) - 2 E_polaron` scales as `1/(epsilon_r L)`; on the `40x40`
control cell the numerical coefficient agrees with the analytic point-charge
coefficient at about the 0.3% level.

Using `E_bind = 2 E_polaron - E_pair`, strict `40x40` pure-tail calculations
give the screening boundaries

- `U = 0.525 eV`, onsite branch: `epsilon_c = 6.89070`;
- `U = 1.000 eV`, intersite-x branch: `epsilon_c = 493.908`.

At the corresponding zero crossings, the cardinal continuum interaction scale
for `a = 7 angstrom` is approximately `0.2985 eV` for `U = 0.525 eV` and
`4.165 meV` for `U = 1.0 eV`. The diagonal intermediate phase observed in the
short-range `U + V1` isotropic model does not rescue binding near dissociation
for the pure `1/r` tail, because the continuum interaction also penalizes
diagonal separations.

All promoted large-cell results use `max structural update < 1e-8 angstrom`,
`max structural gradient < 1e-6 eV/angstrom`, and eigensolver tolerance
`1e-11`. These calculations validate the numerical long-range interaction model;
they do not assign a pentacene dielectric constant. See
`docs/screened-long-range-coulomb.md` for the complete finite-size and branch
record.

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
singlet exchange symmetry, reduced-density-matrix trace/Hermiticity, Hubbard and
nearest-neighbour interaction energies, the exact `dE/dV1 = P_NN`
Hellmann-Feynman identity, long-range interaction matrix elements and disabling
limits, analytic Holstein threshold, finite-difference Holstein/Peierls
gradients, zero-Peierls reduction, rotational symmetry, stationary residual
gradients, periodic pair observables, and finite-size localization. Research
scans are retained under `experiments/`; strict large-cell results are
documented under `docs/`.

## Development stages

1. Validated static one-polaron reference solver and legacy regression tests.
2. Optimized sparse CPU one-polaron implementation.
3. Correlated static singlet bipolaron solver and strict large-cell validation.
4. Extended-Hubbard nearest-neighbour repulsion and anisotropy/isotropy phase validation.
5. Screened long-range Coulomb interaction and finite-size dissociation validation.
6. Combined short-range effective screening plus continuum tail, triplet sector, and material-specific bipolaron phase diagrams.
7. Correlated electron-hole exciton solver with separate HOMO/LUMO parameters.
8. Time-dependent dynamics, transport observables, disorder, and finite temperature.
9. Optional GPU acceleration where two-particle or dynamical workloads justify it.
