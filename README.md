# holstein-peierls-transport

Modern, reproducible implementation of the semiclassical two-dimensional
Holstein-Peierls model used to study charge localization and transport in
molecular organic semiconductors.

## Status

Version `0.7.0a1` closes the current **generic static-physics** phase. The
repository now contains validated stationary implementations for:

- the optimized one-polaron Holstein-Peierls problem;
- the correlated static singlet bipolaron with onsite, shell-resolved
  short-range, and screened long-range repulsion; and
- a distinguishable electron-hole exciton with independent electron/hole
  parameters, attractive interactions, and full shared-lattice relaxation.

The one-polaron path remains the stable production interface. The bipolaron and
exciton implementations are validated research-code APIs under
`holstein_peierls.two_particle` and `holstein_peierls.exciton`, respectively.
The interaction models remain deliberately general and are not assigned to a
specific molecular crystal unless a later material layer explicitly supplies
such parameters.

The static exciton milestone uses an ordered normalized pair wavefunction
`Psi[i_e,i_h]`, separate electron and hole reduced density matrices, a
matrix-free two-particle Hamiltonian, independent electron/hole hopping and
electron-phonon parameters, attractive onsite/nearest-neighbour/long-range
interactions, RPROP relaxation of `u`/`vx`/`vy`, and competing Frenkel,
CT-x, CT-y, diagonal, and separated seeds. Near-degenerate minima are grouped
with an explicit energy tolerance rather than being relabelled by floating-point
noise.

Time-dependent dynamics remain intentionally outside `0.7.0a1`. The next phase
will begin with an audit and benchmark of the archived dynamics, especially the
cost of repeated Hamiltonian eigensolutions, the electronic propagator, the
classical lattice integrator, and the finite-temperature model. No production
dynamical method is selected by this release.

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
repulsion `V1`, explicit optional shell-resolved short-range replacements, and
an optional screened offsite continuum tail

`V(r) = e^2 / (4 pi epsilon_0 epsilon_r r)`.

The long-range interaction requires explicit positive `a_x`, `a_y`, and
`epsilon_r`. Periodic finite cells use minimum-image distances rather than an
Ewald sum. If a nonzero `V1` is combined with the long-range tail, `V1` replaces
the continuum value on the four cardinal nearest-neighbour bonds instead of
being added to it, avoiding implicit short-range double counting. More general
short-range replacements are supplied explicitly by minimum-image shell through
`short_range_shell_overrides`; outside those selected shells the continuum tail
is preserved. Ambiguous double definitions of the same shell are rejected.

The spin-summed bipolaron reduced one-particle density matrix has trace two and
provides the Holstein and Peierls lattice forces. The two-particle Hamiltonian is
applied matrix-free, avoiding construction of an `N^2 x N^2` dense matrix.

The exciton extension instead uses a distinguishable ordered pair
`Psi[i_e,i_h]` with unit normalization. For a fixed lattice,

`H_X = H_e(q) tensor I + I tensor H_h(q) + V_eh`.

Electron and hole parameters are independent in the API even when a symmetric
control is used. Their reduced density matrices each have trace one and both
contribute to the shared Holstein-Peierls lattice forces. User-facing
interaction parameters are positive attraction magnitudes and enter the
Hamiltonian with a negative sign. The exciton interaction layer supports onsite,
cardinal nearest-neighbour, minimum-image screened `1/r`, and shell-specific
replacement controls.

In the validated static long-range models, Coulomb distances are frozen to the
equilibrium lattice geometry. Therefore the interaction changes structural
forces through the electronic state but does not yet add an explicit
electrostatic contribution to the Peierls-coordinate gradients.

Key references underlying the archived one-polaron implementation include:

- E. Mozafari and S. Stafstrom, *Physics Letters A* **376**, 1807-1811 (2012).
- E. Mozafari and S. Stafstrom, *Journal of Chemical Physics* **138**, 184104 (2013).
- E. Mozafari, *A Theoretical Study of Charge Transport in Molecular Crystals*, Linkoping University (2012).

The static theory and validation records are documented under `docs/`, including
`two-particle-static-theory.md`, `screened-long-range-coulomb.md`,
`combined-screening-model.md`, `combined-screening-range-sensitivity.md`,
`static-exciton-reference-model.md`, `static-exciton-validation-results.md`, and
`release-v0.7.0a1.md`.

## Reproducibility and backups

The original Fortran source and generated include files are archived separately
in Google Drive under `CODIGOS/holstein-peierls-transport/legacy`. They are not
silently edited. Consolidated Python versions are additionally archived under
`CODIGOS/holstein-peierls-transport/releases`, so GitHub and Google Drive provide
independent copies of publication-relevant source snapshots.

Historical one-polaron behaviours that may look unusual are represented in an
explicit compatibility path and documented before any modernized alternative is
introduced. The correlated static solvers are validated by stationary energies,
residual gradients, analytic limits, observables, symmetry checks, interaction
identities, and finite-size behavior rather than by reproducing every historical
RPROP iteration.

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

A bandwidth-matched fully isotropic control uses
`Jx = Jy = 0.0575 eV` and `alpha_x = alpha_y = 0.10 eV/angstrom`. In this
regime the x- and y-oriented axial states are rotationally degenerate and a new
controlled phase topology appears for `U = 1.0 eV`:

`axial bipolaron -> diagonal bipolaron -> separated polarons`,

with strict `40x40` boundaries at approximately `3.65887 meV` and
`15.97119 meV`. For `U = 0.525 eV`, the isotropic onsite state dissociates at
approximately `308.57480 meV`.

These short-range calculations demonstrate that hopping anisotropy changes the
stationary energy landscape and phase topology. They do **not** constitute a
material-specific pentacene phase diagram.

### Screened long-range Coulomb validation

Version `0.5.0a1` added a generic isotropic long-range control with

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
`4.165 meV` for `U = 1.0 eV`.

All promoted large-cell results use `max structural update < 1e-8 angstrom`,
`max structural gradient < 1e-6 eV/angstrom`, and eigensolver tolerance
`1e-11`. These calculations validate the numerical long-range interaction model;
they do not assign a pentacene dielectric constant.

### Combined short-/long-range screening validation

Version `0.6.0a1` generalizes the short-range treatment through explicit shell
replacements while retaining the continuum tail elsewhere. A controlled generic
range study assigned every replaced shell

`V_short(r) = eta V_cont(r)`

with `eta = 0.75`. Three replacement ranges were compared:

- `R1`: cardinal nearest neighbours;
- `R2`: `R1` plus the first diagonal shell;
- `R3`: `R2` plus the second axial shell.

Strict `40x40` interpolation gives

| U (eV) | R1 | R2 | R3 |
|---:|---:|---:|---:|
| 0.525 | 5.207723 | 5.177128 | 5.169438 |
| 1.000 | 378.843 | 373.585 | 371.815 |

The range correction is hierarchical: the cardinal shell accounts for most of
the shift, the diagonal correction is smaller, and the second axial correction
is smaller again. Relative to the validated pure-continuum boundaries,
`epsilon_c(R3)/epsilon_c(R0)` is `0.75021` for `U=0.525 eV` and `0.75280` for
`U=1.000 eV`, essentially the imposed `eta=0.75`.

The R1 -> R2 -> R3 hierarchy is sufficiently converged that a generic R4
extension was not justified.

## Static exciton solver

Version `0.7.0a1` adds the generic distinguishable electron-hole sector. The
core API is available under:

```python
from holstein_peierls.exciton import (
    ExcitonParameters,
    exciton_observables,
    relax_exciton_branches,
    relax_static_exciton,
)
```

The reference control uses equal electron/hole one-particle parameters only for
validation. The API itself permits independent hopping, Holstein, and Peierls
couplings for the two carriers.

The binding convention is

`E_bind^X = E_e-pol + E_h-pol - E_X`,

so positive values denote a bound relaxed exciton relative to separately
relaxed electron and hole polarons.

Strict benchmark calculations were performed at `6x6`, `10x10`, and `20x20`.
For the final `20x20` generic control with `Jx=0.100 eV`, `Jy=0.015 eV`,
`alpha_1=3.0 eV/angstrom`, `alpha_2x=alpha_2y=0.4 eV/angstrom`, and a
`0.525 eV` onsite attraction used only as a numerical control scale, the
Frenkel, CT-x, CT-y, and separated seeds converge to the same lowest-energy
minimum within approximately `1e-14 eV`.

That minimum has approximately:

- `E_X = -1.695788941148 eV`;
- `E_bind = 0.884870826 eV`;
- onsite pair probability `0.86043`;
- electron and hole IPR `0.85456`; and
- mean electron-hole separation `0.14360` lattice sites.

The diagonal seed converges to a distinct metastable exciton about `0.41327 eV`
above the common minimum. These values characterize the generic validation
Hamiltonian and are **not** material predictions.

See `docs/static-exciton-reference-model.md`,
`docs/static-exciton-validation-results.md`, and `docs/release-v0.7.0a1.md`.
The reproducible benchmark driver is
`experiments/exciton_reference_branch_benchmark.py`.

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

For the bipolaron solver, regression tests cover the noninteracting limit,
singlet exchange symmetry, reduced-density-matrix trace/Hermiticity, Hubbard and
nearest-neighbour interaction energies, the exact `dE/dV1 = P_NN`
Hellmann-Feynman identity, shell-resolved replacement semantics, long-range
interaction matrix elements and disabling limits, analytic Holstein threshold,
finite-difference Holstein/Peierls gradients, zero-Peierls reduction, rotational
symmetry, stationary residual gradients, periodic pair observables, and
finite-size localization.

For the exciton solver, tests cover the noninteracting Kronecker-sum limit,
unit-trace electron/hole reduced density matrices, atomic Frenkel and
nearest-neighbour CT limits, equal-carrier density symmetry, interaction
replacement semantics, seed topology, finite-difference validation of all three
lattice-coordinate gradients, strict stationary relaxation, binding-energy sign,
and safe branch selection. The final pre-merge suite passed with `154` tests.

## Development stages

Completed static milestones:

1. Validated static one-polaron reference solver and legacy regression tests.
2. Optimized sparse CPU one-polaron implementation.
3. Correlated static singlet bipolaron solver and strict large-cell validation.
4. Extended-Hubbard nearest-neighbour repulsion and anisotropy/isotropy phase validation.
5. Screened long-range Coulomb interaction and finite-size dissociation validation.
6. Shell-resolved combined short-range effective screening plus continuum-tail range validation.
7. Distinguishable electron-hole static exciton with shared-lattice relaxation and strict 20x20 branch validation.

Next phases, deliberately not fixed by `0.7.0a1`:

8. Audit and benchmark the archived dynamics: repeated diagonalization, electronic propagator, lattice integrator, norm/energy conservation, and finite-temperature implementation.
9. Select a reference dynamical algorithm only after controlled accuracy/stability/cost comparisons, including direct time-propagation alternatives that may avoid full diagonalization at every step.
10. Implement and validate unified dynamics for polaron, bipolaron, and exciton, then add electric-field transport observables.
11. Reassess finite-temperature treatment against modern alternatives before production finite-temperature simulations.
12. Add material-specific parameterization, disorder, triplet/exchange physics, and other material layers when justified by the target study.
13. Add multithreaded/GPU acceleration where profiling demonstrates a meaningful benefit.
