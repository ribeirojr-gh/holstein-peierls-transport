# S3 — Paper-1 static parameter campaign design

Date: 2026-09-27  
Status: prospective pilot design; no pilot outcome was used to choose the grid.

## Scope

S3 begins with the correlated singlet bipolaron sector because it is the
Paper-1 sector with an explicit two-dimensional interaction plane.  The pilot
tests a compact `U`, nearest-neighbour `V1`, and global electron-phonon
coupling grid.  It is a workflow and topology pilot, not paper data.

The spin-blind exciton and spin-adapted S/T sectors retain their validated
controls during this pilot.  Their Paper-1 production extensions require
separate manifests because their Hilbert spaces and interaction conventions
must not be mixed with the equal-charge bipolaron phase map.

## Fixed geometry and interaction convention

The pilot uses the previously validated bandwidth-matched isotropic square
control:

- periodic 8x8 lattice for the pilot;
- `Jx = Jy = 0.0575 eV`;
- `K1 = 16.51 eV/A^2`, `K2 = 0.51 eV/A^2`;
- base `alpha_intra = 3.0 eV/A`;
- base `alpha_x = alpha_y = 0.10 eV/A`;
- onsite repulsion `U` plus isotropic nearest-neighbour repulsion `V1`;
- no continuum long-range tail in this first generic map.

The coupling coordinate `g` multiplies all three electron-phonon couplings,
preserving the fixed Holstein/Peierls and x/y ratios.  This makes `g` a global
coupling-strength coordinate rather than an anisotropy coordinate.

## Prospective pilot grid

- `U = {0.525, 1.000} eV`;
- `V1 = {0.000, 0.016, 0.320} eV`;
- `g = {0.8, 1.0}`;
- seeds: onsite, intersite-x, intersite-y, diagonal, separated.

The complete pilot contains 12 parameter points and 60 independently relaxed
branches.  All converged branches are retained.  Seed names describe only
initial conditions.

## Numerical gates

- RPROP coordinate update below `1e-8 A`;
- structural gradient below `1e-6 eV/A`;
- eigensolver tolerance `1e-11`;
- at most 1200 lattice iterations;
- single-thread BLAS policy;
- all five branches required for a complete point.

## Observable-based classification

The selected minimum is the lowest-energy valid branch.  Its phase label is
derived from final pair observables, not its seed:

- separated: total local probability below 0.10 and mean separation above two
  sites, or no energy gain above the separated branch within `1e-8 eV`;
- onsite, intersite-x, intersite-y, or diagonal: the corresponding probability
  channel is largest and at least 0.25; for the fixed isotropic Hamiltonian,
  x/y intersite orientations are reported as the single symmetry-equivalent
  `axial` phase;
- mixed: no local channel reaches 0.25;
- marginal: positive binding below 5 meV.

If the final observables describe a separated state, it remains classified as
separated even when it lies below the particular separated-seed stationary
solution in a small cell.  Such a seed-energy difference is retained for
diagnosis but is not interpreted as a pair-binding energy.

A point is quantitatively admissible at its current size only when all five
branches converge and the promoted minimum satisfies
`max |Delta t_mu| / |J_mu| <= 0.25` in both directions.

## Finite-size policy

Pilot classifications cannot be quoted as Paper-1 phase boundaries.  After
the pilot validates the workflow:

1. freeze a 20x20 production manifest with a refined compact grid;
2. identify every topology change or binding-sign bracket prospectively;
3. repeat the bracket branches on 40x40 cells;
4. promote a boundary only if the competing topologies remain converged,
   inside the linear-Peierls gate, and stable under the 20x20-to-40x40 check.

## Execution and storage

`scripts/run_s3_local_campaign.py` expands the JSON manifest into deterministic
task IDs and checkpoints every branch in its own directory.  An interrupted
run resumes from existing branch JSON files.  The aggregate stores the full
branch count, selected seed as metadata, observable topology, binding against
the same-cell separated branch, convergence gates, Peierls gate, and pending
finite-size status.

Each completed milestone is committed to the S3 GitHub branch.  Raw local run
artifacts are packaged with checksums and uploaded to the project's
`reference-runs` Google Drive hierarchy; generated run directories are not
committed to Git.

## Post-pilot 12x12 screen

After the preregistered pilot completed, the next prospective manifest was
frozen as `s3-paper1-bipolaron-screen-v1`.  It uses:

- `U = {0.525, 0.750, 1.000} eV`;
- `V1 = {0.000, 0.004, 0.016, 0.080, 0.320} eV`;
- `g = {0.8, 0.9, 1.0, 1.1}`;
- the same complete five-seed ensemble;
- a periodic 12x12 lattice.

This gives 60 parameter points and 300 independent branches.  The low-meV V1
values resolve the known isotropic axial/diagonal region, the intermediate U
value tests topology continuity, and the four coupling levels bracket the
weak-coupling separated pilot control.  The screen selects where to spend the
substantially larger 20x20 and 40x40 budgets; it is not paper data.

## Frozen 20x20 production map

The 12x12 screen selects a targeted 20x20 matrix in
`configs/s3-paper1-bipolaron-production-20x20-v1.json`.  Its ten parameter
slices cover 48 parameter points and 240 independently relaxed branches.
Each slice retains the complete five-seed ensemble.

The map refines the low-meV axial/diagonal boundaries, brackets the onsite
dissociation region at `U=0.525 eV`, retains the `g=1.1, U=0.75 eV` points
that failed the linear-Peierls gate at 12x12 for size diagnostics, and includes
one `g=0.8` separated control.  Every point remains subject to the same strict
stationarity and linear-Peierls criteria.  Surviving topology boundaries then
advance to 40x40; all other results remain finite-size-referenced 20x20 data.

## Targeted 40x40 finite-size campaign (completed)

The frozen manifest is
`configs/s3-paper1-bipolaron-finite-size-40x40-v1.json`.  It targets 16 points
around the surviving onsite, axial, diagonal, and separated boundaries, plus
the marginal-binding control rows.  At each point it repeats the observed
bound-topology seed and the same-cell separated seed (32 branch relaxations in
total). All tasks completed; the 16 observable classifications exactly match
20x20, all selected minima passed the linear-Peierls gate, and the maximum
absolute binding change was 0.2872 meV. It is explicitly a targeted finite-size
check, not a repeat of the global five-seed topology search. Detailed results
and limits are recorded in
`docs/s3-paper1-bipolaron-finite-size-40x40-results-20260930.md`.

The two campaign summaries are joined reproducibly by
`scripts/compare_s3_finite_size.py`. The 16 selected points are stable at the
sampled values and the 10 represented topology changes retain their brackets
across sizes; the transition locations inside those brackets remain unresolved.
Before any campaign is paused or closed, update the root `PROJECT_CHECKPOINT.md`
with its status, provenance, stored artifacts, and next action; see
`docs/project-checkpoint-practice.md`.
