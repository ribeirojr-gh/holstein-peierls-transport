# S1R — deterministic spin-adapted singlet root-manifold design

Date: 2026-09-26  
Status: prospective design. S1 remains formally failed. S1P provenance audit may still be running.

## Problem addressed

The canonical S0 electronic initializer diagonalizes the one-body Hamiltonian with `np.linalg.eigh`. The 4x4 checkerboard-gapped square control removes the HOMO-LUMO gap singularity but still contains symmetry-related/near-degenerate one-body subspaces. The eigenvector orientation inside such subspaces is not a physical observable and may depend on LAPACK/library/thread details.

A nonlinear open-shell orbital optimizer can map those numerically different canonical bases into different converged singlet SCF roots. The S1 result shows that strict convergence alone is therefore insufficient to define a production singlet state.

S1R replaces accidental canonical-basis selection by a **finite deterministic root-seed ensemble**. The infinitesimal seed operator is used only to generate starting orbitals; it is never present in the physical Hamiltonian, total energy, lattice force, or reported state.

## Fixed physical control

Retain the exact S1/S0 4x4 control:

- periodic isotropic square lattice;
- Jx=Jy=0.100 eV;
- alpha_intra=alpha_interx=alpha_intery=3.0 eV/A;
- K1=16.51 eV/A^2;
- K2=0.51 eV/A^2;
- U=0.525 eV;
- nearest-neighbour V=0.08 eV;
- checkerboard gap Delta=2.0 eV;
- half-filled neutral reference;
- singlet open-shell state for the primary root-manifold audit;
- structural seeds onsite, bond_x, bond_y with amplitude 1e-3 A;
- orbital-gradient tolerance 1e-8;
- structural update tolerance 1e-8 A;
- structural-gradient tolerance 1e-6 eV/A;
- maximum lattice iterations 2000.

Triplet is retained only as a stable control, not as the root-selection target.

## Deterministic orbital seed construction

Let H be the **physical** one-body matrix at a given lattice geometry.

For root-seed ID r=(dx,dy), with dx,dy in {0,1,2,3}, define a unique site-weight field on the 4x4 cell:

`w0(y,x) = ((4*y+x)+1)/17 - 0.5`.

Translate this fixed weight field periodically by (dx,dy) to obtain `w_r`.

Generate initial orbitals by diagonalizing only the auxiliary seed matrix

`H_seed(r) = H + eta * diag(w_r)`

with fixed

`eta = 1e-8 eV`.

Immediately after the initial orbitals are generated, discard `H_seed`. Every orbital optimization, energy, density matrix and force is evaluated with the unperturbed physical H.

The auxiliary spectrum must be finite and its minimum adjacent eigenvalue separation must exceed `1e-12 eV`. A seed failing this numerical uniqueness gate is invalid; eta is not adjusted post hoc.

The 16 periodic translations form the complete predetermined root-seed set. No additional root seed is added after outcomes are viewed.

## Electronic recovery rule

For an S1R branch, the root-seed ID is fixed for the entire structural trajectory.

At each geometry:

1. try the converged warm-start neutral/excited orbitals from the previous geometry;
2. if a warm state fails its electronic convergence gate, retry from the same deterministic `H_seed(r)` orbitals at the current geometry;
3. permit an increased iteration budget as in the existing recovery path;
4. do **not** fall back to the unperturbed arbitrary `np.linalg.eigh(H)` canonical basis.

Thus cold recovery remains tied to the preregistered root seed instead of reintroducing environment-dependent root selection.

## Structural optimizer matrix

Primary singlet manifold:

- 16 root-seed IDs;
- 3 structural seeds;
- 2 structural optimizers: preconditioned and RPROP.

Total: **96 independent singlet branches**.

Stable triplet control:

- root-seed ID (0,0) only;
- 3 structural seeds;
- both structural optimizers.

Total: **6 triplet branches**.

All branches run independently in GitHub Actions.

## Stored root signatures

Every converged branch stores:

- total referenced energy;
- electronic excitation and lattice energy;
- excitation density vector;
- excitation RDM;
- final u/vx/vy arrays;
- orbital-gradient diagnostics;
- structural update/gradient diagnostics;
- particle-number change;
- root-seed ID and structural seed;
- auxiliary seed-spectrum minimum gap;
- NumPy/SciPy/Python/thread provenance.

## Promotion rule

Within each multiplicity and structural optimizer, the promoted state is the **lowest-energy branch among the complete preregistered deterministic root-seed/structural-seed set that satisfies every strict convergence gate**.

Seed labels are initial conditions only. No state is promoted because it resembles a historical branch.

For the primary singlet comparison define:

`E_S^P,min` = lowest valid preconditioned singlet energy over all 48 branches;

`E_S^R,min` = lowest valid RPROP singlet energy over all 48 branches.

## Locked S1R acceptance criteria

S1R passes as a production root-promotion method only if all are true:

1. all 16 deterministic auxiliary orbital seeds satisfy the >1e-12 eV uniqueness gap at the initial geometries;
2. at least 45/48 singlet branches converge for each structural optimizer;
3. all six triplet control branches converge;
4. `|E_S^P,min-E_S^R,min| <= 1e-5 eV`;
5. promoted singlet particle-number changes are <1e-10;
6. promoted triplet minima from the two optimizers agree within 1e-5 eV;
7. all promoted states have final structural gradient <1e-6 eV/A and last update <1e-8 A;
8. full repository pytest suite passes.

The 45/48 completeness gate allows at most three root-seed/structural-seed failures per optimizer while retaining a broad predetermined root search. Failed branches remain in the artifact and are never replaced.

## Root-manifold diagnostics

In addition to the primary minima, report:

- sorted list of all converged singlet energies;
- multiplicities of energy clusters using a fixed 1e-5 eV clustering scale;
- which root-seed/structural-seed combinations enter each cluster;
- excitation-density and RDM distances between promoted RPROP and preconditioned minima;
- best symmetry-aligned lattice/density differences over the finite periodic square D4 + translation symmetry group;
- historical S0 energies and S1 minima as external regression markers only.

No historical energy is used as a target in the optimizer.

## Production-environment policy

S1P diagnoses why the old arbitrary canonical basis changed. S1R removes that arbitrary basis choice.

The S1R production workflow is prospectively pinned to:

- Python **3.12.14**;
- NumPy **2.5.3**;
- SciPy **1.18.1**;
- pytest **9.1.1** for the validation job;
- `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`.

These versions are fixed before any S1R branch result is generated. Pinning is a reproducibility control, not a substitute for deterministic root enumeration.

The validation job must also run a lightweight preflight for all 16 root-seed IDs on all three structural starting geometries and fail before the production matrix if any auxiliary spectrum has a minimum adjacent eigenvalue separation <= `1e-12 eV`.

## Stop rule

If S1R passes, the deterministic root ensemble becomes the production singlet promotion rule for Paper 1 and S2 may begin.

If S1R fails, do not loosen the 1e-5 eV optimizer-equivalence criterion or expand the root-seed set using observed outcomes. Retain the preconditioned solver as the reference and open a new prospective investigation of the remaining optimizer dependence.
