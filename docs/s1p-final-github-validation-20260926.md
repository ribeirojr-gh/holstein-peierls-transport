# S1P — final GitHub Actions provenance/environment audit (2026-09-26)

## Status

**Execution integrity: PASS.**  
**All 32 environment/source/branch calculations converged.**  
**Historical S0 benchmark reproduction: PASS.**  
**Numerical root-selection sensitivity: established.**

S1 remains formally failed. S1P is a provenance diagnosis and does not promote RPROP or any singlet-triplet splitting.

## Provenance

Workflow: `s1p-s0-provenance-environment-audit`  
Run ID: `36259083908`  
Workflow trigger commit: `04fa15f62bce2ffa75b807177891391ebeb9e1dd`  
Aggregate artifact: `s1p-aggregate-36259083908`  
Aggregate artifact ID: `10911528588`  
Aggregate SHA256: `783b826ca881de8f9ef0d36233aae358c58e170369b78160a127a39d7ca3e204`

The original historical benchmark artifacts from run `33652971202` were also independently recovered and inspected.

## Exact historical reproduction

Historical source commit:

`ac651b7bacd4e19a1bc8f2c7311354028c76b8dc`

Historical numerical environment recovered from the original CI logs:

- Python 3.12.14;
- NumPy 2.5.2;
- SciPy 1.18.1;
- no workflow-level OPENBLAS/OMP/MKL pinning.

The S1P historical-code + NumPy 2.5.2 + historical-unpinned environment reproduces the documented four reference energies with maximum absolute error `2.22e-16 eV`:

- singlet onsite: `1.448028500905174 eV`;
- singlet bond_x: `1.458848887171766 eV`;
- singlet bond_y: `1.4587349849069438 eV`;
- triplet onsite: `1.4467971868815868 eV`.

Therefore the historical `+1.231314 meV` control splitting was a genuine reproducible result of that numerical root-selection environment; it was not a transcription error.

## NumPy patch level is root-selecting

Changing only NumPy `2.5.2 -> 2.5.3` produces six preregistered singlet root changes above the `1e-5 eV` S1 scale.

Maximum observed change:

`1.207035885e-3 eV = 1.207036 meV`.

A particularly direct fixed-source, fixed-thread comparison is the historical-code, single-thread onsite singlet:

- NumPy 2.5.2: `1.448028500905174 eV`;
- NumPy 2.5.3: `1.4468214650197786 eV`.

The triplet reference remains stable.

Thus a patch-level NumPy/LAPACK canonical-basis change can select a different fully converged open-shell singlet root.

## Thread policy is also root-selecting

Changing only the BLAS thread policy produces six singlet changes above `1e-5 eV`, with the same maximum scale of about `1.207036 meV`.

Examples occur at fixed source and fixed NumPy version. Therefore pinning a library version alone is insufficient to define the singlet root.

The thread dependence is numerical root selection, not a physical effect or uncertainty.

## Source-label comparison

The mechanical S1P classifier also flags the source-label factor, with eight comparisons above `1e-5 eV`.

This label must **not** be interpreted as evidence that the physical spin-adapted equations changed. Between the historical S0 commit and the current branch, the relevant promoted preconditioned spin-adapted path is unchanged except for the additive S1 RPROP option/dispatch in `relaxation_control.py`; the underlying open-shell, orbital-optimization, Hamiltonian, parameter and lattice modules retain their historical numerical definitions.

Moreover, S1P current-source jobs checked out the live development branch, so documentation-only branch-head changes could yield different resolved SHAs across jobs while leaving scientific code unchanged. The source-label result is therefore confounded with independent-job numerical root selection and is not promoted as a causal software-change attribution.

The robust causal findings from S1P are the controlled NumPy and thread-policy comparisons.

## Mechanism

The electronic initializer currently obtains canonical starting orbitals from

`np.linalg.eigh(H)`.

The physical Hamiltonian may contain symmetry-related/degenerate subspaces even after the checkerboard control opens the HOMO-LUMO gap. Eigenvectors within such a subspace are not uniquely defined. LAPACK/library/thread details can rotate that basis without changing the one-body eigenvalues.

The subsequent nonlinear open-shell orbital optimization can then converge to distinct singlet stationary roots from those equally valid numerical bases.

All affected branches satisfy strict orbital and structural convergence gates. Therefore convergence is necessary but not sufficient for root identity.

## Consequence

The historical splitting

`E_S-E_T = +1.231314 meV`

and the S1 splittings are all environment-selected stationary-root results. None should be used as the Paper-1 production singlet-triplet value until the root-selection procedure itself is made deterministic and optimizer-independent.

Exact dependency/thread pinning remains necessary for reproducibility, but it is not a sufficient scientific root-selection rule.

## Next stage

Proceed with the preregistered

`docs/s1r-deterministic-singlet-root-manifold-design.md`.

S1R replaces the arbitrary unperturbed canonical basis by a finite predetermined deterministic orbital-root seed ensemble. Auxiliary seed perturbations generate initial orbitals only and are absent from every physical energy and force.

S1R must compare the global promoted minima from the complete predetermined singlet root ensemble between the preconditioned and RPROP structural optimizers under the original `1e-5 eV` optimizer-equivalence criterion.

The T=0 one-polaron frozen baseline remains unaffected.
