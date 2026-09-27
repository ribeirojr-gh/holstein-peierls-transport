# S2 local portability audit (2026-09-27)

## Classification

The canonical S2 GitHub Actions validation remains **PASS** at commit
`3e7eca4e215b8035926d4f165eb4551b70547868`.  The complete local fallback
executed successfully, but its locked aggregate classification is **FAIL**
because one exact spin-adapted regression anchor is not portable to the local
linear-algebra environment.

This result does not replace the canonical S2 decision and no acceptance
threshold or frozen reference was changed.

## Local execution

- artifact directory: `s2-local-validation/20260927T174620Z`;
- Python 3.12.3;
- NumPy 2.5.3;
- SciPy 1.18.1;
- `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`;
- 544 repository tests passed;
- execution integrity: PASS.

Sector results:

- one polaron: PASS;
- correlated bipolaron: PASS;
- spin-blind exciton: PASS;
- spin-adapted exact anchors: FAIL.

The only non-reproduced anchor was the preconditioned singlet with structural
seed `onsite` and root-seed ID 9:

- canonical GitHub energy: `1.446798604442264 eV`;
- local energy: `1.446810301237652 eV`;
- absolute difference: `1.1696795388e-5 eV`;
- local preconditioned/RPROP singlet difference: `1.3110851059e-5 eV`;
- locked optimizer-equivalence threshold: `1.0e-5 eV`.

The local branch nevertheless satisfied the electronic, structural,
particle-number, and finite-value convergence gates.  The other three
spin-adapted anchors reproduced their frozen values within the S2 tolerance.

## Exact-Python control

The sensitive branch was repeated with a locally managed Python 3.12.14
environment and the same pinned NumPy 2.5.3, SciPy 1.18.1, pytest 9.1.1, and
single-thread policy.  It returned exactly the same local energy,
`1.446810301237652 eV`.  The Python patch version is therefore excluded as the
cause.

The remaining sensitivity is consistent with the S1P diagnosis: the
infinitesimal auxiliary root splitter leaves very small spectral separations,
so different BLAS/LAPACK implementations or CPU code paths can orient an
initial near-degenerate subspace differently and select another strictly
converged singlet stationary root.

## Scientific handling

S2 deliberately reruns four exact environment-specific regression anchors;
it is not the Paper-1 production root search.  Production spin promotion
continues to require the complete predetermined S1R ensemble and selection of
the lowest valid stationary branch.  Consequently:

1. preserve this local FAIL as a portability result;
2. retain the successful canonical GitHub S2 PASS as the integration gate;
3. do not relabel root IDs, change frozen energies, or loosen tolerances;
4. classify S3 minima by physical observables rather than seed labels;
5. retain every converged branch so cross-machine root permutations remain
   auditable.

## Artifact integrity

- `aggregate/s2-aggregate.json` SHA-256:
  `36751a0e8c3c1f0d9426d1f6356007a58049dea9f8bcbbe997dd3c53c42f7678`;
- `summary.json` SHA-256:
  `35fc167d0aef6096321db726d5a98a700703e7541efea40c1dd178c4425f750d`;
- exact-Python branch JSON SHA-256:
  `bbf437e5ccdfd43f31a6f8a20290157fec335062390b1acae5a1a3afc0a4381b`.
