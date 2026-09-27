# S1R — final deterministic singlet root-manifold validation (2026-09-26)

## Status

**Execution integrity: PASS.**  
**Deterministic root-manifold completeness: PASS.**  
**RPROP/preconditioned promoted stationary equivalence: PASS.**  
**S1R primary classification: PASS.**

S1R closes the spin-adapted singlet root-selection problem exposed by S1 and S1P.

The production rule is no longer “whatever canonical eigenvectors `np.linalg.eigh` happens to return”. Instead, a finite preregistered deterministic electronic-root seed ensemble is enumerated and the lowest strictly converged stationary branch is promoted.

## GitHub Actions provenance

Workflow: `s1r-deterministic-singlet-root-manifold`  
Run ID: `36260401478`  
Validated commit: `6e73becddf3d0c6e85ff97f1db5e4b099ae79ad8`  
Branch: `isotropic-polaron-barrier`  
Workflow conclusion: **success**

Production numerical environment:

- Python 3.12.14;
- NumPy 2.5.3;
- SciPy 1.18.1;
- OPENBLAS/OMP/MKL threads = 1.

The full repository pytest suite passed in the validation job.

## Design executed

Singlet manifold:

- 16 deterministic auxiliary orbital-root seeds;
- 3 structural seeds: onsite, bond_x, bond_y;
- 2 structural optimizers: preconditioned and RPROP;
- total = **96 independent singlet branches**.

Triplet control:

- root seed 0;
- 3 structural seeds;
- both structural optimizers;
- total = **6 triplet branches**.

Total branch JSON count: **102**.

The auxiliary root perturbation is used only to construct initial orbitals. It is absent from every physical Hamiltonian, energy and force evaluation.

## Completeness

Strictly valid singlets:

- preconditioned: **47/48**;
- RPROP: **48/48**.

The single invalid branch is:

- preconditioned / singlet / bond_x / root seed 15.

It completed but did not satisfy the strict convergence gate and was retained explicitly as invalid. The preregistered minimum requirement was 45/48 per optimizer.

Triplets:

- preconditioned: **3/3 valid**;
- RPROP: **3/3 valid**.

## Promoted singlet minima

Preconditioned promoted singlet:

- structural seed: onsite;
- root seed ID: 9;
- energy: **1.446798604442264 eV**;
- particle-number change: `4.55e-15`;
- final max update: `9.56e-9 A`;
- final max structural gradient: `2.88e-7 eV/A`.

RPROP promoted singlet:

- structural seed: onsite;
- root seed ID: 8;
- energy: **1.4467971903865933 eV**;
- particle-number change: `-1.22e-15`;
- final max update: `9.41e-9 A`;
- final max structural gradient: `4.33e-7 eV/A`.

Absolute optimizer difference:

`|E_S^P-E_S^R| = 1.41405567e-6 eV`.

This is comfortably below the locked `1e-5 eV` equivalence criterion.

## Promoted triplet minima

Preconditioned:

`E_T^P = 1.4467971868828626 eV`.

RPROP:

`E_T^R = 1.4467971868815632 eV`.

Absolute optimizer difference:

`1.2994e-12 eV`.

Thus the triplet remains optimizer-independent to essentially numerical precision.

## Singlet-triplet interpretation

The deterministic root-manifold search finds a singlet state nearly degenerate with the triplet in this validation control.

Using the promoted RPROP minima:

`E_S^R-E_T^R = 3.5050301e-9 eV`.

Using the promoted preconditioned minima:

`E_S^P-E_T^P = 1.4175594e-6 eV`.

These are control-model values, not material predictions.

Because the promoted minima from the two structural optimizers agree within the preregistered energy criterion while their tiny singlet-triplet differences are comparable to or below the residual optimizer/root-manifold scale, Paper 1 should describe the canonical S0 control as **near-degenerate at the promoted minimum** rather than emphasize a precise micro-eV singlet-triplet gap.

The historical `+1.231314 meV` value remains a valid historical environment-selected regression branch, but is superseded for production root promotion by the deterministic root-manifold rule.

## Energy landscape

The complete root ensemble reveals many distinct converged singlet stationary clusters. This confirms the central diagnosis from S1/S1P: the singlet surface is multi-root and strict convergence of a single arbitrary canonical seed is insufficient.

The promoted production result is therefore defined by a complete predetermined finite root search, not by seed label or library-dependent canonical eigenvector orientation.

## Locked gate summary

1. all 16 auxiliary seeds unique above `1e-12 eV`: **PASS**;
2. at least 45/48 preconditioned singlets valid: **PASS, 47/48**;
3. at least 45/48 RPROP singlets valid: **PASS, 48/48**;
4. all six triplet controls valid: **PASS**;
5. promoted singlet minima agree within `1e-5 eV`: **PASS, 1.414e-6 eV**;
6. promoted singlet particle-number changes <`1e-10`: **PASS**;
7. promoted triplet minima agree within `1e-5 eV`: **PASS**;
8. promoted states satisfy strict structural gates: **PASS**;
9. complete pytest suite: **PASS**;
10. production environment exactly pinned: **PASS**.

**S1R: PASS.**

## Production rule for Paper 1

For spin-adapted static Paper-1 calculations:

- pin Python/NumPy/SciPy and BLAS thread policy;
- use the deterministic root-seed construction;
- use multiple preregistered structural seeds;
- retain every valid stationary branch;
- promote the lowest strictly converged state within each spin sector;
- never use seed label as a physical-state label;
- never rely on the arbitrary orientation returned by unseeded `np.linalg.eigh` as the only cold-start root.

RPROP is now an explicitly validated structural option for the spin-adapted static control.

## Next stage

Proceed to **S2 — unified static-sector regression** covering representative strict one-polaron, bipolaron, spin-blind exciton and deterministic-root spin-adapted S/T controls under one pinned GitHub Actions provenance manifest.

S2 is the last integration gate before the Paper-1 parameter campaign and figure/data freeze.
