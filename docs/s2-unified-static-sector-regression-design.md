# S2 — unified static-sector regression preregistration

Date: 2026-09-27  
Status: prospective integration gate for Paper 1.

## Purpose

S2 is the final integration checkpoint before the Paper-1 static parameter campaign.

It does **not** discover new phase boundaries. It reruns one or more previously validated representative stationary controls from each static sector under one exactly pinned GitHub Actions environment and checks that:

- convergence conventions remain explicit;
- binding-energy signs remain consistent;
- state normalization/particle-number conventions remain intact;
- branch/root promotion rules are not silently changed;
- optimizer labels and physical Hilbert spaces remain separated.

## Pinned numerical environment

All S2 jobs use:

- Python 3.12.14;
- NumPy 2.5.3;
- SciPy 1.18.1;
- pytest 9.1.1;
- OPENBLAS_NUM_THREADS=1;
- OMP_NUM_THREADS=1;
- MKL_NUM_THREADS=1.

No dependency is installed from an unbounded version range in the production jobs.

## Sector A — one-polaron regression

Use the generic static reference shared with the validated exciton benchmark:

- 20x20 PBC;
- Jx=0.100 eV, Jy=0.015 eV;
- alpha_intra=3.0 eV/A;
- alpha_interx=alpha_intery=0.4 eV/A;
- K1=16.51 eV/A^2;
- centered legacy localization seed;
- sparse electronic solver;
- optimized O(N) gradient;
- modern all-coordinate RPROP stopping rule.

Reference from the established 20x20 exciton/static benchmark:

`E_pol = -0.405459057756 eV`.

S2 gates:
- all u/vx/vy RPROP coordinate families converged;
- recomputed final maximum structural-gradient component <1e-6 eV/A;
- charge-density sum differs from 1 by <1e-12;
- total energy within 1e-8 eV of the reference;
- IPR finite and >0.

This is a regression anchor, not a new polaron parameter fit.

## Sector B — correlated bipolaron regression

Use the strict 40x40 anisotropic intersite-x control already validated in the static bipolaron study:

- Jx=0.100 eV, Jy=0.015 eV;
- alpha_intra=3.0 eV/A;
- alpha_x=0.10 eV/A, alpha_y=0.12 eV/A;
- U=1.000 eV;
- V1=0;
- no long-range continuum tail in this S2 anchor;
- branch initialization: intersite_x;
- strict RPROP update criterion 1e-8 A;
- strict structural-gradient criterion 1e-6 eV/A;
- eigensolver tolerance 1e-11.

Validated reference energy:

`E_BP = -0.634241741090195 eV`.

S2 gates:
- strict branch converged;
- total energy within 1e-8 eV of the reference;
- binding convention is `E_bind = 2 E_polaron - E_BP`;
- binding is positive and >=5e-3 eV;
- P_NN,x >=0.80;
- mean pair separation <1.20 lattice sites;
- max |Delta t_x|/Jx <0.25 and max |Delta t_y|/Jy <0.25.

The bipolaron one-body RDM trace-2 convention and exchange symmetry remain covered by the full pytest suite.

## Sector C — spin-blind direct electron-hole exciton regression

Use the validated generic 20x20 electron-hole control:

- Jx=0.100 eV, Jy=0.015 eV;
- alpha_intra^e=alpha_intra^h=3.0 eV/A;
- alpha_inter^e=alpha_inter^h=0.4 eV/A;
- K1=16.51, K2=0.51 eV/A^2;
- onsite e-h attraction magnitude=0.525 eV;
- branch seeds: `frenkel` and `diagonal`;
- strict RPROP convergence.

Validated primary minimum reference:

`E_X = -1.695788941148 eV`.

S2 gates:
- both requested branches converge;
- canonical best branch is `frenkel`;
- best total energy within 1e-8 eV of the reference;
- binding convention is `E_bind^X = 2 E_polaron - E_X`;
- best binding >0.80 eV;
- best onsite probability >0.80;
- best mean e-h separation <0.30 lattice sites;
- the diagonal branch remains at least 0.30 eV above the Frenkel-like minimum.

This sector remains explicitly **spin blind** and must not be relabeled singlet/triplet.

## Sector D — spin-adapted deterministic-root regression anchors

S2 does not repeat the full 102-branch S1R discovery matrix. Instead it reruns the four previously promoted S1R branches as exact regression anchors under the same pinned environment:

1. preconditioned singlet: structural seed onsite, root ID 9;
2. RPROP singlet: structural seed onsite, root ID 8;
3. preconditioned triplet: structural seed bond_x, root ID 0;
4. RPROP triplet: structural seed onsite, root ID 0.

Reference energies:

- preconditioned singlet: 1.446798604442264 eV;
- RPROP singlet: 1.4467971903865933 eV;
- preconditioned triplet: 1.4467971868828626 eV;
- RPROP triplet: 1.4467971868815632 eV.

S2 gates:
- all four anchors strictly converge;
- each total energy agrees with its S1R reference within 1e-8 eV;
- particle-number change magnitude <1e-10;
- final update <1e-8 A;
- final structural gradient <1e-6 eV/A;
- repeated singlet optimizer difference remains <=1e-5 eV;
- repeated triplet optimizer difference remains <=1e-5 eV.

These anchors test reproducibility only. Paper-1 production root promotion still requires the complete predetermined deterministic root ensemble, not these four root IDs alone.

## Full-repository integration gates

Before any sector job is accepted:

- pycompile for all S2 drivers;
- full pytest suite passes;
- repository working tree is clean in GitHub Actions;
- exact dependency/thread provenance is recorded.

The aggregate S2 classifier then requires all sector gates to pass simultaneously.

## Locked S2 classification

**S2 PASS** only if:

1. validation/full pytest PASS;
2. one-polaron sector PASS;
3. bipolaron sector PASS;
4. spin-blind exciton sector PASS;
5. spin-adapted regression-anchor sector PASS;
6. all artifacts and exact branch labels are present;
7. all numerical values used in the aggregate are finite.

A green workflow is not by itself S2 PASS; the aggregate scientific/integration classifier is authoritative.

## Stop rule

If S2 passes:
- freeze the static integration baseline;
- begin S3 Paper-1 parameter campaign;
- all Paper-1 figures/data must descend from a single parameter manifest and immutable data artifacts.

If S2 fails:
- do not relax thresholds;
- isolate the failing sector;
- preserve all other sector results;
- repair/revalidate only the failed integration path before rerunning S2.
