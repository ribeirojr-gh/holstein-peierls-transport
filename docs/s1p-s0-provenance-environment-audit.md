# S1P — spin-adapted S0 provenance/environment audit

Date: 2026-09-26  
Status: prospective diagnostic following the formally failed S1 optimizer-equivalence gate.

## Motivation

The historical S0 benchmark was produced at commit

`ac651b7bacd4e19a1bc8f2c7311354028c76b8dc`

by GitHub Actions run `33652971202`.

The historical logs establish:

- Python 3.12.14;
- NumPy 2.5.2;
- SciPy 1.18.1;
- no explicit OPENBLAS/OMP/MKL thread pinning in the workflow.

The S1 run `36256141958` used:

- Python 3.12.14;
- NumPy 2.5.3;
- SciPy 1.18.1;
- OPENBLAS/OMP/MKL threads fixed to 1.

The spin-adapted electronic initializer calls `np.linalg.eigh` and the singlet manifold is known to contain competing stationary roots. Therefore both a patch-level LAPACK/NumPy change and BLAS threading/order can alter the canonical basis chosen inside nearly degenerate subspaces, even when every subsequent optimizer converges strictly.

S1P is an environment/provenance diagnostic only. It does not redefine S1 and cannot promote RPROP.

## Fixed physical calculation

Use the original preconditioned S0 structural optimizer only.

For every environment evaluate:
- singlet onsite;
- singlet bond_x;
- singlet bond_y;
- triplet onsite as a stable reference.

Physical/numerical parameters remain:
- 4x4 isotropic periodic control;
- Jx=Jy=0.100 eV;
- alpha_intra=alpha_interx=alpha_intery=3.0 eV/A;
- K1=16.51 eV/A^2;
- K2=0.51 eV/A^2;
- U=0.525 eV;
- nearest-neighbour V=0.08 eV;
- checkerboard gap=2.0 eV;
- seed amplitude=1e-3 A;
- orbital tolerance=1e-8;
- structural gradient tolerance=1e-6 eV/A;
- structural update tolerance=1e-8 A;
- lattice max iterations=1200, matching the historical workflow.

## Environment matrix

Cross the following factors:

1. source code:
   - historical commit `ac651b7...`;
   - current `isotropic-polaron-barrier` workflow commit.

2. NumPy:
   - 2.5.2;
   - 2.5.3.

3. BLAS thread policy:
   - `historical_unpinned`: do not set OPENBLAS_NUM_THREADS, OMP_NUM_THREADS or MKL_NUM_THREADS in the job;
   - `single_thread`: set all three to 1.

SciPy is fixed to 1.18.1 and Python to 3.12.

This gives 8 environments x 4 branches = 32 independent jobs.

## Historical reference values

From the immutable benchmark document/run:

- singlet onsite: 1.448028500905174 eV;
- singlet bond_x: 1.458848887171766 eV;
- singlet bond_y: 1.458734984906944 eV;
- triplet onsite: 1.446797186881587 eV.

The historical promoted splitting was +1.231314023587 meV.

## Recorded provenance

Each job records:
- source ref and resolved SHA;
- Python/NumPy/SciPy versions;
- `np.show_config()`;
- relevant thread environment variables;
- CPU model and logical core count;
- complete branch JSON.

## Diagnostic thresholds

Define two calculations as belonging to a different **numerical root outcome** when their converged total referenced energies differ by more than `1e-5 eV`, the already preregistered S1 optimizer-equivalence scale.

Historical reproduction is considered exact for this audit when the four branch energies differ from the historical reference by at most `1e-8 eV`.

These thresholds diagnose provenance; they are not physical uncertainties.

## Attribution logic

Report, without changing any physical parameter:

- whether historical-code + NumPy 2.5.2 + unpinned threads reproduces the historical four values;
- NumPy sensitivity at fixed source/thread policy;
- thread-policy sensitivity at fixed source/NumPy;
- source-code sensitivity at fixed NumPy/thread policy.

A factor is labeled root-selecting only if changing that factor alone changes at least one singlet branch by >1e-5 eV while the compared jobs both converge.

If multiple factors change roots, report all of them. If none does, classify the discrepancy as unresolved environment/hardware sensitivity and proceed to explicit deterministic root-manifold seeding.

## Stop rule

S1P cannot rescue S1. Regardless of outcome, the next production-safe spin-adapted method must eliminate accidental dependence on library/thread canonical-basis choices by using an explicitly deterministic root enumeration/promotion rule.

The purpose of S1P is to identify the provenance mechanism before designing that rule.
