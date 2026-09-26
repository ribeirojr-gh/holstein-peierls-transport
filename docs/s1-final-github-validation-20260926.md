# S1 — final GitHub Actions validation and closure (2026-09-26)

## Status

**GitHub execution integrity: PASS.**  
**All 12 optimizer/spin/seed jobs: PASS.**  
**Locked S1 cross-optimizer equivalence criterion: FAIL.**

S1 therefore does **not** promote the new spin-adapted RPROP path as numerically equivalent to the accepted preconditioned structural optimizer under the preregistered `1e-5 eV` stationary-state tolerance.

No tolerance is changed after observing the result.

## GitHub Actions provenance

Workflow: `s1-spin-adapted-rprop-bridge`  
Run ID: `36256141958`  
Run attempt: `1`  
Validated commit: `97ebfc31ec6af6d047a554742fdb0aa02f79ff35`  
Branch: `isotropic-polaron-barrier`  
Workflow conclusion: **success**

Execution layout:
- 1 validation job;
- 12 independent branch jobs = 2 structural optimizers x 2 multiplicities x 3 seeds;
- 1 aggregate job.

All 14 jobs completed successfully.

Validation gates:
- pycompile: PASS;
- focused S1/spin-adapted tests: PASS;
- full repository test suite: **524 passed in 11.20 s**.

## Frozen S1 control

Both optimizers used the same canonical 4x4 spin-adapted control:

- `Jx = Jy = 0.100 eV`;
- `alpha_intra = alpha_interx = alpha_intery = 3.0 eV/A`;
- `K1 = 16.51 eV/A^2`;
- `K2 = 0.51 eV/A^2`;
- `U = 0.525 eV`;
- nearest-neighbour `V = 0.08 eV`;
- checkerboard gap `Delta = 2.0 eV`;
- half-filled neutral reference;
- seed amplitude `1e-3 A`;
- orbital-gradient tolerance `1e-8`;
- structural update tolerance `1e-8 A`;
- structural-gradient tolerance `1e-6 eV/A`.

The checkerboard gap and U/V values remain numerical validation controls, not material parameters.

## Convergence outcome

All six RPROP branches satisfy the strict electronic and structural convergence gates.

All six preconditioned branches also satisfy the same gates.

Particle-number conservation and finite-value gates pass for all RPROP branches.

Thus S1 is not a convergence failure. It is a **stationary-root / optimizer-equivalence failure in the singlet sector**.

## Promoted minima

### Singlet

Lowest converged RPROP branch:

- seed: `bond_y`;
- energy: **1.4468005126646775 eV**;
- iterations: 98;
- final max update: `6.57e-9 A`;
- final max structural gradient: `1.49e-8 eV/A`.

Lowest converged preconditioned branch:

- seed: `onsite`;
- energy: **1.4468214650197786 eV**;
- iterations: 268;
- final max update: `4.42e-9 A`;
- final max structural gradient: `2.77e-7 eV/A`.

Absolute promoted-singlet difference:

`|E_S^R - E_S^P| = 2.0952355101e-5 eV = 20.9524 micro-eV`.

Locked gate: `<= 1.0e-5 eV`.

**Result: FAIL.**

### Triplet

Lowest converged RPROP branch:

`E_T^R = 1.446797186881562 eV`.

Lowest converged preconditioned branch:

`E_T^P = 1.4467971868815868 eV`.

Difference:

`2.49e-14 eV`.

**Triplet equivalence gate: PASS to essentially machine precision.**

## Singlet-triplet splitting

RPROP promoted splitting:

`E_S^R - E_T^R = 3.3257831e-6 eV = 0.003326 meV`.

Preconditioned promoted splitting:

`E_S^P - E_T^P = 2.4278138e-5 eV = 0.024278 meV`.

Absolute difference:

`2.0952355e-5 eV`.

Locked spin-gap equivalence gate: `<=1e-5 eV`.

**Result: FAIL.**

## Branch-level root selection

The branch-by-branch singlet outcomes show that the two structural optimizers access different converged open-shell roots:

| Seed | RPROP energy (eV) | Preconditioned energy (eV) | Absolute difference |
| --- | ---: | ---: | ---: |
| onsite | 1.4474549540088157 | 1.4468214650197786 | 6.33489e-4 eV |
| bond_x | 1.4469485494925460 | 1.4588488871717660 | 1.19003e-2 eV |
| bond_y | 1.4468005126646775 | 1.4588050810956026 | 1.20046e-2 eV |

By contrast, the triplet branch energies agree at approximately `1e-12 eV` or better for corresponding seeds.

This pattern is consistent with the known singlet root-selection sensitivity of the open-shell control and is not evidence that the RPROP update is numerically unstable.

## Important regression finding versus the historical S0 benchmark

The historical benchmark document, introduced at commit

`ac651b7bacd4e19a1bc8f2c7311354028c76b8dc`

reported a promoted preconditioned singlet energy

`1.448028500905174 eV`

and promoted splitting

`E_S-E_T = +1.231314 meV`.

The current S1 run, using the same preconditioned structural path, instead obtains a lower promoted preconditioned singlet

`1.4468214650197786 eV`

with a much smaller splitting of about `0.024278 meV`.

The promoted preconditioned relaxation code itself has no intervening committed change between the historical benchmark and S1 apart from the additive S1 RPROP option. The cause of this changed singlet root is therefore **not established by S1** and requires a dedicated provenance/root-manifold audit. It may reflect deterministic orbital-root recovery/selection, numerical environment, or a state-selection pathway elsewhere in the spin-adapted electronic stack.

Until reconciled, the historical `+1.231314 meV` control splitting must be treated as a historical regression result rather than a current production Paper-1 value.

## Locked S1 gate summary

1. all six RPROP branches converged: **PASS**;
2. all six preconditioned branches converged: **PASS**;
3. promoted singlet energy agrees within `1e-5 eV`: **FAIL**;
4. promoted triplet energy agrees within `1e-5 eV`: **PASS**;
5. promoted spin gap agrees within `1e-5 eV`: **FAIL**;
6. RPROP `|Tr(Delta gamma)| < 1e-10`: **PASS**;
7. all final values finite: **PASS**;
8. full pytest suite: **PASS**.

**S1 primary classification: FAIL.**

## Scientific interpretation

The failed gate is scientifically useful. It shows that the spin-adapted singlet stationary surface contains multiple tightly competing open-shell roots whose selection depends on the structural/electronic optimization path. A single converged branch cannot yet be promoted as an optimizer-independent singlet result.

The triplet sector is substantially more robust and reproduces between optimizers to numerical precision.

## Consequence for Paper 1

Do **not** begin the production singlet/triplet parameter campaign yet.

Do not claim that all static spin-adapted results can be generated interchangeably with RPROP and the preconditioned optimizer.

The next required stage is a separately preregistered root-manifold/provenance audit that:
- reproduces the historical S0 commit/environment where possible;
- enumerates/restarts converged singlet electronic roots at matched geometries;
- cross-seeds RPROP and preconditioned stationary lattices into each other;
- distinguishes structural-basin differences from electronic-root differences;
- determines a deterministic stationary-root promotion rule that is independent of structural optimizer.

The T=0 one-polaron frozen baseline is unaffected.
